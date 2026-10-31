// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Raon-1-mini pipeline bands on the generic mesh executor.
//!
//! The model is split into layer bands ([`crate::band`]), one band bundle per
//! rank on its own device. The executor (`Gpus::submit_mesh_epoch`) sequences
//! the ranks; this module declares what it needs to do so:
//!
//! * [`raon_step_graph`]: a coarse [`StepGraph`] of the trunk program. The only
//!   resource live across a layer boundary is the mHC streams
//!   ([`RAON_STREAMS_RESOURCE`], `[mhc_streams, hidden]` per row), so a cut
//!   between two layers is exactly one `BandXfer` of that resource.
//! * [`raon_manifest`]: the real weight and state declarations (names and
//!   shapes of [`crate::weights::tensor_inventory`], recurrent / conv / latent
//!   state per layer) the splitter places per band.
//! * [`RaonMeshProgram`]: one [`SplitPlan`] per output mode (trunk only for a
//!   prefill chunk that needs no logits, trunk plus head otherwise).
//! * [`RaonMeshBindings`]: the [`MeshStepBindings`] over the per-rank band
//!   bundles. `run_ops` enqueues the rank's band body on its own stream,
//!   `boundary` is the band's [`edge_view`](crate::RaonBundle::edge_view)
//!   (export at band end, import at band start).
//! * [`RaonMeshModel`]: the driver that owns the band bundles and runs
//!   prefill / decode through the executor.
//!
//! # Host and device phases
//!
//! The host work of a band call ([`RaonBundle::mesh_admit`]: context
//! map-ahead, token staging) happens for every rank before the epoch is
//! submitted. Inside the epoch the callbacks only enqueue on the owner's active
//! stream ([`RaonBundle::mesh_enqueue`]). The host mirror of each slot advances
//! in `apply_rank_marks`. One epoch is one prefill chunk (or one decode step)
//! and is drained before the next, because every band owns exactly one set of
//! scratch rows (the edge rows are `scratch.stream_a`).
//!
//! # Refused on a mesh
//!
//! DFlash/speculation, rollback, prefix repair, session snapshots and window
//! checkpoints need the whole model on one device ([`crate::forward`]'s
//! `require_whole_band`); the lifecycle callbacks refuse with a named error and
//! [`MeshStepBindings::admits_prefix_repair`] is `false`.

use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

use hipfire_dispatch::pipeline::step_graph::{
    OpDecl, OpId, OpKind, ResourceClass, ResourceDecl, ResourceId, SplitRule, StateClass, StepGraph,
};
use hipfire_runtime::device_mesh::{DeviceMesh, MeshEpoch};
use hipfire_runtime::mesh_placement::BandPlacement;
use hipfire_runtime::mesh_split::{
    split_program, CollectiveKind, PlannedCollective, RankItem, SplitPlan,
};
use hipfire_runtime::multi_gpu::executor::{
    BoundaryIo, ForwardEpoch, MeshCheckpointId, MeshCheckpointKey, MeshRoute, MeshStepBindings,
};
use hipfire_runtime::multi_gpu::transport::{EdgeDescriptor, EdgeId};
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::weight_manifest::{
    plan_manifest, PinTarget, ShardPolicy, StateEntry, StateKind, WeightEntry,
};
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::band::edge_bytes;
use crate::bundle::{DecodeMode, PrefillMode, RaonBundle};
use crate::config::{FfnKind, LayerKind, RaonConfig};
use crate::forward::BandInput;
use crate::weights::{tensor_inventory, TensorClass};

type Result<T, E = String> = std::result::Result<T, E>;

/// The mHC streams `[mhc_streams, hidden]` per row: the only resource live
/// across a layer boundary, hence the operand of every pipeline `BandXfer`.
/// Always the first resource of [`raon_step_graph`].
pub const RAON_STREAMS_RESOURCE: ResourceId = ResourceId(0);

/// The host-fed token ids (input of the embed op).
pub const RAON_TOKENS_RESOURCE: ResourceId = ResourceId(1);

/// The logits (output of the head op).
pub const RAON_LOGITS_RESOURCE: ResourceId = ResourceId(2);

/// Numerical policy identity pinned for the lifetime of one mesh load: the
/// executor refuses an epoch whose policy changed. Raon has one route (the
/// batched prefill and the band-local decode, bit-identical to the whole model).
pub const RAON_MESH_POLICY_ID: u64 = 0x5241_4f4e_0001;

static NEXT_LOAD_ID: AtomicU64 = AtomicU64::new(1);

fn step_graph_error(error: hipfire_dispatch::pipeline::step_graph::StepGraphError) -> String {
    format!("Raon step graph: {error}")
}

/// What a mesh epoch leaves behind: the trunk only (a prefill chunk that needs
/// no logits), or the trunk and the head (the last prefill chunk, every decode
/// step).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RaonMeshOutput {
    /// Layers only: the last band computes no logits.
    Trunk,
    /// Layers, final norm and `lm_head` of the last row on the last band.
    Logits,
}

impl RaonMeshOutput {
    const ALL: [RaonMeshOutput; 2] = [RaonMeshOutput::Trunk, RaonMeshOutput::Logits];

    fn index(self) -> usize {
        match self {
            RaonMeshOutput::Trunk => 0,
            RaonMeshOutput::Logits => 1,
        }
    }

    fn head(self) -> bool {
        self == RaonMeshOutput::Logits
    }
}

/// Pure declaration of the trunk program the band forward runs: one input op
/// (token embedding plus mHC stream init, attributed to layer 0), per layer
/// two ops (the mixer site: RDA `GatedDeltaNet` or MLA `Attend`; the FFN site:
/// `GatedMlp` or `Moe`; each includes its mHC read and write) and, when `head`,
/// one head op (final norm and `lm_head`, attributed to the last layer).
///
/// Every op carries `layer: Some(_)` so its stage never depends on issue order:
/// the embed op is pinned to the band owning layer 0 and the head op to the band
/// owning the last layer. Every mixer op reads and writes the layer's declared
/// state resources (recurrent + conv for RDA, the latent cache for MLA), which
/// the splitter checks against the manifest's state placement.
pub fn raon_step_graph(cfg: &RaonConfig, head: bool) -> Result<StepGraph> {
    let layers = cfg.n_layers;
    if layers == 0 || cfg.layer_kinds.len() != layers || cfg.ffn_kinds.len() != layers {
        return Err(format!(
            "Raon step graph: {layers} layers with {} mixer and {} ffn kinds",
            cfg.layer_kinds.len(),
            cfg.ffn_kinds.len()
        ));
    }
    let dims = cfg.state_dims();
    let mut graph = StepGraph::new();
    let mut resource = |class: ResourceClass, shape: &[usize]| {
        graph
            .add_resource(ResourceDecl {
                class,
                shape: shape.to_vec(),
                alias: None,
            })
            .map_err(step_graph_error)
    };
    let streams = resource(ResourceClass::Activation, &[cfg.mhc_streams, cfg.hidden])?;
    debug_assert_eq!(streams, RAON_STREAMS_RESOURCE);
    let tokens = resource(ResourceClass::Input, &[1])?;
    debug_assert_eq!(tokens, RAON_TOKENS_RESOURCE);
    let logits = resource(ResourceClass::Output, &[cfg.vocab])?;
    debug_assert_eq!(logits, RAON_LOGITS_RESOURCE);
    let mut layer_state: Vec<Vec<ResourceId>> = Vec::with_capacity(layers);
    for (layer, kind) in cfg.layer_kinds.iter().copied().enumerate() {
        let state = match kind {
            LayerKind::Rda => vec![
                resource(
                    ResourceClass::State {
                        class: StateClass::Recurrent,
                        layer,
                    },
                    &[dims.heads, dims.k_dim, dims.v_dim],
                )?,
                resource(
                    ResourceClass::State {
                        class: StateClass::Conv,
                        layer,
                    },
                    &[dims.conv_channels, dims.conv_hist],
                )?,
            ],
            LayerKind::Mla => vec![resource(
                ResourceClass::State {
                    class: StateClass::Kv,
                    layer,
                },
                &[dims.latent + dims.tail],
            )?],
        };
        layer_state.push(state);
    }

    let mut ops: Vec<OpDecl> = Vec::with_capacity(2 * layers + 2);
    let mut op = |kind: OpKind, layer: usize, reads: Vec<ResourceId>, writes: Vec<ResourceId>| {
        ops.push(OpDecl {
            kind,
            layer: Some(layer),
            rule: SplitRule::Opaque,
            reads,
            writes,
        });
    };
    op(OpKind::Embed, 0, vec![tokens], vec![streams]);
    for (layer, kind) in cfg.layer_kinds.iter().copied().enumerate() {
        let mut reads = vec![streams];
        reads.extend(&layer_state[layer]);
        let mut writes = vec![streams];
        writes.extend(&layer_state[layer]);
        op(
            match kind {
                LayerKind::Rda => OpKind::GatedDeltaNet,
                LayerKind::Mla => OpKind::Attend,
            },
            layer,
            reads,
            writes,
        );
        op(
            match cfg.ffn_kinds[layer] {
                FfnKind::Dense => OpKind::GatedMlp,
                FfnKind::Moe => OpKind::Moe,
            },
            layer,
            vec![streams],
            vec![streams],
        );
    }
    if head {
        op(OpKind::Project, layers - 1, vec![streams], vec![logits]);
    }
    for decl in ops {
        graph.add_op(decl).map_err(step_graph_error)?;
    }
    Ok(graph)
}

/// Ops of `graph` (a [`raon_step_graph`]) a rank owning the global `layers`
/// runs, in issue order: every op attributed to a layer of the band. The embed
/// op (layer 0) is therefore only on the first band and the head op (last
/// layer) only on the last.
pub fn raon_rank_op_ids(graph: &StepGraph, layers: Range<usize>) -> Vec<OpId> {
    graph
        .ops()
        .iter()
        .enumerate()
        .filter(|(_, op)| op.layer.is_some_and(|layer| layers.contains(&layer)))
        .map(|(index, _)| OpId(index as u32))
        .collect()
}

/// The layer number of a `model.layers.N.*` tensor name.
fn tensor_layer(name: &str) -> Option<usize> {
    name.strip_prefix("model.layers.")?
        .split('.')
        .next()?
        .parse()
        .ok()
}

/// The real declarations of the model: every tensor of the checkpoint
/// inventory ([`tensor_inventory`]: original names, stored shapes) with the
/// placement its role implies (the embedding on stage 0, `model.norm` and
/// `lm_head` on the last stage, every layer tensor on the stage owning its
/// layer), and the per-layer state (recurrent + conv per RDA layer, the latent
/// cache per MLA layer). Weight dtypes are the architecture's logical
/// expectation; the loader keeps each source dtype.
pub fn raon_manifest(cfg: &RaonConfig) -> (Vec<WeightEntry>, Vec<StateEntry>) {
    let weights = tensor_inventory(cfg)
        .into_iter()
        .map(|spec| {
            let dtype = match spec.class {
                TensorClass::F32Vec | TensorClass::HostF32 => DType::F32,
                _ => DType::BF16,
            };
            let shape = if spec.shape.is_empty() {
                vec![1]
            } else {
                spec.shape.clone()
            };
            let policy = match spec.name.as_str() {
                "model.embed_tokens.weight" => ShardPolicy::Pin(PinTarget::Embed),
                "model.norm.weight" | "lm_head.weight" => ShardPolicy::Pin(PinTarget::Output),
                _ => ShardPolicy::Replicate,
            };
            match tensor_layer(&spec.name) {
                Some(layer) => WeightEntry::layer(spec.name, layer, shape, dtype, policy),
                None => WeightEntry::model(spec.name, shape, dtype, policy),
            }
        })
        .collect();
    let mut state = Vec::new();
    for (layer, kind) in cfg.layer_kinds.iter().copied().enumerate() {
        match kind {
            LayerKind::Rda => {
                state.push(StateEntry::new(StateKind::Recurrent, layer));
                state.push(StateEntry::new(StateKind::Conv, layer));
            }
            LayerKind::Mla => state.push(StateEntry::new(
                StateKind::Kv {
                    quant: "latent".to_string(),
                },
                layer,
            )),
        }
    }
    (weights, state)
}

/// One output mode's compiled program: the graph, its split and each rank's
/// ops in order.
struct ModeProgram {
    plan: SplitPlan,
    rank_ops: Vec<Vec<OpId>>,
}

/// The compiled mesh program of one model on one band placement: a split plan
/// per [`RaonMeshOutput`]. Pure (no `Gpu`).
pub struct RaonMeshProgram {
    mesh_epoch: MeshEpoch,
    placement: BandPlacement,
    modes: [ModeProgram; 2],
}

impl RaonMeshProgram {
    /// Compile the program of `whole` (the whole model's configuration) for
    /// `mesh` and `placement`: manifest against the placement, then the graph
    /// split per output mode. Each rank's op list is the splitter's, and must
    /// equal the layer-attributed [`raon_rank_op_ids`] of its band.
    pub fn build(whole: &RaonConfig, mesh: &DeviceMesh, placement: &BandPlacement) -> Result<Self> {
        let (weights, state) = raon_manifest(whole);
        let manifest = plan_manifest(&weights, &state, mesh, whole.n_layers, placement)?;
        let mut modes = Vec::with_capacity(2);
        for output in RaonMeshOutput::ALL {
            let graph = raon_step_graph(whole, output.head())?;
            let plan = split_program(&graph, &manifest, mesh)
                .map_err(|error| format!("Raon mesh split: {error}"))?;
            if plan.ranks.len() != placement.stages() {
                return Err(format!(
                    "Raon mesh split has {} ranks for {} bands",
                    plan.ranks.len(),
                    placement.stages()
                ));
            }
            let rank_ops: Vec<Vec<OpId>> = plan
                .ranks
                .iter()
                .map(|rank| {
                    rank.items
                        .iter()
                        .filter_map(|item| match item {
                            RankItem::Op(id) => Some(*id),
                            RankItem::Collective(_) => None,
                        })
                        .collect()
                })
                .collect();
            for (stage, ops) in rank_ops.iter().enumerate() {
                if *ops != raon_rank_op_ids(&graph, placement.band(stage)) {
                    return Err(format!(
                        "Raon mesh split disagrees with the layer attribution on rank {stage}"
                    ));
                }
            }
            modes.push(ModeProgram { plan, rank_ops });
        }
        let modes: [ModeProgram; 2] = modes
            .try_into()
            .map_err(|_| "Raon mesh program needs one plan per output mode".to_string())?;
        Ok(Self {
            mesh_epoch: mesh.epoch(),
            placement: placement.clone(),
            modes,
        })
    }

    /// The split plan of `output`.
    pub fn plan(&self, output: RaonMeshOutput) -> &SplitPlan {
        &self.modes[output.index()].plan
    }

    /// Rank `rank`'s ops of `output`, in order.
    pub fn rank_ops(&self, output: RaonMeshOutput, rank: usize) -> &[OpId] {
        &self.modes[output.index()].rank_ops[rank]
    }

    /// The band placement the program was compiled for.
    pub fn placement(&self) -> &BandPlacement {
        &self.placement
    }

    /// The mesh generation the program was compiled for.
    pub fn mesh_epoch(&self) -> MeshEpoch {
        self.mesh_epoch
    }
}

/// Bytes `Gpus::provision_mesh_edges` must reserve per edge for epochs of up
/// to `chunk_rows` rows of `cfg`'s mHC streams (F32): the largest payload a
/// prefill chunk hands to the next band.
pub fn raon_edge_capacity_bytes(cfg: &RaonConfig, chunk_rows: usize) -> Result<usize> {
    if chunk_rows == 0 {
        return Err("Raon mesh edge of zero rows".into());
    }
    let bytes = chunk_rows
        .checked_mul(cfg.mhc_streams)
        .and_then(|n| n.checked_mul(cfg.hidden))
        .and_then(|n| n.checked_mul(4))
        .ok_or("Raon mesh edge capacity overflows")?;
    if bytes % 16 != 0 || bytes / 16 > i32::MAX as usize {
        return Err(format!(
            "Raon mesh edge capacity {bytes} B is not a 16-byte multiple within the transport limit"
        ));
    }
    Ok(bytes)
}

const NO_MESH: &str = "Raon mesh bands do not run DFlash/speculation, rollback, prefix repair, \
session snapshots or window checkpoints: those need the whole model on one device";

/// An epoch adapter over the per-rank band bundles (rank `r` on device `r`,
/// band `r` of the placement). Borrows the bundles for the submission and
/// never owns a `Gpu`.
pub struct RaonMeshBindings<'a> {
    bands: &'a mut [RaonBundle],
    program: &'a RaonMeshProgram,
    tokens: &'a [u32],
    output: RaonMeshOutput,
    graph_decode: bool,
    edge_epochs: Vec<u64>,
    views: Vec<Option<GpuTensor>>,
    prepared: Option<ForwardEpoch>,
    ran: Vec<bool>,
}

impl<'a> RaonMeshBindings<'a> {
    /// Bindings over `bands` (one per rank); the epoch's inputs are set with
    /// [`set_epoch`](Self::set_epoch) and validated by `prepare_epoch`.
    pub fn new(bands: &'a mut [RaonBundle], program: &'a RaonMeshProgram) -> Self {
        let ranks = bands.len();
        Self {
            bands,
            program,
            tokens: &[],
            output: RaonMeshOutput::Trunk,
            graph_decode: false,
            edge_epochs: Vec::new(),
            views: (0..ranks).map(|_| None).collect(),
            prepared: None,
            ran: vec![false; ranks],
        }
    }

    /// The next epoch's host inputs: the token ids (the rows), what the last
    /// band leaves, whether the decode step takes the per-band graph route, and
    /// the transport epochs of the edges `0->1, 1->2, ...`.
    pub fn set_epoch(
        &mut self,
        tokens: &'a [u32],
        output: RaonMeshOutput,
        graph_decode: bool,
        edge_epochs: Vec<u64>,
    ) {
        self.tokens = tokens;
        self.output = output;
        self.graph_decode = graph_decode;
        self.edge_epochs = edge_epochs;
        self.prepared = None;
    }

    fn require_prepared(&self, epoch: &ForwardEpoch) -> Result<()> {
        if self.prepared != Some(*epoch) {
            return Err("Raon mesh epoch has not been prepared".into());
        }
        Ok(())
    }
}

impl MeshStepBindings for RaonMeshBindings<'_> {
    fn prepare_epoch(&mut self, epoch: &ForwardEpoch) -> Result<()> {
        self.prepared = None;
        let ranks = self.bands.len();
        let rows = epoch.actual_rows;
        if ranks < 2
            || ranks != self.program.placement.stages()
            || self.edge_epochs.len() + 1 != ranks
            || self.edge_epochs.contains(&0)
        {
            return Err(
                "Raon mesh needs one band per rank and one transport epoch per edge".into(),
            );
        }
        if rows == 0
            || epoch.route_rows != rows
            || self.tokens.len() != rows
            || epoch.route == MeshRoute::Verify
            || (epoch.route == MeshRoute::Decode && rows != 1)
            || epoch.mesh_epoch != self.program.mesh_epoch
        {
            return Err("invalid Raon mesh rows, route, tokens or mesh epoch".into());
        }
        if self.graph_decode && epoch.route != MeshRoute::Decode {
            return Err("Raon mesh graph route is a decode step only".into());
        }
        let end = epoch
            .start_position
            .checked_add(rows)
            .ok_or("Raon mesh position overflows")?;
        let vocab = self.bands[0].config.vocab;
        if self.tokens.iter().any(|&token| token as usize >= vocab) {
            return Err("Raon mesh token is outside the vocabulary".into());
        }
        for (rank, band) in self.bands.iter().enumerate() {
            let layers = &band.band().layers;
            if *layers != self.program.placement.band(rank) {
                return Err(format!(
                    "Raon mesh rank {rank} holds layers {layers:?}, the placement says {:?}",
                    self.program.placement.band(rank)
                ));
            }
            if band.n_tokens() != epoch.start_position
                || end > band.max_seq()
                || rows > band.prefill_chunk_rows()
            {
                return Err(format!(
                    "Raon mesh rank {rank} position {} / capacity disagrees with the epoch \
                     (start {}, {rows} rows, context {}, chunk {})",
                    band.n_tokens(),
                    epoch.start_position,
                    band.max_seq(),
                    band.prefill_chunk_rows()
                ));
            }
            self.views[rank] = Some(band.edge_view(rows)?);
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
        if owner >= self.bands.len()
            || self.ran[owner]
            || ops != self.program.rank_ops(self.output, owner)
        {
            return Err(
                "Raon mesh band must run exactly once with its full declared op list".into(),
            );
        }
        let last = owner + 1 == self.bands.len();
        let want_logits = last && self.output.head();
        let token = (owner == 0 && self.graph_decode).then(|| self.tokens[0]);
        self.bands[owner].mesh_enqueue(
            gpu,
            epoch.actual_rows,
            token,
            want_logits,
            self.graph_decode,
        )?;
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
            return Err("Raon mesh only admits mHC stream band transfers".into());
        };
        if src.checked_add(1) != Some(dst)
            || dst >= self.bands.len()
            || (owner != src && owner != dst)
            || collective.operand != RAON_STREAMS_RESOURCE
        {
            return Err("Raon mesh boundary has invalid endpoints or operand".into());
        }
        let view = self
            .views
            .get(owner)
            .and_then(Option::as_ref)
            .ok_or("Raon mesh edge view is not prepared")?;
        Ok(BoundaryIo {
            view,
            into: None,
            descriptor: EdgeDescriptor {
                epoch: *self
                    .edge_epochs
                    .get(src)
                    .ok_or("missing Raon mesh edge epoch")?,
                op: epoch.op_id,
                rows: epoch.actual_rows,
                dtype_bytes: 4,
                bytes: edge_bytes(&self.bands[owner].config, epoch.actual_rows),
            },
            partial_slot: None,
        })
    }

    fn apply_rank_marks(&mut self, owner: usize, epoch: &ForwardEpoch) -> Result<()> {
        self.require_prepared(epoch)?;
        if !self.ran.get(owner).copied().unwrap_or(false) {
            return Err("Raon mesh band marks before its ops ran".into());
        }
        let band = &mut self.bands[owner];
        band.mesh_commit(epoch.actual_rows);
        if band.n_tokens() != epoch.start_position + epoch.actual_rows {
            return Err("Raon mesh band did not advance by the epoch's rows".into());
        }
        Ok(())
    }

    fn admits_prefix_repair(&self) -> bool {
        false
    }

    fn repair_prefix_rank(
        &mut self,
        _owner: usize,
        _gpu: &mut Gpu,
        _epoch: &ForwardEpoch,
        _keep: usize,
        _tokens: &[u32],
    ) -> Result<()> {
        Err(format!("prefix repair: {NO_MESH}"))
    }

    fn capture_session_rank(
        &mut self,
        _owner: usize,
        _gpu: &mut Gpu,
        _epoch: &ForwardEpoch,
    ) -> Result<MeshCheckpointId> {
        Err(format!("session capture: {NO_MESH}"))
    }

    fn probe_session_rank(
        &mut self,
        _owner: usize,
        _epoch: &ForwardEpoch,
        _piece: MeshCheckpointId,
    ) -> Result<()> {
        Err(format!("session restore: {NO_MESH}"))
    }

    fn restore_session_rank(
        &mut self,
        _owner: usize,
        _gpu: &mut Gpu,
        _epoch: &ForwardEpoch,
        _checkpoint: MeshCheckpointId,
    ) -> Result<()> {
        Err(format!("session restore: {NO_MESH}"))
    }

    fn reset_rank(&mut self, owner: usize, gpu: &mut Gpu, _epoch: &ForwardEpoch) -> Result<()> {
        self.prepared = None;
        self.bands
            .get_mut(owner)
            .ok_or("unknown Raon mesh rank")?
            .reset(gpu)
    }

    fn checkpoint_rank(
        &mut self,
        _owner: usize,
        _gpu: &mut Gpu,
        _key: &MeshCheckpointKey,
    ) -> Result<MeshCheckpointId> {
        Err(format!("checkpoint: {NO_MESH}"))
    }

    fn restore_rank(
        &mut self,
        _owner: usize,
        _gpu: &mut Gpu,
        _key: &MeshCheckpointKey,
        _ticket: MeshCheckpointId,
    ) -> Result<()> {
        Err(format!("checkpoint restore: {NO_MESH}"))
    }

    fn validate_finalize_rank(
        &mut self,
        _owner: usize,
        _key: &MeshCheckpointKey,
        _ticket: MeshCheckpointId,
    ) -> Result<()> {
        Err(format!("checkpoint finalize: {NO_MESH}"))
    }

    fn finalize_rank(
        &mut self,
        _owner: usize,
        _key: &MeshCheckpointKey,
        _ticket: MeshCheckpointId,
    ) {
        // Unreachable: `checkpoint_rank` never issues a ticket, and
        // `validate_finalize_rank` refuses before any finalize runs.
    }
}

/// Raon pipeline bands driven through the generic mesh executor: owns the band
/// bundles (rank `r` on `gpus.devices[r]`) and the compiled program.
///
/// The caller owns the [`Gpus`]: one `Gpu` per admitted rank in logical rank
/// order with a launch stream each (`Gpu::ensure_capture_stream`),
/// `gpus.mesh` the admitted mesh, and the edges provisioned with
/// `Gpus::provision_mesh_edges(admitted, placement,
/// `[`RaonMeshModel::edge_capacity_bytes`]`)`.
pub struct RaonMeshModel {
    bands: Vec<RaonBundle>,
    program: RaonMeshProgram,
    load_id: u64,
    request_id: u64,
    next_op: u64,
    chunk_rows: usize,
}

impl RaonMeshModel {
    /// Edge capacity (bytes) to hand `Gpus::provision_mesh_edges` for bands
    /// prefilled in `chunk_rows`-row chunks of `whole`.
    pub fn edge_capacity_bytes(whole: &RaonConfig, chunk_rows: usize) -> Result<usize> {
        raon_edge_capacity_bytes(whole, chunk_rows)
    }

    /// Assemble the model from the loaded band bundles: `bands[r]` is rank
    /// `r`'s bundle (loaded on `gpus.devices[r]` with
    /// [`load_raon_band_bundle`](crate::load_raon_band_bundle)) covering
    /// `placement.band(r)`. Pure: touches no device. On error the bundles are
    /// handed back to the caller through the error path (they are dropped
    /// un-freed only if the caller discards them: call [`free`](Self::free) on
    /// a successfully built model, or free the bundles yourself).
    pub fn new(
        whole: &RaonConfig,
        mesh: &DeviceMesh,
        placement: &BandPlacement,
        bands: Vec<RaonBundle>,
    ) -> std::result::Result<Self, (String, Vec<RaonBundle>)> {
        match Self::validate(whole, &bands, placement) {
            Ok(chunk_rows) => match RaonMeshProgram::build(whole, mesh, placement) {
                Ok(program) => {
                    let load_id = NEXT_LOAD_ID.fetch_add(1, Ordering::Relaxed);
                    Ok(Self {
                        bands,
                        program,
                        load_id,
                        request_id: 1,
                        next_op: 1,
                        chunk_rows,
                    })
                }
                Err(error) => Err((error, bands)),
            },
            Err(error) => Err((error, bands)),
        }
    }

    fn validate(
        whole: &RaonConfig,
        bands: &[RaonBundle],
        placement: &BandPlacement,
    ) -> Result<usize> {
        if bands.len() < 2 || bands.len() != placement.stages() {
            return Err(format!(
                "Raon mesh needs one band bundle per rank (2 or more): {} bundles for {} stages",
                bands.len(),
                placement.stages()
            ));
        }
        if placement.n_trunk_layers != whole.n_layers {
            return Err(format!(
                "Raon mesh placement covers {} layers of a {}-layer model",
                placement.n_trunk_layers, whole.n_layers
            ));
        }
        let chunk_rows = bands[0].prefill_chunk_rows();
        for (rank, band) in bands.iter().enumerate() {
            let layers = &band.band().layers;
            if *layers != placement.band(rank) || band.band().total != whole.n_layers {
                return Err(format!(
                    "Raon mesh rank {rank} holds {} but the placement says layers {:?}",
                    band.band().label(),
                    placement.band(rank)
                ));
            }
            if band.prefill_mode() != PrefillMode::Batched {
                return Err(format!(
                    "Raon mesh rank {rank}: bands run the batched prefill route only"
                ));
            }
            if band.prefill_chunk_rows() != chunk_rows || band.max_seq() != bands[0].max_seq() {
                return Err(
                    "Raon mesh bands disagree on the prefill chunk rows or reserved context".into(),
                );
            }
            if band.decode_mode() != bands[0].decode_mode() {
                return Err("Raon mesh bands disagree on the decode route".into());
            }
        }
        Ok(chunk_rows)
    }

    /// Rows of one prefill chunk (and so the largest epoch).
    pub fn prefill_chunk_rows(&self) -> usize {
        self.chunk_rows
    }

    /// The compiled program.
    pub fn program(&self) -> &RaonMeshProgram {
        &self.program
    }

    /// The band bundles in rank order.
    pub fn bands(&self) -> &[RaonBundle] {
        &self.bands
    }

    /// Tokens consumed so far (every band's slot agrees).
    pub fn n_tokens(&self) -> usize {
        self.bands[0].n_tokens()
    }

    /// Context rows reserved by every band.
    pub fn max_seq(&self) -> usize {
        self.bands[0].max_seq()
    }

    /// Set the decode route of every band (`Graph`: each rank records and
    /// replays its own decode graph on its own `Gpu` inside the epoch;
    /// `Eager`: per-launch dispatch).
    pub fn set_decode_mode(&mut self, mode: DecodeMode) {
        for band in &mut self.bands {
            band.set_decode_mode(mode);
        }
    }

    /// The decode route in effect (`"graph"` / `"eager"`).
    pub fn decode_mode(&self) -> &'static str {
        self.bands[0].decode_mode()
    }

    /// `(captures, replays)` of the decode graph of each rank.
    pub fn decode_graph_stats(&self) -> Vec<(u64, u64)> {
        self.bands.iter().map(|b| b.decode_graph_stats()).collect()
    }

    fn check_gpus(&self, gpus: &Gpus) -> Result<()> {
        if gpus.devices.len() != self.bands.len() {
            return Err(format!(
                "Raon mesh has {} bands but {} devices",
                self.bands.len(),
                gpus.devices.len()
            ));
        }
        if gpus.mesh.epoch() != self.program.mesh_epoch {
            return Err(
                "Raon mesh program was compiled for another mesh than gpus.mesh (set \
                 gpus.mesh to the admitted mesh before provisioning)"
                    .into(),
            );
        }
        Ok(())
    }

    fn edge_epochs(&self, gpus: &Gpus) -> Result<Vec<u64>> {
        (0..self.bands.len() - 1)
            .map(|src| {
                gpus.mesh_edge_epoch(EdgeId {
                    src_device: src,
                    dst_device: src + 1,
                })
                .ok_or_else(|| format!("Raon mesh edge {src}->{} is not provisioned", src + 1))
            })
            .collect()
    }

    /// Start a cold request: drain the mesh, then reset every band's slot on
    /// its own device ([`Gpus::reset_mesh`]). The decode graphs stay valid
    /// (addresses and extents are unchanged).
    pub fn reset(&mut self, gpus: &mut Gpus) -> Result<()> {
        self.check_gpus(gpus)?;
        let epoch = ForwardEpoch {
            mesh_epoch: self.program.mesh_epoch,
            load_id: self.load_id,
            request_id: self.request_id,
            op_id: self.next_op,
            route: MeshRoute::Prefill,
            start_position: self.n_tokens(),
            actual_rows: 0,
            route_rows: 0,
            policy_id: RAON_MESH_POLICY_ID,
        };
        let mut bindings = RaonMeshBindings::new(&mut self.bands, &self.program);
        gpus.reset_mesh(&mut bindings, &epoch)?;
        self.request_id = self
            .request_id
            .checked_add(1)
            .ok_or("Raon mesh request ids are exhausted")?;
        Ok(())
    }

    /// One drained epoch over `tokens` (one prefill chunk or one decode step).
    fn run_epoch(
        &mut self,
        gpus: &mut Gpus,
        tokens: &[u32],
        route: MeshRoute,
        output: RaonMeshOutput,
    ) -> Result<()> {
        let rows = tokens.len();
        let graph_decode = route == MeshRoute::Decode && self.bands[0].decode_mode() == "graph";
        // Host phase of every rank, before anything is enqueued.
        for (rank, band) in self.bands.iter_mut().enumerate() {
            let gpu = &mut gpus.devices[rank];
            gpu.bind_thread().map_err(|e| e.to_string())?;
            let input = if rank == 0 {
                BandInput::Tokens(tokens)
            } else {
                BandInput::Streams { rows }
            };
            band.mesh_admit(gpu, input, graph_decode)?;
        }
        let op_id = self.next_op;
        self.next_op = op_id
            .checked_add(1)
            .ok_or("Raon mesh operation ids are exhausted")?;
        let epoch = ForwardEpoch {
            mesh_epoch: self.program.mesh_epoch,
            load_id: self.load_id,
            request_id: self.request_id,
            op_id,
            route,
            start_position: self.bands[0].n_tokens(),
            actual_rows: rows,
            route_rows: rows,
            policy_id: RAON_MESH_POLICY_ID,
        };
        let edge_epochs = self.edge_epochs(gpus)?;
        let mut bindings = RaonMeshBindings::new(&mut self.bands, &self.program);
        bindings.set_epoch(tokens, output, graph_decode, edge_epochs);
        let ticket = gpus.submit_mesh_epoch(self.program.plan(output), &epoch, &mut bindings)?;
        gpus.drain_mesh_epoch(ticket)
    }

    /// Append `tokens` from the slot's current position in chunks of
    /// [`prefill_chunk_rows`](Self::prefill_chunk_rows), one drained epoch per
    /// chunk; the last chunk's last-row logits are left in
    /// [`logits`](Self::logits). A failure leaves the slot partially advanced
    /// (and a failed submission poisons the mesh): reset or rebuild.
    pub fn prefill(&mut self, gpus: &mut Gpus, tokens: &[u32]) -> Result<()> {
        self.check_gpus(gpus)?;
        if tokens.is_empty() {
            return Err("raon1_mini: prefill of zero tokens".into());
        }
        let start = self.n_tokens();
        if start + tokens.len() > self.max_seq() {
            return Err(format!(
                "raon1_mini: {start} + {} tokens exceed the reserved context of {} rows",
                tokens.len(),
                self.max_seq()
            ));
        }
        let n_chunks = tokens.len().div_ceil(self.chunk_rows);
        for (index, piece) in tokens.chunks(self.chunk_rows).enumerate() {
            let output = if index + 1 == n_chunks {
                RaonMeshOutput::Logits
            } else {
                RaonMeshOutput::Trunk
            };
            self.run_epoch(gpus, piece, MeshRoute::Prefill, output)?;
        }
        Ok(())
    }

    /// One decode step: consume `token`, leave its logits in
    /// [`logits`](Self::logits). One drained epoch.
    pub fn decode_step(&mut self, gpus: &mut Gpus, token: u32) -> Result<()> {
        self.check_gpus(gpus)?;
        if self.n_tokens() + 1 > self.max_seq() {
            return Err(format!(
                "raon1_mini: decode past the reserved context of {} rows",
                self.max_seq()
            ));
        }
        self.run_epoch(gpus, &[token], MeshRoute::Decode, RaonMeshOutput::Logits)
    }

    /// Logits of the last row of the last `prefill` / `decode_step`: F32
    /// `[vocab]` on the last rank's device holding BF16-rounded values.
    pub fn logits(&self) -> &GpuTensor {
        self.bands[self.bands.len() - 1].logits()
    }

    /// Host copy of [`logits`](Self::logits), read on the last rank's device.
    pub fn download_logits(&self, gpus: &Gpus) -> Result<Vec<f32>> {
        let rank = self.bands.len() - 1;
        let gpu = gpus
            .devices
            .get(rank)
            .ok_or("Raon mesh last rank has no device")?;
        gpu.bind_thread().map_err(|e| e.to_string())?;
        self.bands[rank].download_logits(gpu)
    }

    /// End-of-turn token ids (`generation_config.json`), read by the last band.
    pub fn eos(&self) -> &[u32] {
        &self.bands[self.bands.len() - 1].eos
    }

    /// Model generation defaults (sampling and penalty defaults of the card).
    pub fn generation(&self) -> &crate::config::GenerationDefaults {
        &self.bands[0].config.generation
    }

    /// Hidden width, layer count of the whole model and vocabulary.
    pub fn dims(&self) -> (usize, usize, usize) {
        let first = &self.bands[0];
        (first.config.hidden, first.band().total, first.config.vocab)
    }

    /// Grow every band's context-sized state to `rows` rows ahead of a decode
    /// that ends there (each band on its own device). Best effort for the
    /// caller: a failure only leaves growth to the per-step admission.
    pub fn map_ahead(&mut self, gpus: &mut Gpus, rows: usize) -> Result<()> {
        self.check_gpus(gpus)?;
        for (rank, band) in self.bands.iter_mut().enumerate() {
            let gpu = &mut gpus.devices[rank];
            gpu.bind_thread().map_err(|e| e.to_string())?;
            band.state
                .ensure_capacity(gpu, rows)
                .map_err(|e| format!("rank {rank}: {e}"))?;
        }
        Ok(())
    }

    /// Block until every rank's device has retired its queued work.
    pub fn sync_devices(&self, gpus: &Gpus) -> Result<()> {
        for (rank, gpu) in gpus.devices.iter().enumerate() {
            gpu.bind_thread().map_err(|e| e.to_string())?;
            gpu.hip
                .device_synchronize()
                .map_err(|e| format!("raon1_mini: rank {rank} device sync: {e:?}"))?;
        }
        Ok(())
    }

    /// Reset, then prefill `n_tokens` deterministic synthetic ids through the
    /// production [`prefill`](Self::prefill) route and wait for every device.
    /// The same ids as [`RaonBundle::bench_prefill_rows`]. Used by the
    /// loader's prefill benchmark.
    pub fn bench_prefill_rows(&mut self, gpus: &mut Gpus, n_tokens: usize) -> Result<()> {
        if n_tokens == 0 {
            return Err("raon1_mini: bench prefill of zero tokens".into());
        }
        self.reset(gpus)?;
        let vocab = self.bands[0].config.vocab as u64;
        let lo = 1024u64.min(vocab.saturating_sub(1));
        let span = vocab - lo;
        let tokens: Vec<u32> = (0..n_tokens as u64)
            .map(|i| (lo + (i.wrapping_mul(2654435761).wrapping_add(17)) % span) as u32)
            .collect();
        self.prefill(gpus, &tokens)?;
        self.sync_devices(gpus)
    }

    /// Tear the model down: drain the mesh edges (all devices, so no in-flight
    /// work reads a band) and free every band on its own device. A failed
    /// drain quarantines: the bands are leaked rather than freed under
    /// possibly in-flight work.
    pub fn free(self, gpus: &mut Gpus) -> Result<()> {
        let Self { bands, .. } = self;
        if let Err(error) = gpus.drain_mesh_edges() {
            std::mem::forget(bands);
            return Err(format!(
                "Raon mesh edge drain failed ({error}): bands leaked until process restart"
            ));
        }
        for (rank, band) in bands.into_iter().enumerate() {
            let gpu = &mut gpus.devices[rank];
            gpu.bind_thread().map_err(|e| e.to_string())?;
            band.free_all(gpu);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::device_mesh::DimKind;
    use hipfire_runtime::mesh_split::ShardSpec;
    use hipfire_runtime::weight_manifest::ManifestPlan;

    fn released() -> RaonConfig {
        RaonConfig::from_json(&serde_json::json!({"model_type": "raon1_mini"})).unwrap()
    }

    fn pp(stages: usize) -> DeviceMesh {
        DeviceMesh::rect(&[(DimKind::Pp, stages)]).unwrap()
    }

    fn program(counts: &[usize]) -> (RaonConfig, DeviceMesh, BandPlacement, RaonMeshProgram) {
        let cfg = released();
        let mesh = pp(counts.len());
        let placement = BandPlacement::from_counts(&mesh, cfg.n_layers, counts).unwrap();
        let program = RaonMeshProgram::build(&cfg, &mesh, &placement).unwrap();
        (cfg, mesh, placement, program)
    }

    #[test]
    fn graph_has_two_ops_per_layer_plus_embed_and_head() {
        let cfg = released();
        let trunk = raon_step_graph(&cfg, false).unwrap();
        let head = raon_step_graph(&cfg, true).unwrap();
        assert_eq!(trunk.ops().len(), 1 + 2 * cfg.n_layers);
        assert_eq!(head.ops().len(), 2 + 2 * cfg.n_layers);
        assert_eq!(head.ops()[0].kind, OpKind::Embed);
        assert_eq!(head.ops()[0].layer, Some(0));
        let last = head.ops().last().unwrap();
        assert_eq!(last.kind, OpKind::Project);
        assert_eq!(last.layer, Some(cfg.n_layers - 1));
        assert_eq!(
            head.resources()[RAON_STREAMS_RESOURCE.0 as usize].shape,
            vec![cfg.mhc_streams, cfg.hidden]
        );
        assert_eq!(
            head.resources()[RAON_STREAMS_RESOURCE.0 as usize].class,
            ResourceClass::Activation
        );
        // Mixer op kinds follow the layer schedule, ffn kinds the ffn schedule.
        for layer in 0..cfg.n_layers {
            let mixer = &head.ops()[1 + 2 * layer];
            let ffn = &head.ops()[2 + 2 * layer];
            assert_eq!(mixer.layer, Some(layer));
            assert_eq!(ffn.layer, Some(layer));
            assert_eq!(
                mixer.kind,
                match cfg.layer_kinds[layer] {
                    LayerKind::Rda => OpKind::GatedDeltaNet,
                    LayerKind::Mla => OpKind::Attend,
                }
            );
            assert_eq!(
                ffn.kind,
                match cfg.ffn_kinds[layer] {
                    FfnKind::Dense => OpKind::GatedMlp,
                    FfnKind::Moe => OpKind::Moe,
                }
            );
            // Only the mixer touches state.
            assert!(mixer.reads.len() > 1 && ffn.reads == vec![RAON_STREAMS_RESOURCE]);
        }
    }

    #[test]
    fn streams_are_the_only_resource_read_across_layers() {
        // Every op reads the streams, an input, or its own layer's state; no
        // activation other than the streams crosses an op boundary.
        let cfg = released();
        let graph = raon_step_graph(&cfg, true).unwrap();
        for (index, op) in graph.ops().iter().enumerate() {
            for r in &op.reads {
                let class = &graph.resources()[r.0 as usize].class;
                match class {
                    ResourceClass::Activation => {
                        assert_eq!(*r, RAON_STREAMS_RESOURCE, "op {index}")
                    }
                    ResourceClass::State { layer, .. } => assert_eq!(Some(*layer), op.layer),
                    ResourceClass::Input => assert_eq!(*r, RAON_TOKENS_RESOURCE),
                    other => panic!("op {index} reads {other:?}"),
                }
            }
        }
    }

    #[test]
    fn each_cut_is_exactly_one_streams_band_xfer() {
        for counts in [
            vec![1, 24],
            vec![13, 12],
            vec![24, 1],
            vec![1, 12, 12],
            vec![5, 5, 5, 5, 5],
        ] {
            let (cfg, _, placement, program) = program(&counts);
            for output in RaonMeshOutput::ALL {
                let plan = program.plan(output);
                let graph = plan_graph(&cfg, output);
                let bands = counts.len();
                assert_eq!(plan.collectives.len(), bands - 1, "{counts:?}");
                for (edge, c) in plan.collectives.iter().enumerate() {
                    assert_eq!(
                        c.kind,
                        CollectiveKind::BandXfer {
                            src: edge,
                            dst: edge + 1
                        }
                    );
                    assert_eq!(c.operand, RAON_STREAMS_RESOURCE);
                    assert_eq!(c.group, vec![edge, edge + 1]);
                    // The producer is the last op of the source band (its FFN site).
                    let producer = &graph.ops()[c.producer.0 as usize];
                    assert_eq!(producer.layer, Some(placement.band(edge).end - 1));
                    assert_eq!(
                        producer.kind,
                        match cfg.ffn_kinds[placement.band(edge).end - 1] {
                            FfnKind::Dense => OpKind::GatedMlp,
                            FfnKind::Moe => OpKind::Moe,
                        }
                    );
                }
                // Rank item order: ops, then publish (not last), receive (not first).
                for (rank, program_rank) in plan.ranks.iter().enumerate() {
                    let collectives: Vec<_> = program_rank
                        .items
                        .iter()
                        .filter(|item| matches!(item, RankItem::Collective(_)))
                        .collect();
                    let expected = usize::from(rank > 0) + usize::from(rank + 1 < bands);
                    assert_eq!(collectives.len(), expected, "{counts:?} rank {rank}");
                    if rank > 0 {
                        assert!(matches!(program_rank.items[0], RankItem::Collective(_)));
                    }
                    if rank + 1 < bands {
                        assert!(matches!(
                            program_rank.items.last().unwrap(),
                            RankItem::Collective(_)
                        ));
                    }
                }
            }
        }
    }

    fn plan_graph(cfg: &RaonConfig, output: RaonMeshOutput) -> StepGraph {
        raon_step_graph(cfg, output.head()).unwrap()
    }

    #[test]
    fn every_op_runs_on_exactly_one_rank_embed_first_head_last() {
        for counts in [vec![1, 24], vec![13, 12], vec![24, 1], vec![7, 9, 9]] {
            let (cfg, _, placement, program) = program(&counts);
            for output in RaonMeshOutput::ALL {
                let graph = plan_graph(&cfg, output);
                let mut seen = vec![0usize; graph.ops().len()];
                for rank in 0..counts.len() {
                    let ops = program.rank_ops(output, rank);
                    assert_eq!(
                        ops,
                        raon_rank_op_ids(&graph, placement.band(rank)),
                        "{counts:?} rank {rank}"
                    );
                    for id in ops {
                        seen[id.0 as usize] += 1;
                    }
                    // Ops within a rank stay in issue order.
                    assert!(ops.windows(2).all(|pair| pair[0] < pair[1]));
                }
                assert!(seen.iter().all(|&n| n == 1), "{counts:?}");
                // Embed pinned to rank 0.
                assert!(program.rank_ops(output, 0).contains(&OpId(0)));
                assert!(counts.len() < 2 || !program.rank_ops(output, 1).contains(&OpId(0)));
                // Head (the last op) only on the last rank, and only with the head.
                let last_rank = counts.len() - 1;
                let head_op = OpId(graph.ops().len() as u32 - 1);
                assert_eq!(
                    program.rank_ops(output, last_rank).contains(&head_op)
                        && graph.ops()[head_op.0 as usize].kind == OpKind::Project,
                    output.head()
                );
                for rank in 0..last_rank {
                    assert!(!program.rank_ops(output, rank).contains(&head_op));
                }
            }
        }
    }

    #[test]
    fn state_follows_its_layer_and_a_misplaced_state_is_refused() {
        let (cfg, mesh, placement, _) = program(&[13, 12]);
        let (weights, state) = raon_manifest(&cfg);
        let manifest: ManifestPlan =
            plan_manifest(&weights, &state, &mesh, cfg.n_layers, &placement).unwrap();
        for (entry, devices) in &manifest.state {
            let layer = entry.layer().unwrap();
            assert_eq!(devices, &vec![placement.stage_of_layer(layer)]);
        }
        // Per-layer state counts: two entries per RDA layer, one per MLA layer.
        assert_eq!(
            state.len(),
            2 * cfg.n_rda() + cfg.n_mla(),
            "recurrent + conv per RDA layer, latent per MLA layer"
        );
        // Embedding on the first stage, output head on the last.
        let place = |name: &str| {
            manifest
                .weights
                .iter()
                .find(|w| w.name == name)
                .unwrap()
                .devices
                .clone()
        };
        assert_eq!(place("model.embed_tokens.weight"), vec![0]);
        assert_eq!(place("model.norm.weight"), vec![1]);
        assert_eq!(place("lm_head.weight"), vec![1]);
        assert_eq!(place("model.layers.12.input_layernorm.weight"), vec![0]);
        assert_eq!(place("model.layers.13.input_layernorm.weight"), vec![1]);
    }

    #[test]
    fn plan_resources_keep_the_streams_whole_on_the_producer() {
        let (_, _, _, program) = program(&[13, 12]);
        let plan = program.plan(RaonMeshOutput::Logits);
        // The streams exit the program owned by the last writer's rank.
        let streams = plan.resources[RAON_STREAMS_RESOURCE.0 as usize];
        assert!(matches!(
            streams.exit,
            ShardSpec::Owner { device: 1 } | ShardSpec::Whole
        ));
    }

    #[test]
    fn edge_capacity_is_the_chunk_of_streams_in_f32() {
        let cfg = released();
        let bytes = raon_edge_capacity_bytes(&cfg, 512).unwrap();
        assert_eq!(bytes, 512 * cfg.mhc_streams * cfg.hidden * 4);
        assert_eq!(bytes % 16, 0);
        assert_eq!(
            RaonMeshModel::edge_capacity_bytes(&cfg, 512).unwrap(),
            bytes
        );
        assert!(raon_edge_capacity_bytes(&cfg, 0).is_err());
        // One row is a whole number of 16-byte units.
        assert_eq!(edge_bytes(&cfg, 1) % 16, 0);
    }

    #[test]
    fn mesh_refuses_repair_session_and_window_lifecycle() {
        let (_, _, _, program) = program(&[13, 12]);
        let mut bands: Vec<RaonBundle> = Vec::new();
        let mut bindings = RaonMeshBindings::new(&mut bands[..], &program);
        assert!(!bindings.admits_prefix_repair());
        let epoch = ForwardEpoch {
            mesh_epoch: program.mesh_epoch(),
            load_id: 1,
            request_id: 1,
            op_id: 1,
            route: MeshRoute::Prefill,
            start_position: 0,
            actual_rows: 1,
            route_rows: 1,
            policy_id: RAON_MESH_POLICY_ID,
        };
        let key = MeshCheckpointKey {
            mesh_epoch: program.mesh_epoch(),
            load_gen: 1,
            reset_gen: 0,
            request_gen: 1,
            window_gen: 0,
        };
        let ticket = MeshCheckpointId::issue(key, 0, 0);
        let error = bindings.probe_session_rank(0, &epoch, ticket).unwrap_err();
        assert!(error.contains("whole model on one device"), "{error}");
        let error = bindings
            .validate_finalize_rank(0, &key, ticket)
            .unwrap_err();
        assert!(error.contains("whole model on one device"), "{error}");
        // An unprepared epoch is refused by every device callback.
        assert!(bindings
            .boundary(
                0,
                &program.plan(RaonMeshOutput::Logits).collectives[0],
                &epoch
            )
            .is_err());
    }

    #[test]
    fn a_split_with_a_single_band_or_a_wrong_cover_is_refused() {
        let cfg = released();
        let mesh = pp(2);
        // A placement of another model size is refused before planning.
        let placement = BandPlacement::from_counts(&mesh, 24, &[12, 12]).unwrap();
        assert!(RaonMeshProgram::build(&cfg, &mesh, &placement).is_err());
    }
}
