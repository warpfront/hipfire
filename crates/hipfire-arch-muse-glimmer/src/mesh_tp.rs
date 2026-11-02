// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Muse Glimmer dense tensor parallelism on the generic mesh executor
//! (#927 item 5).
//!
//! Each TP rank holds an even contiguous block of the query and KV heads and
//! of the FFN intermediate ([`TpSlice`], loaded by
//! [`GlimmerWeights::load_shard`]). Both sublayers end in the sandwich
//! epilogue `x + post_norm(out)`, and an RMS post-norm cannot take a partial
//! sum, so the program ([`glimmer_tp_plan`]) splits each sublayer at that
//! boundary: the attention (`SplitRule::HeadMixer`) writes this rank's output
//! projection partial into [`GLIMMER_MIX_OUT_RESOURCE`], the MLP
//! (`SplitRule::GatedMlp`, `SandwichSteps` ladder) into
//! [`GLIMMER_FFN_OUT_RESOURCE`]; the whole-row epilogue after each takes one
//! rank-ordered `AllReduce` of the partial and then runs the single-card
//! post-norm residual kernels on the reduced rows. Embedding and head run
//! replicated.
//!
//! One program serves both routes: a decode epoch over one token on the
//! decode state, a prefill epoch over up to `max_batch` tokens on the rank's
//! rows scratch (the logits are the last row's). Sessions, checkpoints and
//! prefix repair are refused.

use crate::config::GlimmerConfig;
use crate::forward::{self, RowsScratch};
use crate::glimmer::{GlimmerState, GlimmerWeights, TpSlice};
use crate::program::{self, LayerIo};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::gated_mlp::MlpLadder;
use hipfire_dispatch::pipeline::sandwich::{
    execute_sandwich_attention_partial, execute_sandwich_epilogue, execute_sandwich_mlp_partial,
    windowed_q8_partials_elems,
};
use hipfire_dispatch::pipeline::step_graph::{
    OpDecl, OpId, OpKind, ResourceClass, ResourceDecl, ResourceId, SplitRule, StepGraph,
};
use hipfire_dispatch::pipeline::{execute_steps, RowsRoute, Step};
use hipfire_runtime::device_mesh::{DeviceMesh, DimKind};
use hipfire_runtime::mesh_placement::BandPlacement;
use hipfire_runtime::mesh_split::{
    split_program, CollectiveKind, PlannedCollective, RankItem, SplitPlan,
};
use hipfire_runtime::multi_gpu::executor::{
    BoundaryIo, ForwardEpoch, MeshCheckpointId, MeshCheckpointKey, MeshRoute, MeshStepBindings,
};
use hipfire_runtime::multi_gpu::transport::EdgeDescriptor;
use hipfire_runtime::weight_manifest::{plan_manifest, ShardPolicy, WeightEntry};
use rdna_compute::{DType, Gpu, GpuTensor};

type Result<T, E = String> = std::result::Result<T, E>;

/// The attention output projection: a per-rank partial on a TP mesh.
pub const GLIMMER_MIX_OUT_RESOURCE: ResourceId = ResourceId(3);
/// The MLP down projection: a per-rank partial on a TP mesh.
pub const GLIMMER_FFN_OUT_RESOURCE: ResourceId = ResourceId(4);
/// Packed-quant block along a row-sharded (input) axis: whole G256 groups.
const ROW_GROUP: usize = 256;
/// The batched epochs' row cap (the single-card prefill chunk cap).
const MAX_BATCH: usize = 512;

/// What one op of the program runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TpOp {
    /// Token embedding and positions.
    Embed,
    /// The layer's gated attention over this rank's heads, up to `o_proj`.
    Attention { layer: usize },
    /// The layer's MLP over this rank's intermediate block, up to `down`.
    Mlp { layer: usize },
    /// `out = x + post_norm(reduced sublayer output)` of the attention
    /// (`mlp == false`) or MLP sublayer.
    Epilogue { layer: usize, mlp: bool },
    /// Final norm and LM head into the logits.
    Head,
}

/// One declared weight: manifest name, logical shape, policy, row group.
type WeightDecl = (&'static str, [usize; 2], ShardPolicy, usize);

/// The layer's projections in [`layer_dtypes`] order: attention gate, q, k,
/// v, o, then MLP gate, up, down.
fn layer_weights(cfg: &GlimmerConfig) -> [WeightDecl; 8] {
    let (dim, inner, hd) = (cfg.dim, cfg.hidden_dim, cfg.head_dim);
    let heads = |n_heads| ShardPolicy::HeadSharded {
        n_heads,
        head_dim: hd,
    };
    let (q, kv) = (cfg.q_dim(), cfg.kv_dim());
    let row = ShardPolicy::RowShard { axis: 1 };
    let column = ShardPolicy::ColumnShard { axis: 0 };
    [
        ("self_attn.gate_proj", [q, dim], heads(cfg.n_heads), 1),
        ("self_attn.q_proj", [q, dim], heads(cfg.n_heads), 1),
        ("self_attn.k_proj", [kv, dim], heads(cfg.n_kv_heads), 1),
        ("self_attn.v_proj", [kv, dim], heads(cfg.n_kv_heads), 1),
        ("self_attn.o_proj", [dim, q], row.clone(), ROW_GROUP),
        ("mlp.gate_proj", [inner, dim], column.clone(), 1),
        ("mlp.up_proj", [inner, dim], column, 1),
        ("mlp.down_proj", [dim, inner], row, ROW_GROUP),
    ]
}

/// The step graph of the model, what each op runs and the weight manifest.
/// Built together so the three cannot drift.
fn tp_program(
    cfg: &GlimmerConfig,
    dtypes: &[[DType; 8]],
) -> Result<(StepGraph, Vec<TpOp>, Vec<WeightEntry>)> {
    if cfg.n_layers == 0 || dtypes.len() != cfg.n_layers {
        return Err("Glimmer TP program: layer count disagrees".into());
    }
    let error = |e| format!("Glimmer TP step graph: {e}");
    let mut graph = StepGraph::new();
    let resource = |graph: &mut StepGraph, class, shape: &[usize]| {
        graph
            .add_resource(ResourceDecl {
                class,
                shape: shape.to_vec(),
                alias: None,
            })
            .map_err(error)
    };
    let dim = cfg.dim;
    let token = resource(&mut graph, ResourceClass::Input, &[1])?;
    // The two residual streams the sublayers alternate (`x` -> `residual`
    // -> `x`), then the two sublayer partials.
    let x = resource(&mut graph, ResourceClass::Activation, &[dim])?;
    let residual = resource(&mut graph, ResourceClass::Activation, &[dim])?;
    let mix = resource(&mut graph, ResourceClass::Activation, &[dim])?;
    let ffn = resource(&mut graph, ResourceClass::Activation, &[dim])?;
    debug_assert_eq!(
        (mix, ffn),
        (GLIMMER_MIX_OUT_RESOURCE, GLIMMER_FFN_OUT_RESOURCE)
    );
    let logits = resource(&mut graph, ResourceClass::Output, &[cfg.vocab_size])?;

    let mut ops = Vec::with_capacity(4 * cfg.n_layers + 2);
    let mut entries = Vec::with_capacity(8 * cfg.n_layers);
    let mut op = |graph: &mut StepGraph,
                  what: TpOp,
                  kind: OpKind,
                  layer: Option<usize>,
                  rule: SplitRule,
                  reads: &[ResourceId],
                  writes: &[ResourceId]| {
        graph
            .add_op(OpDecl {
                kind,
                layer,
                rule,
                reads: reads.to_vec(),
                writes: writes.to_vec(),
            })
            .map_err(error)?;
        ops.push(what);
        Ok::<_, String>(())
    };
    let embed = (TpOp::Embed, OpKind::Embed, None, SplitRule::Opaque);
    op(
        &mut graph,
        embed.0,
        embed.1,
        embed.2,
        embed.3,
        &[token],
        &[x],
    )?;
    for (layer, dtypes) in dtypes.iter().enumerate() {
        let mut w = Vec::with_capacity(8);
        for (&(name, shape, ref policy, group), &dtype) in layer_weights(cfg).iter().zip(dtypes) {
            let class = ResourceClass::Weight {
                name: name.to_string(),
                layer: Some(layer),
                group,
            };
            w.push(resource(&mut graph, class, &shape)?);
            let entry = WeightEntry::layer(name, layer, shape.to_vec(), dtype, policy.clone());
            entries.push(entry);
        }
        let l = Some(layer);
        let rule = SplitRule::HeadMixer {
            input: x,
            out_proj: w[4],
            output: mix,
            heads: cfg.n_heads,
            kv_heads: cfg.n_kv_heads,
        };
        let reads = [x, w[0], w[1], w[2], w[3], w[4]];
        let attention = TpOp::Attention { layer };
        op(
            &mut graph,
            attention,
            OpKind::SandwichAttention,
            l,
            rule,
            &reads,
            &[mix],
        )?;
        let epilogue = TpOp::Epilogue { layer, mlp: false };
        let (add, whole) = (OpKind::ResidualAdd, SplitRule::RowComplete);
        op(&mut graph, epilogue, add, l, whole, &[x, mix], &[residual])?;
        let rule = SplitRule::GatedMlp {
            input: residual,
            gate: w[5],
            up: w[6],
            down: w[7],
            output: ffn,
            ladder: MlpLadder::SandwichSteps,
        };
        let reads = [residual, w[5], w[6], w[7]];
        op(
            &mut graph,
            TpOp::Mlp { layer },
            OpKind::SandwichMlp,
            l,
            rule,
            &reads,
            &[ffn],
        )?;
        let epilogue = TpOp::Epilogue { layer, mlp: true };
        op(&mut graph, epilogue, add, l, whole, &[residual, ffn], &[x])?;
    }
    let head = (TpOp::Head, OpKind::Gemv, SplitRule::RowComplete);
    op(&mut graph, head.0, head.1, None, head.2, &[x], &[logits])?;
    Ok((graph, ops, entries))
}

/// The TP split plan on a pure TP `mesh`; `dtypes` as [`layer_dtypes`].
/// Refuses (`Geometry`) a mesh the query or KV heads, the attention output
/// width or the FFN intermediate do not divide into whole G256 blocks.
pub fn glimmer_tp_plan(
    cfg: &GlimmerConfig,
    dtypes: &[[DType; 8]],
    mesh: &DeviceMesh,
) -> Result<SplitPlan> {
    if mesh.n_devices() < 2 || mesh.size_of(DimKind::Tp) != mesh.n_devices() {
        return Err("Glimmer TP plan needs a pure TP mesh of two or more devices".into());
    }
    let (graph, _, entries) = tp_program(cfg, dtypes)?;
    let placement = BandPlacement::uniform(mesh, cfg.n_layers)?;
    let plan = plan_manifest(&entries, &[], mesh, cfg.n_layers, &placement)?;
    split_program(&graph, &plan, mesh).map_err(|error| format!("Glimmer TP split: {error}"))
}

/// Per layer, the payload dtypes of the projections in [`layer_weights`]
/// order.
pub fn layer_dtypes(weights: &GlimmerWeights) -> Vec<[DType; 8]> {
    weights
        .layers
        .iter()
        .map(|l| {
            [
                &l.attn_gate_proj,
                &l.q_proj,
                &l.k_proj,
                &l.v_proj,
                &l.o_proj,
                &l.gate_proj,
                &l.up_proj,
                &l.down_proj,
            ]
            .map(|w| w.gpu_dtype)
        })
        .collect()
}

/// The config one rank of a `parts`-way mesh runs: its heads and its FFN
/// block. Refuses a split that is not even (whole heads, whole G256 blocks of
/// every row-sharded input).
pub fn rank_config(global: &GlimmerConfig, parts: usize) -> Result<GlimmerConfig> {
    let even = |extent: usize, unit: usize| parts != 0 && extent % (parts * unit) == 0;
    if !(even(global.n_heads, 1)
        && even(global.n_kv_heads, 1)
        && even(global.q_dim(), ROW_GROUP)
        && even(global.hidden_dim, ROW_GROUP))
    {
        return Err(format!(
            "Glimmer TP{parts}: {} query heads, {} KV heads, attention width {} and FFN \
             intermediate {} do not split evenly",
            global.n_heads,
            global.n_kv_heads,
            global.q_dim(),
            global.hidden_dim
        ));
    }
    let mut cfg = global.clone();
    cfg.n_heads /= parts;
    cfg.n_kv_heads /= parts;
    cfg.hidden_dim /= parts;
    Ok(cfg)
}

/// One TP rank's model shard and state, resident on the rank's device. Only
/// [`Self::load`] builds one, so the slice [`GlimmerTpBindings::new`] checks
/// is the slice the rank holds.
pub struct GlimmerTpRank {
    /// Rank-local config ([`rank_config`]).
    config: GlimmerConfig,
    slice: TpSlice,
    weights: GlimmerWeights,
    /// Decode state and KV; its `logits` are the rank's output on both routes.
    state: GlimmerState,
    /// Prefill rows (`MAX_BATCH` rows).
    rows: RowsScratch,
    /// Tokens committed so far.
    position: usize,
}

impl GlimmerTpRank {
    /// Rank `rank` of a `parts`-way TP mesh of `global` on `gpu`: loads the
    /// rank's weight slice and allocates its state over `max_seq` positions
    /// and its prefill rows. On error everything allocated so far is freed.
    pub fn load(
        hfq: &hipfire_runtime::hfq::HfqFile,
        global: &GlimmerConfig,
        gpu: &mut Gpu,
        rank: usize,
        parts: usize,
        max_seq: usize,
    ) -> Result<Self> {
        if rank >= parts {
            return Err(format!("Glimmer TP rank {rank} is outside {parts} parts"));
        }
        gpu.bind_thread().map_err(|e| e.to_string())?;
        let config = rank_config(global, parts)?;
        let slice = TpSlice { rank, parts };
        let weights = GlimmerWeights::load_shard(hfq, global, gpu, Some(slice))?;
        let state = match GlimmerState::new_with_max_seq(gpu, &config, max_seq) {
            Ok(state) => state,
            Err(error) => {
                weights.free_gpu(gpu);
                return Err(error);
            }
        };
        let mut rank = Self {
            rows: match RowsScratch::alloc(gpu, &config, MAX_BATCH) {
                Ok(rows) => rows,
                Err(error) => {
                    state.free_gpu(gpu);
                    weights.free_gpu(gpu);
                    return Err(error);
                }
            },
            config,
            slice,
            weights,
            state,
            position: 0,
        };
        // The single card allocates these lazily on first use; a mesh
        // callback must not allocate. Neither changes the kernel route.
        let (heads, head_dim) = (rank.config.n_heads, rank.config.head_dim);
        let flash = windowed_q8_partials_elems(heads, head_dim, max_seq);
        let bf16 = MAX_BATCH * rank.config.hidden_dim;
        let staged = gpu.alloc_tensor(&[flash], DType::F32).and_then(|partials| {
            rank.state.prefill_flash_partials = Some(partials);
            gpu.zeros(&[bf16], DType::BF16)
        });
        match staged {
            Ok(bf16) => {
                rank.state.prefill_bf16_scratch = Some(bf16);
                Ok(rank)
            }
            Err(error) => {
                rank.free_gpu(gpu);
                Err(format!("Glimmer TP rank scratch: {error:?}"))
            }
        }
    }

    /// The rank's logits (last row of a prefill epoch, or the decode token).
    pub fn logits(&self) -> &GpuTensor {
        &self.state.logits
    }

    /// Tokens committed so far.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Free everything the rank holds on `gpu`.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.bind_thread();
        self.rows.free(gpu);
        self.state.free_gpu(gpu);
        self.weights.free_gpu(gpu);
        gpu.drain_pool();
    }
}

/// One epoch adapter over the TP ranks (rank `i` on device `i`): borrows the
/// ranks and the plan they were validated against, never a GPU.
pub struct GlimmerTpBindings<'a> {
    ranks: &'a mut [GlimmerTpRank],
    plan: &'a SplitPlan,
    ops: Vec<TpOp>,
    /// Per rank, the op ranges of its program between collectives.
    segments: Vec<Vec<std::ops::Range<u32>>>,
    /// Per rank, the next segment to run.
    cursor: Vec<usize>,
    tokens: &'a [u32],
    tp_epoch: u64,
    prepared: Option<ForwardEpoch>,
}

impl<'a> GlimmerTpBindings<'a> {
    /// Validate before any submission: two or more ranks of one global config,
    /// rank `i` loaded (by [`GlimmerTpRank::load`]) as slice `i` of
    /// `ranks.len()`, one weight dtype set, and `plan` the TP plan of exactly
    /// these shards. `tp_epoch` is `Gpus::mesh_tp_epoch()`.
    pub fn new(
        global: &GlimmerConfig,
        ranks: &'a mut [GlimmerTpRank],
        plan: &'a SplitPlan,
        tp_epoch: u64,
    ) -> Result<Self> {
        let parts = ranks.len();
        if parts < 2 || tp_epoch == 0 {
            return Err("Glimmer TP needs two or more ranks and a provisioned TP group".into());
        }
        let dtypes = layer_dtypes(&ranks[0].weights);
        for (index, rank) in ranks.iter().enumerate() {
            if rank.slice != (TpSlice { rank: index, parts })
                || layer_dtypes(&rank.weights) != dtypes
            {
                return Err(format!(
                    "Glimmer TP rank {index} was not loaded as rank {index} of {parts}"
                ));
            }
        }
        let (_, ops, _) = tp_program(global, &dtypes)?;
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, parts)]).map_err(|e| format!("{e:?}"))?;
        let expected = glimmer_tp_plan(global, &dtypes, &mesh)?;
        if expected.ranks != plan.ranks || expected.collectives != plan.collectives {
            return Err("Glimmer TP plan is not the plan of these shards".into());
        }
        let segments = plan
            .ranks
            .iter()
            .map(|rank| {
                let mut out = Vec::new();
                let mut run: Option<std::ops::Range<u32>> = None;
                for item in &rank.items {
                    match *item {
                        RankItem::Op(OpId(id)) => match &mut run {
                            Some(range) if range.end == id => range.end = id + 1,
                            Some(range) => {
                                out.push(range.clone());
                                *range = id..id + 1;
                            }
                            None => run = Some(id..id + 1),
                        },
                        RankItem::Collective(_) => out.extend(run.take()),
                    }
                }
                out.extend(run);
                out
            })
            .collect();
        Ok(Self {
            ranks,
            plan,
            ops,
            segments,
            cursor: vec![0; parts],
            tokens: &[],
            tp_epoch,
            prepared: None,
        })
    }

    /// The tokens of the next epoch: one for decode, the prompt chunk for
    /// prefill.
    pub fn set_tokens(&mut self, tokens: &'a [u32]) {
        self.tokens = tokens;
        self.prepared = None;
    }

    fn require_prepared(&self, epoch: &ForwardEpoch) -> Result<()> {
        if self.prepared != Some(*epoch) {
            return Err("Glimmer TP epoch has not been prepared".into());
        }
        Ok(())
    }

    fn unsupported(what: &str) -> String {
        format!("Glimmer TP does not admit {what} yet")
    }

    fn run_op(
        rank: &mut GlimmerTpRank,
        gpu: &mut Gpu,
        op: TpOp,
        epoch: &ForwardEpoch,
        tokens: &[u32],
    ) -> Result<()> {
        let dispatch = |e: hipfire_dispatch::types::DispatchError| format!("Glimmer TP: {e:?}");
        let prefill = epoch.route == MeshRoute::Prefill;
        let (start, n) = (rank.position, epoch.actual_rows);
        // A prefill epoch's rows: the scratch cut to exactly `n` rows, as the
        // single card allocates it.
        let rows = prefill.then(|| rank.rows.rows(&rank.config, n));
        if let TpOp::Embed = op {
            let (cfg, w, s) = (&rank.config, &rank.weights, &mut rank.state);
            return match &rows {
                Some(r) => {
                    forward::stage_rows(gpu, cfg, s, r, tokens, start, RowsRoute::Prefill)?;
                    forward::embed_rows(gpu, cfg, w, s, r, tokens)
                }
                None => {
                    forward::embed_lookup(cfg, w, s, gpu, tokens[0])?;
                    forward::stage_decode_position(gpu, cfg, s, start as u32)
                }
            };
        }
        let ctx = DispatchCtx::new(gpu);
        let (cfg, s) = (&rank.config, &rank.state);
        let io = match &rows {
            Some(r) => forward::rows_io(s, r, n, RowsRoute::Prefill, start),
            None => LayerIo::decode(s, forward::decode_tiled(s)),
        };
        if let TpOp::Head = op {
            let lm_head = rank.weights.lm_head.dispatch_ref();
            let last;
            let normed = if prefill {
                last = io.x.sub_offset((n - 1) * cfg.dim, cfg.dim);
                &last
            } else {
                io.normed
            };
            let mut steps = vec![io.final_norm(cfg, &rank.weights)];
            steps.extend(program::lm_head_steps(
                cfg,
                &lm_head,
                normed,
                &rank.state.logits,
            ));
            return execute_steps(gpu, &ctx, &steps).map_err(dispatch);
        }
        let layer = match op {
            TpOp::Attention { layer } | TpOp::Mlp { layer } | TpOp::Epilogue { layer, .. } => layer,
            TpOp::Embed | TpOp::Head => unreachable!("handled above"),
        };
        let lw = &rank.weights.layers[layer];
        let [attention, mlp] = program::layer_steps(cfg, layer, lw, &rank.state, &io);
        let (Step::SandwichAttention(attention), Step::SandwichMlp(mlp)) = (attention, mlp) else {
            return Err("Glimmer TP: the layer program is not attention then MLP".into());
        };
        match op {
            TpOp::Attention { .. } => {
                attention.validate_for_gpu(gpu).map_err(dispatch)?;
                execute_sandwich_attention_partial(gpu, &ctx, &attention)
            }
            TpOp::Mlp { .. } => {
                mlp.validate_for_gpu(gpu).map_err(dispatch)?;
                execute_sandwich_mlp_partial(gpu, &ctx, &mlp)
            }
            TpOp::Epilogue { mlp: false, .. } => {
                let s = &attention.stream;
                execute_sandwich_epilogue(gpu, s, s.normed, attention.post_norm)
            }
            TpOp::Epilogue { mlp: true, .. } => {
                execute_sandwich_epilogue(gpu, &mlp.stream, mlp.out, mlp.post_norm)
            }
            TpOp::Embed | TpOp::Head => unreachable!("handled above"),
        }
        .map_err(dispatch)
    }
}

impl MeshStepBindings for GlimmerTpBindings<'_> {
    fn prepare_epoch(&mut self, epoch: &ForwardEpoch) -> Result<()> {
        self.prepared = None;
        let rows = epoch.actual_rows;
        let admitted = match epoch.route {
            MeshRoute::Decode => rows == 1,
            MeshRoute::Prefill => (1..=MAX_BATCH).contains(&rows),
            _ => false,
        };
        if !admitted || epoch.route_rows != rows {
            return Err(format!(
                "Glimmer TP admits single-token decode and prefill epochs of at most \
                 {MAX_BATCH} rows only, got {:?} over {rows} rows",
                epoch.route
            ));
        }
        if self.tokens.len() != rows {
            return Err("Glimmer TP epoch tokens disagree with its rows".into());
        }
        for rank in self.ranks.iter() {
            let end = rank.position.checked_add(rows);
            if rank.position != epoch.start_position
                || end.is_none_or(|end| end > rank.state.max_seq)
            {
                return Err("Glimmer TP rank position or KV capacity disagrees".into());
            }
        }
        self.cursor.fill(0);
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
        let segment = self
            .segments
            .get(owner)
            .and_then(|segments| segments.get(self.cursor[owner]))
            .ok_or("Glimmer TP rank ran past its program")?;
        if !ops.iter().map(|op| op.0).eq(segment.clone()) {
            return Err("Glimmer TP segment does not continue its rank program in order".into());
        }
        for op in ops {
            let op = self.ops[op.0 as usize];
            Self::run_op(&mut self.ranks[owner], gpu, op, epoch, self.tokens)?;
        }
        self.cursor[owner] += 1;
        Ok(())
    }

    fn boundary(
        &self,
        owner: usize,
        collective: &PlannedCollective,
        epoch: &ForwardEpoch,
    ) -> Result<BoundaryIo<'_>> {
        self.require_prepared(epoch)?;
        if collective.kind != CollectiveKind::AllReduce
            || !self.plan.collectives.contains(collective)
        {
            return Err("Glimmer TP only reduces sublayer partials in place".into());
        }
        let rank = self.ranks.get(owner).ok_or("unknown Glimmer TP rank")?;
        let rows = epoch.actual_rows;
        let (s, r) = (&rank.state, &rank.rows);
        let view = match (epoch.route, collective.operand) {
            (MeshRoute::Decode, GLIMMER_MIX_OUT_RESOURCE) => &s.tmp,
            (MeshRoute::Decode, GLIMMER_FFN_OUT_RESOURCE) => &s.ffn_out,
            (MeshRoute::Prefill, GLIMMER_MIX_OUT_RESOURCE) => &r.normed,
            (MeshRoute::Prefill, GLIMMER_FFN_OUT_RESOURCE) => &r.ffn_out,
            _ => return Err("Glimmer TP collective operand is not a sublayer partial".into()),
        };
        Ok(BoundaryIo {
            view,
            into: None,
            descriptor: EdgeDescriptor {
                epoch: self.tp_epoch,
                op: epoch.op_id,
                rows,
                dtype_bytes: 4,
                bytes: rows * rank.config.dim * std::mem::size_of::<f32>(),
            },
            partial_slot: None,
        })
    }

    fn apply_rank_marks(&mut self, owner: usize, epoch: &ForwardEpoch) -> Result<()> {
        self.require_prepared(epoch)?;
        if self.cursor.get(owner) != self.segments.get(owner).map(Vec::len).as_ref() {
            return Err("Glimmer TP rank did not run its whole program".into());
        }
        let rank = &mut self.ranks[owner];
        if rank.position != epoch.start_position {
            return Err("Glimmer TP rank position moved during the epoch".into());
        }
        rank.position += epoch.actual_rows;
        rank.state.n_tokens = rank.position;
        Ok(())
    }

    fn reset_rank(&mut self, owner: usize, _gpu: &mut Gpu, _epoch: &ForwardEpoch) -> Result<()> {
        self.prepared = None;
        let rank = self.ranks.get_mut(owner).ok_or("unknown Glimmer TP rank")?;
        rank.state.reset();
        rank.position = 0;
        Ok(())
    }

    /// Refused before the first callback, so the mesh stays healthy.
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
        Err(Self::unsupported("prefix repair"))
    }

    fn capture_session_rank(
        &mut self,
        _owner: usize,
        _gpu: &mut Gpu,
        _epoch: &ForwardEpoch,
    ) -> Result<MeshCheckpointId> {
        Err(Self::unsupported("session capture"))
    }

    fn probe_session_rank(
        &mut self,
        _owner: usize,
        _epoch: &ForwardEpoch,
        _piece: MeshCheckpointId,
    ) -> Result<()> {
        Err(Self::unsupported("session restore"))
    }

    fn restore_session_rank(
        &mut self,
        _owner: usize,
        _gpu: &mut Gpu,
        _epoch: &ForwardEpoch,
        _checkpoint: MeshCheckpointId,
    ) -> Result<()> {
        Err(Self::unsupported("session restore"))
    }

    fn checkpoint_rank(
        &mut self,
        _owner: usize,
        _gpu: &mut Gpu,
        _key: &MeshCheckpointKey,
    ) -> Result<MeshCheckpointId> {
        Err(Self::unsupported("checkpoints"))
    }

    fn restore_rank(
        &mut self,
        _owner: usize,
        _gpu: &mut Gpu,
        _key: &MeshCheckpointKey,
        _ticket: MeshCheckpointId,
    ) -> Result<()> {
        Err(Self::unsupported("checkpoints"))
    }

    fn validate_finalize_rank(
        &mut self,
        _owner: usize,
        _key: &MeshCheckpointKey,
        _ticket: MeshCheckpointId,
    ) -> Result<()> {
        Err(Self::unsupported("checkpoints"))
    }

    /// Unreachable: `validate_finalize_rank` refuses every ticket first.
    fn finalize_rank(&mut self, _: usize, _: &MeshCheckpointKey, _: MeshCheckpointId) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `[L, L, L, G]` config: `heads` query / `kv` KV heads of 128,
    /// intermediate `inner`.
    fn config(heads: usize, kv: usize, inner: usize) -> GlimmerConfig {
        let tc = serde_json::json!({
            "hidden_size": 512, "num_hidden_layers": 4, "vocab_size": 1000,
            "num_attention_heads": heads, "num_key_value_heads": kv, "head_dim": 128,
            "intermediate_size": inner,
            "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
            "layer_rope_theta": [500000.0, 500000.0, 500000.0, 0.0],
        });
        let meta = serde_json::json!({ "config": { "text_config": tc } });
        GlimmerConfig::from_metadata_json(&meta.to_string()).unwrap()
    }

    fn dtypes(cfg: &GlimmerConfig) -> Vec<[DType; 8]> {
        vec![[DType::MQ4G256; 8]; cfg.n_layers]
    }

    fn tp(parts: usize) -> DeviceMesh {
        DeviceMesh::rect(&[(DimKind::Tp, parts)]).unwrap()
    }

    /// Every rank runs the whole program; each sublayer's partial is reduced
    /// in place by one rank-ordered `AllReduce` right before its epilogue.
    #[test]
    fn tp_plan_reduces_each_sublayer_partial_before_its_epilogue() {
        let cfg = config(8, 4, 2048);
        let (_, ops, _) = tp_program(&cfg, &dtypes(&cfg)).unwrap();
        for parts in [2, 4] {
            let plan = glimmer_tp_plan(&cfg, &dtypes(&cfg), &tp(parts)).unwrap();
            assert_eq!(plan.collectives.len(), 2 * cfg.n_layers);
            let mut expected = Vec::new();
            let mut next = 0;
            for (index, op) in ops.iter().enumerate() {
                if let TpOp::Epilogue { mlp, .. } = op {
                    let c = &plan.collectives[next];
                    let operand = if *mlp {
                        GLIMMER_FFN_OUT_RESOURCE
                    } else {
                        GLIMMER_MIX_OUT_RESOURCE
                    };
                    assert_eq!((c.kind, c.operand), (CollectiveKind::AllReduce, operand));
                    assert_eq!(c.group, (0..parts).collect::<Vec<_>>());
                    expected.push(RankItem::Collective(c.id));
                    next += 1;
                }
                expected.push(RankItem::Op(OpId(index as u32)));
            }
            for rank in &plan.ranks {
                assert_eq!(rank.items, expected);
            }
        }
    }

    /// The Glimmer 30B shape splits over two ranks only; uneven head or
    /// G256 splits and non-TP meshes are refused before any load.
    #[test]
    fn tp_plan_refusals() {
        let glimmer30b = config(32, 2, 19968);
        let d = dtypes(&glimmer30b);
        assert!(glimmer_tp_plan(&glimmer30b, &d, &tp(2)).is_ok());
        let local = rank_config(&glimmer30b, 2).unwrap();
        assert_eq!(
            (local.n_heads, local.n_kv_heads, local.hidden_dim),
            (16, 1, 9984)
        );
        // TP4: two KV heads do not split four ways.
        let error = glimmer_tp_plan(&glimmer30b, &d, &tp(4)).unwrap_err();
        assert!(error.contains("n_heads=2 not divisible by Tp=4"), "{error}");
        assert!(rank_config(&glimmer30b, 4).is_err());
        // An intermediate whose halves are not whole G256 groups.
        let ragged = config(4, 4, 1280);
        let error = glimmer_tp_plan(&ragged, &dtypes(&ragged), &tp(2)).unwrap_err();
        assert!(
            error.contains("extent 1280 is not divisible by parts 2 x align 256"),
            "{error}"
        );
        assert!(rank_config(&ragged, 2).is_err());
        // Not a pure TP mesh.
        let pp = DeviceMesh::rect(&[(DimKind::Pp, 2)]).unwrap();
        let error = glimmer_tp_plan(&glimmer30b, &d, &pp).unwrap_err();
        assert!(error.contains("pure TP mesh"), "{error}");
    }
}

/// GPU parity of the TP route (#927 item 5). The executor arm (one emulated
/// or real device per rank) must be bitwise equal to the single-device
/// rank-order oracle: the same rank programs on one device, each partial
/// folded `p0 + p1 + …` in rank order with the executor's kernels. Both are
/// reported against the plain single card (tokens, max logit difference).
/// Every arm runs the same prompt and the plain card's greedy tokens
/// (teacher-forced).
#[cfg(test)]
mod hw_tests {
    use super::*;
    use crate::forward::{decode_step, prefill_with_capture};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::mesh_placement::{AdmittedMesh, AdmittedRank};
    use hipfire_runtime::multi_gpu::Gpus;
    use hipfire_runtime::tokenizer::Tokenizer;

    const MODEL_ENV: &str = "HIPFIRE_GLIMMER_TP_MODEL";
    /// Forced tokens that run as 1-row Prefill epochs instead of decode: one
    /// right after the prompt, one after decode epochs.
    const ONE_ROW_PREFILL: [usize; 2] = [0, 8];

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    }

    fn download(gpu: &Gpu, t: &GpuTensor, n: usize) -> Vec<f32> {
        gpu.bind_thread().unwrap();
        gpu.hip.device_synchronize().unwrap();
        gpu.download_f32(t).unwrap()[..n].to_vec()
    }

    fn argmax(row: &[f32]) -> u32 {
        (0..row.len()).fold(0, |best, i| if row[i] > row[best] { i } else { best }) as u32
    }

    fn fnv(rows: &[Vec<f32>]) -> u64 {
        rows.iter()
            .flatten()
            .fold(0xcbf2_9ce4_8422_2325u64, |h, v| {
                (h ^ u64::from(v.to_bits())).wrapping_mul(0x100_0000_01b3)
            })
    }

    fn epoch(
        mesh: &DeviceMesh,
        op_id: u64,
        route: MeshRoute,
        start: usize,
        rows: usize,
    ) -> ForwardEpoch {
        ForwardEpoch {
            mesh_epoch: mesh.epoch(),
            load_id: 1,
            request_id: 1,
            op_id,
            route,
            start_position: start,
            actual_rows: rows,
            route_rows: rows,
            policy_id: 1,
        }
    }

    /// The epochs of the TP arms: the prompt as prefill epochs of at most
    /// `chunk` rows, then one epoch per forced token (a 1-row Prefill at
    /// [`ONE_ROW_PREFILL`], else a decode). Keeps the last prompt epoch's
    /// logits and every forced token's, the rows the plain card produces.
    fn rows_of(
        prompt: &[u32],
        chunk: usize,
        forced: &[u32],
        mut run: impl FnMut(&[u32], MeshRoute, usize) -> Vec<f32>,
    ) -> Vec<Vec<f32>> {
        let mut last = Vec::new();
        for (index, epoch) in prompt.chunks(chunk).enumerate() {
            last = run(epoch, MeshRoute::Prefill, index * chunk);
        }
        let mut rows = vec![last];
        for (step, token) in forced.iter().enumerate() {
            let route = if ONE_ROW_PREFILL.contains(&step) {
                MeshRoute::Prefill
            } else {
                MeshRoute::Decode
            };
            rows.push(run(std::slice::from_ref(token), route, prompt.len() + step));
        }
        rows
    }

    /// One epoch of the rank programs on one device, in plan order: each op
    /// segment on every rank, then each collective as the rank-order fold
    /// `sum = p0; sum += p1; …` copied back into every rank's partial.
    fn oracle_epoch(
        gpu: &mut Gpu,
        bindings: &mut GlimmerTpBindings<'_>,
        plan: &SplitPlan,
        e: &ForwardEpoch,
        sum: &GpuTensor,
        dim: usize,
    ) {
        let parts = plan.ranks.len();
        bindings.prepare_epoch(e).unwrap();
        let items = &plan.ranks[0].items;
        let mut i = 0;
        while i < items.len() {
            match items[i] {
                RankItem::Op(_) => {
                    let mut ops = Vec::new();
                    while let Some(RankItem::Op(op)) = items.get(i) {
                        ops.push(*op);
                        i += 1;
                    }
                    for r in 0..parts {
                        bindings.run_ops(r, gpu, e, &ops).unwrap();
                    }
                }
                RankItem::Collective(id) => {
                    let c = &plan.collectives[id.0 as usize];
                    let count = e.actual_rows * dim;
                    let views: Vec<GpuTensor> = (0..parts)
                        .map(|r| {
                            bindings
                                .boundary(r, c, e)
                                .unwrap()
                                .view
                                .sub_offset(0, count)
                        })
                        .collect();
                    let total = sum.sub_offset(0, count);
                    gpu.copy_f32_buffer(&total, &views[0], count).unwrap();
                    for addend in &views[1..] {
                        gpu.add_inplace_f32(&total, addend).unwrap();
                    }
                    for view in &views {
                        gpu.copy_f32_buffer(view, &total, count).unwrap();
                    }
                    i += 1;
                }
            }
        }
        for r in 0..parts {
            bindings.apply_rank_marks(r, e).unwrap();
        }
    }

    /// `HIPFIRE_GLIMMER_TP_ROWS` prompt rows (default 16, the text cycled) as
    /// prefill epochs of at most `HIPFIRE_GLIMMER_TP_CHUNK` rows (default 16),
    /// then `HIPFIRE_GLIMMER_TP_GREEDY` forced tokens (default 480), two of
    /// them as 1-row Prefill epochs.
    #[test]
    #[ignore = "GPU: HIPFIRE_GLIMMER_TP_MODEL=<.mq4> with HIPFIRE_EMULATE_GPUS=N (one card) or HIPFIRE_GLIMMER_TP_PARTS=N (N cards)"]
    fn tp_prefill_decode_matches_rank_order_oracle() {
        let parts = std::env::var("HIPFIRE_GLIMMER_TP_PARTS")
            .ok()
            .and_then(|parts| parts.parse().ok())
            .or(hipfire_runtime::config::get().emulate_gpus)
            .expect("set HIPFIRE_EMULATE_GPUS or HIPFIRE_GLIMMER_TP_PARTS");
        let path = std::env::var(MODEL_ENV).expect(MODEL_ENV);
        let hfq = HfqFile::open(std::path::Path::new(&path)).unwrap();
        let global = GlimmerConfig::from_hfq(&hfq).unwrap();
        let (vocab, dim) = (global.vocab_size, global.dim);
        let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).unwrap();
        let text = "The Roman Empire was the post-Republican state of ancient Rome, ruling large territories around the Mediterranean";
        let rows = env_usize("HIPFIRE_GLIMMER_TP_ROWS", 16);
        let chunk = env_usize("HIPFIRE_GLIMMER_TP_CHUNK", 16).min(MAX_BATCH);
        let greedy = env_usize("HIPFIRE_GLIMMER_TP_GREEDY", 480);
        let max_seq = (rows + greedy + 16).next_multiple_of(64);
        let prompt: Vec<u32> = tokenizer
            .encode(text)
            .into_iter()
            .cycle()
            .take(rows)
            .collect();

        // Plain single card, unsharded: batched prefill, then greedy decode;
        // its greedy tokens drive every arm.
        let mut gpu = Gpu::init().unwrap();
        let weights = GlimmerWeights::load(&hfq, &global, &mut gpu).unwrap();
        let mut state = GlimmerState::new_with_max_seq(&mut gpu, &global, max_seq).unwrap();
        let first = prefill_with_capture(
            &global,
            &weights,
            &mut state,
            &mut gpu,
            &prompt,
            0,
            &[],
            &mut Vec::new(),
        )
        .unwrap();
        let mut plain_rows = vec![first[..vocab].to_vec()];
        let mut forced = Vec::new();
        for step in 0..greedy {
            let token = argmax(plain_rows.last().unwrap());
            forced.push(token);
            let pos = (rows + step) as u32;
            let logits = decode_step(&global, &weights, &mut state, &mut gpu, token, pos).unwrap();
            plain_rows.push(logits[..vocab].to_vec());
        }
        state.free_gpu(&mut gpu);
        weights.free_gpu(&mut gpu);
        gpu.drain_pool();
        // A fresh device handle: per-`Gpu` weight caches are keyed by buffer
        // address, which the freed plain weights may hand to the ranks.
        drop(gpu);
        let mut gpu = Gpu::init().unwrap();

        // The rank-order oracle: every rank's shard on this one device.
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, parts)]).unwrap();
        let oracle_rows = {
            let mut ranks: Vec<_> = (0..parts)
                .map(|r| GlimmerTpRank::load(&hfq, &global, &mut gpu, r, parts, max_seq).unwrap())
                .collect();
            let dtypes = layer_dtypes(&ranks[0].weights);
            let plan = glimmer_tp_plan(&global, &dtypes, &mesh).unwrap();
            let sum = gpu.alloc_tensor(&[MAX_BATCH * dim], DType::F32).unwrap();
            let mut op_id = 1;
            let out = rows_of(&prompt, chunk, &forced, |tokens, route, start| {
                op_id += 1;
                let mut bindings = GlimmerTpBindings::new(&global, &mut ranks, &plan, 1).unwrap();
                bindings.set_tokens(tokens);
                let e = epoch(&mesh, op_id, route, start, tokens.len());
                oracle_epoch(&mut gpu, &mut bindings, &plan, &e, &sum, dim);
                drop(bindings);
                let rows: Vec<_> = ranks
                    .iter()
                    .map(|r| download(&gpu, r.logits(), vocab))
                    .collect();
                assert!(
                    rows.iter().all(|row| row == &rows[0]),
                    "oracle ranks disagree"
                );
                rows[0].clone()
            });
            let _ = gpu.free_tensor(sum);
            for rank in ranks {
                rank.free_gpu(&mut gpu);
            }
            out
        };
        drop(gpu);

        // The same shards, one per (emulated) device, through the executor.
        let mut gpus = Gpus::init_tp(parts, 1).unwrap();
        for gpu in &mut gpus.devices {
            gpu.ensure_capture_stream().unwrap();
        }
        let tp_mesh = gpus.mesh.clone();
        let admitted = AdmittedMesh {
            mesh: tp_mesh.clone(),
            ranks: gpus
                .devices
                .iter()
                .enumerate()
                .map(|(logical, gpu)| AdmittedRank {
                    logical,
                    device_ordinal: gpu.device_id,
                    pci_bus_id: None,
                    arch: gpu.arch.clone(),
                })
                .collect(),
            edges: Vec::new(),
            family: "glimmer-tp".into(),
            numerical_policy: "f32".into(),
        };
        let max_bytes = (chunk * dim * std::mem::size_of::<f32>()).next_multiple_of(16);
        let placement = BandPlacement::uniform(&tp_mesh, 1).unwrap();
        gpus.provision_mesh_edges(&admitted, &placement, max_bytes)
            .unwrap();
        let mut ranks: Vec<_> = (0..parts)
            .map(|r| {
                GlimmerTpRank::load(&hfq, &global, &mut gpus.devices[r], r, parts, max_seq).unwrap()
            })
            .collect();
        let dtypes = layer_dtypes(&ranks[0].weights);
        let tp_plan = glimmer_tp_plan(&global, &dtypes, &tp_mesh).unwrap();
        let tp_epoch = gpus.mesh_tp_epoch().unwrap();
        // Refusals before any launch leave every rank at position 0 and the
        // mesh healthy: a prefill epoch past the batch cap, an epoch whose
        // tokens disagree with its rows, a rank set loaded out of order,
        // session capture and prefix repair.
        {
            let long = vec![prompt[0]; MAX_BATCH + 1];
            let mut bindings =
                GlimmerTpBindings::new(&global, &mut ranks, &tp_plan, tp_epoch).unwrap();
            bindings.set_tokens(&long);
            let e = epoch(&tp_mesh, 1, MeshRoute::Prefill, 0, MAX_BATCH + 1);
            let error = gpus
                .submit_mesh_epoch(&tp_plan, &e, &mut bindings)
                .unwrap_err();
            assert!(error.contains("at most"), "{error}");
            bindings.set_tokens(&prompt[..2]);
            let e = epoch(&tp_mesh, 1, MeshRoute::Prefill, 0, 1);
            let error = gpus
                .submit_mesh_epoch(&tp_plan, &e, &mut bindings)
                .unwrap_err();
            assert!(error.contains("tokens disagree"), "{error}");
            let e = epoch(&tp_mesh, 1, MeshRoute::Decode, 0, 1);
            let error = bindings
                .capture_session_rank(0, &mut gpus.devices[0], &e)
                .unwrap_err();
            assert!(error.contains("does not admit session capture"), "{error}");
            let error = gpus
                .repair_mesh_prefix(&mut bindings, &e, 0, &[])
                .unwrap_err();
            assert!(error.contains("not admitted"), "{error}");
        }
        ranks.swap(0, 1);
        let error = GlimmerTpBindings::new(&global, &mut ranks, &tp_plan, tp_epoch)
            .err()
            .unwrap();
        assert!(error.contains("was not loaded as rank 0"), "{error}");
        ranks.swap(0, 1);
        assert!(
            gpus.mesh_tp_epoch().is_some(),
            "a refusal quarantined the mesh"
        );
        assert!(ranks.iter().all(|rank| rank.position == 0));
        let mut op_id = 1;
        let tp_rows = rows_of(&prompt, chunk, &forced, |tokens, route, start| {
            op_id += 1;
            let mut bindings =
                GlimmerTpBindings::new(&global, &mut ranks, &tp_plan, tp_epoch).unwrap();
            bindings.set_tokens(tokens);
            let e = epoch(&tp_mesh, op_id, route, start, tokens.len());
            let ticket = gpus.submit_mesh_epoch(&tp_plan, &e, &mut bindings).unwrap();
            drop(bindings);
            gpus.drain_mesh_epoch(ticket).unwrap();
            let rows: Vec<_> = gpus
                .devices
                .iter()
                .zip(&ranks)
                .map(|(gpu, r)| download(gpu, r.logits(), vocab))
                .collect();
            assert!(rows.iter().all(|row| row == &rows[0]), "TP ranks disagree");
            rows[0].clone()
        });
        assert!(ranks.iter().all(|rank| rank.position == rows + greedy));
        gpus.drain_mesh_edges().unwrap();
        for (rank, gpu) in ranks.into_iter().zip(&mut gpus.devices) {
            rank.free_gpu(gpu);
        }

        let ids = |rows: &[Vec<f32>]| rows.iter().map(|row| argmax(row)).collect::<Vec<_>>();
        let max_diff = tp_rows
            .iter()
            .flatten()
            .zip(plain_rows.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let (tp_ids, plain_ids) = (ids(&tp_rows), ids(&plain_rows));
        let same = tp_ids
            .iter()
            .zip(&plain_ids)
            .filter(|(a, b)| a == b)
            .count();
        eprintln!(
            "TP{parts} {} rows={rows} chunk={chunk} greedy={greedy}: {} collectives/epoch; oracle fnv64 {:016x}, tp fnv64 {:016x}, plain fnv64 {:016x}; \
             argmax {same}/{} equal to plain; max |logit diff| vs plain {max_diff:e}",
            std::path::Path::new(&path).file_name().unwrap().to_string_lossy(),
            tp_plan.collectives.len(),
            fnv(&oracle_rows),
            fnv(&tp_rows),
            fnv(&plain_rows),
            tp_rows.len(),
        );
        eprintln!("TP{parts} argmax {tp_ids:?}");
        eprintln!("plain argmax {plain_ids:?}");
        eprintln!(
            "TP{parts} text: {:?} | plain text: {:?}",
            tokenizer.decode(&tp_ids),
            tokenizer.decode(&plain_ids),
        );
        let bits = |rows: &[Vec<f32>]| {
            rows.iter()
                .map(|row| row.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        };
        assert!(
            bits(&tp_rows) == bits(&oracle_rows),
            "TP{parts} executor vs single-device rank-order oracle"
        );
    }
}
