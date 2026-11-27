// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Gemma 4's engine-loop family contract ([`MeshFamily`]) on a one-owner
//! mesh: the declared step program, the prompt schedule of
//! [`Gemma4Bundle::prefill`] and the bindings that run one epoch as the
//! legacy forward does. The runner (`hipfire_runtime::mesh_family`) plans,
//! submits, drains and resets.
//!
//! Every epoch runs the same forward the legacy loop runs for those rows
//! (batched prompt chunk, one-row eager step, captured-graph decode) with
//! its logits left on the device, so the engine's tokens are the legacy
//! loop's.

use super::{Gemma4Bundle, SessionDriver};
use crate::config::Gemma4Config;
use crate::forward;
use crate::gemma4::GEMMA4_FORWARD_BATCH_MAX;
use hipfire_dispatch::pipeline::step_graph::{
    OpDecl, OpId, OpKind, ResourceClass, ResourceDecl, SplitRule, StateClass, StepGraph,
};
use hipfire_runtime::mesh_family::{EpochHead, EpochInput, MeshFamily, MeshLanes};
use hipfire_runtime::mesh_split::PlannedCollective;
use hipfire_runtime::multi_gpu::executor::{
    BoundaryIo, ForwardEpoch, MeshCheckpointId, MeshCheckpointKey, MeshRoute, MeshStepBindings,
    Result,
};
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::session_cache::SessionRoute;
use hipfire_runtime::weight_manifest::{StateEntry, StateKind, WeightEntry};
use rdna_compute::Gpu;

/// The one band of a one-owner mesh: device 0.
pub(super) fn device(gpus: &mut Gpus) -> Result<&mut Gpu> {
    match gpus.devices.as_mut_slice() {
        [gpu] => Ok(gpu),
        _ => Err("a Gemma 4 load runs on a one-owner mesh".into()),
    }
}

/// The KV cache state of every layer that writes its own rows (a KV-sharing
/// layer reads its source layer's).
pub(super) fn gemma4_state_manifest(cfg: &Gemma4Config) -> Vec<StateEntry> {
    (0..cfg.n_layers)
        .filter(|&layer| !cfg.is_kv_shared_layer(layer))
        .map(|layer| StateEntry::new(StateKind::Kv { quant: "q8".into() }, layer))
        .collect()
}

/// The declared trunk of one output mode: the embedding (layer 0), per layer
/// the sandwich attention over its KV state, the MLP (dense or parallel MoE)
/// and, on E-series checkpoints, the per-layer input; the head (final norm,
/// LM head, soft cap) and its argmax on the last layer. Whole-trunk ops
/// (`Opaque`): a one-owner mesh runs them as one forward.
fn gemma4_step_graph(cfg: &Gemma4Config, head: bool, top1: bool) -> Result<StepGraph> {
    let layers = cfg.n_layers;
    if layers == 0 || cfg.layer_types.len() != layers {
        return Err(format!(
            "Gemma 4 step graph: {layers} layers with {} layer types",
            cfg.layer_types.len()
        ));
    }
    if top1 && !head {
        return Err("Gemma 4 step graph: top-1 rows require the LM head".into());
    }
    let error = |e| format!("Gemma 4 step graph: {e}");
    let mut graph = StepGraph::new();
    let mut resource = |class: ResourceClass, shape: &[usize]| {
        graph
            .add_resource(ResourceDecl {
                class,
                shape: shape.to_vec(),
                alias: None,
            })
            .map_err(error)
    };
    let x = resource(ResourceClass::Activation, &[cfg.dim])?;
    let tokens = resource(ResourceClass::Input, &[1])?;
    let logits = resource(ResourceClass::Output, &[cfg.vocab_size])?;
    let top1_rows = resource(ResourceClass::Output, &[1])?;
    let kv = (0..layers)
        .map(|layer| {
            (!cfg.is_kv_shared_layer(layer))
                .then(|| {
                    resource(
                        ResourceClass::State {
                            class: StateClass::Kv,
                            layer,
                        },
                        &[cfg.max_kv_dim()],
                    )
                })
                .transpose()
        })
        .collect::<Result<Vec<_>>>()?;
    let mut ops = Vec::new();
    let mut op = |kind, layer, reads: Vec<_>, writes: Vec<_>| {
        ops.push(OpDecl {
            kind,
            layer: Some(layer),
            rule: SplitRule::Opaque,
            reads,
            writes,
        })
    };
    op(OpKind::Embed, 0, vec![tokens], vec![x]);
    for layer in 0..layers {
        let attention = match kv[layer] {
            Some(own) => (vec![x, own], vec![x, own]),
            None => {
                let source = cfg
                    .kv_shared_source_layer_idx(layer)
                    .and_then(|source| kv[source])
                    .ok_or_else(|| format!("Gemma 4 layer {layer} has no KV source"))?;
                (vec![x, source], vec![x])
            }
        };
        op(OpKind::SandwichAttention, layer, attention.0, attention.1);
        let mlp = if cfg.enable_moe_block {
            OpKind::ParallelMoeMlp
        } else {
            OpKind::GatedMlp
        };
        op(mlp, layer, vec![x], vec![x]);
        if cfg.hidden_size_per_layer_input != 0 {
            op(OpKind::PerLayerInput, layer, vec![x], vec![x]);
        }
    }
    if head {
        op(OpKind::Project, layers - 1, vec![x], vec![logits]);
        if top1 {
            op(OpKind::Argmax, layers - 1, vec![logits], vec![top1_rows]);
        }
    }
    for decl in ops {
        graph.add_op(decl).map_err(error)?;
    }
    Ok(graph)
}

impl MeshFamily for Gemma4Bundle {
    fn step_graph(&self, head: bool, top1: bool) -> Result<StepGraph> {
        gemma4_step_graph(&self.config, head, top1)
    }

    /// The one-owner mesh places no weight (they stay where the load put
    /// them); a multi-owner Gemma 4 mesh would declare them here.
    fn manifest(&self) -> (&[WeightEntry], &[StateEntry]) {
        (&[], &self.mesh_state)
    }

    fn n_layers(&self) -> usize {
        self.config.n_layers
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn admit(&self, gpus: &Gpus) -> Result<()> {
        match gpus.devices.len() {
            1 => Ok(()),
            n => Err(format!(
                "a Gemma 4 load runs on a one-owner mesh, not {n} devices"
            )),
        }
    }

    fn position(&self) -> usize {
        self.state.n_tokens
    }

    /// As [`Gemma4Bundle::prefill`] begins: resume, restore or start cold.
    fn begin_prompt(&mut self, gpus: &mut Gpus, prompt: &[u32], reused: usize) -> Result<()> {
        if reused >= prompt.len() {
            return Err(format!(
                "gemma4 prefill: reusing {reused} of {} prompt tokens leaves nothing to compute",
                prompt.len()
            ));
        }
        let gpu = device(gpus)?;
        match self.with_session(|session, bundle| {
            session.begin(gpu, bundle, prompt, SessionRoute::Ar, reused, false)
        }) {
            Some(result) => result.map(drop),
            None if reused == 0 => {
                self.state.reset();
                Ok(())
            }
            None => Err(format!(
                "gemma4 session cache is not attached; cannot reuse {reused} tokens"
            )),
        }
    }

    fn next_prompt_rows(&mut self, _gpus: &mut Gpus, prompt: &[u32]) -> Result<usize> {
        let stop = self.session.as_ref().and_then(SessionDriver::next_boundary);
        Ok(prompt_epoch_rows(
            self.state.n_tokens,
            stop,
            prompt.len(),
            self.prefill_rows,
        ))
    }

    fn prompt_rows_done(&mut self, gpus: &mut Gpus, prompt: &[u32], end: usize) -> Result<()> {
        if self.session.as_ref().and_then(SessionDriver::next_boundary) != Some(end) {
            return Ok(());
        }
        let gpu = device(gpus)?;
        self.with_session(|session, bundle| session.at_boundary(gpu, bundle, &prompt[..end]))
            .unwrap_or(Ok(()))
    }

    /// Only a snapshot boundary needs the drain; the chunks between queue back
    /// to back, as the legacy prefill (HIPF-589).
    fn prompt_epoch_drains(&self, end: usize) -> bool {
        self.session.as_ref().and_then(SessionDriver::next_boundary) == Some(end)
    }

    fn prepare_epoch(&mut self, _gpus: &mut Gpus, _route: MeshRoute, rows: usize) -> Result<usize> {
        Ok(rows)
    }

    fn with_bindings(
        &mut self,
        input: EpochInput<'_>,
        run: &mut dyn FnMut(&mut dyn MeshStepBindings) -> Result<()>,
    ) -> Result<()> {
        if input.device_token.is_some() {
            return Err("a Gemma 4 decode epoch takes its token from the host".into());
        }
        run(&mut Gemma4MeshBindings {
            bundle: self,
            tokens: input.tokens,
            head: input.head,
            prepared: None,
            ran: false,
        })
    }

    /// The staged engine lanes (`serve.vmm_batch` on the engine loop).
    fn mesh_lanes(&mut self) -> Option<&mut dyn MeshLanes> {
        self.lanes.is_some().then_some(self as &mut dyn MeshLanes)
    }
}

const NO_MESH: &str = "the Gemma 4 one-owner mesh keeps its session and rollback on the bundle";

/// One epoch of a Gemma 4 load on its one-owner mesh.
struct Gemma4MeshBindings<'a> {
    bundle: &'a mut Gemma4Bundle,
    tokens: &'a [u32],
    head: Option<EpochHead<'a>>,
    prepared: Option<ForwardEpoch>,
    ran: bool,
}

impl Gemma4MeshBindings<'_> {
    fn require_prepared(&self, epoch: &ForwardEpoch) -> Result<()> {
        if self.prepared != Some(*epoch) {
            return Err("Gemma 4 mesh epoch has not been prepared".into());
        }
        Ok(())
    }
}

impl MeshStepBindings for Gemma4MeshBindings<'_> {
    fn prepare_epoch(&mut self, epoch: &ForwardEpoch) -> Result<()> {
        self.prepared = None;
        let rows = epoch.actual_rows;
        let state = &self.bundle.state;
        if rows == 0
            || rows > GEMMA4_FORWARD_BATCH_MAX
            || epoch.route_rows != rows
            || self.tokens.len() != rows
            || epoch.route == MeshRoute::Verify
            || (epoch.route == MeshRoute::Decode && rows != 1)
            || epoch.start_position != state.n_tokens
            || epoch.start_position + rows > state.max_seq
        {
            return Err(format!(
                "invalid Gemma 4 mesh epoch: {:?} of {rows} rows at {} (state at {}, capacity {})",
                epoch.route, epoch.start_position, state.n_tokens, state.max_seq
            ));
        }
        if self
            .tokens
            .iter()
            .any(|&token| token as usize >= self.bundle.config.vocab_size)
        {
            return Err("Gemma 4 mesh token is outside the vocabulary".into());
        }
        self.ran = false;
        self.prepared = Some(*epoch);
        Ok(())
    }

    fn run_ops(
        &mut self,
        owner: usize,
        gpu: &mut Gpu,
        epoch: &ForwardEpoch,
        _ops: &[OpId],
    ) -> Result<()> {
        self.require_prepared(epoch)?;
        if owner != 0 || self.ran {
            return Err("the Gemma 4 trunk runs once, on owner 0".into());
        }
        self.ran = true;
        let Gemma4Bundle {
            config,
            weights,
            state,
            ..
        } = &mut *self.bundle;
        let position = epoch.start_position;
        match (epoch.route, self.tokens) {
            (MeshRoute::Decode, &[token]) => {
                forward::decode_step_with_graph_device(
                    config,
                    weights,
                    state,
                    gpu,
                    token,
                    position as u32,
                )?;
            }
            (_, &[token]) => {
                forward::decode_step_device(config, weights, state, gpu, token, position as u32)?
            }
            (_, tokens) => forward::forward_rows_device(
                config,
                weights,
                state,
                gpu,
                tokens,
                position,
                self.head.is_some(),
            )?,
        }
        if let Some(head) = self.head {
            let vocab = config.vocab_size;
            gpu.memcpy_dtod_auto(&head.logits.buf, &state.logits.buf, vocab * 4)
                .map_err(|e| format!("gemma4 mesh head logits: {e:?}"))?;
            if let Some(top1) = head.top1 {
                gpu.argmax_f32_batched(&state.logits, top1, vocab, 1)
                    .map_err(|e| format!("gemma4 mesh head argmax: {e:?}"))?;
            }
        }
        Ok(())
    }

    fn boundary(
        &self,
        _owner: usize,
        _collective: &PlannedCollective,
        _epoch: &ForwardEpoch,
    ) -> Result<BoundaryIo<'_>> {
        Err("a Gemma 4 one-owner mesh has no band boundary".into())
    }

    fn apply_rank_marks(&mut self, owner: usize, epoch: &ForwardEpoch) -> Result<()> {
        self.require_prepared(epoch)?;
        if owner != 0 || !self.ran {
            return Err("Gemma 4 mesh marks before its trunk ran".into());
        }
        if self.bundle.state.n_tokens != epoch.start_position + epoch.actual_rows {
            return Err("Gemma 4 mesh state did not advance by the epoch's rows".into());
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

    /// The rollback: drop the turn's rows and the live session state, as
    /// `ArchModel::reset_session_state` does.
    fn reset_rank(&mut self, owner: usize, _gpu: &mut Gpu, _epoch: &ForwardEpoch) -> Result<()> {
        if owner != 0 {
            return Err("unknown Gemma 4 mesh rank".into());
        }
        self.prepared = None;
        self.bundle.state.reset();
        if let Some(session) = self.bundle.session.as_mut() {
            session.forget_live();
        }
        Ok(())
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
        // Unreachable: `checkpoint_rank` never issues a ticket.
    }
}

/// The rows of the prompt epoch at `position` as [`Gemma4Bundle::prefill`]
/// chunks them: up to the next snapshot boundary `stop` in
/// [`GEMMA4_FORWARD_BATCH_MAX`]-row chunks, the segment that ends the prompt
/// in prefill-`width` chunks (one row each on per-token prefill). The same
/// chunks run the same forwards, so the KV bits are the legacy loop's.
fn prompt_epoch_rows(position: usize, stop: Option<usize>, len: usize, width: usize) -> usize {
    let stop = stop.unwrap_or(len).min(len);
    let width = if stop == len || width == 1 {
        width
    } else {
        GEMMA4_FORWARD_BATCH_MAX
    };
    stop.saturating_sub(position).min(width)
}

#[cfg(test)]
mod tests {
    use super::{gemma4_state_manifest, gemma4_step_graph, prompt_epoch_rows};
    use crate::config::Gemma4Config;
    use hipfire_runtime::device_mesh::DeviceMesh;
    use hipfire_runtime::mesh_placement::BandPlacement;
    use hipfire_runtime::mesh_split::split_program;
    use hipfire_runtime::weight_manifest::plan_manifest;

    /// Every output mode of a KV-sharing checkpoint plans on a one-owner mesh
    /// (the engine's `MeshRunner::begin`).
    #[test]
    fn step_program_plans_on_a_one_owner_mesh() {
        let meta = serde_json::json!({"config": {"text_config": {
            "hidden_size": 16, "num_hidden_layers": 4, "vocab_size": 16,
            "num_attention_heads": 2, "head_dim": 8, "num_key_value_heads": 2,
            "global_head_dim": 8, "num_global_key_value_heads": 2,
            "attention_k_eq_v": true, "intermediate_size": 32,
            "num_kv_shared_layers": 2,
            "layer_types": ["sliding_attention", "full_attention",
                            "sliding_attention", "full_attention"],
        }}});
        let cfg = Gemma4Config::from_metadata_json(&meta.to_string()).unwrap();
        assert!(cfg.is_kv_shared_layer(3), "the test covers a shared layer");
        let mesh = DeviceMesh::single().unwrap();
        let placement = BandPlacement::uniform(&mesh, cfg.n_layers).unwrap();
        let state = gemma4_state_manifest(&cfg);
        assert_eq!(state.len(), 2);
        let planned = plan_manifest(&[], &state, &mesh, cfg.n_layers, &placement).unwrap();
        for (head, top1) in [(false, false), (true, false), (true, true)] {
            let graph = gemma4_step_graph(&cfg, head, top1).unwrap();
            split_program(&graph, &planned, &mesh).unwrap();
        }
    }

    /// Epoch rows from 0 to `len`, with the boundaries the session yields.
    fn walk(len: usize, width: usize, boundaries: &[usize]) -> Vec<usize> {
        let (mut position, mut rows) = (0, Vec::new());
        while position < len {
            let stop = boundaries.iter().copied().find(|&b| b > position);
            let n = prompt_epoch_rows(position, stop, len, width);
            assert!(n > 0, "an epoch always advances");
            rows.push(n);
            position += n;
        }
        rows
    }

    #[test]
    fn epochs_follow_the_legacy_prefill_chunks() {
        // A stride segment runs in 256-row chunks, the last segment in the
        // prefill width; a lone remainder row is its own (decode-route) epoch.
        assert_eq!(
            walk(1300, 64, &[1024]),
            [&[256; 4][..], &[64; 4], &[20]].concat()
        );
        assert_eq!(walk(1025, 256, &[1024]), [&[256; 4][..], &[1]].concat());
        assert_eq!(walk(513, 64, &[513]), [&[64; 8][..], &[1]].concat());
        // Per-token prefill stays per token on both segments.
        assert_eq!(walk(5, 1, &[3]), [1; 5]);
        // No session: one segment.
        assert_eq!(walk(300, 256, &[]), [256, 44]);
    }
}
