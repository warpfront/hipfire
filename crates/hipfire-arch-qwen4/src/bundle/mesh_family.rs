// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4's engine-loop family contract ([`MeshFamily`]): what the engine
//! needs to run a Single load on a one-owner mesh. Declarations and adapters
//! only; the runner (`hipfire_runtime::mesh_family::MeshRunner`) plans,
//! submits, drains, recovers and resets.

use super::*;
use crate::gpu_forward::Qwen4MeshHead;
use crate::mesh_bindings::{Qwen4MeshBindings, Qwen4RankBand};
use crate::program::qwen4_step_graph_with_output;
use hipfire_dispatch::pipeline::step_graph::StepGraph;
use hipfire_runtime::mesh_family::{EpochInput, MeshFamily};
use hipfire_runtime::multi_gpu::executor::{MeshStepBindings, Result};
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::weight_manifest::{StateEntry, WeightEntry};

fn text(error: impl std::fmt::Display) -> String {
    error.to_string()
}

/// The one band of a one-owner mesh: device 0.
fn device(gpus: &mut Gpus) -> Result<&mut Gpu> {
    match gpus.devices.as_mut_slice() {
        [gpu] => Ok(gpu),
        _ => Err("a Qwen4 Single load runs on a one-owner mesh".into()),
    }
}

impl Qwen4Bundle {
    /// The Single load's forward, taken out for a call that also borrows the
    /// bundle, and put back.
    fn with_forward<R>(
        &mut self,
        body: impl FnOnce(&mut Self, &mut Qwen4GpuForward) -> Result<R>,
    ) -> Result<R> {
        let mut forward = self
            .execution
            .take()
            .ok_or("Qwen4 forward resources are not attached")?;
        let result = body(self, &mut forward);
        self.execution = Some(forward);
        result
    }
}

impl MeshFamily for Qwen4Bundle {
    fn step_graph(&self, head: bool, top1: bool) -> Result<StepGraph> {
        qwen4_step_graph_with_output(&self.config, MeshRoute::Prefill, head, top1)
    }

    fn manifest(&self) -> (&[WeightEntry], &[StateEntry]) {
        let manifest = Qwen4Bundle::manifest(self);
        (&manifest.weights, &manifest.state)
    }

    fn n_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn admit(&self, gpus: &Gpus) -> Result<()> {
        let [gpu] = gpus.devices.as_slice() else {
            return Err("a Qwen4 Single load runs on a one-owner mesh".into());
        };
        if self.slowlink_profile().is_some() {
            return Err("the S1 slow-link prompt route is not a mesh program".into());
        }
        let forward = self
            .execution
            .as_ref()
            .ok_or("Qwen4 forward resources are not attached")?;
        forward.mesh_admission(self, gpu).map_err(text)
    }

    fn position(&self) -> usize {
        self.state.position
    }

    fn begin_prompt(&mut self, gpus: &mut Gpus, prompt: &[u32], reused: usize) -> Result<()> {
        self.session_begin(device(gpus)?, prompt, reused, SessionRoute::Ar)
            .map_err(text)
    }

    /// The Single prompt schedule: with the NPU trunk planned, its pair
    /// groups and session cuts (`prefill_schedule_rows`); otherwise the next
    /// session boundary, one forward chunk at most.
    fn next_prompt_rows(&mut self, gpus: &mut Gpus, prompt: &[u32]) -> Result<usize> {
        let gpu = device(gpus)?;
        self.poll_npu_trunk(gpu).map_err(text)?;
        let position = self.state.position;
        let chunk = self
            .spec_chunk_rows()
            .ok_or("Qwen4 forward resources are not attached")?;
        let end = if self.npu_trunk_planned() {
            position + self.prefill_schedule_rows(position, prompt.len(), chunk)
        } else {
            self.session_next_boundary()
                .unwrap_or(prompt.len())
                .min(position + chunk)
        };
        self.set_ple_lookahead(&prompt[end.min(prompt.len())..]);
        Ok(end.saturating_sub(position))
    }

    fn prompt_rows_done(&mut self, gpus: &mut Gpus, prompt: &[u32], end: usize) -> Result<()> {
        let gpu = device(gpus)?;
        if self.session_next_boundary() == Some(end) {
            self.session_at_boundary(gpu, &prompt[..end])
                .map_err(text)?;
        }
        if end == prompt.len() {
            // The terminal fence of the ordinary prompt phase, as `prefill_final`.
            self.seal_gdn_streams(gpu).map_err(text)?;
        }
        Ok(())
    }

    /// An NPU fault latched the route off: the prompt reruns GPU-only from
    /// the re-planned session start (`npu_trunk::containment`).
    fn take_prompt_rerun(&mut self, prompt: &[u32]) -> Option<usize> {
        self.take_npu_trunk_fault()
            .then(|| self.rank_session_plan(prompt, SessionRoute::Ar, prompt.len()))
    }

    fn reserve_decode(&mut self, gpus: &mut Gpus, new_tokens: usize) {
        if let Ok(gpu) = device(gpus) {
            let position = self.state.position;
            Qwen4Bundle::map_ahead_for_decode(self, gpu, position, new_tokens, 0);
        }
    }

    fn prepare_epoch(&mut self, gpus: &mut Gpus, route: MeshRoute, rows: usize) -> Result<usize> {
        let gpu = device(gpus)?;
        let layout = self.with_forward(|bundle, forward| {
            let layout = forward
                .mesh_layout(bundle, gpu, route, rows)
                .map_err(text)?;
            let end = bundle.state.position + rows;
            forward
                .prepare_mesh_capacity(bundle, gpu, end, layout.route_rows)
                .map_err(text)?;
            Ok(layout)
        })?;
        self.mesh_epoch_layout = Some(layout);
        Ok(layout.route_rows)
    }

    fn with_bindings(
        &mut self,
        input: EpochInput<'_>,
        run: &mut dyn FnMut(&mut dyn MeshStepBindings) -> Result<()>,
    ) -> Result<()> {
        let layout = self.mesh_epoch_layout.take();
        self.with_forward(|bundle, forward| {
            let mut bindings = Qwen4MeshBindings::new(vec![Qwen4RankBand { bundle, forward }])?;
            bindings.set_tokens(input.tokens);
            bindings.set_edge_epochs(input.edge_epochs);
            if let Some(layout) = layout {
                bindings.set_rank_layout(0, layout)?;
            }
            bindings.set_head(input.head.map(|head| Qwen4MeshHead {
                logits: head.logits,
                top1: head.top1,
                rows: crate::gpu_forward::Qwen4OutputRows::Final,
            }));
            run(&mut bindings)
        })
    }
}
