// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The Single bundle's forward on a one-owner mesh (G12 engine loop).
//!
//! The bundle stays the Single bundle (its session driver, scope and NPU-free
//! schedule are unchanged); only the forward goes through the generic mesh
//! executor: one band covering the whole trunk on `gpus.devices[0]`, which
//! must be the `Gpu` the bundle was loaded on. Every call submits one epoch and
//! drains it before returning, so the caller observes complete outputs.

use super::*;
use crate::gpu_forward::Qwen4MeshHead;
use crate::mesh_bindings::{Qwen4MeshBindings, Qwen4RankBand};
use crate::program::qwen4_step_graph_with_output;
use hipfire_runtime::mesh_placement::{AdmittedMesh, AdmittedRank, BandPlacement};
use hipfire_runtime::mesh_split::{split_program, SplitPlan};
use hipfire_runtime::multi_gpu::executor::ForwardEpoch;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::weight_manifest::plan_manifest;

/// Numerical-policy identity of every one-owner epoch of this build.
const SINGLE_POLICY_ID: u64 = 1;

/// Split plans and epoch identity of a provisioned one-owner mesh.
pub struct Qwen4SingleMesh {
    /// `[trunk only, logits, logits + top-1]`.
    plans: [SplitPlan; 3],
    load_id: u64,
    request_id: u64,
    next_op: u64,
}

impl Qwen4SingleMesh {
    /// Provision `gpus` (one device, the bundle's) as a one-owner mesh and plan
    /// the bundle's whole-trunk program on it. Installs the device's launch
    /// stream: mesh work never runs on the null stream.
    pub fn provision(bundle: &Qwen4Bundle, gpus: &mut Gpus) -> Result<Self, BundleError> {
        let error =
            |message: String| BundleError::Forward(format!("qwen4 one-owner mesh: {message}"));
        if gpus.devices.len() != 1 || bundle.pp.is_some() || bundle.ep.is_some() {
            return Err(error("needs exactly one device and a Single bundle".into()));
        }
        let gpu = &mut gpus.devices[0];
        gpu.ensure_capture_stream().map_err(BundleError::Hip)?;
        let config = &bundle.config;
        let mesh = gpus.mesh.clone();
        let placement = BandPlacement::uniform(&mesh, config.num_hidden_layers).map_err(error)?;
        let admitted = AdmittedMesh {
            mesh: mesh.clone(),
            ranks: vec![AdmittedRank {
                logical: 0,
                device_ordinal: gpu.device_id,
                pci_bus_id: None,
                arch: gpu.arch.clone(),
            }],
            edges: Vec::new(),
            family: "qwen4".into(),
            numerical_policy: "single".into(),
        };
        // No edge or TP group exists; the capacity only has to be valid.
        gpus.provision_mesh_edges(&admitted, &placement, 16)
            .map_err(error)?;
        let manifest = bundle.manifest();
        let planned = plan_manifest(
            &manifest.weights,
            &manifest.state,
            &mesh,
            config.num_hidden_layers,
            &placement,
        )
        .map_err(error)?;
        let plan = |head, top1| -> Result<SplitPlan, BundleError> {
            let graph = qwen4_step_graph_with_output(config, MeshRoute::Prefill, head, top1)
                .map_err(|e| error(e.to_string()))?;
            split_program(&graph, &planned, &mesh).map_err(|e| error(e.to_string()))
        };
        Ok(Self {
            plans: [plan(false, false)?, plan(true, false)?, plan(true, true)?],
            load_id: super::pp::next_load_id().map_err(error)?,
            request_id: 1,
            next_op: 1,
        })
    }
}

impl Qwen4Bundle {
    /// A Single load (no pipeline mesh, no expert parallelism): the shape the
    /// one-owner mesh serves.
    pub fn single_mesh_eligible(&self) -> bool {
        self.pp.is_none() && self.ep.is_none()
    }

    /// Begin `prompt` on the one-owner mesh: the Single session driver
    /// continues live state, restores the planned `reused` prefix, or resets.
    pub fn single_mesh_prefill_begin(
        &mut self,
        gpus: &mut Gpus,
        prompt: &[u32],
        reused: usize,
    ) -> Result<(), BundleError> {
        self.session_begin(&mut gpus.devices[0], prompt, reused, SessionRoute::Ar)
    }

    /// One prefill tile of `prompt` from the bundle's position: up to the next
    /// session boundary, at most one forward chunk. The final tile writes the
    /// last logits row into `logits` and seals the prompt. Returns the new
    /// position. The same cuts and session captures as `prefill_final`.
    pub fn single_mesh_prefill_step(
        &mut self,
        mesh: &mut Qwen4SingleMesh,
        gpus: &mut Gpus,
        prompt: &[u32],
        logits: &GpuTensor,
    ) -> Result<usize, BundleError> {
        let pos = self.state.position;
        let chunk = self
            .spec_chunk_rows()
            .ok_or_else(|| BundleError::Forward("qwen4 forward is not attached".into()))?;
        if pos >= prompt.len() {
            return Err(BundleError::Forward(format!(
                "qwen4 prefill step at {pos} is past the {}-token prompt",
                prompt.len()
            )));
        }
        let end = self
            .session_next_boundary()
            .unwrap_or(prompt.len())
            .min(pos + chunk)
            .min(prompt.len());
        self.set_ple_lookahead(&prompt[end..]);
        let head = (end == prompt.len()).then_some(Qwen4MeshHead {
            logits,
            top1: None,
            rows: crate::gpu_forward::Qwen4OutputRows::Final,
        });
        self.single_mesh_forward(mesh, gpus, &prompt[pos..end], MeshRoute::Prefill, head)?;
        let gpu = &mut gpus.devices[0];
        if self.session_next_boundary() == Some(end) {
            self.session_at_boundary(gpu, &prompt[..end])?;
        }
        if end == prompt.len() {
            // The terminal fence of the ordinary-prefill phase, as `prefill_final`.
            self.seal_gdn_streams(gpu)?;
        }
        Ok(end)
    }

    /// One decode token through the mesh: `logits` gets its row and the
    /// returned token is the device argmax of it (read back via `top1`).
    pub fn single_mesh_decode(
        &mut self,
        mesh: &mut Qwen4SingleMesh,
        gpus: &mut Gpus,
        token: u32,
        logits: &GpuTensor,
        top1: Option<&GpuTensor>,
    ) -> Result<Option<u32>, BundleError> {
        let head = Qwen4MeshHead {
            logits,
            top1,
            rows: crate::gpu_forward::Qwen4OutputRows::Final,
        };
        self.single_mesh_forward(mesh, gpus, &[token], MeshRoute::Decode, Some(head))?;
        let Some(top1) = top1 else {
            return Ok(None);
        };
        let bytes = gpus.devices[0]
            .download_raw_bytes(top1)
            .map_err(BundleError::Hip)?;
        let id = bytes
            .get(..4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .ok_or_else(|| BundleError::Forward("qwen4 top-1 readback is short".into()))?;
        Ok(Some(id))
    }

    /// One epoch of `tokens` (at most the forward's chunk) on `route` at the
    /// bundle's position, through the mesh executor. `head` names the output
    /// rows (`logits`, optional device argmax `top1`); `None` commits trunk
    /// state only. Returns after the epoch drained.
    pub fn single_mesh_forward(
        &mut self,
        mesh: &mut Qwen4SingleMesh,
        gpus: &mut Gpus,
        tokens: &[u32],
        route: MeshRoute,
        head: Option<Qwen4MeshHead<'_>>,
    ) -> Result<(), BundleError> {
        let error =
            |message: String| BundleError::Forward(format!("qwen4 one-owner mesh: {message}"));
        let mut forward = self
            .execution
            .take()
            .ok_or_else(|| error("forward resources are not attached".into()))?;
        let result = (|| {
            let gpu = &mut gpus.devices[0];
            let layout = forward
                .mesh_layout(self, gpu, route, tokens.len())
                .map_err(|e| error(e.to_string()))?;
            let start = self.state.position;
            forward
                .prepare_mesh_capacity(self, gpu, start + tokens.len(), layout.route_rows)
                .map_err(|e| error(e.to_string()))?;
            let mode = match &head {
                None => 0,
                Some(head) => 1 + usize::from(head.top1.is_some()),
            };
            let op_id = mesh.next_op;
            mesh.next_op = op_id
                .checked_add(1)
                .ok_or_else(|| error("operation ids are exhausted".into()))?;
            let epoch = ForwardEpoch {
                mesh_epoch: gpus.mesh.epoch(),
                load_id: mesh.load_id,
                request_id: mesh.request_id,
                op_id,
                route,
                start_position: start,
                actual_rows: tokens.len(),
                route_rows: layout.route_rows,
                policy_id: SINGLE_POLICY_ID,
            };
            let mut bindings = Qwen4MeshBindings::new(vec![Qwen4RankBand {
                bundle: &mut *self,
                forward: &mut forward,
            }])
            .map_err(error)?;
            bindings.set_tokens(tokens);
            bindings.set_rank_layout(0, layout).map_err(error)?;
            bindings.set_head(head);
            let ticket = gpus
                .submit_mesh_epoch(&mesh.plans[mode], &epoch, &mut bindings)
                .map_err(error)?;
            gpus.drain_mesh_epoch(ticket).map_err(error)
        })();
        self.execution = Some(forward);
        result
    }
}
