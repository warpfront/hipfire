// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Lab harness of the pinned-host canonical expert owner (expert cache slice S6b). Feature
//! `lab`; consumed by `examples/fn_ds4_mapped_parity.rs`.
//!
//! The production loader fills a whole model; the parity example also needs one layer's routed
//! experts loaded each way (resident device blobs, or the pinned host-mapped blobs of a
//! [`Ds4ExpertSource`]) and one routed-FFN launch over a caller-supplied route. Both go through
//! the production loader functions and the production selected-experts dispatch
//! ([`hipfire_dispatch::families::moe::MoeFamily::run_selected`], the route the
//! heterogeneous G2 split already executes), so the harness adds no kernel path of its own.

use hipfire_dispatch::families::moe::MoeSelectedParams;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::arch::{DeepseekV4, MappedLoad};
use crate::backend::Mq2rBackend;
use crate::deepseek4::{DeepseekV4Config, DeepseekV4LayerWeights};
use crate::expert_source::{
    fresh_owner_generation, owner_id_for, Ds4ExpertSource, Ds4Geometry, HfqIndex, MappedGroup,
    MtpSource, TensorIndex,
};

/// One layer's routed experts loaded by one of the two loaders.
pub struct ProbeLayer {
    layer: DeepseekV4LayerWeights,
    group: Option<MappedGroup>,
}

impl ProbeLayer {
    pub fn layer(&self) -> &DeepseekV4LayerWeights {
        &self.layer
    }

    /// The pinned blobs of a mapped load; `None` for the resident loader.
    pub fn group(&self) -> Option<&MappedGroup> {
        self.group.as_ref()
    }

    /// Release every buffer the layer owns.
    pub fn free(self, gpu: &mut Gpu) {
        self.layer.free_gpu(gpu);
    }
}

/// The trunk-only source of `hfq` (index reads only; no MTP head), under a fresh owner
/// generation.
pub fn trunk_source(hfq: &HfqFile, cfg: &DeepseekV4Config) -> Result<Ds4ExpertSource, String> {
    let trunk = HfqIndex(hfq);
    Ds4ExpertSource::build(
        owner_id_for(&trunk.fingerprint()),
        fresh_owner_generation(),
        Ds4Geometry {
            hidden: cfg.hidden_size,
            moe_intermediate: cfg.moe_intermediate_size,
            n_experts: cfg.n_routed_experts,
            trunk_layers: cfg.num_hidden_layers,
        },
        &trunk,
        MtpSource::None,
    )
    .map_err(|error| error.to_string())
}

/// Load layer `layer_idx`'s routed experts with the resident loader (`source: None`) or into
/// the pinned host-mapped blobs of `source`.
pub fn load_probe_layer(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    cfg: &DeepseekV4Config,
    layer_idx: usize,
    source: Option<&Ds4ExpertSource>,
) -> Result<ProbeLayer, String> {
    let prefix = format!("layers.{layer_idx}");
    let mut layer = DeepseekV4LayerWeights::new_empty(0);
    let mut load = source.map(|source| MappedLoad {
        source: source.clone(),
        groups: Vec::new(),
    });
    let result = DeepseekV4::upload_layer_routed_experts(
        hfq,
        gpu,
        &prefix,
        cfg.n_routed_experts,
        &mut layer,
        None,
        None,
        load.as_mut(),
    );
    match result {
        Ok(()) => Ok(ProbeLayer {
            layer,
            group: load.and_then(|mut load| load.groups.pop()),
        }),
        Err(error) => {
            layer.free_gpu(gpu);
            Err(error)
        }
    }
}

/// Run `layer`'s routed experts over the route (`ids` / `route_weights`, `k` of each) and the
/// rotated input `x_rot` (`hidden` floats) and return the `hidden`-float combined output. The
/// route is supplied, never recomputed, so two loads of the same experts see identical input.
pub fn run_selected(
    cfg: &DeepseekV4Config,
    gpu: &mut Gpu,
    layer: &DeepseekV4LayerWeights,
    x_rot: &[f32],
    ids: &[i32],
    route_weights: &[f32],
) -> Result<Vec<f32>, String> {
    let (k, im, hidden) = (
        cfg.num_experts_per_tok,
        cfg.moe_intermediate_size,
        cfg.hidden_size,
    );
    if ids.len() != k || route_weights.len() != k || x_rot.len() != hidden {
        return Err(format!(
            "run_selected: route of {} ids / {} weights and input of {} floats, expected {k} / \
             {k} / {hidden}",
            ids.len(),
            route_weights.len(),
            x_rot.len()
        ));
    }
    let (Some(gate_up_ptrs), Some(down_ptrs)) = (
        layer.expert_gate_up_ptrs.as_ref(),
        layer.expert_w2_ptrs.as_ref(),
    ) else {
        return Err("run_selected: the layer has no routed expert pointer tables".into());
    };
    let backend = if cfg.mq2r {
        Mq2rBackend::for_verified_mq2r(gpu)
    } else {
        Mq2rBackend::Portable
    };
    let err = |what: &str, e: hip_bridge::HipError| format!("run_selected: {what}: {e:?}");
    let ids_bits: Vec<f32> = ids.iter().map(|&id| f32::from_bits(id as u32)).collect();
    let mut scratch: Vec<GpuTensor> = Vec::new();
    let result = (|| -> Result<Vec<f32>, String> {
        let mut take = |tensor: Result<GpuTensor, hip_bridge::HipError>,
                        what: &str|
         -> Result<usize, String> {
            scratch.push(tensor.map_err(|e| err(what, e))?);
            Ok(scratch.len() - 1)
        };
        let x = take(gpu.upload_f32(x_rot, &[hidden]), "x_rot")?;
        let topk_indices = take(gpu.upload_f32(&ids_bits, &[k]), "topk indices")?;
        let topk_weights = take(gpu.upload_f32(route_weights, &[k]), "topk weights")?;
        let gate_batch = take(gpu.alloc_tensor(&[k, im], DType::F32), "gate batch")?;
        let up_batch = take(gpu.alloc_tensor(&[k, im], DType::F32), "up batch")?;
        let rot_batch = take(gpu.alloc_tensor(&[k, im], DType::F32), "rot batch")?;
        let down_expanded = take(gpu.alloc_tensor(&[k, hidden], DType::F32), "down expanded")?;
        let out = take(gpu.zeros(&[hidden], DType::F32), "out")?;
        let params = MoeSelectedParams {
            hidden,
            mi: im,
            k_top: k,
            swiglu_limit: cfg.swiglu_limit,
            uses_atomic_moe_down: backend.uses_atomic_moe_down(),
            native_mq2_backend: backend.bias_aware_native_backend(),
            nonowned_gate_up_dummy: layer.expert_gate_up_dummy.as_ref(),
            batch_size: 1,
            x_rot: &scratch[x],
            ffn_out: &scratch[out],
            expert_gate_up_ptrs: gate_up_ptrs,
            expert_down_ptrs: down_ptrs,
            topk_indices: &scratch[topk_indices],
            topk_weights: &scratch[topk_weights],
            gate_batch: &scratch[gate_batch],
            up_batch: &scratch[up_batch],
            rot_batch: &scratch[rot_batch],
            down_expanded: &scratch[down_expanded],
        };
        hipfire_runtime::llama::moe_family()
            .run_selected(gpu, &params)
            .map_err(|e| format!("run_selected: selected experts: {e}"))?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| err("device synchronize", e))?;
        gpu.download_f32(&scratch[out])
            .map_err(|e| err("download output", e))
    })();
    for tensor in scratch {
        let _ = gpu.free_tensor(tensor);
    }
    result
}
