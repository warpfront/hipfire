// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MoE-specific PaRo weight loaders for Qwen3.5.
//!
//! These cannot live in `hipfire-runtime` because they return/accept
//! Qwen3.5-specific types: `MoeParoSidecars`, `MoeFfnWeights`, `Qwen35Config`,
//! `SharedExpertWeights`, and `ExpertWeights`.

use crate::qwen35::weights::{MoeExpertSourceRecord, MoeProjectionSource};
use crate::qwen35::{ExpertWeights, MoeFfnWeights, MoeParoSidecars, Qwen35Config};
use hip_bridge::{HipError, HipResult};
use hipfire_runtime::llama::WeightTensor;
use hipfire_runtime::model_source::ModelSource;
use hipfire_runtime::paro::{
    alias_paro_rotation, load_fp16_weight_from_source, paro_load_wt, paro_repack_moe_projection,
    paro_text_prefix,
};
use rdna_compute::{DType, Gpu, GpuTensor};

fn paro_metadata_source(
    source: &dyn ModelSource,
    name: &str,
    fingerprint: &str,
) -> HipResult<MoeProjectionSource> {
    let info = source.tensor_info(name).ok_or_else(|| {
        HipError::new(0, &format!("ParoQuant sealed MoE source not found: {name}"))
    })?;
    let shape = info.shape.clone();
    let rows = *shape
        .first()
        .ok_or_else(|| HipError::new(0, "ParoQuant source has empty shape"))?;
    if rows == 0 || info.data_size == 0 || info.data_size % rows != 0 {
        return Err(HipError::new(
            0,
            &format!("ParoQuant source has invalid encoded row stride: {name}"),
        ));
    }
    let dtype = match info.dtype.as_str() {
        "F16" => DType::F16,
        "BF16" => DType::BF16,
        "F32" => DType::F32,
        _ => DType::F32,
    };
    Ok(MoeProjectionSource {
        name: info.name.clone(),
        fingerprint: fingerprint.to_owned(),
        shape,
        dtype,
        encoded_bytes: info.data_size,
        row_stride: info.data_size / rows,
        alignment: 1,
        quant_tag: format!("paro:{}", info.dtype),
        basis: format!("{:?}", hipfire_dispatch::types::dtype_rotation_plan(dtype)),
        sidecars: Box::new([]),
    })
}

fn paro_source_projection(
    source: &dyn ModelSource,
    name: String,
    rows: usize,
    cols: usize,
    encoded_bytes: usize,
    row_stride: usize,
    sidecars: &[String],
    fingerprint: &str,
) -> HipResult<crate::qwen35::weights::MoeProjectionSource> {
    let info = source.tensor_info(&name).ok_or_else(|| {
        HipError::new(0, &format!("ParoQuant sealed MoE source not found: {name}"))
    })?;
    if rows == 0 || rows % 8 != 0 || cols == 0 {
        return Err(HipError::new(
            0,
            &format!("ParoQuant sealed MoE source dimensions are invalid for {name}"),
        ));
    }
    let packed_shape = vec![cols, rows / 8];
    if info.shape != packed_shape {
        return Err(HipError::new(
            0,
            &format!(
                "ParoQuant sealed MoE source shape {:?} != packed {:?} for {name}",
                info.shape, packed_shape
            ),
        ));
    }
    if info.data_size == 0 || encoded_bytes == 0 || row_stride == 0 {
        return Err(HipError::new(
            0,
            &format!("ParoQuant sealed MoE source has empty encoding: {name}"),
        ));
    }
    Ok(crate::qwen35::weights::MoeProjectionSource {
        name,
        fingerprint: fingerprint.to_owned(),
        shape: vec![rows, cols],
        dtype: DType::ParoQ4G128,
        encoded_bytes,
        row_stride,
        alignment: 1,
        quant_tag: "paro:q4g128".to_string(),
        basis: "Givens".to_string(),
        sidecars: sidecars.to_vec().into_boxed_slice(),
    })
}

/// Upload the per-layer shared PARO rotation sidecars (one tuple for gate||up,
/// one for down). All 256 experts will reference these via non-owning
/// `ParoRotation` aliases.
pub(crate) fn paro_load_moe_shared_sidecars(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    p: &str,
) -> HipResult<MoeParoSidecars> {
    let mp = paro_text_prefix(source)?;
    let base = format!("{mp}.{p}.mlp.experts");
    let qc = source
        .quant_config()
        .ok_or_else(|| HipError::new(0, "ParoQuant: quant_config required"))?;
    let gate_up_sidecar_names = Vec::from([
        format!("{base}.gate_up_weight_pairs"),
        format!("{base}.gate_up_weight_theta"),
        format!("{base}.gate_up_weight_channel_scales"),
    ])
    .into_boxed_slice();
    let down_sidecar_names = Vec::from([
        format!("{base}.down_weight_pairs"),
        format!("{base}.down_weight_theta"),
        format!("{base}.down_weight_channel_scales"),
    ])
    .into_boxed_slice();

    // Sidecars are shared by every routed expert, so stage them as one
    // transaction. If any upload fails, release the already-owned tensors
    // before returning the initiating error.
    let mut loaded = Vec::with_capacity(6);
    for suffix in [
        "gate_up_weight_pairs",
        "gate_up_weight_theta",
        "gate_up_weight_channel_scales",
        "down_weight_pairs",
        "down_weight_theta",
        "down_weight_channel_scales",
    ] {
        let full = format!("{base}.{suffix}");
        let tensor = match source.tensor_data(&full) {
            Some((_, data)) => gpu.upload_raw(data, &[data.len()]),
            None => Err(HipError::new(
                0,
                &format!("ParoQuant MoE shared sidecar not found: {full}"),
            )),
        };
        match tensor {
            Ok(tensor) => loaded.push(tensor),
            Err(error) => {
                let _ = gpu.hip.device_synchronize();
                for tensor in loaded.drain(..) {
                    let _ = gpu.free_tensor(tensor);
                }
                return Err(error);
            }
        }
    }
    let mut loaded = loaded.into_iter();
    Ok(MoeParoSidecars {
        gate_up_pairs: loaded.next().expect("six Paro sidecars staged"),
        gate_up_theta: loaded.next().expect("six Paro sidecars staged"),
        gate_up_channel_scales: loaded.next().expect("six Paro sidecars staged"),
        down_pairs: loaded.next().expect("six Paro sidecars staged"),
        down_theta: loaded.next().expect("six Paro sidecars staged"),
        down_channel_scales: loaded.next().expect("six Paro sidecars staged"),
        krot: qc.krot as u32,
        group_size: qc.group_size,
        gate_up_sidecar_names,
        down_sidecar_names,
        source_fingerprint: crate::qwen35::weights::model_source_fingerprint(source),
    })
}

/// Load the full ParoQuant MoE FFN block for one layer:
///   - dense FP16 router (`mlp.gate.weight [n_exp, hidden]`)
///   - dense FP16 shared-expert scalar gate (`mlp.shared_expert_gate.weight [1, hidden]`)
///   - shared expert (three per-projection PARO tensors: gate, up, down)
///   - routed experts, each with a fused gate||up HFQ4G128 buffer + a down
///     HFQ4G128 buffer, all referencing layer-shared PARO sidecars
pub(crate) fn paro_load_moe_ffn(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    p: &str,
    config: &Qwen35Config,
    layer_idx: u16,
) -> HipResult<MoeFfnWeights> {
    if config.reap_keep.is_some() {
        return Err(HipError::new(
            0,
            "qwen35: ParoQuant REAP compact routing is unsupported; use an HFQ source",
        ));
    }

    let n_exp = config.num_experts;
    let mi = config.moe_intermediate_size;
    let smi = config.shared_expert_intermediate_size;
    let dim = config.dim;
    let qc = source
        .quant_config()
        .ok_or_else(|| HipError::new(0, "ParoQuant MoE requires quant_config"))?;
    let gs = qc.group_size;
    let kr = qc.krot;
    let mp = paro_text_prefix(source)?;
    let group_size = usize::try_from(gs)
        .map_err(|_| HipError::new(0, "ParoQuant group size does not fit usize"))?;
    if group_size == 0 || dim == 0 || mi == 0 || dim % group_size != 0 || mi % group_size != 0 {
        return Err(HipError::new(
            0,
            "ParoQuant MoE dimensions are incompatible with group size",
        ));
    }
    let table_len = n_exp
        .checked_mul(2)
        .ok_or_else(|| HipError::new(0, "ParoQuant expert pointer table size overflows"))?;
    let bytes_per_row_hidden = (dim / group_size)
        .checked_mul(72)
        .ok_or_else(|| HipError::new(0, "ParoQuant hidden row size overflows"))?;
    let bytes_per_row_mi = (mi / group_size)
        .checked_mul(72)
        .ok_or_else(|| HipError::new(0, "ParoQuant intermediate row size overflows"))?;
    let expected_gate_bytes = mi
        .checked_mul(bytes_per_row_hidden)
        .ok_or_else(|| HipError::new(0, "ParoQuant gate bytes overflow"))?;
    let expected_down_bytes = dim
        .checked_mul(bytes_per_row_mi)
        .ok_or_else(|| HipError::new(0, "ParoQuant down bytes overflow"))?;
    let expected_gate_up_bytes = expected_gate_bytes
        .checked_mul(2)
        .ok_or_else(|| HipError::new(0, "ParoQuant fused gate/up bytes overflow"))?;

    let base = format!("{mp}.{p}.mlp.experts");
    let gate_up_sidecars = [
        format!("{base}.gate_up_weight_pairs"),
        format!("{base}.gate_up_weight_theta"),
        format!("{base}.gate_up_weight_channel_scales"),
    ];
    let down_sidecars = [
        format!("{base}.down_weight_pairs"),
        format!("{base}.down_weight_theta"),
        format!("{base}.down_weight_channel_scales"),
    ];
    let source_fingerprint = crate::qwen35::weights::model_source_fingerprint(source);
    let mut source_records = Vec::with_capacity(n_exp);
    for x in 0..n_exp {
        let gate_prefix = format!("{base}.{x}.gate_proj");
        let up_prefix = format!("{base}.{x}.up_proj");
        let down_prefix = format!("{base}.{x}.down_proj");
        let gate = paro_source_projection(
            source,
            format!("{gate_prefix}.qweight"),
            mi,
            dim,
            expected_gate_bytes,
            bytes_per_row_hidden,
            &gate_up_sidecars,
            &source_fingerprint,
        )?;
        let up = paro_source_projection(
            source,
            format!("{up_prefix}.qweight"),
            mi,
            dim,
            expected_gate_bytes,
            bytes_per_row_hidden,
            &gate_up_sidecars,
            &source_fingerprint,
        )?;
        let down = paro_source_projection(
            source,
            format!("{down_prefix}.qweight"),
            dim,
            mi,
            expected_down_bytes,
            bytes_per_row_mi,
            &down_sidecars,
            &source_fingerprint,
        )?;
        source_records.push(MoeExpertSourceRecord {
            gate_up: None,
            gate: Some(gate),
            up: Some(up),
            down,
        });
    }
    let router_source = paro_metadata_source(
        source,
        &format!("{mp}.{p}.mlp.gate.weight"),
        &source_fingerprint,
    )?;
    let mut sidecar_sources = Vec::with_capacity(gate_up_sidecars.len() + down_sidecars.len());
    for name in gate_up_sidecars.iter().chain(down_sidecars.iter()) {
        sidecar_sources.push(paro_metadata_source(source, name, &source_fingerprint)?);
    }
    let (expert_execution_plan, expert_table, expert_binding) =
        crate::qwen35::weights::build_expert_binding(
            source_records.into_boxed_slice(),
            router_source,
            sidecar_sources,
            layer_idx as usize,
            config.n_layers,
            crate::qwen35::weights::ExpertBindingTarget::Single {
                physical_device: gpu.device_id,
            },
        )?;

    // All allocations after this point are owned by `pending`; every error
    // routes through its rollback so a failed layer can be retried.
    let mut pending = crate::qwen35::load::PendingMoeFfn::new(layer_idx, n_exp);

    // ── Router (FP16 dense in shisa-ai's PARO checkpoint) ──
    pending.router = Some(
        match load_fp16_weight_from_source(
            source,
            gpu,
            &format!("{mp}.{p}.mlp.gate.weight"),
            n_exp,
            dim,
        ) {
            Ok(weight) => weight,
            Err(error) => return Err(pending.rollback(gpu, error)),
        },
    );

    // Scalar gate on the shared-expert add — also FP16 dense.
    pending.shared_gate_scalar = Some(
        match load_fp16_weight_from_source(
            source,
            gpu,
            &format!("{mp}.{p}.mlp.shared_expert_gate.weight"),
            1,
            dim,
        ) {
            Ok(weight) => weight,
            Err(error) => return Err(pending.rollback(gpu, error)),
        },
    );

    // ── Shared expert ──
    pending.shared_gate = Some(
        match paro_load_wt(
            source,
            gpu,
            &format!("{p}.mlp.shared_expert.gate_proj"),
            smi,
            dim,
            gs,
            kr,
        ) {
            Ok(weight) => weight,
            Err(error) => return Err(pending.rollback(gpu, error)),
        },
    );
    pending.shared_up = Some(
        match paro_load_wt(
            source,
            gpu,
            &format!("{p}.mlp.shared_expert.up_proj"),
            smi,
            dim,
            gs,
            kr,
        ) {
            Ok(weight) => weight,
            Err(error) => return Err(pending.rollback(gpu, error)),
        },
    );
    pending.shared_down = Some(
        match paro_load_wt(
            source,
            gpu,
            &format!("{p}.mlp.shared_expert.down_proj"),
            dim,
            smi,
            gs,
            kr,
        ) {
            Ok(weight) => weight,
            Err(error) => return Err(pending.rollback(gpu, error)),
        },
    );

    // ── Routed experts ──
    pending.paro_shared = Some(match paro_load_moe_shared_sidecars(source, gpu, p) {
        Ok(sidecars) => sidecars,
        Err(error) => return Err(pending.rollback(gpu, error)),
    });

    for x in 0..n_exp {
        let gate_prefix = format!("{mp}.{p}.mlp.experts.{x}.gate_proj");
        let up_prefix = format!("{mp}.{p}.mlp.experts.{x}.up_proj");
        let down_prefix = format!("{mp}.{p}.mlp.experts.{x}.down_proj");

        let gate_bytes = match paro_repack_moe_projection(source, &gate_prefix, mi, dim, group_size)
        {
            Ok(bytes) => bytes,
            Err(error) => return Err(pending.rollback(gpu, error)),
        };
        if gate_bytes.len() != expected_gate_bytes {
            return Err(pending.rollback(
                gpu,
                HipError::new(
                    0,
                    &format!(
                        "ParoQuant gate bytes {} != expected {expected_gate_bytes} for expert {x}",
                        gate_bytes.len()
                    ),
                ),
            ));
        }
        let up_bytes = match paro_repack_moe_projection(source, &up_prefix, mi, dim, group_size) {
            Ok(bytes) => bytes,
            Err(error) => return Err(pending.rollback(gpu, error)),
        };
        if up_bytes.len() != expected_gate_bytes {
            return Err(pending.rollback(
                gpu,
                HipError::new(
                    0,
                    &format!(
                        "ParoQuant up bytes {} != expected {expected_gate_bytes} for expert {x}",
                        up_bytes.len()
                    ),
                ),
            ));
        }
        let gate_up_len = match gate_bytes.len().checked_add(up_bytes.len()) {
            Some(len) if len == expected_gate_up_bytes => len,
            Some(len) => {
                return Err(pending.rollback(
                    gpu,
                    HipError::new(
                        0,
                        &format!(
                            "ParoQuant fused gate/up bytes {len} != expected {expected_gate_up_bytes} for expert {x}"
                        ),
                    ),
                ))
            }
            None => {
                return Err(pending.rollback(
                    gpu,
                    HipError::new(0, "ParoQuant fused gate/up bytes overflow"),
                ))
            }
        };
        let mut gate_up_bytes = Vec::with_capacity(gate_up_len);
        gate_up_bytes.extend_from_slice(&gate_bytes);
        gate_up_bytes.extend_from_slice(&up_bytes);
        let gate_up_buf = match gpu.upload_raw(&gate_up_bytes, &[gate_up_len]) {
            Ok(buffer) => buffer,
            Err(error) => return Err(pending.rollback(gpu, error)),
        };

        let down_bytes = match paro_repack_moe_projection(source, &down_prefix, dim, mi, group_size)
        {
            Ok(bytes) => bytes,
            Err(error) => {
                let _ = gpu.free_tensor(gate_up_buf);
                return Err(pending.rollback(gpu, error));
            }
        };
        if down_bytes.len() != expected_down_bytes {
            let _ = gpu.free_tensor(gate_up_buf);
            return Err(pending.rollback(
                gpu,
                HipError::new(
                    0,
                    &format!(
                        "ParoQuant down bytes {} != expected {expected_down_bytes} for expert {x}",
                        down_bytes.len()
                    ),
                ),
            ));
        }
        let down_buf = match gpu.upload_raw(&down_bytes, &[down_bytes.len()]) {
            Ok(buffer) => buffer,
            Err(error) => {
                let _ = gpu.free_tensor(gate_up_buf);
                return Err(pending.rollback(gpu, error));
            }
        };

        let gate_up = {
            let shared = pending.paro_shared.as_ref().expect("Paro sidecars staged");
            WeightTensor {
                buf: gate_up_buf,
                gpu_dtype: DType::ParoQ4G128,
                m: 2 * mi,
                k: dim,
                row_stride: bytes_per_row_hidden,
                paro: Some(alias_paro_rotation(
                    &shared.gate_up_pairs,
                    &shared.gate_up_theta,
                    &shared.gate_up_channel_scales,
                    shared.krot,
                    shared.group_size,
                )),
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
            }
        };
        let down = {
            let shared = pending.paro_shared.as_ref().expect("Paro sidecars staged");
            WeightTensor {
                buf: down_buf,
                gpu_dtype: DType::ParoQ4G128,
                m: dim,
                k: mi,
                row_stride: bytes_per_row_mi,
                paro: Some(alias_paro_rotation(
                    &shared.down_pairs,
                    &shared.down_theta,
                    &shared.down_channel_scales,
                    shared.krot,
                    shared.group_size,
                )),
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
            }
        };
        pending.experts.push(ExpertWeights { gate_up, down });
    }

    // ── Device-side expert pointer tables ──
    let mut gu_ptrs = Vec::with_capacity(n_exp);
    let mut dn_ptrs = Vec::with_capacity(n_exp);
    for expert in &pending.experts {
        gu_ptrs.push(expert.gate_up.buf.buf.as_ptr() as u64);
        dn_ptrs.push(expert.down.buf.buf.as_ptr() as u64);
    }
    let gu_bytes: Vec<u8> = gu_ptrs.iter().flat_map(|ptr| ptr.to_ne_bytes()).collect();
    let dn_bytes: Vec<u8> = dn_ptrs.iter().flat_map(|ptr| ptr.to_ne_bytes()).collect();
    pending.expert_gate_up_ptrs = Some(match gpu.alloc_tensor(&[table_len], DType::F32) {
        Ok(tensor) => tensor,
        Err(error) => return Err(pending.rollback(gpu, error)),
    });
    let gate_up_copy = {
        let tensor = pending
            .expert_gate_up_ptrs
            .as_ref()
            .expect("pending gate/up pointer table");
        gpu.hip.memcpy_htod(&tensor.buf, &gu_bytes)
    };
    if let Err(error) = gate_up_copy {
        return Err(pending.rollback(gpu, error));
    }
    pending.expert_down_ptrs = Some(match gpu.alloc_tensor(&[table_len], DType::F32) {
        Ok(tensor) => tensor,
        Err(error) => return Err(pending.rollback(gpu, error)),
    });
    let down_copy = {
        let tensor = pending
            .expert_down_ptrs
            .as_ref()
            .expect("pending down pointer table");
        gpu.hip.memcpy_htod(&tensor.buf, &dn_bytes)
    };
    if let Err(error) = down_copy {
        return Err(pending.rollback(gpu, error));
    }

    pending.commit(expert_execution_plan, expert_table, expert_binding, gpu)
}
