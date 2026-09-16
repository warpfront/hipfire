// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Transparent ParoQuant safetensors-Dir weight loading for Cohere2-MoE.
//!
//! Mirrors the qwen35 live Dir path (`qwen35::ParoSource` + `paro_moe`): the
//! heavy lifting (paro→fused-HFQ4G128 repack, rotation-sidecar aliasing) is done
//! by the SHARED `hipfire_runtime::paro` primitives; this module only maps
//! cohere2moe's tensor names + layer structure (parallel block, dense prefix
//! layer, sliding/full layers) and assembles cohere2moe's weight structs.
//!
//! Differences from qwen35's `paro_load_moe_ffn`: Cohere2-MoE has NO shared
//! expert (`num_shared_experts=0`), and uses **plain** RMSNorm (no `+1` bake),
//! so norms load via `paro_load_f32`, not qwen35's `paro_load_norm`.

use crate::cohere2moe::{
    build_moe_static_binding, model_source_metadata, source_fingerprint, source_row_stride,
    Cohere2MoeLayerWeights, Cohere2MoeParoSidecars, Cohere2MoeWeights, DenseFfn, ExpertWeights,
    Ffn, LoadTransaction, MoeFfn,
};
use crate::config::Cohere2MoeConfig;
use hipfire_runtime::llama::WeightTensor;
use hipfire_runtime::model_source::ModelSource;
use hipfire_runtime::paro::{
    alias_paro_rotation, load_fp16_weight_from_source, paro_load_f32, paro_load_wt,
    paro_repack_moe_projection, paro_text_prefix,
};
use rdna_compute::{DType, Gpu};

fn e<T: std::fmt::Debug>(ctx: &str) -> impl Fn(T) -> String + '_ {
    move |err| format!("cohere2moe Dir: {ctx}: {err:?}")
}

/// Load the per-layer shared PARO sidecars for a MoE layer. Every upload is
/// registered before the next source lookup, so a missing sidecar rolls back
/// all sidecars already acquired by this layer.
fn load_moe_sidecars(
    source: &dyn ModelSource,
    gpu: &Gpu,
    mp: &str,
    p: &str,
    tx: &mut LoadTransaction,
) -> Result<Cohere2MoeParoSidecars, String> {
    let base = format!("{mp}.{p}.mlp.experts");
    let load = |name: &str, tx: &mut LoadTransaction| -> Result<usize, String> {
        let full = format!("{base}.{name}");
        let (_, data) = source
            .tensor_data(&full)
            .ok_or_else(|| format!("cohere2moe Dir: MoE sidecar not found: {full}"))?;
        let tensor = gpu.upload_raw(data, &[data.len()]).map_err(e("sidecar"))?;
        Ok(tx.hold_tensor(tensor))
    };
    let gate_up_pairs = load("gate_up_weight_pairs", tx)?;
    let gate_up_theta = load("gate_up_weight_theta", tx)?;
    let gate_up_channel_scales = load("gate_up_weight_channel_scales", tx)?;
    let down_pairs = load("down_weight_pairs", tx)?;
    let down_theta = load("down_weight_theta", tx)?;
    let down_channel_scales = load("down_weight_channel_scales", tx)?;
    Ok(Cohere2MoeParoSidecars {
        gate_up_pairs: tx.take_tensor(gate_up_pairs)?,
        gate_up_theta: tx.take_tensor(gate_up_theta)?,
        gate_up_channel_scales: tx.take_tensor(gate_up_channel_scales)?,
        down_pairs: tx.take_tensor(down_pairs)?,
        down_theta: tx.take_tensor(down_theta)?,
        down_channel_scales: tx.take_tensor(down_channel_scales)?,
    })
}

/// Load the full ParoQuant weights for Cohere2-MoE from a safetensors Dir.
pub fn load_from_source(
    source: &dyn ModelSource,
    cfg: &Cohere2MoeConfig,
    gpu: &mut Gpu,
) -> Result<Cohere2MoeWeights, String> {
    let hidden = cfg.hidden_size;
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let dense_inter = cfg.dense_intermediate_size;
    let moe_inter = cfg.moe_intermediate_size;
    let n_exp = cfg.num_experts;
    let gate_up_rows = moe_inter
        .checked_mul(2)
        .ok_or_else(|| "cohere2moe Dir: gate/up dimension overflows".to_string())?;
    let ptr_count = n_exp
        .checked_mul(2)
        .ok_or_else(|| "cohere2moe Dir: expert pointer-table length overflows".to_string())?;

    let qc = source.quant_config().ok_or_else(|| {
        "cohere2moe Dir: ParoQuant model requires quantization_config".to_string()
    })?;
    let gs = qc.group_size;
    let kr = qc.krot;
    let group_size = usize::try_from(gs)
        .map_err(|_| "cohere2moe Dir: Paro group size does not fit usize".to_string())?;
    if group_size == 0 || hidden % group_size != 0 || moe_inter % group_size != 0 {
        return Err(format!(
            "cohere2moe Dir: Paro dimensions hidden={hidden} moe_inter={moe_inter} are incompatible with group size {group_size}"
        ));
    }
    let bytes_per_row_hidden = (hidden / group_size)
        .checked_mul(72)
        .ok_or_else(|| "cohere2moe Dir: hidden encoded row size overflows".to_string())?;
    let bytes_per_row_mi = (moe_inter / group_size)
        .checked_mul(72)
        .ok_or_else(|| "cohere2moe Dir: expert encoded row size overflows".to_string())?;
    let mp = paro_text_prefix(source).map_err(e("text prefix"))?;
    let mut tx = LoadTransaction::new(gpu);

    let embed_name = format!("{mp}.embed_tokens.weight");
    let (_, embed_bytes) = source
        .tensor_data(&embed_name)
        .ok_or_else(|| "cohere2moe Dir: embed_tokens.weight not found".to_string())?;
    let embed_slot = tx.hold_tensor(
        gpu.upload_raw(embed_bytes, &[embed_bytes.len()])
            .map_err(e("upload embed"))?,
    );
    let lm_head_slot = tx.hold_weight(
        load_fp16_weight_from_source(source, gpu, &embed_name, cfg.vocab_size, hidden)
            .map_err(e("lm_head"))?,
    );
    let embed_dtype = tx
        .weight_ref(lm_head_slot)
        .map(|weight| weight.gpu_dtype)
        .ok_or_else(|| "cohere2moe Dir: lm_head transaction slot is not a weight".to_string())?;
    let final_norm_slot =
        tx.hold_tensor(paro_load_f32(source, gpu, "norm.weight", hidden).map_err(e("final norm"))?);

    let mut layer_slots = Vec::with_capacity(cfg.num_hidden_layers);
    for l in 0..cfg.num_hidden_layers {
        let p = format!("layers.{l}");
        let input_norm_slot = tx.hold_tensor(
            paro_load_f32(source, gpu, &format!("{p}.input_layernorm.weight"), hidden)
                .map_err(e("input norm"))?,
        );
        let wq_slot = tx.hold_weight(
            paro_load_wt(
                source,
                gpu,
                &format!("{p}.self_attn.q_proj"),
                q_dim,
                hidden,
                gs,
                kr,
            )
            .map_err(e("wq"))?,
        );
        let wk_slot = tx.hold_weight(
            paro_load_wt(
                source,
                gpu,
                &format!("{p}.self_attn.k_proj"),
                kv_dim,
                hidden,
                gs,
                kr,
            )
            .map_err(e("wk"))?,
        );
        let wv_slot = tx.hold_weight(
            paro_load_wt(
                source,
                gpu,
                &format!("{p}.self_attn.v_proj"),
                kv_dim,
                hidden,
                gs,
                kr,
            )
            .map_err(e("wv"))?,
        );
        let wo_slot = tx.hold_weight(
            paro_load_wt(
                source,
                gpu,
                &format!("{p}.self_attn.o_proj"),
                hidden,
                q_dim,
                gs,
                kr,
            )
            .map_err(e("wo"))?,
        );

        let ffn = if cfg.is_dense_ffn(l) {
            let gate_slot = tx.hold_weight(
                paro_load_wt(
                    source,
                    gpu,
                    &format!("{p}.mlp.gate_proj"),
                    dense_inter,
                    hidden,
                    gs,
                    kr,
                )
                .map_err(e("dense gate"))?,
            );
            let up_slot = tx.hold_weight(
                paro_load_wt(
                    source,
                    gpu,
                    &format!("{p}.mlp.up_proj"),
                    dense_inter,
                    hidden,
                    gs,
                    kr,
                )
                .map_err(e("dense up"))?,
            );
            let down_slot = tx.hold_weight(
                paro_load_wt(
                    source,
                    gpu,
                    &format!("{p}.mlp.down_proj"),
                    hidden,
                    dense_inter,
                    gs,
                    kr,
                )
                .map_err(e("dense down"))?,
            );
            Ffn::Dense(DenseFfn {
                gate: tx.take_weight(gate_slot)?,
                up: tx.take_weight(up_slot)?,
                down: tx.take_weight(down_slot)?,
            })
        } else {
            let router_slot = tx.hold_weight(
                load_fp16_weight_from_source(
                    source,
                    gpu,
                    &format!("{mp}.{p}.mlp.gate.weight"),
                    n_exp,
                    hidden,
                )
                .map_err(e("router"))?,
            );
            let shared = load_moe_sidecars(source, gpu, &mp, &p, &mut tx)?;
            let shared_slot = tx.hold_sidecars(shared);
            let gate_up_sidecars = vec![
                format!("{mp}.{p}.mlp.experts.gate_up_weight_pairs"),
                format!("{mp}.{p}.mlp.experts.gate_up_weight_theta"),
                format!("{mp}.{p}.mlp.experts.gate_up_weight_channel_scales"),
            ];
            let down_sidecars = vec![
                format!("{mp}.{p}.mlp.experts.down_weight_pairs"),
                format!("{mp}.{p}.mlp.experts.down_weight_theta"),
                format!("{mp}.{p}.mlp.experts.down_weight_channel_scales"),
            ];
            // Build the immutable runtime plan from exact source metadata before
            // the first routed expert is repacked or uploaded.
            let source_capacity = n_exp
                .checked_mul(3)
                .and_then(|count| count.checked_add(7))
                .ok_or_else(|| format!("cohere2moe Dir L{l}: source-name capacity overflows"))?;
            let router_name = format!("{mp}.{p}.mlp.gate.weight");
            let mut source_names = Vec::with_capacity(source_capacity);
            source_names.push(router_name.clone());
            source_names.extend(gate_up_sidecars.iter().cloned());
            source_names.extend(down_sidecars.iter().cloned());
            for x in 0..n_exp {
                let base = format!("{mp}.{p}.mlp.experts.{x}");
                source_names.push(format!("{base}.gate_proj.qweight"));
                source_names.push(format!("{base}.up_proj.qweight"));
                source_names.push(format!("{base}.down_proj.qweight"));
            }
            let fingerprint = source_fingerprint(source, &source_names);
            let source_dtype = |name: &str| -> Result<DType, String> {
                let info = source.tensor_info(name).ok_or_else(|| {
                    format!("cohere2moe Dir L{l}: source tensor not found: {name}")
                })?;
                Ok(match info.dtype.as_str() {
                    "F16" => DType::F16,
                    "BF16" => DType::BF16,
                    "F32" => DType::F32,
                    _ => DType::F32,
                })
            };
            let router_info = source
                .tensor_info(&router_name)
                .ok_or_else(|| format!("cohere2moe Dir L{l}: router source not found"))?;
            let router_rows = *router_info
                .shape
                .first()
                .ok_or_else(|| format!("cohere2moe Dir L{l}: router source shape is empty"))?;
            let router_stride = source_row_stride(router_info.data_size, router_rows)?;
            let router_source = model_source_metadata(
                source,
                &router_name,
                router_info.shape.clone(),
                router_info.data_size,
                router_stride,
                source_dtype(&router_name)?,
                &fingerprint,
                format!("paro:{}", router_info.dtype),
                format!(
                    "{:?}",
                    hipfire_dispatch::types::dtype_rotation_plan(source_dtype(&router_name)?)
                ),
                Vec::new(),
            )?;
            let mut gate_sources = Vec::with_capacity(n_exp);
            let mut up_sources = Vec::with_capacity(n_exp);
            let mut down_sources = Vec::with_capacity(n_exp);
            for x in 0..n_exp {
                let base = format!("{mp}.{p}.mlp.experts.{x}");
                let gate_name = format!("{base}.gate_proj.qweight");
                let up_name = format!("{base}.up_proj.qweight");
                let down_name = format!("{base}.down_proj.qweight");
                let gate_bytes = moe_inter
                    .checked_mul(bytes_per_row_hidden)
                    .ok_or_else(|| format!("cohere2moe Dir L{l}E{x}: gate bytes overflow"))?;
                let down_bytes = hidden
                    .checked_mul(bytes_per_row_mi)
                    .ok_or_else(|| format!("cohere2moe Dir L{l}E{x}: down bytes overflow"))?;
                gate_sources.push(model_source_metadata(
                    source,
                    &gate_name,
                    vec![moe_inter, hidden],
                    gate_bytes,
                    bytes_per_row_hidden,
                    DType::ParoQ4G128,
                    &fingerprint,
                    format!("paro:q4g128:g{gs}"),
                    "Givens".to_string(),
                    gate_up_sidecars.to_vec(),
                )?);
                up_sources.push(model_source_metadata(
                    source,
                    &up_name,
                    vec![moe_inter, hidden],
                    gate_bytes,
                    bytes_per_row_hidden,
                    DType::ParoQ4G128,
                    &fingerprint,
                    format!("paro:q4g128:g{gs}"),
                    "Givens".to_string(),
                    gate_up_sidecars.to_vec(),
                )?);
                down_sources.push(model_source_metadata(
                    source,
                    &down_name,
                    vec![hidden, moe_inter],
                    down_bytes,
                    bytes_per_row_mi,
                    DType::ParoQ4G128,
                    &fingerprint,
                    format!("paro:q4g128:g{gs}"),
                    "Givens".to_string(),
                    down_sidecars.to_vec(),
                )?);
            }
            let mut sidecar_sources =
                Vec::with_capacity(gate_up_sidecars.len() + down_sidecars.len());
            for name in gate_up_sidecars.iter().chain(down_sidecars.iter()) {
                let info = source.tensor_info(name).ok_or_else(|| {
                    format!("cohere2moe Dir L{l}: sidecar source not found: {name}")
                })?;
                let rows = *info.shape.first().ok_or_else(|| {
                    format!("cohere2moe Dir L{l}: sidecar shape is empty: {name}")
                })?;
                let stride = source_row_stride(info.data_size, rows)?;
                let dtype = source_dtype(name)?;
                sidecar_sources.push(model_source_metadata(
                    source,
                    name,
                    info.shape.clone(),
                    info.data_size,
                    stride,
                    dtype,
                    &fingerprint,
                    format!("paro:{}", info.dtype),
                    format!("{:?}", hipfire_dispatch::types::dtype_rotation_plan(dtype)),
                    Vec::new(),
                )?);
            }
            let sealed = build_moe_static_binding(
                router_source,
                gate_sources,
                up_sources,
                down_sources,
                sidecar_sources,
                l,
                cfg.num_hidden_layers,
                gpu.device_id,
            )
            .map_err(|error| format!("cohere2moe Dir L{l}: {error}"))?;
            let mut expert_slots = Vec::with_capacity(n_exp);
            for x in 0..n_exp {
                let gate_prefix = format!("{mp}.{p}.mlp.experts.{x}.gate_proj");
                let up_prefix = format!("{mp}.{p}.mlp.experts.{x}.up_proj");
                let down_prefix = format!("{mp}.{p}.mlp.experts.{x}.down_proj");
                let gate_bytes = paro_repack_moe_projection(
                    source,
                    &gate_prefix,
                    moe_inter,
                    hidden,
                    gs as usize,
                )
                .map_err(e("repack gate"))?;
                let up_bytes =
                    paro_repack_moe_projection(source, &up_prefix, moe_inter, hidden, gs as usize)
                        .map_err(e("repack up"))?;
                let gate_up_len =
                    gate_bytes
                        .len()
                        .checked_add(up_bytes.len())
                        .ok_or_else(|| {
                            format!("cohere2moe Dir L{l}E{x}: gate/up byte length overflows")
                        })?;
                let mut gate_up_bytes = Vec::with_capacity(gate_up_len);
                gate_up_bytes.extend_from_slice(&gate_bytes);
                gate_up_bytes.extend_from_slice(&up_bytes);
                let gate_up_slot = tx.hold_tensor(
                    gpu.upload_raw(&gate_up_bytes, &[gate_up_len])
                        .map_err(e("upload gate_up"))?,
                );
                let down_bytes = paro_repack_moe_projection(
                    source,
                    &down_prefix,
                    hidden,
                    moe_inter,
                    gs as usize,
                )
                .map_err(e("repack down"))?;
                let down_len = down_bytes.len();
                let down_slot = tx.hold_tensor(
                    gpu.upload_raw(&down_bytes, &[down_len])
                        .map_err(e("upload down"))?,
                );
                let gate_up_row_stride = source_row_stride(gate_up_len, gate_up_rows)?;
                let down_row_stride = source_row_stride(down_len, hidden)?;
                let (gate_up_paro, down_paro) = {
                    let sidecars = tx
                        .sidecars_ref(shared_slot)
                        .ok_or_else(|| format!("cohere2moe Dir L{l}: missing sidecar owner"))?;
                    (
                        alias_paro_rotation(
                            &sidecars.gate_up_pairs,
                            &sidecars.gate_up_theta,
                            &sidecars.gate_up_channel_scales,
                            kr as u32,
                            gs,
                        ),
                        alias_paro_rotation(
                            &sidecars.down_pairs,
                            &sidecars.down_theta,
                            &sidecars.down_channel_scales,
                            kr as u32,
                            gs,
                        ),
                    )
                };
                let gate_up = WeightTensor {
                    buf: tx.take_tensor(gate_up_slot)?,
                    gpu_dtype: DType::ParoQ4G128,
                    m: gate_up_rows,
                    k: hidden,
                    row_stride: gate_up_row_stride,
                    paro: Some(gate_up_paro),
                    awq_scale: None,
                    lloyd_lut_e4m3: None,
                    lloyd_lut_f16: None,
                    lloyd_lut_c16: None,
                };
                let down = WeightTensor {
                    buf: tx.take_tensor(down_slot)?,
                    gpu_dtype: DType::ParoQ4G128,
                    m: hidden,
                    k: moe_inter,
                    row_stride: down_row_stride,
                    paro: Some(down_paro),
                    awq_scale: None,
                    lloyd_lut_e4m3: None,
                    lloyd_lut_f16: None,
                    lloyd_lut_c16: None,
                };
                let expert = ExpertWeights { gate_up, down };
                expert_slots.push(tx.hold_expert(expert));
            }

            let gu_bytes: Vec<u8> = expert_slots
                .iter()
                .map(|slot| {
                    tx.expert_ref(*slot)
                        .ok_or_else(|| {
                            format!("cohere2moe Dir L{l}: missing expert transaction slot")
                        })
                        .map(|expert| (expert.gate_up.buf.buf.as_ptr() as u64).to_ne_bytes())
                })
                .collect::<Result<Vec<[u8; 8]>, String>>()?
                .into_iter()
                .flatten()
                .collect();
            let dn_bytes: Vec<u8> = expert_slots
                .iter()
                .map(|slot| {
                    tx.expert_ref(*slot)
                        .ok_or_else(|| {
                            format!("cohere2moe Dir L{l}: missing expert transaction slot")
                        })
                        .map(|expert| (expert.down.buf.buf.as_ptr() as u64).to_ne_bytes())
                })
                .collect::<Result<Vec<[u8; 8]>, String>>()?
                .into_iter()
                .flatten()
                .collect();
            let expert_gate_up_ptrs_slot = tx.hold_tensor(
                gpu.alloc_tensor(&[ptr_count], DType::F32)
                    .map_err(e("alloc gu_ptrs"))?,
            );
            let expert_down_ptrs_slot = tx.hold_tensor(
                gpu.alloc_tensor(&[ptr_count], DType::F32)
                    .map_err(e("alloc dn_ptrs"))?,
            );
            let gate_up_ptrs = tx
                .tensor_ref(expert_gate_up_ptrs_slot)
                .ok_or_else(|| format!("cohere2moe Dir L{l}: missing gate/up pointer table"))?;
            gpu.hip
                .memcpy_htod(&gate_up_ptrs.buf, &gu_bytes)
                .map_err(e("htod gu_ptrs"))?;
            let down_ptrs = tx
                .tensor_ref(expert_down_ptrs_slot)
                .ok_or_else(|| format!("cohere2moe Dir L{l}: missing down pointer table"))?;
            gpu.hip
                .memcpy_htod(&down_ptrs.buf, &dn_bytes)
                .map_err(e("htod dn_ptrs"))?;
            let router = tx.take_weight(router_slot)?;
            let mut experts = Vec::with_capacity(expert_slots.len());
            for slot in expert_slots {
                experts.push(tx.take_expert(slot)?);
            }
            let expert_gate_up_ptrs = tx.take_tensor(expert_gate_up_ptrs_slot)?;
            let expert_down_ptrs = tx.take_tensor(expert_down_ptrs_slot)?;
            let paro_shared = tx.take_sidecars(shared_slot)?;
            let moe_slot = tx.hold_moe(MoeFfn {
                router,
                experts,
                expert_gate_up_ptrs,
                expert_down_ptrs,
                sealed,
                paro_shared: Some(paro_shared),
            });
            tx.moe_mut(moe_slot)
                .ok_or_else(|| format!("cohere2moe Dir L{l}: missing staged MoE owner"))?
                .bind_live()?;
            Ffn::Moe(tx.take_moe(moe_slot)?)
        };

        let input_norm = tx.take_tensor(input_norm_slot)?;
        let wq = tx.take_weight(wq_slot)?;
        let wk = tx.take_weight(wk_slot)?;
        let wv = tx.take_weight(wv_slot)?;
        let wo = tx.take_weight(wo_slot)?;
        let layer_slot = tx.hold_layer(Cohere2MoeLayerWeights {
            input_norm,
            wq,
            wk,
            wv,
            wo,
            ffn,
            attn_kind: cfg.attn_kind(l),
        });
        layer_slots.push(layer_slot);
    }

    let mut layers = Vec::with_capacity(layer_slots.len());
    for slot in layer_slots {
        layers.push(tx.take_layer(slot)?);
    }
    let embed = tx.take_tensor(embed_slot)?;
    let final_norm = tx.take_tensor(final_norm_slot)?;
    let lm_head = tx.take_weight(lm_head_slot)?;
    tx.commit();
    Ok(Cohere2MoeWeights {
        embed,
        embed_dtype,
        final_norm,
        lm_head,
        layers,
    })
}
