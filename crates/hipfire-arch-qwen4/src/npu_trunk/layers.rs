// SPDX-License-Identifier: Apache-2.0
//! First-light bank bindings reuse the ordinary typed layer operations.
use super::*;
pub(super) struct BankProgram<'a> {
    /// Heap, not an inline `SmallVec<[Step; 384]>` (~567 KiB moved through the
    /// builder and runner frames; safe only on the main thread's stack).
    pub steps: Vec<Step<'a>>,
    pub ranges: SmallVec<[std::ops::Range<usize>; 64]>,
    pub ple_rows_f16: bool,
    pub rows: usize,
}
pub(super) fn build_program<'a>(
    owner: &'a Qwen4GpuForward,
    scratch: &'a Qwen4GpuForwardScratch,
    bundle: &'a Qwen4Bundle,
    gpu: &Gpu,
    ctx: &'a DispatchCtx,
    scratch_desc: &'a Qwen4LayerScratch<'a>,
    offset: usize,
    n: usize,
) -> Result<BankProgram<'a>, Qwen4GpuForwardError> {
    let config = &bundle.config;
    let dims = program_dims(config);
    let ple_layer_index = config.ple.as_ref().map(|_| config.ple_layer_ids[0] - 1);
    let (ple_head_count, ple_row_width) = config
        .ple
        .as_ref()
        .map_or((0, 0), |p| (p.head_count, p.row_width));
    let ep = owner.ep.as_ref();
    let ep_gather: Option<&dyn MoeRowGather> = None;
    let expert_stage = None;
    let bf16_state = gpu.qwen4_bf16_streams(n);
    let route_n = gpu.route_rows(n);
    // The admitted exact few-row cap; one-row banks always sit within it.
    let few_row_limit = gpu.few_row_limit();
    let mut steps = Vec::with_capacity(QWEN4_STEP_INLINE_CAPACITY);
    let mut ranges = SmallVec::new();
    let mut gdn_slot = 0;
    let mut qsa_slot = 0;
    let mut ple_rows_f16 = false;
    for layer_index in 0..config.num_hidden_layers {
        let begin = steps.len();
        let layer = bundle
            .weights
            .layer(layer_index)
            .ok_or_else(|| invalid(format!("NPU trunk layer {layer_index} is not owned")))?;
        if ple_layer_index == Some(layer_index) {
            let ple_weights = layer
                .ple
                .as_ref()
                .ok_or_else(|| invalid("PLE weights missing"))?;
            if steps.len() >= QWEN4_STEP_INLINE_CAPACITY {
                return Err(invalid(
                    "Qwen4 generic layer program inline capacity exhausted",
                ));
            }
            let ple = ple_desc(&bundle.weights, ple_weights)?;
            ple_rows_f16 =
                qwen4_ple_rows_f16(gpu, bf16_state, dims.hc_count, config.hidden_size, &ple, n);
            steps.push(Step::GroupedDepthwise(GroupedDepthwiseOp {
                rotation: &scratch.rotation,
                state_bf16: bf16_state,
                rows_f16: ple_rows_f16,
                key: ple.key,
                value: ple.value,
                norm_key: ple.norm_key,
                norm_query: ple.norm_query,
                norm_conv: ple.norm_conv,
                conv: ple.conv,
                state: bundle
                    .state
                    .ple_conv
                    .as_ref()
                    .ok_or_else(|| invalid("PLE conv state missing"))?,
                streams: &scratch.streams,
                rows_tensor: &scratch.ple_rows,
                query: &scratch.ple_query,
                key_scratch: &scratch.ple_key,
                value_scratch: &scratch.ple_value,
                gated: &scratch.ple_gated,
                normed: &scratch.ple_normed,
                output: &scratch.ple_output,
                rows: n,
                branches: dims.hc_count,
                hidden: config.hidden_size,
                embed_dim: ple_head_count * ple_row_width,
                kernel_size: dims.ple_conv_kernel_dim,
                dilation: config.ple_conv_dilation(),
                epsilon: EPSILON,
            }));
        }

        let layer_moe = match ep {
            Some(ep) if route_n > 1 => ep.moe_full.get(layer_index),
            _ => owner.moe.get(layer_index).and_then(Option::as_ref),
        };
        let mut description = layer_desc(
            &bundle.weights,
            layer,
            layer_moe,
            &config,
            &owner.decode_q8,
            route_n,
            few_row_limit,
            expert_stage,
        )?;
        if let (Some(gather), Qwen4FfnBinding::Moe(binding)) = (ep_gather, &mut description.ffn) {
            binding.row_gather = Some(gather);
        }
        let attn_read = &description.attn_hyper.read;
        if steps.len() >= QWEN4_STEP_INLINE_CAPACITY {
            return Err(invalid(
                "Qwen4 generic layer program inline capacity exhausted",
            ));
        }
        steps.push(Step::HyperRead(HyperReadOp {
            rotation: &scratch.rotation,
            state_bf16: bf16_state,
            input: &scratch.streams,
            norm_weight: attn_read.norm,
            input_mix_down: decode_copy(
                &owner.decode_q8,
                layer_index * DECODE_Q8_PER_LAYER,
                route_n,
                few_row_limit,
                attn_read.input_mix_down,
            ),
            input_mix_up: decode_copy(
                &owner.decode_q8,
                layer_index * DECODE_Q8_PER_LAYER + 1,
                route_n,
                few_row_limit,
                attn_read.input_mix_up,
            ),
            normalized: &scratch.hc_normalized,
            low: &scratch.hc_low,
            up: &scratch.hc_up,
            mixed: &scratch.hc_mixed,
            rows: n,
            branches: dims.hc_count,
            hidden: config.hidden_size,
            low_rank: dims.hc_lowrank,
        }));

        match layer.kind {
            LayerType::LinearAttention => {
                let state = bundle
                    .state
                    .gdn
                    .get(gdn_slot)
                    .ok_or_else(|| invalid("GDN state slot missing"))?;
                gdn_slot += 1;
                let weights = match &description.attention {
                    Qwen4AttentionWeights::Linear(weights) => weights,
                    _ => return Err(invalid("GDN descriptor kind mismatch")),
                };
                steps.push(Step::GatedDeltaNet(GatedDeltaNetOp {
                    rotation: &scratch.rotation,
                    qkv: weights.qkv,
                    conv: weights.conv,
                    in_proj_a: weights.in_proj_a,
                    in_proj_b: weights.in_proj_b,
                    a_log: weights.a_log,
                    dt_bias: weights.dt_bias,
                    z: weights.z,
                    norm: weights.norm,
                    output: weights.output,
                    recurrent: &state.recurrent,
                    format: bundle.state.gdn_format(),
                    residual: state.ef_residual.as_ref(),
                    conv_state: &state.conv,
                    projection: &scratch.projection,
                    projection2: &scratch.projection2,
                    a: &scratch.gdn_a,
                    b: &scratch.gdn_b,
                    gate: &scratch.gdn_gate,
                    beta: &scratch.gdn_beta,
                    recurrent_output: &scratch.gdn_recurrent_output,
                    bf16_scratch: &scratch.gdn_bf16,
                    z_output: &scratch.gdn_z,
                    output_scratch: &scratch.gdn_output,
                    input: &scratch.hc_mixed,
                    output_tensor: &scratch.attention_output,
                    rows: n,
                    start_position: bundle.state.position + offset,
                    key_heads: dims.linear_num_key_heads,
                    value_heads: dims.linear_num_value_heads,
                    key_dim: dims.linear_key_head_dim,
                    value_dim: dims.linear_value_head_dim,
                    conv_kernel: dims.linear_conv_kernel_dim,
                    input_width: dims.hidden,
                    row_capture: bundle.state.gdn_row_capture(gdn_slot - 1, n),
                    zba_fold: owner.trunk_zba.get(layer_index).and_then(Option::as_ref),
                    trunk_a4: owner.trunk_a4.get(layer_index).copied().unwrap_or(0),
                    phase: MixerPhase::Whole,
                }));
            }
            LayerType::FullAttention => {
                let state = bundle
                    .state
                    .qsa
                    .get(qsa_slot)
                    .ok_or_else(|| invalid("QSA state slot missing"))?;
                qsa_slot += 1;
                let weights = match &description.attention {
                    Qwen4AttentionWeights::Full(weights) => weights,
                    _ => return Err(invalid("QSA descriptor kind mismatch")),
                };
                steps.push(Step::IndexedAttention(IndexedAttentionOp {
                    rotation: &scratch.rotation,
                    mode: IndexedAttentionMode::Full,
                    indexer_qk: weights.indexer_qk,
                    indexer_q_norm: weights.indexer_q_norm,
                    indexer_k_norm: weights.indexer_k_norm,
                    q: weights.q,
                    k: weights.k,
                    v: weights.v,
                    q_norm: weights.q_norm,
                    k_norm: weights.k_norm,
                    output: weights.output,
                    state: IndexedAttentionState {
                        format: state.format,
                        full_keys: &state.full_keys,
                        full_values: &state.full_values,
                        raw_index_keys: &state.raw_index_keys,
                        pooled_keys: &state.pooled_keys,
                        selected_indices: &state.selected_indices,
                        // Decode and speculative verify never consume the arena.
                        f16_arena: if owner.prefill_scope {
                            state.f16_arena_ref()
                        } else {
                            None
                        },
                        pooled_bf16: state.pooled_bf16_ref(),
                        full_capacity: state.full_capacity,
                        raw_capacity: state.raw_capacity,
                        pooled_capacity: state.pooled_capacity,
                        selected_capacity: state.selected_capacity,
                        position_capacity: state.position_capacity,
                        full_len: state.full_len,
                        raw_len: state.raw_len,
                        pooled_len: state.pooled_len,
                        selected_len: state.selected_len,
                        position: state.position,
                    },
                    input: &scratch.hc_mixed,
                    index_scratch: &scratch.qsa_index,
                    qgate_scratch: &scratch.qsa_qgate,
                    k_scratch: &scratch.qsa_k,
                    v_scratch: &scratch.qsa_v,
                    qsa_output: &scratch.qsa_output,
                    selected_scratch: &scratch.qsa_selected,
                    attention_output: &scratch.attention_output,
                    bf16_scratch: &scratch.gdn_bf16,
                    rows: n,
                    index_heads: dims.indexer_n_heads,
                    index_kv_heads: dims.indexer_kv_heads,
                    index_dim: dims.indexer_head_dim,
                    budget: dims.indexer_budget,
                    compress: dims.indexer_compress_ratio,
                    heads: dims.num_attention_heads,
                    kv_heads: dims.num_key_value_heads,
                    head_dim: dims.head_dim,
                    rotary_dim: config.rotary_dim(),
                    rope: config.rope(),
                    mrope: None,
                    input_width: dims.hidden,
                    trunk_a4: owner.trunk_a4.get(layer_index).copied().unwrap_or(0),
                    phase: MixerPhase::Whole,
                }));
            }
        }

        let attn_write = &description.attn_hyper.write;
        steps.push(Step::HyperWrite(HyperWriteOp {
            rotation: &scratch.rotation,
            state_bf16: bf16_state,
            input: &scratch.streams,
            norm_weight: attn_write.norm,
            block_inject: attn_write.block_inject,
            normalized: &scratch.hc_normalized,
            mixed: &scratch.attention_output,
            gates: &scratch.hc_gates,
            output: &scratch.streams,
            rows: n,
            branches: dims.hc_count,
            hidden: config.hidden_size,
        }));

        let mlp_read = &description.mlp_hyper.read;
        steps.push(Step::HyperRead(HyperReadOp {
            rotation: &scratch.rotation,
            state_bf16: bf16_state,
            input: &scratch.streams,
            norm_weight: mlp_read.norm,
            input_mix_down: decode_copy(
                &owner.decode_q8,
                layer_index * DECODE_Q8_PER_LAYER + 2,
                route_n,
                few_row_limit,
                mlp_read.input_mix_down,
            ),
            input_mix_up: decode_copy(
                &owner.decode_q8,
                layer_index * DECODE_Q8_PER_LAYER + 3,
                route_n,
                few_row_limit,
                mlp_read.input_mix_up,
            ),
            normalized: &scratch.hc_normalized,
            low: &scratch.hc_low,
            up: &scratch.hc_up,
            mixed: &scratch.hc_mixed,
            rows: n,
            branches: dims.hc_count,
            hidden: config.hidden_size,
            low_rank: dims.hc_lowrank,
        }));

        match &description.ffn {
            Qwen4FfnBinding::Moe(moe) => {
                let sealed =
                    crate::program::seal_moe_for_layer(ctx, dims, *moe, scratch_desc, n, route_n)
                        .map_err(|error| {
                        Qwen4GpuForwardError::Dispatch(format!("seal Qwen4 MoE: {error:?}"))
                    })?;
                steps.push(Step::Clear(ClearOp {
                    tensor: &scratch.ffn_output,
                    elements: n * config.hidden_size,
                }));
                steps.push(Step::Moe(sealed));
            }
            Qwen4FfnBinding::Dense(dense) => {
                steps.push(Step::GatedMlp(GatedMlpOp {
                    rows: n,
                    hidden: config.hidden_size,
                    intermediate: dense.intermediate,
                    x: &scratch.hc_mixed,
                    pre_norm: None,
                    gate: dense.gate,
                    up: dense.up,
                    down: dense.down,
                    activation: Activation::Silu,
                    storage: MlpStorage::Bf16Source,
                    output: MlpOutput::Write(&scratch.ffn_output),
                    ladder: MlpLadder::Trunk,
                    scratch: GatedMlpScratch {
                        gate: &scratch.moe_gate_batch,
                        up: &scratch.moe_up_batch,
                        act: &scratch.moe_gate_batch,
                        x_rot: &scratch.rotation,
                    },
                }));
            }
        }

        let mlp_write = &description.mlp_hyper.write;
        steps.push(Step::HyperWrite(HyperWriteOp {
            rotation: &scratch.rotation,
            state_bf16: bf16_state,
            input: &scratch.streams,
            norm_weight: mlp_write.norm,
            block_inject: mlp_write.block_inject,
            normalized: &scratch.hc_normalized,
            mixed: &scratch.ffn_output,
            gates: &scratch.hc_gates,
            output: &scratch.streams,
            rows: n,
            branches: dims.hc_count,
            hidden: config.hidden_size,
        }));
        ranges.push(begin..steps.len());
    }
    if offset != 0 {
        for step in &mut steps {
            if let Step::IndexedAttention(op) = step {
                // Advance the initial view by the rows before this bank, not its live rows.
                let rows = op.rows;
                op.rows = offset;
                let (full, raw, pooled, selected, position) = op.next_lengths().map_err(fail)?;
                op.rows = rows;
                op.state.full_len = full;
                op.state.raw_len = raw;
                op.state.pooled_len = pooled;
                op.state.selected_len = selected;
                op.state.position = position;
            }
        }
    }
    validate_steps(gpu, &steps).map_err(fail)?;
    Ok(BankProgram {
        steps,
        ranges,
        ple_rows_f16,
        rows: n,
    })
}
