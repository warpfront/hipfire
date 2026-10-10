// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Exact VMM batched step: byte-identical, per request, to the default
// singleton route.
//
// Decode rows run the qwen35 super-op layer program
// (`qwen35::forward::{lower_variant, variant_of}` + `Qwen35Bindings`), whose
// bindings call the same `hipfire_dispatch::pipeline::hybrid` op stages as the
// singleton's decode Steps. The
// projection super-ops are batched across requests, each column
// byte-identical to the singleton's kernel:
// - plain (QKVZA / QKV / gate+up): the exact batched RMSNorm+FWHT rotate,
//   then the PM multi-column V2 GEMV (`Gpu::gemv_mq4g256v2_xbatch_pm`);
// - residual (`wo`, `w_down`): each row's rotated input is staged by the
//   singleton's own prepass (gated-norm rotate / GEMV-family rotate / fused
//   SiLU·up rotate), then the PM multi-column residual GEMV
//   (`Gpu::gemv_mq4g256v2_residual_xbatch`) adds into the residual stream.
// (Oracle/probe evidence in runs/route-probe; kernel oracles in
// kernels/pm-decode/gfx1201.) Every other super-op — DeltaNet prep,
// recurrence, gated norm, attention (on the request's own VMM `KvCache`) —
// is the singleton binding itself, run once per request on the shared
// scratch with that request's rows copied in and out. Prefill chunks are the
// singleton prefill (`forward_prefill_batch`) on the request's own KV/DN with
// the singleton route's chunk lengths (`exact_prefill_chunk_len`).

use super::{Qwen35RequestState, Qwen35VmmStore, RequestStepKind};
use crate::forward_slots::final_logits_per_slot;
use crate::qwen35::forward::{lower_variant, qwen35_fa_epilogue_enabled, variant_of, Qwen35Bindings};
use crate::qwen35::program::hybrid_dims;
use crate::qwen35::StateQuant;
use crate::qwen35::{LayerWeights, Qwen35Config, Qwen35Scratch, Qwen35Weights};
use hip_bridge::{HipError, HipResult};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::ops::{delta_net, pm_xbatch};
use hipfire_dispatch::families::gemv::RotateInputs;
use hipfire_dispatch::pipeline::hybrid;
use hipfire_dispatch::pipeline::superop::{dispatch_super_op, SuperOpKind};
use hipfire_runtime::llama::{fused_rmsnorm_rotate_mq_batched_for, fused_silu_mul_rotate_mq_for, WeightTensor};
use hipfire_runtime::slot_batch::BatchStepPlan;
use rdna_compute::pm_xbatch::PM_XBATCH_MAX;
use rdna_compute::{DType, Gpu, GpuTensor};

/// The exact route reproduces the default singleton route only where its
/// batched projection super-ops are the singleton's norm+GEMV: dense
/// DeltaNet/FullAttention layers, uniform MQ4G256V2 projections, gfx1201.
pub(super) fn supports(gpu: &Gpu, weights: &Qwen35Weights) -> Result<(), String> {
    if !gpu.arch_caps.is_gfx1201() {
        return Err(format!("exact VMM route: arch {} not admitted (gfx1201 only)", gpu.arch));
    }
    let v2 = |w: &WeightTensor| w.gpu_dtype == DType::MQ4G256V2;
    for (i, layer) in weights.layers.iter().enumerate() {
        let ok = match layer {
            LayerWeights::DeltaNet(l) => [&l.wqkv, &l.wz, &l.w_beta, &l.w_alpha, &l.w_gate, &l.w_up, &l.wo, &l.w_down].into_iter().all(v2),
            LayerWeights::FullAttn(l) => [&l.wq, &l.wk, &l.wv, &l.w_gate, &l.w_up, &l.wo, &l.w_down].into_iter().all(v2),
            _ => false,
        };
        if !ok {
            return Err(format!("exact VMM route: layer {i} is not a dense uniform-MQ4G256V2 layer"));
        }
    }
    Ok(())
}

/// Singleton route's outer prefill chunk for a request with `remaining`
/// prompt rows (hipfire-generate ar.rs no-eviction path): the ordinary
/// chunk ceiling for this request's own state, then the serve tail rule.
pub(super) fn prefill_chunk_len(
    gpu: &Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    st: &Qwen35RequestState,
    remaining: usize,
) -> Result<usize, String> {
    let ceiling = crate::qwen35::ordinary_prefill_chunk_limit(gpu, weights, config, &st.dn, &st.kv, None)
        .map_err(|e| format!("exact prefill chunk limit: {e}"))?;
    Ok(crate::qwen35::prefill::ordinary_serve_prefill_chunk_len(remaining, ceiling)
        .unwrap_or(remaining.min(ceiling).max(1)))
}

fn copy_row(gpu: &Gpu, dst: &GpuTensor, dst_row: usize, src: &GpuTensor, src_row: usize, elems: usize) -> HipResult<()> {
    gpu.memcpy_dtod_at_auto(&dst.buf, dst_row * elems * 4, &src.buf, src_row * elems * 4, elems * 4)
}

/// Batched plain projection super-op `ordinal` (0 = attention projection,
/// 1 = gate+up) for every decode row.
fn batched_proj(
    gpu: &mut Gpu,
    st: &Qwen35VmmStore,
    layer: &LayerWeights,
    config: &Qwen35Config,
    ordinal: usize,
    n: usize,
) -> HipResult<()> {
    let p = &st.pbs;
    let (norm, ws): (&GpuTensor, Vec<(&WeightTensor, &GpuTensor)>) = match (layer, ordinal) {
        (LayerWeights::DeltaNet(l), 0) => (
            &l.attn_norm,
            vec![(&l.wqkv, &p.dn_qkv_batch), (&l.wz, &p.dn_z_batch), (&l.w_beta, &p.dn_beta_batch), (&l.w_alpha, &p.dn_alpha_batch)],
        ),
        (LayerWeights::FullAttn(l), 0) => (
            &l.attn_norm,
            vec![(&l.wq, &p.fa_q_full_batch), (&l.wk, &p.fa_k_batch), (&l.wv, &p.fa_v_batch)],
        ),
        (LayerWeights::DeltaNet(l), 1) => (&l.ffn_norm, vec![(&l.w_gate, &p.gate_ffn_batch), (&l.w_up, &p.up_batch)]),
        (LayerWeights::FullAttn(l), 1) => (&l.ffn_norm, vec![(&l.w_gate, &p.gate_ffn_batch), (&l.w_up, &p.up_batch)]),
        _ => return Err(HipError::new(0, "exact VMM step: unsupported projection super-op")),
    };
    fused_rmsnorm_rotate_mq_batched_for(gpu, &p.x_batch, norm, ws[0].0, &p.x_rot_batch, config.dim, config.norm_eps, n)?;
    for (w, y) in ws {
        for (b0, nb) in xbatch_chunks(n) {
            pm_xbatch::plain(
                gpu,
                &w.buf,
                &p.x_rot_batch.sub_offset(b0 * w.k, nb * w.k),
                &y.sub_offset(b0 * w.m, nb * w.m),
                w.m,
                w.k,
                nb,
            )?;
        }
    }
    Ok(())
}

/// `(first row, rows)` launches covering `n` rows, at most
/// [`PM_XBATCH_MAX`] each.
fn xbatch_chunks(n: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..n).step_by(PM_XBATCH_MAX).map(move |b0| (b0, PM_XBATCH_MAX.min(n - b0)))
}

/// Residual projection of super-op run `ordinal` (0 = attention/DeltaNet
/// `wo`, 1 = `w_down`): its weight and the batch buffer that stages each
/// row's rotated input contiguously as `[n][k]` (a buffer no row scratch
/// view aliases).
fn residual_target<'a>(st: &'a Qwen35VmmStore, layer: &'a LayerWeights, ordinal: usize) -> HipResult<(&'a WeightTensor, &'a GpuTensor)> {
    let p = &st.pbs;
    match (layer, ordinal) {
        (LayerWeights::DeltaNet(l), 0) => Ok((&l.wo, &p.dn_normed_rot_batch)),
        (LayerWeights::FullAttn(l), 0) => Ok((&l.wo, &p.fa_attn_out_rot_batch)),
        (LayerWeights::DeltaNet(l), 1) => Ok((&l.w_down, &p.ffn_hidden_batch)),
        (LayerWeights::FullAttn(l), 1) => Ok((&l.w_down, &p.ffn_hidden_batch)),
        _ => Err(HipError::new(0, "exact VMM step: unsupported residual super-op")),
    }
}

/// One decode row's view of the singleton scratch: the fields its binding
/// run reads or writes as its own row of the step's batch buffers (residual
/// stream, projection outputs, position) alias that row; every other field
/// is the shared scratch. Removes the per-row copy in/out of those rows.
struct RowScratch(std::mem::ManuallyDrop<Qwen35Scratch>);

impl RowScratch {
    fn new(s: &Qwen35Scratch, st: &Qwen35VmmStore, config: &Qwen35Config, r: usize) -> Self {
        let p = &st.pbs;
        let row = |t: &GpuTensor, len: usize| t.sub_offset(r * len, len);
        let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
            + config.linear_num_value_heads * config.linear_value_head_dim;
        let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let nv = config.linear_num_value_heads;
        let q_full = config.n_heads * config.head_dim * 2;
        let kv_row = config.n_kv_heads * config.head_dim;
        let hidden = s.gate_ffn.numel();
        // SAFETY: a bitwise copy of the shared scratch that is never
        // dropped (`ManuallyDrop`); only the replaced fields below are owned
        // by this value and dropped in `Drop`. Every replacement is a
        // non-owning alias of a live batch buffer row.
        unsafe {
            let mut v = std::mem::ManuallyDrop::new(std::ptr::read(s));
            std::ptr::write(&mut v.x, row(&p.x_batch, config.dim));
            std::ptr::write(&mut v.dn_qkv, row(&p.dn_qkv_batch, qkv_dim));
            std::ptr::write(&mut v.dn_z, row(&p.dn_z_batch, v_dim));
            std::ptr::write(&mut v.dn_beta, row(&p.dn_beta_batch, nv));
            std::ptr::write(&mut v.dn_alpha, row(&p.dn_alpha_batch, nv));
            std::ptr::write(&mut v.fa_q_full, row(&p.fa_q_full_batch, q_full));
            std::ptr::write(&mut v.fa_k, row(&p.fa_k_batch, kv_row));
            std::ptr::write(&mut v.fa_v, row(&p.fa_v_batch, kv_row));
            std::ptr::write(&mut v.gate_ffn, row(&p.gate_ffn_batch, hidden));
            std::ptr::write(&mut v.up, row(&p.up_batch, hidden));
            std::ptr::write(&mut v.fa_attn_out, row(&p.fa_attn_out_batch, q_full / 2));
            std::ptr::write(&mut v.fa_q, row(&p.fa_q_batch, q_full / 2));
            std::ptr::write(&mut v.fa_gate, row(&p.fa_gate_batch, q_full / 2));
            std::ptr::write(&mut v.pos_buf, p.positions.sub_offset(r, 1).buf);
            Self(v)
        }
    }
}

impl std::ops::Deref for RowScratch {
    type Target = Qwen35Scratch;
    fn deref(&self) -> &Qwen35Scratch {
        &self.0
    }
}

impl Drop for RowScratch {
    fn drop(&mut self) {
        // SAFETY: drop exactly the aliases written in `new`; the remaining
        // fields are bitwise copies owned by the shared scratch.
        unsafe {
            std::ptr::drop_in_place(&mut self.0.x);
            std::ptr::drop_in_place(&mut self.0.dn_qkv);
            std::ptr::drop_in_place(&mut self.0.dn_z);
            std::ptr::drop_in_place(&mut self.0.dn_beta);
            std::ptr::drop_in_place(&mut self.0.dn_alpha);
            std::ptr::drop_in_place(&mut self.0.fa_q_full);
            std::ptr::drop_in_place(&mut self.0.fa_k);
            std::ptr::drop_in_place(&mut self.0.fa_v);
            std::ptr::drop_in_place(&mut self.0.gate_ffn);
            std::ptr::drop_in_place(&mut self.0.up);
            std::ptr::drop_in_place(&mut self.0.fa_attn_out);
            std::ptr::drop_in_place(&mut self.0.fa_q);
            std::ptr::drop_in_place(&mut self.0.fa_gate);
            std::ptr::drop_in_place(&mut self.0.pos_buf);
        }
    }
}

/// Stage decode row `r`'s rotated residual-projection input — exactly the
/// vector the singleton's residual super-op feeds `gemv_mq4g256v2_residual`
/// (RESID_WO: the gated-norm rotate output or the GEMV family's rotate of
/// the raw input; RESID_DOWN_SWIGLU: the fused SiLU·up rotate) — into row
/// `r` of the staging buffer.
fn stage_residual_input(
    gpu: &mut Gpu,
    st: &Qwen35VmmStore,
    s: &Qwen35Scratch,
    config: &Qwen35Config,
    layer: &LayerWeights,
    ordinal: usize,
    r: usize,
) -> HipResult<()> {
    let (w, stage) = residual_target(st, layer, ordinal)?;
    let k = w.k;
    let dst = stage.sub_offset(r * k, k);
    let rotate_raw = |gpu: &mut Gpu, x: &GpuTensor| -> HipResult<()> {
        let ctx = DispatchCtx::new(gpu);
        let xr = hipfire_runtime::llama::gemv_family()
            .rotate(&ctx, gpu, &w.dispatch_ref(), x, &RotateInputs::default())
            .map_err(|e| HipError::new(0, &e.to_string()))?
            .into_buf();
        copy_row(gpu, &dst, 0, &xr, 0, k)
    };
    match (layer, ordinal) {
        (LayerWeights::DeltaNet(_), 0) => {
            if hybrid::gated_norm_mq_rotate(gpu, &hybrid_dims(config), &w.dispatch_ref()) {
                copy_row(gpu, &dst, 0, &s.x_rot, 0, k)
            } else {
                rotate_raw(gpu, &s.dn_normed)
            }
        }
        (LayerWeights::FullAttn(_), 0) => rotate_raw(gpu, &s.fa_attn_out),
        (_, 1) => {
            gpu.ensure_mq_signs()?;
            fused_silu_mul_rotate_mq_for(gpu, w, &s.gate_ffn, &s.up, &dst, k)
        }
        _ => Err(HipError::new(0, "exact VMM step: unsupported residual super-op")),
    }
}

/// Batched residual projection `x_batch[r] += W · staged[r]` for every
/// decode row: per column the singleton's `gemv_mq4g256v2_residual`.
fn batched_residual(gpu: &mut Gpu, st: &Qwen35VmmStore, layer: &LayerWeights, ordinal: usize, n: usize) -> HipResult<()> {
    let (w, stage) = residual_target(st, layer, ordinal)?;
    let x_batch = &st.pbs.x_batch;
    for (b0, nb) in xbatch_chunks(n) {
        pm_xbatch::residual(
            gpu,
            &w.buf,
            &stage.sub_offset(b0 * w.k, nb * w.k),
            &x_batch.sub_offset(b0 * w.m, nb * w.m),
            w.m,
            w.k,
            nb,
        )?;
    }
    Ok(())
}

/// Staged row tables of CbVmm's rows-batched twins for this step.
struct RowTwins {
    /// GDN + conv + gated-norm twins apply to every DeltaNet layer.
    dn: bool,
    /// FA prep twin applies to every FullAttention layer.
    fa_prep: bool,
    gdn: GpuTensor,
    conv: GpuTensor,
    gnorm_table: GpuTensor,
    fa_table: GpuTensor,
    /// Table rows per layer (the store width).
    stride: usize,
}

impl RowTwins {
    fn gdn_table(&self, d: usize) -> GpuTensor {
        self.gdn.sub_offset(d * self.stride * 72 / 4, self.stride * 72 / 4)
    }
    fn conv_table(&self, d: usize) -> GpuTensor {
        self.conv.sub_offset(d * self.stride * 40 / 4, self.stride * 40 / 4)
    }
}

/// Build and upload the step's row tables: every row points straight at
/// its own batch rows and its own request state. `None` when no twin
/// applies (the per-row singleton bindings run instead). The GDN twin
/// reserves consecutive requant frames in row order — exactly the frames
/// back-to-back per-row singleton launches take when error feedback is on;
/// with it off each request keeps its own frame, so the DN twins are used
/// only when every row has error feedback.
fn stage_row_twins(
    gpu: &mut Gpu,
    st: &mut Qwen35VmmStore,
    config: &Qwen35Config,
    rows: &[(usize, usize)],
    row_scratch: &[RowScratch],
) -> HipResult<Option<RowTwins>> {
    use rdna_compute::attention::rows_batched::{ConvRow, FaPrepRow, GatedNormRow, GdnRow, GDN_HD};
    let Some(all) = st.rows_tables.as_ref() else {
        return Ok(None);
    };
    if !gpu.arch_caps.is_gfx1201() {
        return Ok(None);
    }
    let n = rows.len();
    let ms = st.max_slots;
    let n_dn = config.layer_types.iter().filter(|t| **t == crate::qwen35::LayerType::LinearAttention).count();
    let nv = config.linear_num_value_heads;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = nv * config.linear_value_head_dim;
    let qkv_dim = 2 * k_dim + v_dim;
    let req = |slot: usize| st.slots[slot].as_ref().expect("provisioned slot");
    let quant = req(rows[0].0).dn.quant;
    let dn = rows.iter().all(|&(slot, _)| {
        let d = &req(slot).dn;
        d.quant == quant && d.s_ef_residual.len() == n_dn && !d.s_ef_residual.is_empty()
    }) && hybrid::gdn_compact_qk_div(gpu, &hybrid_dims(config), quant == StateQuant::Q8) == Some(3)
        && nv == 48
        && config.linear_value_head_dim == GDN_HD
        && config.linear_key_head_dim == GDN_HD;
    let fa_prep = config.n_heads == 24 && config.n_kv_heads == 4 && config.head_dim == 256;
    if !dn && !fa_prep {
        return Ok(None);
    }
    let tags: Vec<u64> = rows.iter().map(|&(slot, _)| req(slot).epoch.request_tag).collect();
    let p = &st.pbs;
    let ptr = |t: &GpuTensor, row: usize, len: usize| t.buf.as_ptr() as u64 + (row * len * 4) as u64;
    let conv_off = n_dn * ms * 72;
    let gnorm_off = conv_off + n_dn * ms * 40;
    let fa_off = gnorm_off + ms * 24;
    let view = |off: usize, bytes: usize| all.sub_offset(off / 4, bytes / 4);
    let t = RowTwins {
        dn,
        fa_prep,
        gdn: view(0, n_dn * ms * 72),
        conv: view(conv_off, n_dn * ms * 40),
        gnorm_table: view(gnorm_off, ms * 24),
        fa_table: view(fa_off, ms * 40),
        stride: ms,
    };
    let key: Vec<(usize, hipfire_runtime::slot_batch::RequestEpoch)> =
        rows.iter().map(|&(slot, _)| (slot, req(slot).epoch)).collect();
    if st.rows_tables_key == key {
        return Ok(Some(t));
    }
    st.rows_tables_key.clear();
    let req = |slot: usize| st.slots[slot].as_ref().expect("provisioned slot");
    let p = &st.pbs;
    if dn {
        for d in 0..n_dn {
            let mut gdn_rows = Vec::with_capacity(n);
            let mut conv_rows = Vec::with_capacity(n);
            for (r, &(slot, _)) in rows.iter().enumerate() {
                let s = &req(slot).dn;
                gdn_rows.push(GdnRow {
                    q: ptr(&p.dn_q_raw_batch, r, k_dim),
                    k: ptr(&p.dn_k_raw_batch, r, k_dim),
                    v: ptr(&p.dn_v_batch, r, v_dim),
                    gate: ptr(&p.dn_alpha_batch, r, nv),
                    beta: ptr(&p.dn_beta_batch, r, nv),
                    s_q8: s.s_matrices[d].buf.as_ptr() as u64,
                    s_scales: s.s_scales[d].buf.as_ptr() as u64,
                    output: ptr(&p.dn_attn_out_batch, r, v_dim),
                    s_ef_residual: s.s_ef_residual[d].buf.as_ptr() as u64,
                });
                conv_rows.push(ConvRow {
                    q_out: ptr(&p.dn_q_raw_batch, r, k_dim),
                    k_out: ptr(&p.dn_k_raw_batch, r, k_dim),
                    v_out: ptr(&p.dn_v_batch, r, v_dim),
                    input: ptr(&p.dn_qkv_batch, r, qkv_dim),
                    state: s.conv_states[d].buf.as_ptr() as u64,
                });
            }
            gpu.stage_gated_delta_net_q8_compact3_b2_rows(&t.gdn_table(d), &gdn_rows, &tags, &tags, 1, nv)?;
            gpu.stage_conv1d_silu_split_qknorm_b256_rows(&t.conv_table(d), &conv_rows, &tags, &tags, k_dim, v_dim)?;
        }
        let gnorm_rows: Vec<GatedNormRow> = (0..n)
            .map(|r| GatedNormRow {
                x: ptr(&p.dn_attn_out_batch, r, v_dim),
                z: ptr(&p.dn_z_batch, r, v_dim),
                x_rot: ptr(&p.dn_normed_rot_batch, r, v_dim),
            })
            .collect();
        gpu.stage_gated_norm_mq_rotate_awq_k6144_rows(&t.gnorm_table, &gnorm_rows, &tags, &tags)?;
    }
    if fa_prep {
        let fa_rows: Vec<FaPrepRow> = row_scratch
            .iter()
            .map(|rs| FaPrepRow {
                q_interleaved: rs.fa_q_full.buf.as_ptr() as u64,
                q: rs.fa_q.buf.as_ptr() as u64,
                gate: rs.fa_gate.buf.as_ptr() as u64,
                k: rs.fa_k.buf.as_ptr() as u64,
                pos: rs.pos_buf.as_ptr() as u64,
            })
            .collect();
        gpu.stage_qwen36_27b_fa_prep_rows(&t.fa_table, &fa_rows, &tags, &tags)?;
    }
    st.rows_tables_key = key;
    Ok(Some(t))
}

/// Exact decode for the AR rows of `members` (plan request indices). Rows
/// are laid out in `pbs` in member order. Fills `picks`.
pub(super) fn decode(
    gpu: &mut Gpu,
    st: &mut Qwen35VmmStore,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    s: &Qwen35Scratch,
    plan: &BatchStepPlan,
    members: &[usize],
    picks: &mut [u32],
) -> HipResult<()> {
    let n = members.len();
    if n == 0 {
        return Ok(());
    }
    let dim = config.dim;
    let pb = &plan.batch;
    // Per-row slot / position in member order.
    let rows: Vec<(usize, usize)> = members
        .iter()
        .map(|&i| {
            let r = &plan.requests[i];
            (pb.row_slot[r.rows.begin] as usize, pb.positions[r.rows.begin] as usize)
        })
        .collect();
    let tokens: Vec<i32> = members.iter().map(|&i| pb.tokens[plan.requests[i].rows.begin] as i32).collect();
    let tb: Vec<u8> = tokens.iter().flat_map(|t| t.to_ne_bytes()).collect();
    gpu.hip.memcpy_htod(&st.pbs.tokens.buf, &tb)?;
    // Row positions, uploaded once: each row's scratch view reads its own.
    let pos_bytes: Vec<u8> = rows.iter().flat_map(|&(_, p)| (p as i32).to_ne_bytes()).collect();
    gpu.hip.memcpy_htod(&st.pbs.positions.buf, &pos_bytes)?;
    gpu.embedding_lookup_q8_batched(&weights.token_embd, &st.pbs.x_batch, &st.pbs.tokens, n, dim)?;
    let row_scratch: Vec<RowScratch> = (0..n).map(|r| RowScratch::new(s, st, config, r)).collect();
    // Row→slot map and flash plan for the row-batched attention.
    let slot_bytes: Vec<u8> = rows.iter().flat_map(|&(slot, _)| (slot as i32).to_ne_bytes()).collect();
    gpu.hip.memcpy_htod(&st.row_slot_dev.buf, &slot_bytes)?;
    let max_ctx = rows.iter().map(|&(_, p)| p + 1).max().unwrap_or(1);
    let flash = gpu.vmm_flash_decode_plan(st.format, config.n_heads, config.n_kv_heads, config.head_dim, st.model_max_seq, max_ctx)?;
    let twins = stage_row_twins(gpu, st, config, &rows, &row_scratch)?;

    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;
    let mut delta_layer_idx = 0usize;
    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        let program = lower_variant(variant_of(layer));
        let mut proj_ordinal = 0usize;
        let mut last_proj: Option<usize> = None;
        // Whether the run just executed staged the next residual's inputs.
        let mut staged = false;
        let mut i = 0usize;
        while i < program.len() {
            if program[i].kind == SuperOpKind::Proj {
                batched_proj(gpu, st, layer, config, proj_ordinal, n)?;
                last_proj = Some(proj_ordinal);
                proj_ordinal += 1;
                staged = false;
                i += 1;
                continue;
            }
            if program[i].kind == SuperOpKind::ResidualGemv {
                let ordinal = last_proj.ok_or_else(|| HipError::new(0, "exact VMM step: residual before projection"))?;
                if !staged {
                    // Directly after its projection (down after gate+up):
                    // one batched fused SiLU·up rotate over every row.
                    let (w, stage) = residual_target(st, layer, ordinal)?;
                    if ordinal != 1 {
                        return Err(HipError::new(0, "exact VMM step: unstaged attention residual"));
                    }
                    gpu.ensure_mq_signs()?;
                    hipfire_runtime::llama::fused_silu_mul_rotate_mq_batched_for(
                        gpu,
                        w,
                        &st.pbs.gate_ffn_batch,
                        &st.pbs.up_batch,
                        stage,
                        w.k,
                        n,
                    )?;
                }
                batched_residual(gpu, st, layer, ordinal, n)?;
                // Later runs of this layer start from the residual stream.
                last_proj = None;
                staged = false;
                i += 1;
                continue;
            }
            // A run of non-batched super-ops: the singleton bindings, per row.
            let end = (i..program.len())
                .find(|&j| matches!(program[j].kind, SuperOpKind::Proj | SuperOpKind::ResidualGemv))
                .unwrap_or(program.len());
            let stage_for = (end < program.len() && program[end].kind == SuperOpKind::ResidualGemv)
                .then_some(last_proj)
                .flatten();
            if let LayerWeights::FullAttn(l) = layer {
                // Row-batched attention, per row the singleton's: the fused
                // FA prep per row, then the batched VMM KV write + flash
                // decode (bitwise the singleton decode per row) and gate.
                // Only where the singleton runs exactly that sequence.
                let batched_attend = end == i + 1
                    && program[i].kind == SuperOpKind::Attend
                    && stage_for == Some(0)
                    && hybrid::fused_attention_prep(gpu, &hybrid_dims(config))
                    && !hipfire_runtime::triattn::tap_enabled()
                    && !qwen35_fa_epilogue_enabled(gpu, config, &l.wo)
                    && rows
                        .iter()
                        .all(|&(slot, _)| st.slots[slot].as_ref().is_some_and(|q| q.kv.compact_offset == 0));
                if batched_attend {
                    if let Some(t) = twins.as_ref().filter(|t| t.fa_prep) {
                        gpu.qwen36_27b_fa_prep_rows(&t.fa_table, n, &l.q_norm, &l.k_norm, config.norm_eps, config.rope_theta)?;
                    } else {
                        for rs in &row_scratch {
                            gpu.qwen35_fa_prep_gfx1100(
                                &rs.fa_q_full,
                                &rs.fa_q,
                                &rs.fa_gate,
                                &rs.fa_k,
                                &l.q_norm,
                                &l.k_norm,
                                &rs.pos_buf,
                                config.norm_eps,
                                config.rope_theta,
                                config.n_heads,
                                config.n_kv_heads,
                            )?;
                        }
                    }
                    let kv_l = st
                        .kv_layer_ids
                        .iter()
                        .position(|&x| x == layer_idx)
                        .ok_or_else(|| HipError::new(0, "exact VMM step: FA layer without a KV layer"))?;
                    super::vmm_kv_write_attend(
                        gpu,
                        config,
                        &super::VmmLayerKv {
                            format: st.format,
                            descs: &st.descs_dev[kv_l],
                            row_slot: &st.row_slot_dev,
                            flash: &flash,
                            partials: &st.flash_partials,
                        },
                        &st.pbs,
                        n,
                    )?;
                    let q = config.n_heads * config.head_dim;
                    gpu.sigmoid_mul_f32(&st.pbs.fa_attn_out_batch.sub_offset(0, n * q), &st.pbs.fa_gate_batch.sub_offset(0, n * q))?;
                    let (_, stage) = residual_target(st, layer, 0)?;
                    hipfire_runtime::llama::rotate_x_mq_batched_for(gpu, &l.wo, &st.pbs.fa_attn_out_batch, stage, l.wo.k, n)?;
                    staged = true;
                    i = end;
                    continue;
                }
            }
            // DeltaNet prep: where the singleton's ATTEND_DN_PREP is the
            // plain sigmoid-alpha gate + fused conv/QK-norm (compact GDN),
            // the stateless gate runs once over all rows and each row runs
            // only its stateful conv before the rest of the run.
            let split_dn_prep = match layer {
                LayerWeights::DeltaNet(l) => {
                    let quant = st.slots[rows[0].0].as_ref().expect("provisioned slot").dn.quant;
                    let (dims, q8) = (hybrid_dims(config), quant == StateQuant::Q8);
                    program[i].kind == SuperOpKind::Attend
                        && rows.iter().all(|&(slot, _)| st.slots[slot].as_ref().is_some_and(|q| q.dn.quant == quant))
                        && !hybrid::qkvza_scalar_prep(
                            gpu,
                            &dims,
                            q8,
                            &l.wqkv.dispatch_ref(),
                            &l.wz.dispatch_ref(),
                            &l.w_beta.dispatch_ref(),
                            &l.w_alpha.dispatch_ref(),
                        )
                        && !hybrid::conv_scalar_prep(gpu, &dims, q8)
                        && hybrid::conv_qknorm(gpu, &dims, q8)
                        && hybrid::gdn_compact_qk_div(gpu, &dims, q8).is_some()
                }
                _ => false,
            };
            if let (true, LayerWeights::DeltaNet(l)) = (split_dn_prep, layer) {
                delta_net::sigmoid_alpha_gate_batched(
                    gpu,
                    &st.pbs.dn_beta_batch,
                    &st.pbs.dn_alpha_batch,
                    &l.dt_bias,
                    &l.a_log,
                    n_v_heads,
                    n,
                )?;
            }
            // The whole DeltaNet attention run on the rows twins: conv,
            // GDN recurrence and gated-norm rotate, each one launch over all
            // rows, the gated norm writing each row's `wo` input in place.
            if let (true, LayerWeights::DeltaNet(l), Some(t)) = (split_dn_prep, layer, twins.as_ref()) {
                let kinds: Vec<SuperOpKind> = program[i..end].iter().map(|o| o.kind).collect();
                if t.dn
                    && stage_for == Some(0)
                    && kinds == [SuperOpKind::Attend, SuperOpKind::Recurrent, SuperOpKind::Norm]
                    && l.wo.awq_scale.is_some()
                    && hybrid::gated_norm_mq_rotate(gpu, &hybrid_dims(config), &l.wo.dispatch_ref())
                {
                    gpu.conv1d_silu_split_qknorm_b256_rows(
                        &t.conv_table(delta_layer_idx),
                        n,
                        &l.conv_weight,
                        k_dim,
                        v_dim,
                        config.linear_num_key_heads,
                        hd,
                        1.0 / (hd as f32).sqrt(),
                        config.norm_eps,
                    )?;
                    gpu.gated_delta_net_q8_compact3_b2_rows(
                        &t.gdn_table(delta_layer_idx),
                        n,
                        1,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?;
                    gpu.gated_norm_mq_rotate_awq_k6144_rows(
                        &t.gnorm_table,
                        n,
                        &l.norm_weight,
                        l.wo.awq_scale.as_ref().expect("checked"),
                        config.norm_eps,
                    )?;
                    staged = true;
                    i = end;
                    continue;
                }
            }
            let first_op = if split_dn_prep { i + 1 } else { i };
            for (r, &(slot, pos)) in rows.iter().enumerate() {
                let rs = &row_scratch[r];
                let req = st.slots[slot].as_mut().expect("provisioned slot");
                let frame = req.dn.s_ef_residual.is_empty().then(|| {
                    let g = rdna_compute::norm::gdn_requant_frame_checkpoint();
                    rdna_compute::norm::restore_gdn_requant_frame_checkpoint(req.gdn_frame);
                    g
                });
                {
                    let ctx = DispatchCtx::new(gpu);
                    let mut bind = Qwen35Bindings {
                        layer,
                        s: rs,
                        config,
                        kv_cache: &mut req.kv,
                        dn_state: &req.dn,
                        pos,
                        layer_idx,
                        delta_layer_idx,
                        k_dim,
                        v_dim,
                        n_v_heads,
                        hd,
                        precomputed_attn_x_rot: false,
                        fa_output_prerotated: false,
                        defer_routed_combine: false,
                    };
                    if let (true, LayerWeights::DeltaNet(l)) = (split_dn_prep, layer) {
                        gpu.conv1d_silu_split_qknorm(
                            &rs.dn_q_raw,
                            &rs.dn_k_raw,
                            &rs.dn_v,
                            &rs.dn_qkv,
                            &l.conv_weight,
                            &req.dn.conv_states[delta_layer_idx],
                            k_dim,
                            v_dim,
                            config.linear_num_key_heads,
                            hd,
                            1.0 / (hd as f32).sqrt(),
                            config.norm_eps,
                        )?;
                    }
                    for op in &program[first_op..end] {
                        dispatch_super_op(gpu, &ctx, op, &mut bind)
                            .map_err(|e| HipError::new(0, &e.to_string()))?;
                    }
                }
                if let Some(g) = frame {
                    req.gdn_frame = rdna_compute::norm::gdn_requant_frame_checkpoint();
                    rdna_compute::norm::restore_gdn_requant_frame_checkpoint(g);
                }
                // Attention `wo` inputs stage batched after the row loop.
                let batched_stage = matches!(layer, LayerWeights::FullAttn(_)) && stage_for == Some(0);
                if let Some(ordinal) = stage_for.filter(|_| !batched_stage) {
                    stage_residual_input(gpu, st, rs, config, layer, ordinal, r)?;
                }
            }
            if let (LayerWeights::FullAttn(l), Some(0)) = (layer, stage_for) {
                let (_, stage) = residual_target(st, layer, 0)?;
                hipfire_runtime::llama::rotate_x_mq_batched_for(gpu, &l.wo, &st.pbs.fa_attn_out_batch, stage, l.wo.k, n)?;
            }
            staged = stage_for.is_some();
            i = end;
        }
        if matches!(layer, LayerWeights::DeltaNet(_) | LayerWeights::DeltaNetMoe(_)) {
            delta_layer_idx += 1;
        }
    }

    // Final norm + head. With 2+ rows on gfx1201: per row the singleton's
    // rmsnorm_f32 + lm_head input rotate (`Step::Gemv { input: Raw }`), then
    // the PM multi-column twin of the singleton `gemv_mq4g256v2_multirow_r2`
    // once per run of consecutive slots. One row keeps the singleton head.
    let out = &weights.output;
    let vocab = config.vocab_size;
    if n >= 2 && out.gpu_dtype == DType::MQ4G256V2 && gpu.arch_caps.is_gfx1201() {
        for (r, rs) in row_scratch.iter().enumerate() {
            gpu.rmsnorm_f32(&rs.x, &weights.output_norm, &s.tmp, config.norm_eps)?;
            let ctx = DispatchCtx::new(gpu);
            let xr = hipfire_runtime::llama::gemv_family()
                .rotate(&ctx, gpu, &out.dispatch_ref(), &s.tmp, &RotateInputs::default())
                .map_err(|e| HipError::new(0, &e.to_string()))?
                .into_buf();
            copy_row(gpu, &st.pbs.x_rot_batch, r, &xr, 0, dim)?;
        }
        let mut r0 = 0usize;
        while r0 < n {
            let mut r1 = r0 + 1;
            while r1 < n && r1 - r0 < PM_XBATCH_MAX && rows[r1].0 == rows[r1 - 1].0 + 1 {
                r1 += 1;
            }
            pm_xbatch::multirow_r2(
                gpu,
                &out.buf,
                &st.pbs.x_rot_batch.sub_offset(r0 * dim, (r1 - r0) * dim),
                &st.logits.sub_offset(rows[r0].0 * vocab, (r1 - r0) * vocab),
                vocab,
                dim,
                r1 - r0,
            )?;
            r0 = r1;
        }
    } else {
        let mut b = hipfire_runtime::slot_batch::SlotBatch::default();
        let n_slots = pb.m_per_slot.len();
        b.m_per_slot = vec![0; n_slots];
        for &(slot, pos) in &rows {
            b.m_per_slot[slot] = 1;
            b.tokens.push(0);
            b.positions.push(pos as i32);
            b.row_slot.push(slot as i32);
        }
        // final_logits_per_slot walks slots in order; rows are in slot order.
        st.skip.clear();
        st.skip.resize(n_slots, true);
        for &(slot, _) in &rows {
            st.skip[slot] = false;
        }
        final_logits_per_slot(gpu, weights, config, &b, &st.pbs, s, &st.logits, &st.skip)?;
    }
    for &i in members {
        super::sample_head(gpu, st, config, s, plan, i, picks)?;
    }
    Ok(())
}

/// Exact prefill chunk for plan request `i`: the singleton prefill on the
/// request's own KV/DN; the chunk that completes the prompt picks from the
/// singleton's last-token logits.
pub(super) fn prefill(
    gpu: &mut Gpu,
    st: &mut Qwen35VmmStore,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    s: &Qwen35Scratch,
    plan: &BatchStepPlan,
    i: usize,
    picks: &mut [u32],
) -> HipResult<()> {
    let r = &plan.requests[i];
    if r.kind != RequestStepKind::Prefill {
        return Err(HipError::new(0, "exact VMM step: not a prefill request"));
    }
    let pb = &plan.batch;
    let slot = pb.row_slot[r.rows.begin] as usize;
    let pos = pb.positions[r.rows.begin] as usize;
    let tokens = &pb.tokens[r.rows.begin..r.rows.begin + r.rows.len];
    let head = st.wants_head(r);
    {
        let req = st.slots[slot].as_mut().expect("provisioned slot");
        let frame = req.dn.s_ef_residual.is_empty().then(|| {
            let g = rdna_compute::norm::gdn_requant_frame_checkpoint();
            rdna_compute::norm::restore_gdn_requant_frame_checkpoint(req.gdn_frame);
            g
        });
        crate::qwen35::forward_prefill_batch(
            gpu, weights, config, tokens, pos, &mut req.kv, &mut req.dn, s, None, None, None, None,
        )?;
        if let Some(g) = frame {
            req.gdn_frame = rdna_compute::norm::gdn_requant_frame_checkpoint();
            rdna_compute::norm::restore_gdn_requant_frame_checkpoint(g);
        }
    }
    if head {
        let view = st.logits.sub_offset(slot * config.vocab_size, config.vocab_size);
        gpu.memcpy_dtod_at_auto(&view.buf, 0, &s.logits.buf, 0, config.vocab_size * 4)?;
        super::sample_head(gpu, st, config, s, plan, i, picks)?;
    }
    let _ = DType::F32;
    Ok(())
}
