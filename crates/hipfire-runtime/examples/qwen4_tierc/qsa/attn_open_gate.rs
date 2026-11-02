//! Open-gate attention-batch helpers for `case_attention_batch*` and
//! `wide_attention_f32`: the arm label of an admitted decode call and the n
//! one-row public calls an open-gate batch must reproduce bit for bit.

use super::*;
use rdna_compute::tensor_ops::qsa_decode_attention_eager_differs_from_recorded;

/// Attention arm the public route takes for this call (label only), derived
/// under the call's own launch state: `scope` and the verify gate
/// `attn_public` opens for `rows >= 2` (`with_wide_gate`). An admitted decode
/// call (`1..=few_row_limit` rows, `route_rows <= few_row_limit`) is ONE
/// implementation: `qsa_decode_attention_eager_differs_from_recorded` is true
/// exactly when the eager policy is split-K (the recorded one is always
/// per-head), else the per-head kernels. Past the limit or under the prefill
/// scope the launcher's row-routed predicates apply: `route_rows >= 16` + 12
/// heads per KV head (hg12), `route_rows >= 2` + head group % 4 (hg4), else the
/// per-head parallel kernel. `max_selected` is the launch's selected bound at
/// `end` cached tokens.
pub(super) fn attn_arm(
    gpu: &mut Gpu,
    g: AttnGeom,
    rows: usize,
    scope: bool,
    end: usize,
) -> Result<&'static str, String> {
    with_scope(gpu, scope, |gp| {
        Ok(with_wide_gate(gp, rows, |gp| {
            let limit = gp.few_row_limit();
            let route_rows = gp.route_rows(rows);
            if (1..=limit).contains(&rows) && route_rows <= limit {
                let max_selected = end.min(SEL_CAP).min(cap_for(BUDGET_BLOCKS));
                if qsa_decode_attention_eager_differs_from_recorded(
                    gp,
                    QsaKvFormat::F32,
                    g.head_dim,
                    g.heads,
                    g.kv_heads,
                    max_selected,
                ) {
                    "splitk_partial+combine"
                } else {
                    "per_head"
                }
            } else if g.head_dim == 256 && g.heads == g.kv_heads * 12 && route_rows >= 16 {
                "hg12"
            } else if g.head_dim == 256 && (g.heads / g.kv_heads) % 4 == 0 && route_rows >= 2 {
                "hg4"
            } else {
                "generic_parallel"
            }
        }))
    })
}

/// The `rows` one-row public calls (`attn_public_parts`, rows 1, gate closed)
/// at `position_start + r` on `t`'s K/V, with row r's own q slice
/// (`2 * g.out_w()` floats) and selection slice (`SEL_CAP` ints), concatenated:
/// the decode an admitted n-row call must reproduce.
#[allow(clippy::too_many_arguments)]
pub(super) fn attn_one_row_calls(
    gpu: &mut Gpu,
    t: &AttnTensors,
    q: &[f32],
    selected: &[i32],
    g: AttnGeom,
    rows: usize,
    position_start: usize,
    end: usize,
) -> Result<Vec<f32>, String> {
    let q_row = 2 * g.out_w();
    let mut one = Vec::with_capacity(rows * g.out_w());
    for r in 0..rows {
        let q_t = up_f32(gpu, &q[r * q_row..(r + 1) * q_row])?;
        let sel_t = raw_i32(gpu, &selected[r * SEL_CAP..(r + 1) * SEL_CAP])?;
        one.extend(attn_public_parts(
            gpu,
            &q_t,
            &t.k,
            &t.v,
            &sel_t,
            g,
            1,
            position_start + r,
            end,
        )?);
        free_all(gpu, vec![q_t, sel_t])?;
    }
    Ok(one)
}
