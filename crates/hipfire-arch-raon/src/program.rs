// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Declarative Raon-1-mini layer program (#666 G18).
//!
//! Each site of the layer schedule in [`crate::forward`] binds resident
//! weights, slot state and scratch to typed `hipfire_dispatch` steps; the step
//! executors own every launch. One program serves decode, prefill and verify:
//! the call's [`RowsRoute`] ([`rows_route`]) is resolved once from its row
//! count and the device's row-exact verify width, and it selects the
//! projection, FFN and MoE kernels.
//!
//! ```text
//! embedding  Embed -> StreamMix(Broadcast)
//! site       StreamMix(RouteAggregate) -> mixer | FFN -> StreamMix(Postmix)
//! RDA        BoundaryProject(q,k,v,g; one A8 input) -> BoundaryProject(b,w,f_a; one F16 input)
//!            -> BoundaryProject(f_b) -> Rda -> BoundaryProject(o)
//! MLA        BoundaryProject(q_a,kv_a,g) -> LatentAttention(NormRows) -> BoundaryProject(q_b)
//!            -> LatentAttention(SplitKv) -> [GatherLatent, BoundaryProject(kv_b), Decompress]*
//!            -> LatentAttention(Attention) -> LatentAttention(Gate) -> BoundaryProject(o)
//! FFN        GatedFfn (dense layer 0) | GroupedMoe (sigmoid top-6, shared expert)
//! head       StreamMix(Mean + final norm) -> BoundaryProject(lm_head)
//! advance    LatentAttention(Advance)
//! ```
//!
//! Every projection output is rounded to BF16 ([`OutRound::Bf16`]) except the
//! `kv_b` expansion, which the mirror scatter rounds itself.

use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::boundary_project::{
    BoundaryProjectOp, BoundaryProjection, OutRound, ProjectShare,
};
use hipfire_dispatch::pipeline::latent_attention::{
    ExpandedMirror, KvExpansion, LatentAttentionOp, LatentCache, LatentGeometry, RowRange,
};
use hipfire_dispatch::pipeline::rda::{
    RdaLayer, RdaLayerState, RdaOp, RdaProjected, RdaStage, RdaTape,
};
use hipfire_dispatch::pipeline::stream_mix::{
    StreamChart, StreamMixOp, StreamMixStage, StreamNorm, StreamRounding,
};
use hipfire_dispatch::pipeline::{
    EmbeddingOp, GatedFfnOp, GroupedMoeProtocol, PrefillActTier, RowsRoute, Step,
};
use hipfire_runtime::llama::WeightTensor;
use rdna_compute::GpuTensor;

use crate::config::RaonConfig;
use crate::forward::RaonWork;
use crate::mla::{raon_mla_scale, MlaRoute, RAON_MLA_EPS};
use crate::moe::{raon_dense_op, raon_moe_op, RaonDenseRefs, RaonMoeLayerRefs};
use crate::precision::RaonPrecisionSite;
use crate::projection::{is_mq8, rows_exact};
use crate::state::{RaonPrefillScratch, RaonState, MLA_TILE_ROWS};
use crate::weights::{MhcSiteWeights, MlaWeights, RaonFfn, RaonLayerWeights, RdaWeights};

/// Route of a call of `rows` rows on `arch`: one row decodes, a block the
/// device verifies row-exactly ([`rows_exact`]) verifies, anything wider
/// prefills.
pub(crate) fn rows_route(arch: &str, rows: usize) -> RowsRoute {
    if rows == 1 {
        RowsRoute::Decode
    } else if rows_exact(arch, rows) {
        RowsRoute::Verify
    } else {
        RowsRoute::Prefill
    }
}

/// Route of a head over rows no decode step produces (the drafter's rows):
/// the batched GEMMs whatever the row-exact width says.
pub(crate) fn batched_route(rows: usize) -> RowsRoute {
    if rows == 1 {
        RowsRoute::Decode
    } else {
        RowsRoute::Prefill
    }
}

/// How the FFN of a call on `route` runs: the MoE core, whether the dense MLP
/// runs row-exactly, and which weight copy (by row count) the dense and shared
/// projections read. A verify block runs every row as the decode step does.
pub(crate) fn ffn_plan(route: RowsRoute, rows: usize) -> (GroupedMoeProtocol, bool, usize) {
    match route {
        RowsRoute::Decode => (GroupedMoeProtocol::Indexed, false, rows),
        RowsRoute::Prefill => (GroupedMoeProtocol::Grouped, false, rows),
        RowsRoute::Verify => (GroupedMoeProtocol::RowsExact, true, 1),
    }
}

fn out<'a>(
    w: &'a WeightTensor,
    output: &'a GpuTensor,
    round: OutRound,
    tier: PrefillActTier,
) -> BoundaryProjection<'a> {
    BoundaryProjection {
        weight: w.dispatch_ref(),
        output,
        round,
        tier,
    }
}

fn project<'a>(
    input: &'a GpuTensor,
    rows: usize,
    route: RowsRoute,
    share: ProjectShare,
    projections: Vec<BoundaryProjection<'a>>,
) -> Step<'a> {
    Step::BoundaryProject(BoundaryProjectOp {
        input,
        rows,
        route,
        share,
        projections,
    })
}

/// One projection of `input` with its own route.
pub(crate) fn project_one<'a>(
    w: &'a WeightTensor,
    input: &'a GpuTensor,
    output: &'a GpuTensor,
    rows: usize,
    route: RowsRoute,
    round: OutRound,
    tier: PrefillActTier,
) -> Step<'a> {
    project(
        input,
        rows,
        route,
        ProjectShare::Own,
        vec![out(w, output, round, tier)],
    )
}

fn mix<'a>(work: &RaonWork, rows: usize, stage: StreamMixStage<'a>) -> Step<'a> {
    let shape = &work.mhc_shape;
    Step::StreamMix(StreamMixOp {
        rows,
        streams: shape.streams,
        width: shape.width,
        rounding: shape.rounding,
        stage,
    })
}

/// Embedding lookup of the first `rows` staged ids into `out` (`[rows, hidden]`
/// F32): the raw table rows, no scaling.
pub(crate) fn embed_step<'a>(
    cfg: &RaonConfig,
    table: &'a GpuTensor,
    scratch: &'a RaonPrefillScratch,
    rows: usize,
    out: &'a GpuTensor,
) -> Step<'a> {
    Step::Embed(EmbeddingOp {
        table,
        // Only a packed MQ4 table stages rows here; Raon tables are Q8_0 or F32.
        rotated: out,
        token_ids: &scratch.token_ids,
        output: out,
        rows,
        dim: cfg.hidden,
    })
}

/// [`embed_step`] into `embedded`, broadcast to every stream of `streams_out`.
pub(crate) fn embed_steps<'a>(
    cfg: &RaonConfig,
    table: &'a GpuTensor,
    scratch: &'a RaonPrefillScratch,
    work: &RaonWork,
    rows: usize,
    embedded: &'a GpuTensor,
    streams_out: &'a GpuTensor,
) -> [Step<'a>; 2] {
    [
        embed_step(cfg, table, scratch, rows, embedded),
        mix(
            work,
            rows,
            StreamMixStage::Broadcast {
                input: embedded,
                input_stride: cfg.hidden,
                streams_out,
            },
        ),
    ]
}

/// Controls of one mHC site from the OLD `streams` and its branch input
/// `out = RMSNorm(round(sum_s pre_s X_s))`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn site_route<'a>(
    work: &'a RaonWork,
    site: &'a MhcSiteWeights,
    rows: usize,
    streams: &'a GpuTensor,
    gamma: &'a GpuTensor,
    eps: f32,
    out: &'a GpuTensor,
) -> Step<'a> {
    let shape = &work.mhc_shape;
    mix(
        work,
        rows,
        StreamMixStage::RouteAggregate {
            streams,
            chart: StreamChart::HelmertBounded {
                selector: &site.selector.buf,
                params: &site.params,
                post_factor: shape.post_factor,
                selector_eps: shape.selector_eps,
            },
            selector_out: work.mhc.selector_out(),
            controls: work.mhc.controls(),
            norm: Some(StreamNorm { gamma, eps }),
            out,
        },
    )
}

/// New streams from the OLD `streams` and the site's `branch` output.
pub(crate) fn site_postmix<'a>(
    work: &'a RaonWork,
    rows: usize,
    streams: &'a GpuTensor,
    branch: &'a GpuTensor,
    streams_out: &'a GpuTensor,
) -> Step<'a> {
    mix(
        work,
        rows,
        StreamMixStage::Postmix {
            streams,
            branch,
            controls: work.mhc.controls(),
            streams_out,
        },
    )
}

/// The `rows` rows (`width` wide) of `streams` into `dst` at row stride
/// `dst_row_stride`, column `dst_col` (the drafter's carrier capture, or an
/// observer's F32 copy of BF16 streams). One row of all streams is one
/// `width`-wide "stream" here: the copy does not mix.
pub(crate) fn capture_streams<'a>(
    rows: usize,
    width: usize,
    streams: &'a GpuTensor,
    dst: &'a GpuTensor,
    dst_row_stride: usize,
    dst_col: usize,
) -> Step<'a> {
    Step::StreamMix(StreamMixOp {
        rows,
        streams: 1,
        width,
        rounding: StreamRounding::F32,
        stage: StreamMixStage::Capture {
            streams,
            dst,
            dst_row_stride,
            dst_col,
        },
    })
}

/// Final mean over the streams + RMSNorm (`model.norm`) into `normed`, then
/// the BF16-rounded `lm_head` into `logits`, over `rows` stream rows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn head_steps<'a>(
    work: &RaonWork,
    lm_head: &'a WeightTensor,
    gamma: &'a GpuTensor,
    eps: f32,
    rows: usize,
    route: RowsRoute,
    streams: &'a GpuTensor,
    normed: &'a GpuTensor,
    logits: &'a GpuTensor,
) -> [Step<'a>; 2] {
    [
        mix(
            work,
            rows,
            StreamMixStage::Mean {
                streams,
                norm: Some(StreamNorm { gamma, eps }),
                out: normed,
            },
        ),
        project_one(
            lm_head,
            normed,
            logits,
            rows,
            route,
            OutRound::Bf16,
            PrefillActTier::F16,
        ),
    ]
}

/// Device-side `seq_len += rows` after every stage that read the length.
pub(crate) fn advance_step(s: &RaonState, rows: usize) -> Step<'_> {
    Step::LatentAttention(LatentAttentionOp::Advance {
        pos: &s.seq_len,
        rows,
    })
}

/// RDA layer `ri`: projections of `scratch.branch_in` (BF16-rounded), the RDA
/// core, then the `o` projection into `scratch.branch_out`. With a `tape` the
/// raw q/k/v projections and the prepare outputs land in its planes (same
/// launches, different pointers).
#[allow(clippy::too_many_arguments)]
pub(crate) fn rda_steps<'a>(
    w: &'a RdaWeights,
    ri: usize,
    s: &'a RaonState,
    scratch: &'a RaonPrefillScratch,
    work: &'a RaonWork,
    rows: usize,
    route: RowsRoute,
    tape: Option<&'a RdaTape>,
) -> Vec<Step<'a>> {
    let x = &scratch.branch_in;
    let b = OutRound::Bf16;
    let (q_raw, k_raw, v_raw) = match tape {
        Some(t) => (&t.q_raw, &t.k_raw, &t.v_raw),
        None => (&scratch.rda_q_raw, &scratch.rda_k_raw, &scratch.rda_v_raw),
    };
    let qkvg = work.site_tiers.get(RaonPrecisionSite::RdaQkvg);
    let f16 = PrefillActTier::F16;
    // b / w / f_a read `x` through the F16 WMMA GEMM: one F16 copy of it serves all three.
    // `scratch.x_rot` is never read by the Raon forward: its leading words hold that plane.
    let bwf_share = match scratch.x_rot.as_f16_view(rows * w.b_proj.k) {
        Some(plane) => ProjectShare::F16 { plane, tier: f16 },
        None => ProjectShare::Own,
    };
    vec![
        // One A8 quantization of `x` serves q/k/v/g (same tier, same input).
        project(
            x,
            rows,
            route,
            ProjectShare::A8 { tier: qkvg },
            vec![
                out(&w.q_proj, q_raw, b, qkvg),
                out(&w.k_proj, k_raw, b, qkvg),
                out(&w.v_proj, v_raw, b, qkvg),
                out(&w.g_proj, &scratch.rda_z, b, qkvg),
            ],
        ),
        project(
            x,
            rows,
            route,
            bwf_share,
            vec![
                out(&w.b_proj, &scratch.rda_b, b, f16),
                out(&w.rda_w_proj, &scratch.rda_delta, b, f16),
                out(&w.f_a_proj, &scratch.rda_f_lat, b, f16),
            ],
        ),
        project_one(
            &w.f_b_proj,
            &scratch.rda_f_lat,
            &scratch.rda_f,
            rows,
            route,
            b,
            f16,
        ),
        Step::Rda(RdaStage::Layer(RdaOp {
            geom: work.rda_geom,
            layer: RdaLayer {
                conv_w_q: &w.q_conv,
                conv_w_k: &w.k_conv,
                conv_w_v: &w.v_conv,
                a_log: &w.a_log,
                dt_bias: &w.dt_bias,
                o_norm_gamma: &w.o_norm,
            },
            state: RdaLayerState {
                state: &s.rda[ri],
                hist_q: &s.conv_q[ri],
                hist_k: &s.conv_k[ri],
                hist_v: &s.conv_v[ri],
                slot: 0,
                state_slot_stride: work.rda_geom.state_elems(),
            },
            x: RdaProjected {
                q_raw,
                k_raw,
                v_raw,
                z: &scratch.rda_z,
                b: &scratch.rda_b,
                delta: &scratch.rda_delta,
                f: &scratch.rda_f,
            },
            scratch: &work.rda,
            gated: &scratch.rda_gated,
            rows,
            tape,
        })),
        project_one(
            &w.o_proj,
            &scratch.rda_gated,
            &scratch.branch_out,
            rows,
            route,
            b,
            work.site_tiers.get(RaonPrecisionSite::RdaO),
        ),
    ]
}

pub(crate) fn latent_geometry(cfg: &RaonConfig) -> LatentGeometry {
    LatentGeometry {
        n_heads: cfg.n_heads,
        nope: cfg.qk_nope,
        tail: cfg.qk_rope,
        v_dim: cfg.v_head,
        latent: cfg.kv_lora,
    }
}

/// Expand cached latent rows `range` of one MLA layer into the K/V mirror
/// through the layer's own (quantized) `kv_b` projection: gather -> `kv_b`
/// (exact F32) -> mirror scatter. `range` spans `count <= MLA_TILE_ROWS` rows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mla_expand_steps<'a>(
    arch: &str,
    cfg: &RaonConfig,
    w: &'a MlaWeights,
    scratch: &'a RaonPrefillScratch,
    cache: LatentCache<'a>,
    mirror: ExpandedMirror<'a>,
    range: RowRange,
    count: usize,
    tier: PrefillActTier,
) -> [Step<'a>; 3] {
    let geometry = latent_geometry(cfg);
    [
        Step::LatentAttention(LatentAttentionOp::GatherLatent {
            geometry,
            cache,
            range,
            out: &scratch.mla_latent_plane,
        }),
        project_one(
            &w.kv_b_proj,
            &scratch.mla_latent_plane,
            &scratch.mla_kv_tile,
            count,
            rows_route(arch, count),
            OutRound::Exact,
            tier,
        ),
        Step::LatentAttention(LatentAttentionOp::Decompress {
            geometry,
            cache,
            mirror,
            expansion: KvExpansion::External {
                projected: &scratch.mla_kv_tile,
            },
            range,
        }),
    ]
}

/// MLA layer `mi` over the expanded mirror (`MlaRoute::Decompressed`) or the
/// latent cache alone (`Absorbed`). Reads `scratch.branch_in`, writes
/// `scratch.branch_out`. `start` is the host copy of the slot's position.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mla_steps<'a>(
    arch: &str,
    cfg: &RaonConfig,
    w: &'a MlaWeights,
    mi: usize,
    s: &'a RaonState,
    scratch: &'a RaonPrefillScratch,
    work: &'a RaonWork,
    rows: usize,
    start: usize,
    mla_route: MlaRoute,
) -> Result<Vec<Step<'a>>, String> {
    let x = &scratch.branch_in;
    let b = OutRound::Bf16;
    let route = rows_route(arch, rows);
    let geometry = latent_geometry(cfg);
    let mirror = match mla_route {
        MlaRoute::Decompressed => {
            let m = s
                .expanded_mirror()
                .ok_or("raon1_mini: MLA attention needs a backed expanded K/V mirror")?;
            Some(ExpandedMirror {
                knope: &m.knope[mi],
                vmir: &m.vmir[mi],
                capacity_rows: m.rows,
                q8: m.q8,
            })
        }
        MlaRoute::Absorbed => None,
    };
    let cache = LatentCache {
        rows: &s.mla_cache[mi],
        capacity_rows: s.capacity_rows(),
        pos: &s.seq_len,
        host_pos: Some(start),
    };
    let q_tier = work.site_tiers.get(RaonPrecisionSite::MlaQ);
    let kv_tier = work.site_tiers.get(RaonPrecisionSite::MlaKv);
    let go_tier = work.site_tiers.get(RaonPrecisionSite::MlaGo);
    // q_a / kv_a / g read the same `x`: quantize it once for whichever of them runs A8, or for
    // the MQ8G256 weights (whose rotated int8 route does not depend on the tier).
    let share = if is_mq8(&w.q_a_proj) || [q_tier, kv_tier, go_tier].contains(&PrefillActTier::A8) {
        ProjectShare::A8 {
            tier: PrefillActTier::A8,
        }
    } else {
        ProjectShare::Own
    };
    let mut steps = vec![
        project(
            x,
            rows,
            route,
            share,
            vec![
                out(&w.q_a_proj, &scratch.mla_qa, b, q_tier),
                out(&w.kv_a_proj, &scratch.mla_kv_a, b, kv_tier),
                out(&w.g_proj, &scratch.mla_g, b, go_tier),
            ],
        ),
        Step::LatentAttention(LatentAttentionOp::NormRows {
            x: &scratch.mla_qa,
            gamma: &w.q_a_norm,
            out: &scratch.mla_qa_norm,
            rows,
            width: cfg.q_lora,
            eps: RAON_MLA_EPS,
        }),
        project_one(
            &w.q_b_proj,
            &scratch.mla_qa_norm,
            &scratch.mla_q,
            rows,
            route,
            b,
            q_tier,
        ),
        Step::LatentAttention(LatentAttentionOp::SplitKv {
            geometry,
            cache,
            kv_a: &scratch.mla_kv_a,
            gamma: &w.kv_a_norm,
            rows,
            eps: RAON_MLA_EPS,
        }),
    ];
    if let Some(m) = mirror {
        if rows <= MLA_TILE_ROWS {
            steps.extend(mla_expand_steps(
                arch,
                cfg,
                w,
                scratch,
                cache,
                m,
                RowRange::NewRows { count: rows },
                rows,
                kv_tier,
            ));
        } else {
            // A chunk wider than the decompression tile: expand the new rows tile by tile at
            // their absolute positions (`start` is the host copy of the device position
            // `NewRows` would read). Prefill only, never captured.
            let mut done = 0;
            while done < rows {
                let count = MLA_TILE_ROWS.min(rows - done);
                steps.extend(mla_expand_steps(
                    arch,
                    cfg,
                    w,
                    scratch,
                    cache,
                    m,
                    RowRange::Absolute {
                        start: start + done,
                        count,
                    },
                    count,
                    kv_tier,
                ));
                done += count;
            }
        }
    }
    steps.extend([
        Step::LatentAttention(LatentAttentionOp::Attention {
            geometry,
            cache,
            mirror,
            q: &scratch.mla_q,
            y: &scratch.mla_attn,
            rows,
            scale: raon_mla_scale(&geometry),
            route: work.mla.attention_route(mla_route, mi, rows)?,
        }),
        Step::LatentAttention(LatentAttentionOp::Gate {
            y: &scratch.mla_attn,
            g: &scratch.mla_g,
            out: &scratch.mla_attn,
            elements: rows * cfg.n_heads * cfg.v_head,
        }),
        project_one(
            &w.o_proj,
            &scratch.mla_attn,
            &scratch.branch_out,
            rows,
            route,
            b,
            go_tier,
        ),
    ]);
    Ok(steps)
}

/// The FFN of one layer over `scratch.branch_in` into `scratch.ffn_out`: the
/// dense layer-0 MLP or the routed MoE. A verify block runs every row as the
/// decode step does: the single-row weight copies, the row-exact protocol.
pub(crate) fn ffn_step<'a>(
    ctx: &'a DispatchCtx,
    work: &'a RaonWork,
    layer: &'a RaonLayerWeights,
    scratch: &'a RaonPrefillScratch,
    rows: usize,
    route: RowsRoute,
) -> Result<Step<'a>, String> {
    let (protocol, exact, weight_rows) = ffn_plan(route, rows);
    let (x, y) = (&scratch.branch_in, &scratch.ffn_out);
    match &layer.ffn {
        RaonFfn::Dense(d) => {
            if rows > work.dense.max_rows {
                return Err(format!(
                    "raon dense ffn: {rows} rows exceed the scratch capacity {}",
                    work.dense.max_rows
                ));
            }
            let (gate, up, down) = d.projections(weight_rows);
            let refs =
                RaonDenseRefs::from_parts(gate, up, down)?.with_iu4_symmetric(d.iu4_symmetric);
            Ok(Step::GatedFfn(GatedFfnOp {
                rows_exact: exact,
                ..raon_dense_op(ctx, &work.moe_dims, &refs, rows, x, &work.dense, y, y)
            }))
        }
        RaonFfn::Moe(m) => {
            if rows > work.moe.max_rows {
                return Err(format!(
                    "raon moe: {rows} rows exceed the scratch capacity {}",
                    work.moe.max_rows
                ));
            }
            let (shared_gate, shared_up, shared_down) = m.shared.projections(weight_rows);
            let refs = RaonMoeLayerRefs::from_parts(
                &m.router,
                &m.e_bias,
                shared_gate,
                shared_up,
                shared_down,
                &m.expert_gate_up_ptrs,
                &m.expert_down_ptrs,
                m.gate_up_awq.as_ref(),
                m.expert_down_awq_ptrs.as_ref(),
            )
            .with_iu4_symmetric(m.iu4_symmetric, m.shared.iu4_symmetric);
            Ok(Step::GroupedMoe(raon_moe_op(
                ctx,
                &work.moe_dims,
                &refs,
                protocol,
                rows,
                x,
                &work.moe,
                y,
                None,
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default row-exact widths (`HIPFIRE_RAON_VERIFY_ROWEXACT` unset): gfx1201 up
    /// to 4 rows, gfx1100 up to 9, every other arch none. Every op of one call takes
    /// the route this resolves, so a wrong edge silently changes verify kernels.
    #[test]
    fn rows_route_follows_the_row_exact_width() {
        use RowsRoute::{Decode, Prefill, Verify};
        for (arch, want) in [
            (
                "gfx1151",
                [
                    Decode, Prefill, Prefill, Prefill, Prefill, Prefill, Prefill, Prefill,
                ],
            ),
            (
                "gfx1201",
                [
                    Decode, Verify, Verify, Prefill, Prefill, Prefill, Prefill, Prefill,
                ],
            ),
            (
                "gfx1100",
                [
                    Decode, Verify, Verify, Verify, Verify, Prefill, Prefill, Prefill,
                ],
            ),
        ] {
            for (rows, want) in [1, 2, 4, 5, 9, 10, 16, 17].into_iter().zip(want) {
                assert_eq!(rows_route(arch, rows), want, "{arch} rows {rows}");
            }
        }
        for rows in [2, 4, 9, 16] {
            assert_eq!(batched_route(rows), Prefill);
        }
        assert_eq!(batched_route(1), Decode);
    }

    #[test]
    fn verify_ffn_runs_every_row_as_a_decode_row() {
        assert_eq!(
            ffn_plan(RowsRoute::Verify, 4),
            (GroupedMoeProtocol::RowsExact, true, 1)
        );
        assert_eq!(
            ffn_plan(RowsRoute::Prefill, 64),
            (GroupedMoeProtocol::Grouped, false, 64)
        );
        assert_eq!(
            ffn_plan(RowsRoute::Decode, 1),
            (GroupedMoeProtocol::Indexed, false, 1)
        );
    }
}
