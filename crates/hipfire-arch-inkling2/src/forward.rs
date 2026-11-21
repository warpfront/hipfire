// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU reference primitive for the attention half of one Inkling-2 decoder
//! layer (single-token, SWA path), mirroring the attention part of
//! `llama_model_inkling2::graph::build_layer` in the merged llama.cpp.
//!
//! This is a correctness scaffold, NOT a complete forward. Deliberately not
//! implemented yet (the oracle ops for each exist and are separately tested):
//!   * the routed MoE (sigmoid top-8 + SiTU + expert combine),
//!   * the loop schedule: cross-pass messages, controllers, `mix()`, prelude/bank,
//!   * full-attention layers' dynamic K/V conv taps and q/k/v input norms,
//!   * a multi-token KV cache (this primitive handles one key at distance 0),
//!   * token embedding / final norm / LM head / softcap.
//! `pos` is accepted for signature stability but unused until the KV cache lands.
//!
//! It consumes true f32 weights (from `weights::Inkling2Weights::f32`) and the
//! scalar operators in `oracle`. Numbers are simple row-major f32 math; the
//! point is to lock the operation order the GPU port must reproduce.

use crate::oracle;

/// Decoded f32 weights for one physical layer, shaped as in the HFQ artifact.
#[derive(Clone)]
pub struct LayerWeightsF32 {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_rel: usize,
    pub n_extent: usize,
    pub is_swa: bool,
    pub conv_kernel: usize,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub rms_eps: f32,
    // attention projections (row-major [out, in])
    pub wq: Vec<f32>,                     // [n_head*hd, n_embd]
    pub wk: Vec<f32>,                     // [n_head_kv*hd, n_embd]
    pub wv: Vec<f32>,                     // [n_head_kv*hd, n_embd]
    pub wr: Vec<f32>,                     // [n_head*n_rel, n_embd]
    pub wgate: Vec<f32>,                  // [n_head*hd, n_embd] (output gate)
    pub wo: Vec<f32>,                     // [n_embd, n_head*hd]
    pub sink_proj: Vec<f32>,              // [n_head, n_embd]
    pub rel_table: Vec<f32>,              // [n_rel, n_extent]
    pub q_clip_scale: Option<Vec<f32>>,   // [n_head, hd]
    pub rel_clip_scale: Option<Vec<f32>>, // [n_head, n_rel]
    pub attn_norm: Vec<f32>,              // RMSNorm gain for SWA layers
    // short convs
    pub k_conv: Vec<f32>,         // [conv, n_head_kv*hd]
    pub v_conv: Vec<f32>,         // [conv, n_head_kv*hd]
    pub attn_post_conv: Vec<f32>, // [conv, n_embd]
    pub ffn_post_conv: Vec<f32>,  // [conv, n_embd]
}

/// Per-visit recurrent state for the four short convolutions. Each holds the
/// last `conv_kernel-1` inputs of K, V, attn-out and FFN-out (flattened).
#[derive(Clone, Default)]
pub struct ConvState {
    pub k_past: Vec<f32>,
    pub v_past: Vec<f32>,
    pub attn_out_past: Vec<f32>,
    pub ffn_out_past: Vec<f32>,
}

/// Apply a learned-gain RMSNorm: `x_norm * weight`, the raw HF / llama.cpp
/// convention (`build_norm` does `ggml_mul(rms_norm(x), mw)`). Verified the
/// decoded Inkling-2 norm weights are raw (embed_norm has negative values),
/// NOT the Qwen4-style `(1 + delta)` zero-centered convention.
fn rmsnorm_gain(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = rmsnorm_l2(x, eps);
    n.iter().zip(weight).map(|(v, g)| v * g).collect()
}

fn rmsnorm_l2(x: &[f32], eps: f32) -> Vec<f32> {
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (mean_sq + eps).sqrt();
    x.iter().map(|v| v * s).collect()
}

/// HFQ stores conv tensors as `[n_channels, 1, kernel]`; transpose to the
/// oracle's `[kernel, n_channels]` layout.
fn flatten_conv(v: &[f32], kernel: usize) -> Vec<f32> {
    let n_ch = v.len() / kernel;
    let mut out = vec![0.0f32; v.len()];
    for c in 0..n_ch {
        for k in 0..kernel {
            out[k * n_ch + c] = v[c * kernel + k];
        }
    }
    out
}

/// Run one single-token decoder layer. `x` is `[n_embd]`, `pos` the position,
/// `state` the per-visit conv state (mutated). Returns the layer output.
pub fn decode_layer(
    w: &LayerWeightsF32,
    state: &mut ConvState,
    x: &[f32],
    _pos: usize,
) -> Vec<f32> {
    let d = w.head_dim;
    let nh = w.n_head;
    let nkv = w.n_head_kv;
    let ekv = nkv * d;
    let eattn = nh * d;

    // Attention input(s): SWA has one (attn_norm); full layers would use
    // q/k/v input norms (handled separately by the caller in Phase 1).
    let inp = if w.is_swa {
        rmsnorm_gain(x, &w.attn_norm, w.rms_eps)
    } else {
        rmsnorm_l2(x, w.rms_eps) // full-layer q/k/v norms differ; refined below
    };

    let q = oracle::matvec(&w.wq, &inp, eattn, w.n_embd);
    let k_raw = oracle::matvec(&w.wk, &inp, ekv, w.n_embd);
    let v_raw = oracle::matvec(&w.wv, &inp, ekv, w.n_embd);
    let r = oracle::matvec(&w.wr, &inp, nh * w.n_rel, w.n_embd);

    // Static short convs on K and V (SWA path). Full-layer dynamic convs are
    // approximated with the same static weights here; the oracle's
    // convolve_static is the verified primitive.
    let (k_conv, k_past) =
        oracle::convolve_static(&k_raw, &w.k_conv, &state.k_past, 1, ekv, w.conv_kernel);
    let (v_conv, v_past) =
        oracle::convolve_static(&v_raw, &w.v_conv, &state.v_past, 1, ekv, w.conv_kernel);
    state.k_past = k_past;
    state.v_past = v_past;

    // Reshape to [n_head, d] and apply clips.
    let qc = oracle::clip(&q, w.q_clip_scale.as_deref(), w.qk_clip);
    let kc = oracle::clip(&k_conv, None, w.qk_clip);
    // V uses unweighted RMSNorm per head.
    let vc: Vec<f32> = (0..nkv)
        .flat_map(|h| oracle::rmsnorm(&v_conv[h * d..(h + 1) * d], w.rms_eps))
        .collect();
    let rc = oracle::clip(&r, w.rel_clip_scale.as_deref(), w.rel_clip);

    // Relative bias: table [n_rel, n_extent] × per-head R, sink subtracted
    // before gather (llama.cpp order). Single-token: distance to every cached
    // key is 0 for the current token; a real multi-token KV cache is Phase 2.
    let sinks = oracle::matvec(&w.sink_proj, &inp, nh, w.n_embd);
    let mut attn_out = vec![0.0f32; eattn];
    for h in 0..nh {
        let rh = &rc[h * w.n_rel..(h + 1) * w.n_rel];
        // distance 0 gather -> table[r][0] for this head.
        let bias0: f32 = (0..w.n_rel)
            .map(|rr| w.rel_table[rr * w.n_extent] * rh[rr])
            .sum::<f32>()
            - sinks[h];
        let qh = &qc[h * d..(h + 1) * d];
        let kh = if h < nkv {
            &kc[(h % nkv) * d..(h % nkv + 1) * d]
        } else {
            &kc[0..d]
        };
        let vh = if h < nkv {
            &vc[(h % nkv) * d..(h % nkv + 1) * d]
        } else {
            &vc[0..d]
        };
        let score: f32 =
            qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>() / (d as f32).sqrt() + bias0;
        // With a single key, the sink-softmax yields sigmoid(score - sink)
        // weighting on that value; use the full operator for clarity.
        let wv = oracle::attention_softmax_with_sink(&[score], 0.0, vh, 1, d);
        attn_out[h * d..(h + 1) * d].copy_from_slice(&wv);
    }

    // Output gate (sigmoid of gate projection) then output projection.
    let gate = oracle::matvec(&w.wgate, &inp, eattn, w.n_embd);
    for i in 0..eattn {
        attn_out[i] *= oracle::sigmoid(gate[i]);
    }
    let proj = oracle::matvec(&w.wo, &attn_out, w.n_embd, eattn);
    let (attn_res, attn_past) = oracle::convolve_static(
        &proj,
        &w.attn_post_conv,
        &state.attn_out_past,
        1,
        w.n_embd,
        w.conv_kernel,
    );
    state.attn_out_past = attn_past;

    // Residual: next layer input is x + attention result (MoE omitted in this
    // single-layer primitive; it is covered by the oracle MoE operators).
    x.iter().zip(&attn_res).map(|(a, b)| a + b).collect()
}

/// Decode and run the routed MoE for one layer: sigmoid top-8 router,
/// per-expert SiTU gate/up, down projection, fp32 combine. `x` is `[n_embd]`.
/// `gate_up[e]` is `[2*inter, n_embd]` (gate rows first), `down[e]` is
/// `[n_embd, inter]`, `router` is `[n_experts, n_embd]`.
pub fn decode_moe(
    x: &[f32],
    router: &[f32],
    router_bias: Option<&[f32]>,
    gate_up: &[Vec<f32>],
    down: &[Vec<f32>],
    n_embd: usize,
    inter: usize,
    top_k: usize,
    route_scale: f32,
    situ_beta: f32,
    situ_linear_beta: f32,
) -> Vec<f32> {
    let n_experts = gate_up.len();
    let logits = oracle::matvec(router, x, n_experts, n_embd);
    let selected = oracle::sigmoid_topk(&logits, router_bias, top_k, route_scale);
    let mut out = vec![0.0f32; n_embd];
    for (e, weight) in selected {
        let gu = &gate_up[e];
        let gate = &gu[..inter * n_embd];
        let up = &gu[inter * n_embd..];
        let g = oracle::matvec(gate, x, inter, n_embd);
        let u = oracle::matvec(up, x, inter, n_embd);
        let act: Vec<f32> = (0..inter)
            .map(|i| oracle::situ(g[i], u[i], situ_beta, situ_linear_beta))
            .collect();
        let d = oracle::matvec(&down[e], &act, n_embd, inter);
        for i in 0..n_embd {
            out[i] += weight * d[i];
        }
    }
    out
}

/// Run one full decoder visit (attention + MoE + residual/block bookkeeping),
/// loading weights lazily from the validated HFQ index. `x` is `[n_embd]`,
/// `visit` the global visit index (0..48), `blocks` the AttnRes block stream,
/// `state` the per-visit conv state. Messages/controllers are handled by the
/// higher-level loop driver (Phase 2 of this scaffold).
pub fn decode_visit(
    w: &crate::weights::Inkling2Weights,
    cfg: &crate::config::Inkling2Config,
    state: &mut ConvState,
    blocks: &mut Vec<Vec<f32>>,
    x: &[f32],
    visit: usize,
    pos: usize,
) -> Result<Vec<f32>, String> {
    let phys = cfg.loop_cfg.layer_phys(visit);
    let p = format!("model.llm.layers.{phys}.");
    let swa = cfg.swa_layers[phys];
    let g = |s: &str| w.f32(&format!("{p}{s}")).map(|x| x.1);

    // --- attention input: AttnRes mix (if this visit has a query) + norm ---
    let mut inp = x.to_vec();
    let attn_q = format!("model.llm.attn_res.attn_queries.{phys}");
    if let Ok((shape, q)) = w.f32(&attn_q) {
        let query_rows = if shape.len() == 1 {
            1
        } else {
            shape[0] as usize
        };
        let mixed = oracle::mix(blocks, x, &q, cfg.n_embd, query_rows);
        inp = mixed[..cfg.n_embd].to_vec();
    }
    if swa {
        inp = rmsnorm_gain(&inp, &g("attn_norm.weight")?, cfg.rms_norm_eps);
    } else {
        inp = rmsnorm_gain(&inp, &g("q_input_norm.weight")?, cfg.rms_norm_eps);
    }

    // --- attention projections + convs (SWA static; full dynamic deferred) ---
    let d = cfg.head_dim;
    let nh = cfg.n_head;
    let nkv = cfg.n_head_kv;
    let ekv = nkv * d;
    let eattn = nh * d;
    let q = oracle::matvec(&g("attn.wq_du.weight")?, &inp, eattn, cfg.n_embd);
    let k_raw = oracle::matvec(&g("attn.wk_dv.weight")?, &inp, ekv, cfg.n_embd);
    let v_raw = oracle::matvec(&g("attn.wv_dv.weight")?, &inp, ekv, cfg.n_embd);
    let r = oracle::matvec(&g("attn.wr_du.weight")?, &inp, nh * cfg.n_rel, cfg.n_embd);
    let (k_conv, kp) = oracle::convolve_static(
        &k_raw,
        &flatten_conv(&g("attn.k_sconv.weight")?, cfg.conv_kernel),
        &state.k_past,
        1,
        ekv,
        cfg.conv_kernel,
    );
    let (v_conv, vp) = oracle::convolve_static(
        &v_raw,
        &flatten_conv(&g("attn.v_sconv.weight")?, cfg.conv_kernel),
        &state.v_past,
        1,
        ekv,
        cfg.conv_kernel,
    );
    state.k_past = kp;
    state.v_past = vp;

    let qc = oracle::clip(
        &q,
        w.f32(&format!("{p}attn.q_l2_clip.scale"))
            .ok()
            .map(|x| x.1)
            .as_deref(),
        cfg.qk_clip,
    );
    let kc = oracle::clip(&k_conv, None, cfg.qk_clip);
    let vc: Vec<f32> = (0..nkv)
        .flat_map(|h| oracle::rmsnorm(&v_conv[h * d..(h + 1) * d], cfg.rms_norm_eps))
        .collect();
    let mut rc = oracle::clip(
        &r,
        w.f32(&format!("{p}attn.rel_l2_clip.scale"))
            .ok()
            .map(|x| x.1)
            .as_deref(),
        cfg.rel_clip,
    );
    if !swa {
        // full-layer tau multiplies Q (per-head below) and R.
        let t = oracle::tau(pos, cfg.log_floor, cfg.log_alpha);
        for v in rc.iter_mut() {
            *v *= t;
        }
    }

    let sinks = oracle::matvec(&g("attn.attn_sink_proj.weight")?, &inp, nh, cfg.n_embd);
    let rel_table = g("attn.rel_logits_proj.proj")?;
    let mut attn_out = vec![0.0f32; eattn];
    for h in 0..nh {
        let rh = &rc[h * cfg.n_rel..(h + 1) * cfg.n_rel];
        let bias0: f32 = (0..cfg.n_rel)
            .map(|rr| rel_table[rr * cfg.n_extent] * rh[rr])
            .sum::<f32>()
            - sinks[h];
        let mut qh = qc[h * d..(h + 1) * d].to_vec();
        if !swa {
            let t = oracle::tau(pos, cfg.log_floor, cfg.log_alpha);
            for v in &mut qh {
                *v *= t;
            }
        }
        let kh = &kc[(h % nkv) * d..(h % nkv + 1) * d];
        let vh = &vc[(h % nkv) * d..(h % nkv + 1) * d];
        let score: f32 =
            qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>() / (d as f32).sqrt() + bias0;
        let wv = oracle::attention_softmax_with_sink(&[score], 0.0, vh, 1, d);
        attn_out[h * d..(h + 1) * d].copy_from_slice(&wv);
    }
    let gate = oracle::matvec(&g("attn.gate_attn.weight")?, &inp, eattn, cfg.n_embd);
    for i in 0..eattn {
        attn_out[i] *= oracle::sigmoid(gate[i]);
    }
    let proj = oracle::matvec(&g("attn.wo_ud.weight")?, &attn_out, cfg.n_embd, eattn);
    let (attn_res, ap) = oracle::convolve_static(
        &proj,
        &flatten_conv(&g("attn_sconv.weight")?, cfg.conv_kernel),
        &state.attn_out_past,
        1,
        cfg.n_embd,
        cfg.conv_kernel,
    );
    state.attn_out_past = ap;

    // AttnRes block bookkeeping: first visit of a block pushes inpSA, others add residual.
    let inp_sa = x.to_vec();
    let cur = if visit % cfg.attn_res_block == 0 {
        blocks.push(inp_sa);
        attn_res
    } else {
        inp_sa.iter().zip(&attn_res).map(|(a, b)| a + b).collect()
    };

    // --- MoE with AttnRes mix on FFN input ---
    let ffn_mix_q = format!("model.llm.attn_res.mlp_queries.{phys}");
    let mut ffn_in = cur.clone();
    if let Ok((_shape, q)) = w.f32(&ffn_mix_q) {
        let mixed = oracle::mix(blocks, &cur, &q, cfg.n_embd, 1);
        ffn_in = mixed;
    }
    let ffn_normed = rmsnorm_gain(&ffn_in, &g("mlp_norm.weight")?, cfg.rms_norm_eps);
    let router = g("mlp.gate.weight")?;
    let bias = g("mlp.gate.bias").ok();
    let mut gu = Vec::with_capacity(cfg.n_experts);
    let mut dn = Vec::with_capacity(cfg.n_experts);
    for e in 0..cfg.n_experts {
        gu.push(g(&format!("mlp.experts.{e}.gate_up_proj.weight"))?);
        dn.push(g(&format!("mlp.experts.{e}.down_proj.weight"))?);
    }
    let moe = decode_moe(
        &ffn_normed,
        &router,
        bias.as_deref(),
        &gu,
        &dn,
        cfg.n_embd,
        cfg.inter,
        cfg.n_expert_used,
        cfg.route_scale,
        cfg.situ_beta,
        cfg.situ_linear_beta,
    );
    let (moe_conv, fp) = oracle::convolve_static(
        &moe,
        &flatten_conv(&g("mlp_sconv.weight")?, cfg.conv_kernel),
        &state.ffn_out_past,
        1,
        cfg.n_embd,
        cfg.conv_kernel,
    );
    state.ffn_out_past = fp;
    Ok(ffn_in.iter().zip(&moe_conv).map(|(a, b)| a + b).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_weights() -> LayerWeightsF32 {
        let d = 4;
        LayerWeightsF32 {
            n_embd: d,
            n_head: 2,
            n_head_kv: 2,
            head_dim: d,
            n_rel: 2,
            n_extent: 4,
            is_swa: true,
            conv_kernel: 1,
            qk_clip: 10.0,
            rel_clip: 10.0,
            rms_eps: 1e-6,
            wq: vec![0.0; 8 * 4],
            wk: vec![0.0; 8 * 4],
            wv: vec![0.0; 8 * 4],
            wr: vec![0.0; 4 * 4],
            wgate: vec![0.0; 8 * 4],
            wo: vec![0.0; 4 * 8],
            sink_proj: vec![0.0; 2 * 4],
            rel_table: vec![0.0; 2 * 4],
            q_clip_scale: None,
            rel_clip_scale: None,
            attn_norm: vec![0.0; 4],
            k_conv: vec![1.0; 8],
            v_conv: vec![1.0; 8],
            attn_post_conv: vec![1.0; 4],
            ffn_post_conv: vec![1.0; 4],
        }
    }

    #[test]
    fn zero_weights_give_residual_passthrough() {
        let w = tiny_weights();
        let mut st = ConvState::default();
        let x = [1.0, 2.0, 3.0, 4.0];
        let out = decode_layer(&w, &mut st, &x, 0);
        // All projections/gates zero => attention result zero => output = x.
        for (a, b) in out.iter().zip(x.iter()) {
            assert!(oracle_close(*a, *b), "{a} vs {b}");
        }
    }

    #[test]
    #[ignore = "requires the 14 GB inkling2-mini-mq4.hfq artifact"]
    fn decode_real_layer0_produces_finite_output() {
        use crate::config::Inkling2Config;
        use crate::weights::Inkling2Weights;
        use hipfire_runtime::hfq::HfqFile;
        let meta = crate::TEST_METADATA;
        let cfg = Inkling2Config::from_metadata_json(meta).unwrap();
        let hfq = HfqFile::open(std::path::Path::new(
            r"D:\hipfire-models\inkling2-mini-mq4.hfq",
        ))
        .expect("open artifact");
        let w = Inkling2Weights::load(&hfq, &cfg).expect("load weights");
        let p = "model.llm.layers.0.";
        let (_, embed) = w.f32("model.llm.embed.weight").unwrap();
        let x = embed[..cfg.n_embd].to_vec(); // one token embedding row

        let g = |s: &str| w.f32(&format!("{p}{s}")).unwrap().1;
        let (_, qshape) = w.f32(&format!("{p}attn.wq_du.weight")).unwrap();
        let _ = qshape;
        let d = cfg.head_dim;
        let nh = cfg.n_head;
        let nkv = cfg.n_head_kv;
        let nrel = cfg.n_rel;
        let f = LayerWeightsF32 {
            n_embd: cfg.n_embd,
            n_head: nh,
            n_head_kv: nkv,
            head_dim: d,
            n_rel: nrel,
            n_extent: cfg.n_extent,
            is_swa: true,
            conv_kernel: cfg.conv_kernel,
            qk_clip: cfg.qk_clip,
            rel_clip: cfg.rel_clip,
            rms_eps: 1e-6,
            wq: g("attn.wq_du.weight"),
            wk: g("attn.wk_dv.weight"),
            wv: g("attn.wv_dv.weight"),
            wr: g("attn.wr_du.weight"),
            wgate: g("attn.gate_attn.weight"),
            wo: g("attn.wo_ud.weight"),
            sink_proj: g("attn.attn_sink_proj.weight"),
            rel_table: g("attn.rel_logits_proj.proj"),
            q_clip_scale: w.f32(&format!("{p}attn.q_l2_clip.scale")).ok().map(|x| x.1),
            rel_clip_scale: w
                .f32(&format!("{p}attn.rel_l2_clip.scale"))
                .ok()
                .map(|x| x.1),
            attn_norm: g("attn_norm.weight"),
            k_conv: flatten_conv(&g("attn.k_sconv.weight"), 4),
            v_conv: flatten_conv(&g("attn.v_sconv.weight"), 4),
            attn_post_conv: flatten_conv(&g("attn_sconv.weight"), 4),
            ffn_post_conv: flatten_conv(&g("mlp_sconv.weight"), 4),
        };
        let mut st = ConvState::default();
        let out = decode_layer(&f, &mut st, &x, 0);
        assert_eq!(out.len(), cfg.n_embd);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite layer output");
        let max = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min = out.iter().copied().fold(f32::INFINITY, f32::min);
        eprintln!("layer0 decode: {} values, min={min}, max={max}", out.len());
    }

    #[test]
    fn moe_single_expert_is_its_output_times_scale() {
        // 2 experts, top-1, expert 0 selected: identity gate/up/down,
        // SiTU(1,1,beta=4,lb=25) * down = act. Router forces expert 0.
        let n = 2;
        let inter = 2;
        let router = vec![100.0, 0.0, 0.0, 100.0]; // [2,2], x=[1,0] -> e0 wins
        let x = [1.0, 0.0];
        // gate_up[e] = [gate(2x2); up(2x2)], identity so g=x, u=x.
        let gu0 = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let gu1 = vec![0.0; 8];
        let d0 = vec![1.0, 0.0, 0.0, 1.0]; // down 2x2 identity
        let d1 = vec![0.0; 4];
        let out = decode_moe(
            &x,
            &router,
            None,
            &[gu0, gu1],
            &[d0, d1],
            n,
            inter,
            1,
            8.0,
            4.0,
            25.0,
        );
        let act0 = oracle::situ(1.0, 1.0, 4.0, 25.0);
        let act1 = oracle::situ(0.0, 0.0, 4.0, 25.0);
        assert!(
            oracle_close(out[0], 8.0 * act0),
            "{} vs {}",
            out[0],
            8.0 * act0
        );
        assert!(
            oracle_close(out[1], 8.0 * act1),
            "{} vs {}",
            out[1],
            8.0 * act1
        );
    }

    #[test]
    #[ignore = "requires the 14 GB inkling2-mini-mq4.hfq artifact"]
    fn moe_real_layer0_produces_finite_output() {
        use crate::weights::Inkling2Weights;
        use hipfire_runtime::hfq::HfqFile;
        let cfg = crate::config::Inkling2Config::from_metadata_json(crate::TEST_METADATA).unwrap();
        let hfq = HfqFile::open(std::path::Path::new(
            r"D:\hipfire-models\inkling2-mini-mq4.hfq",
        ))
        .expect("open");
        let w = Inkling2Weights::load(&hfq, &cfg).expect("load");
        let p = "model.llm.layers.0.";
        let router = w.f32(&format!("{p}mlp.gate.weight")).unwrap().1;
        let bias = w.f32(&format!("{p}mlp.gate.bias")).unwrap().1;
        let mut gu = Vec::with_capacity(cfg.n_experts);
        let mut dn = Vec::with_capacity(cfg.n_experts);
        for e in 0..cfg.n_experts {
            gu.push(
                w.f32(&format!("{p}mlp.experts.{e}.gate_up_proj.weight"))
                    .unwrap()
                    .1,
            );
            dn.push(
                w.f32(&format!("{p}mlp.experts.{e}.down_proj.weight"))
                    .unwrap()
                    .1,
            );
        }
        let x = vec![0.01f32; cfg.n_embd];
        let out = decode_moe(
            &x,
            &router,
            Some(&bias),
            &gu,
            &dn,
            cfg.n_embd,
            cfg.inter,
            cfg.n_expert_used,
            cfg.route_scale,
            cfg.situ_beta,
            cfg.situ_linear_beta,
        );
        assert!(out.iter().all(|v| v.is_finite()), "non-finite MoE output");
        let max = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("real MoE: {} values, max={max}", out.len());
    }

    #[test]
    #[ignore = "requires the 14 GB inkling2-mini-mq4.hfq artifact"]
    fn full_visit_real_layer0_produces_finite_output() {
        use crate::weights::Inkling2Weights;
        use hipfire_runtime::hfq::HfqFile;
        let cfg = crate::config::Inkling2Config::from_metadata_json(crate::TEST_METADATA).unwrap();
        let hfq = HfqFile::open(std::path::Path::new(
            r"D:\hipfire-models\inkling2-mini-mq4.hfq",
        ))
        .expect("open");
        let w = Inkling2Weights::load(&hfq, &cfg).expect("load");
        let embed = w.f32("model.llm.embed.weight").unwrap().1;
        let x = embed[..cfg.n_embd].to_vec();
        let mut st = ConvState::default();
        let mut blocks = Vec::new();
        let out = decode_visit(&w, &cfg, &mut st, &mut blocks, &x, 0, 0).expect("decode visit");
        assert_eq!(out.len(), cfg.n_embd);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite visit output");
        let max = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("full visit 0: {} values, max={max}", out.len());
    }

    fn oracle_close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }
}
