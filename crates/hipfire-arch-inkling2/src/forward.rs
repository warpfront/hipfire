// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire â€” see LICENSE and NOTICE in the project root.

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

/// Load a conv tensor into `[kernel, n_channels]` oracle layout, honoring the
/// checkpoint layouts:
///  * SWA layers: `[n_channels, 1, kernel]` (plain `.weight`) â€” transpose only.
///  * full layers: `.static_w` is `[kernel, n_channels]`; the merged converter
///    applies `flip(0).T` to produce the runtime conv, so flip the kernel axis.
fn load_conv<W: crate::weights::WeightSource>(
    w: &W,
    name: &str,
    kernel: usize,
) -> Result<Vec<f32>, String> {
    let (shape, v) = w.f32(name)?;
    let n_ch = v.len() / kernel;
    let mut out = vec![0.0f32; v.len()];
    if shape.len() == 3 {
        // [channels, 1, kernel] -> [kernel, channels]
        for c in 0..n_ch {
            for k in 0..kernel {
                out[k * n_ch + c] = v[c * kernel + k];
            }
        }
    } else if name.ends_with(".static_w") {
        // [kernel, channels], flipped by the converter along the kernel axis.
        for k in 0..kernel {
            for c in 0..n_ch {
                out[k * n_ch + c] = v[(kernel - 1 - k) * n_ch + c];
            }
        }
    } else {
        for c in 0..n_ch {
            for k in 0..kernel {
                out[k * n_ch + c] = v[c * kernel + k];
            }
        }
    }
    Ok(out)
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
    let qc = oracle::clip(&q, w.q_clip_scale.as_deref(), w.qk_clip, w.head_dim);
    let kc = oracle::clip(&k_conv, None, w.qk_clip, w.head_dim);
    // V uses unweighted RMSNorm per head.
    let vc: Vec<f32> = (0..nkv)
        .flat_map(|h| oracle::rmsnorm(&v_conv[h * d..(h + 1) * d], w.rms_eps))
        .collect();
    let rc = oracle::clip(&r, w.rel_clip_scale.as_deref(), w.rel_clip, w.n_rel);

    // Relative bias: table [n_rel, n_extent] Ã— per-head R, sink subtracted
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
            &kc[(h / (nh / nkv)) * d..(h / (nh / nkv) + 1) * d]
        } else {
            &kc[0..d]
        };
        let vh = if h < nkv {
            &vc[(h / (nh / nkv)) * d..(h / (nh / nkv) + 1) * d]
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
    decode_moe_lazy(
        x,
        router,
        router_bias,
        n_embd,
        inter,
        top_k,
        route_scale,
        situ_beta,
        situ_linear_beta,
        &mut |e| (gate_up[e].clone(), down[e].clone()),
    )
}

/// Like [`decode_moe`] but decodes expert weights lazily via `get(e)`, so a
/// 128-expert layer only decodes its top-8 (critical for the 48-visit CPU run).
pub fn decode_moe_lazy<F>(
    x: &[f32],
    router: &[f32],
    router_bias: Option<&[f32]>,
    n_embd: usize,
    inter: usize,
    top_k: usize,
    route_scale: f32,
    situ_beta: f32,
    situ_linear_beta: f32,
    get: &mut F,
) -> Vec<f32>
where
    F: FnMut(usize) -> (Vec<f32>, Vec<f32>),
{
    let n_experts = router.len() / n_embd;
    let logits = oracle::matvec(router, x, n_experts, n_embd);
    let selected = oracle::sigmoid_topk(&logits, router_bias, top_k, route_scale);
    let mut out = vec![0.0f32; n_embd];
    for (e, weight) in selected {
        let (gu, dn) = get(e);
        let gate = &gu[..inter * n_embd];
        let up = &gu[inter * n_embd..];
        let g = oracle::matvec(gate, x, inter, n_embd);
        let u = oracle::matvec(up, x, inter, n_embd);
        let act: Vec<f32> = (0..inter)
            .map(|i| oracle::situ(g[i], u[i], situ_beta, situ_linear_beta))
            .collect();
        let d = oracle::matvec(&dn, &act, n_embd, inter);
        for i in 0..n_embd {
            out[i] += weight * d[i];
        }
    }
    out
}

/// A controller's three weight matrices for one of Q/K/V/FFN:
/// `down [2*n_embd, rank]`, `gain [n_embd, rank]`, `forget [n_embd, rank]`.
pub struct ControllerWeights {
    pub down: Vec<f32>,
    pub gain: Vec<f32>,
    pub forget: Vec<f32>,
}

/// Pack the global cross-pass query tensors into per-core-layer message
/// queries. The stored `middle_cycle_cross_pass_queries` and `_kv_queries`
/// are each `[2*core_len, n_embd]`; per core layer i the four queries are
/// `[q[2i], kv[2i], kv[2i+1], q[2i+1]]` = [Q, K, V, FFN] messages.
pub fn pack_message_queries<W: crate::weights::WeightSource>(
    w: &W,
    cfg: &crate::config::Inkling2Config,
) -> Result<Vec<[Vec<f32>; 4]>, String> {
    let (_, q) = w.f32("model.llm.middle_cycle_cross_pass_queries")?;
    let (_, kv) = w.f32("model.llm.middle_cycle_cross_pass_kv_queries")?;
    let e = cfg.n_embd;
    let core = cfg.loop_cfg.core_end - cfg.loop_cfg.core_start;
    if q.len() != 2 * core * e || kv.len() != q.len() {
        return Err(format!(
            "inkling2: cross-pass queries shape mismatch: q={} kv={} expected {}",
            q.len(),
            kv.len(),
            2 * core * e
        ));
    }
    let row = |v: &[f32], i: usize| v[i * e..(i + 1) * e].to_vec();
    let mut out = Vec::with_capacity(core);
    for i in 0..core {
        out.push([
            row(&q, 2 * i),
            row(&kv, 2 * i),
            row(&kv, 2 * i + 1),
            row(&q, 2 * i + 1),
        ]);
    }
    Ok(out)
}

/// Run one full decoder visit (attention + MoE + residual/block bookkeeping),
/// loading weights lazily from the validated HFQ index. `x` is `[n_embd]`,
/// `visit` the global visit index (0..48), `blocks` the AttnRes block stream,
/// `state` the per-visit conv state. `messages`/`ctrl` are present for loop
/// visits after pass 0 (four messages Q/K/V/FFN and their controllers).
pub fn decode_visit<W: crate::weights::WeightSource>(
    w: &W,
    cfg: &crate::config::Inkling2Config,
    state: &mut ConvState,
    blocks: &mut Vec<Vec<f32>>,
    x: &[f32],
    visit: usize,
    pos: usize,
    messages: Option<&([Vec<f32>; 4], [ControllerWeights; 4])>,
    mut debug: Option<&mut std::collections::HashMap<String, Vec<f32>>>,
) -> Result<Vec<f32>, String> {
    let phys = cfg.loop_cfg.layer_phys(visit);
    let p = format!("model.llm.layers.{phys}.");
    let swa = cfg.swa_layers[phys];
    let g = |s: &str| w.f32(&format!("{p}{s}")).map(|x| x.1);

    // --- attention inputs: AttnRes mix (if this visit has a query) ---
    // SWA layers have 1 query (all three inputs equal); full layers have 3
    // queries (one each for Q, K, V). Then loop messages are applied per input.
    // Off-by-one: GGUF has no attn_res_query on block 0; the HFQ checkpoint
    // names it `attn_queries.0` but it feeds block 1. Our visit uses `phys`
    // directly, so skip the mix on visit 0 (matches llama.cpp: build_layer(0)
    // has no attn_res_query), and use query phys-1 on later visits.
    let (base_q, base_k, base_v) = if phys == 0 {
        (x.to_vec(), x.to_vec(), x.to_vec())
    } else {
        let attn_q = format!("model.llm.attn_res.attn_queries.{}", phys - 1);
        let (shape, q) = w.f32(&attn_q)?;
        let query_rows = if shape.len() == 1 {
            1
        } else {
            shape[0] as usize
        };
        let mixed = oracle::mix(blocks, x, &q, cfg.n_embd, query_rows);
        let row = |j: usize| mixed[j * cfg.n_embd..(j + 1) * cfg.n_embd].to_vec();
        if query_rows >= 3 {
            (row(0), row(1), row(2))
        } else {
            let r = row(0);
            (r.clone(), r.clone(), r)
        }
    };
    // Pass 0 (no messages): inputs are the AttnRes-mixed bases. Later passes:
    // each is `control(base, msg[j], ctrl[j])`.
    let (q_ctrl, k_ctrl, v_ctrl) = match messages {
        Some((msgs, ctrls)) => {
            let apply = |base: &[f32], j: usize, c: &ControllerWeights| {
                oracle::control(
                    base,
                    &msgs[j],
                    &c.down,
                    &c.gain,
                    &c.forget,
                    cfg.n_embd,
                    cfg.loop_cfg.gate_rank,
                )
            };
            (
                apply(&base_q, 0, &ctrls[0]),
                apply(&base_k, 1, &ctrls[1]),
                apply(&base_v, 2, &ctrls[2]),
            )
        }
        None => (base_q.clone(), base_k.clone(), base_v.clone()),
    };
    let (q_inp, k_inp, v_inp) = if swa {
        let norm = g("attn_norm.weight")?;
        (
            rmsnorm_gain(&q_ctrl, &norm, cfg.rms_norm_eps),
            rmsnorm_gain(&k_ctrl, &norm, cfg.rms_norm_eps),
            rmsnorm_gain(&v_ctrl, &norm, cfg.rms_norm_eps),
        )
    } else {
        (
            rmsnorm_gain(&q_ctrl, &g("q_input_norm.weight")?, cfg.rms_norm_eps),
            rmsnorm_gain(&k_ctrl, &g("k_input_norm.weight")?, cfg.rms_norm_eps),
            rmsnorm_gain(&v_ctrl, &g("v_input_norm.weight")?, cfg.rms_norm_eps),
        )
    };

    if let Some(d) = debug.as_deref_mut() {
        d.insert("x_in".into(), x.to_vec());
        d.insert("base_q".into(), base_q.clone());
        d.insert("q_ctrl".into(), q_ctrl.clone());
        d.insert("q_inp".into(), q_inp.clone());
        d.insert("k_inp".into(), k_inp.clone());
        d.insert("v_inp".into(), v_inp.clone());
    }
    // --- attention projections + convs (SWA static; full dynamic deferred) ---
    let d = cfg.head_dim;
    let nh = cfg.n_head;
    let nkv = cfg.n_head_kv;
    let ekv = nkv * d;
    let eattn = nh * d;
    let q = oracle::matvec(&g("attn.wq_du.weight")?, &q_inp, eattn, cfg.n_embd);
    let k_raw = oracle::matvec(&g("attn.wk_dv.weight")?, &k_inp, ekv, cfg.n_embd);
    let v_raw = oracle::matvec(&g("attn.wv_dv.weight")?, &v_inp, ekv, cfg.n_embd);
    let r = oracle::matvec(&g("attn.wr_du.weight")?, &q_inp, nh * cfg.n_rel, cfg.n_embd);
    if let Some(db) = debug.as_deref_mut() {
        db.insert("k_raw".into(), k_raw.clone());
        db.insert("v_raw".into(), v_raw.clone());
    }
    // SWA layers store a static `k_sconv.weight [ch,1,kernel]`; full layers
    // store `k_sconv.static_w` (same shape) plus a `weight_proj` for dynamic
    // taps. The dynamic taps are deferred; here use the static weights.
    let k_conv_name = if w.f32(&format!("{p}attn.k_sconv.weight")).is_ok() {
        format!("{p}attn.k_sconv.weight")
    } else {
        format!("{p}attn.k_sconv.static_w")
    };
    let v_conv_name = if w.f32(&format!("{p}attn.v_sconv.weight")).is_ok() {
        format!("{p}attn.v_sconv.weight")
    } else {
        format!("{p}attn.v_sconv.static_w")
    };
    let (k_conv, v_conv): (Vec<f32>, Vec<f32>);
    if !swa {
        // Full layers: dynamic K/V convs. static_w is stored [kernel, channels]
        // verbatim; the merged converter applies flip(0).T -> torch [C,kernel]
        // with tap (kernel-1)=current. Transpose static_w to that torch view.
        let (sw_k_shape, sw_k) = w.f32(&format!("{p}attn.k_sconv.static_w"))?;
        let (sw_v_shape, sw_v) = w.f32(&format!("{p}attn.v_sconv.static_w"))?;
        let torch_conv = |shape: &[usize], v: &[f32]| -> Vec<f32> {
            let kk = cfg.conv_kernel;
            let ch = v.len() / kk;
            // HFQ stores static_w verbatim [kernel, channels] (converter's
            // flip(0).T not applied) -> need flip here. The GgufF32 source
            // already returns the converted torch view [channels, kernel].
            let already_torch = shape.len() >= 2 && shape[0] == ch;
            let mut t = vec![0.0f32; v.len()];
            for c in 0..ch {
                for l in 0..kk {
                    t[c * kk + l] = if already_torch {
                        v[c * kk + l]
                    } else {
                        v[(kk - 1 - l) * ch + c]
                    };
                }
            }
            t
        };
        let wkt = torch_conv(&sw_k_shape, &sw_k);
        let wvt = torch_conv(&sw_v_shape, &sw_v);
        let kproj = g("attn.k_sconv.weight_proj")?;
        let vproj = g("attn.v_sconv.weight_proj")?;
        let (kc2, kp2) = oracle::convolve_dynamic(
            &k_raw,
            &wkt,
            &kproj,
            &k_inp,
            &state.k_past,
            nkv,
            d,
            cfg.conv_kernel,
            cfg.n_embd,
        );
        let (vc2, vp2) = oracle::convolve_dynamic(
            &v_raw,
            &wvt,
            &vproj,
            &v_inp,
            &state.v_past,
            nkv,
            d,
            cfg.conv_kernel,
            cfg.n_embd,
        );
        state.k_past = kp2;
        state.v_past = vp2;
        k_conv = kc2;
        v_conv = vc2;
    } else {
        let k_conv_w = load_conv(w, &k_conv_name, cfg.conv_kernel)?;
        let v_conv_w = load_conv(w, &v_conv_name, cfg.conv_kernel)?;
        let (kc2, kp2) =
            oracle::convolve_static(&k_raw, &k_conv_w, &state.k_past, 1, ekv, cfg.conv_kernel);
        let (vc2, vp2) =
            oracle::convolve_static(&v_raw, &v_conv_w, &state.v_past, 1, ekv, cfg.conv_kernel);
        state.k_past = kp2;
        state.v_past = vp2;
        k_conv = kc2;
        v_conv = vc2;
    }

    let qc = oracle::clip(
        &q,
        w.f32(&format!("{p}attn.q_l2_clip.scale"))
            .ok()
            .map(|x| x.1)
            .as_deref(),
        cfg.qk_clip,
        d,
    );
    let kc = oracle::clip(&k_conv, None, cfg.qk_clip, d);
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
        cfg.n_rel,
    );
    if !swa {
        // full-layer tau multiplies Q (per-head below) and R.
        let t = oracle::tau(pos, cfg.log_floor, cfg.log_alpha);
        for v in rc.iter_mut() {
            *v *= t;
        }
    }

    let sinks = oracle::matvec(&g("attn.attn_sink_proj.weight")?, &q_inp, nh, cfg.n_embd);
    let rel_table = g("attn.rel_logits_proj.proj")?;
    let mut rel_bias_debug: Vec<f32> = Vec::with_capacity(nh);
    let mut attn_out = vec![0.0f32; eattn];
    for h in 0..nh {
        let rh = &rc[h * cfg.n_rel..(h + 1) * cfg.n_rel];
        let bias0: f32 = (0..cfg.n_rel)
            .map(|rr| rel_table[rr * cfg.n_extent] * rh[rr])
            .sum::<f32>()
            - sinks[h];
        rel_bias_debug.push(bias0);
        let mut qh = qc[h * d..(h + 1) * d].to_vec();
        if !swa {
            let t = oracle::tau(pos, cfg.log_floor, cfg.log_alpha);
            for v in &mut qh {
                *v *= t;
            }
        }
        let kh = &kc[(h / (nh / nkv)) * d..(h / (nh / nkv) + 1) * d];
        let vh = &vc[(h / (nh / nkv)) * d..(h / (nh / nkv) + 1) * d];
        let score: f32 =
            qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>() / (d as f32).sqrt() + bias0;
        let wv = oracle::attention_softmax_with_sink(&[score], 0.0, vh, 1, d);
        attn_out[h * d..(h + 1) * d].copy_from_slice(&wv);
    }
    if let Some(d) = debug.as_deref_mut() {
        d.insert("k_conv".into(), k_conv.clone());
        d.insert("v_conv".into(), v_conv.clone());
        d.insert("qc".into(), qc.clone());
        d.insert("kc".into(), kc.clone());
        d.insert("vc".into(), vc.clone());
        d.insert("rc".into(), rc.clone());
        d.insert("sinks".into(), sinks.clone());
        d.insert("attn_heads".into(), attn_out.clone());
    }
    let gate = oracle::matvec(&g("attn.gate_attn.weight")?, &q_inp, eattn, cfg.n_embd);
    for i in 0..eattn {
        attn_out[i] *= oracle::sigmoid(gate[i]);
    }
    let proj = oracle::matvec(&g("attn.wo_ud.weight")?, &attn_out, cfg.n_embd, eattn);
    if let Some(db) = debug.as_deref_mut() {
        db.insert("wo_out".into(), proj.clone());
        db.insert("rel_bias".into(), rel_bias_debug.clone());
    }
    let (attn_res, ap) = oracle::convolve_static(
        &proj,
        &load_conv(w, &format!("{p}attn_sconv.weight"), cfg.conv_kernel)?,
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
    // FFN message controller (index 3 = LOOP_CTRL_FFN), applied before norm.
    if let Some((msgs, ctrls)) = messages {
        ffn_in = oracle::control(
            &ffn_in,
            &msgs[3],
            &ctrls[3].down,
            &ctrls[3].gain,
            &ctrls[3].forget,
            cfg.n_embd,
            cfg.loop_cfg.gate_rank,
        );
    }
    let ffn_normed = rmsnorm_gain(&ffn_in, &g("mlp_norm.weight")?, cfg.rms_norm_eps);
    if let Some(d) = debug.as_deref_mut() {
        d.insert("ffn_mix".into(), ffn_in.clone());
        d.insert("ffn_normed".into(), ffn_normed.clone());
    }
    let router = g("mlp.gate.weight")?;
    let bias = g("mlp.gate.bias").ok();
    // Route first, decode only the selected experts (lazy) â€” a 48-visit run
    // touches ~384 experts instead of all 6144.
    let expert_prefix = format!("{p}mlp.experts.");
    let moe = decode_moe_lazy(
        &ffn_normed,
        &router,
        bias.as_deref(),
        cfg.n_embd,
        cfg.inter,
        cfg.n_expert_used,
        cfg.route_scale,
        cfg.situ_beta,
        cfg.situ_linear_beta,
        &mut |e| {
            let gu = w
                .f32(&format!("{expert_prefix}{e}.gate_up_proj.weight"))
                .expect("expert gate_up")
                .1;
            let dn = w
                .f32(&format!("{expert_prefix}{e}.down_proj.weight"))
                .expect("expert down")
                .1;
            (gu, dn)
        },
    );
    let (moe_conv, fp) = oracle::convolve_static(
        &moe,
        &load_conv(w, &format!("{p}mlp_sconv.weight"), cfg.conv_kernel)?,
        &state.ffn_out_past,
        1,
        cfg.n_embd,
        cfg.conv_kernel,
    );
    state.ffn_out_past = fp;
    if let Some(d) = debug.as_deref_mut() {
        d.insert("moe_raw".into(), moe.clone());
        d.insert("moe_conv".into(), moe_conv.clone());
        d.insert("cur".into(), cur.clone());
        d.insert("cur_attn".into(), cur.clone());
    }
    // llama.cpp build_layer: `ffn_inp = cur` (post-attention stream), then
    // `cur = MoE(mix(ffn_inp))`, `return cur + ffn_inp`. The residual base is
    // the post-attention `cur`, NOT the mixed/controlled `ffn_in`.
    Ok(cur.iter().zip(&moe_conv).map(|(a, b)| a + b).collect())
}

/// Compute logits from the final hidden state: LM head GEMV, fp32, final
/// softcap 30, then slice to the unpadded vocabulary (200058 of 201024).
pub fn lm_head_logits<W: crate::weights::WeightSource>(
    w: &W,
    cfg: &crate::config::Inkling2Config,
    hidden: &[f32],
) -> Result<Vec<f32>, String> {
    let head = w.f32("model.llm.unembed.weight")?.1;
    let vocab_pad = cfg.vocab_size;
    let logits = oracle::matvec(&head, hidden, vocab_pad, cfg.n_embd);
    let capped: Vec<f32> = logits
        .iter()
        .map(|&v| oracle::softcap(v, cfg.softcap))
        .collect();
    Ok(capped[..cfg.unpadded_vocab_size].to_vec())
}

/// Run all 48 visits for one token, from token embedding to final normalized
/// hidden state (before the LM head). Mirrors
/// `llama_model_inkling2_loop::graph::graph`: prelude layers 0..6 once, core
/// layers 6..18 for `num_passes` passes (later passes get cross-pass messages
/// from the previous pass's bank), post-loop layers 18..24 once.
pub fn forward_token<W: crate::weights::WeightSource>(
    w: &W,
    cfg: &crate::config::Inkling2Config,
    token_embed: &[f32],
    pos: usize,
    mut trace: Option<&mut Vec<Vec<f32>>>,
    mut debug: Option<&mut (usize, std::collections::HashMap<String, Vec<f32>>)>,
) -> Result<Vec<f32>, String> {
    // build_inputs: token embedding, RMSNormed (token path only).
    let tok_norm = w.f32("model.llm.embed_norm.weight")?.1;
    let mut x = rmsnorm_gain(token_embed, &tok_norm, cfg.rms_norm_eps);

    // One state per visit slot (48); physical weights are shared but KV/conv
    // state is per-visit.
    let mut states: Vec<ConvState> = (0..cfg.loop_cfg.n_slot())
        .map(|_| ConvState::default())
        .collect();
    let mut blocks: Vec<Vec<f32>> = Vec::new();
    let mut visit = 0usize;

    // Prelude layers 0..core_start (6).
    for _ in 0..cfg.loop_cfg.core_start {
        let out = decode_visit(
            w,
            cfg,
            &mut states[visit],
            &mut blocks,
            &x,
            visit,
            pos,
            None,
            debug
                .as_mut()
                .and_then(|d| if d.0 == visit { Some(&mut d.1) } else { None }),
        )?;
        x = out;
        if let Some(t) = trace.as_deref_mut() {
            t.push(x.clone());
        }
        visit += 1;
    }
    let prelude_x = x.clone();
    let prelude_blocks = blocks.clone();

    let msg_queries = pack_message_queries(w, cfg)?;
    let core_len = cfg.loop_cfg.core_end - cfg.loop_cfg.core_start;

    // Loop passes. pass 0 has no messages; later passes use the previous
    // pass's bank (blocks after the first core block + its output).
    let mut prev_blocks = prelude_blocks.clone();
    let mut prev_out = prelude_x.clone();
    for pass in 0..cfg.loop_cfg.num_passes {
        x = prelude_x.clone();
        blocks = prelude_blocks.clone();
        // Messages/bank come from the previous pass (pass 0: none).
        let bank: Vec<Vec<f32>> = if pass > 0 {
            prev_blocks
                .iter()
                .skip(prelude_blocks.len() + 1)
                .cloned()
                .collect()
        } else {
            Vec::new()
        };
        for i in 0..core_len {
            let phys = cfg.loop_cfg.core_start + i;
            let message = if pass > 0 {
                // mix() appends `cur` itself: states = [bank..., prev_out].
                let msgs: Vec<Vec<f32>> = msg_queries[i]
                    .iter()
                    .map(|q| oracle::mix(&bank, &prev_out, q, cfg.n_embd, 1))
                    .collect();
                let msgs_arr: [Vec<f32>; 4] = [
                    msgs[0].clone(),
                    msgs[1].clone(),
                    msgs[2].clone(),
                    msgs[3].clone(),
                ];
                let ctrls = load_controllers(w, phys, cfg)?;
                Some((msgs_arr, ctrls))
            } else {
                None
            };
            let out = decode_visit(
                w,
                cfg,
                &mut states[visit],
                &mut blocks,
                &x,
                visit,
                pos,
                message.as_ref(),
                debug
                    .as_mut()
                    .and_then(|d| if d.0 == visit { Some(&mut d.1) } else { None }),
            )?;
            x = out;
            if let Some(t) = trace.as_deref_mut() {
                t.push(x.clone());
            }
            visit += 1;
        }
        prev_blocks = blocks.clone();
        prev_out = x.clone();
    }

    // Post-loop layers core_end..n_phys once.
    for _ in cfg.loop_cfg.core_end..cfg.n_phys {
        let out = decode_visit(
            w,
            cfg,
            &mut states[visit],
            &mut blocks,
            &x,
            visit,
            pos,
            None,
            debug
                .as_mut()
                .and_then(|d| if d.0 == visit { Some(&mut d.1) } else { None }),
        )?;
        x = out;
        if let Some(t) = trace.as_deref_mut() {
            t.push(x.clone());
        }
        visit += 1;
    }
    debug_assert_eq!(visit, cfg.loop_cfg.n_slot());
    let _ = &visit;

    // Final norm.
    // build_output: mix the collected AttnRes blocks with the output query,
    // then final RMSNorm. `blocks` still holds all visit block states.
    if let Ok(out_q) = w.f32("model.llm.attn_res.output_query") {
        let mixed = oracle::mix(&blocks, &x, &out_q.1, cfg.n_embd, 1);
        x = mixed;
    }
    let final_norm = w.f32("model.llm.norm.weight")?.1;
    Ok(rmsnorm_gain(&x, &final_norm, cfg.rms_norm_eps))
}

fn load_controllers<W: crate::weights::WeightSource>(
    w: &W,
    phys: usize,
    cfg: &crate::config::Inkling2Config,
) -> Result<[ControllerWeights; 4], String> {
    let p = format!("model.llm.layers.{phys}.controllers.");
    let load = |name: &str| -> Result<ControllerWeights, String> {
        Ok(ControllerWeights {
            down: w.f32(&format!("{p}{name}.down_weight"))?.1,
            gain: w.f32(&format!("{p}{name}.gain_weight"))?.1,
            forget: w.f32(&format!("{p}{name}.forget_weight"))?.1,
        })
    };
    let rank = cfg.loop_cfg.gate_rank;
    for name in ["attn", "k", "v", "mlp"] {
        let c = load(name)?;
        if c.down.len() != 2 * cfg.n_embd * rank {
            return Err(format!(
                "inkling2: controller {phys}.{name} down shape mismatch"
            ));
        }
    }
    Ok([load("attn")?, load("k")?, load("v")?, load("mlp")?])
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
            k_conv: load_conv(&w, &format!("{p}attn.k_sconv.weight"), 4).unwrap(),
            v_conv: load_conv(&w, &format!("{p}attn.v_sconv.weight"), 4).unwrap(),
            attn_post_conv: load_conv(&w, &format!("{p}attn_sconv.weight"), 4).unwrap(),
            ffn_post_conv: load_conv(&w, &format!("{p}mlp_sconv.weight"), 4).unwrap(),
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
        let out = decode_visit(&w, &cfg, &mut st, &mut blocks, &x, 0, 0, None, None)
            .expect("decode visit");
        assert_eq!(out.len(), cfg.n_embd);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite visit output");
        let max = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("full visit 0: {} values, max={max}", out.len());
    }

    #[test]
    #[ignore = "requires the 14 GB artifact; runs all 48 visits (slow)"]
    fn full_48_visit_forward_produces_finite_output() {
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
        let out = forward_token(&w, &cfg, &x, 0, None, None).expect("forward token");
        assert_eq!(out.len(), cfg.n_embd);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite model output");
        let max = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("48-visit forward: {} values, max={max}", out.len());
    }

    #[test]
    #[ignore = "requires the 14 GB artifact; full 48-visit forward + LM head"]
    fn end_to_end_logits_are_finite_and_sized() {
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
        let hidden = forward_token(&w, &cfg, &x, 0, None, None).expect("forward");
        let logits = lm_head_logits(&w, &cfg, &hidden).expect("lm head");
        assert_eq!(logits.len(), cfg.unpadded_vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite logits");
        let (argmax, _) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        eprintln!(
            "end-to-end: {} logits, argmax={argmax}, max={max}",
            logits.len()
        );
    }

    #[test]
    #[ignore = "parity: dump CPU logits for token 0 to compare with llama.cpp oracle"]
    fn dump_cpu_logits_token0() {
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
        let hidden = forward_token(&w, &cfg, &x, 0, None, None).expect("forward");
        let logits = lm_head_logits(&w, &cfg, &hidden).expect("lm head");
        let mut s = String::new();
        for (i, v) in logits.iter().enumerate() {
            s.push_str(&format!("{} {:08x}\n", i, v.to_bits()));
        }
        std::fs::write(r"D:\hipfire-models\cpu-logits-tok0.txt", s).unwrap();
        eprintln!("dumped {} CPU logits", logits.len());
    }

    #[test]
    #[ignore = "parity: dump CPU logits on RAW checkpoint weights"]
    fn dump_raw_logits_token0() {
        use crate::weights::{GgufF32, WeightSource};
        let cfg = crate::config::Inkling2Config::from_metadata_json(crate::TEST_METADATA).unwrap();
        let w = GgufF32::open(std::path::Path::new(r"D:\hipfire-models\gguf-f32.bin"))
            .expect("open gguf f32");
        let embed = w.f32("model.llm.embed.weight").unwrap().1;
        let x = embed[..cfg.n_embd].to_vec();
        let hidden = forward_token(&w, &cfg, &x, 0, None, None).expect("forward");
        let logits = lm_head_logits(&w, &cfg, &hidden).expect("lm head");
        let mut s = String::new();
        for (i, v) in logits.iter().enumerate() {
            s.push_str(&format!("{} {:08x}\n", i, v.to_bits()));
        }
        std::fs::write(r"D:\hipfire-models\cpu-raw-logits-tok0.txt", s).unwrap();
        let (argmax, _) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        eprintln!("raw: dumped {} logits, argmax={argmax}", logits.len());
    }

    #[test]
    #[ignore = "parity: dump per-visit hidden states on F32 weights"]
    fn dump_visits_f32() {
        use crate::weights::{GgufF32, WeightSource};
        let cfg = crate::config::Inkling2Config::from_metadata_json(crate::TEST_METADATA).unwrap();
        let w = GgufF32::open(std::path::Path::new(r"D:\hipfire-models\gguf-f32.bin")).unwrap();
        let embed = w.f32("model.llm.embed.weight").unwrap().1;
        let x = embed[..cfg.n_embd].to_vec();
        let mut trace = Vec::new();
        let _ = forward_token(&w, &cfg, &x, 0, Some(&mut trace), None).expect("forward");
        let dir = std::path::Path::new(r"D:\hipfire-models\rust-visits-tok0");
        std::fs::create_dir_all(dir).unwrap();
        for (i, v) in trace.iter().enumerate() {
            let mut s = String::new();
            for x in v {
                s.push_str(&format!("{}\n", x.to_bits()));
            }
            std::fs::write(dir.join(format!("visit{i:02}.txt")), s).unwrap();
        }
        eprintln!("dumped {} visits", trace.len());
    }

    #[test]
    #[ignore = "compare visit-0 intermediates against validated reference dumps (HFQ path)"]
    fn visit0_intermediates_vs_reference() {
        use crate::weights::{GgufF32, WeightSource};
        let cfg = crate::config::Inkling2Config::from_metadata_json(crate::TEST_METADATA).unwrap();
        let w = GgufF32::open(std::path::Path::new(r"D:\hipfire-models\gguf-f32.bin")).unwrap();
        let emb = w.f32("model.llm.embed.weight").unwrap().1;
        let x = rmsnorm_gain(
            &emb[..cfg.n_embd],
            &w.f32("model.llm.embed_norm.weight").unwrap().1,
            cfg.rms_norm_eps,
        );

        // visit 0 = physical layer 0, SWA, no messages, no AttnRes query on layer 0.
        let p = "model.llm.layers.0.";
        let an = w.f32(&format!("{p}attn_norm.weight")).unwrap().1;
        let q_inp = rmsnorm_gain(&x, &an, cfg.rms_norm_eps);
        let wq = w.f32(&format!("{p}attn.wq_du.weight")).unwrap().1;
        let q_raw = oracle::matvec(&wq, &q_inp, cfg.n_head * cfg.head_dim, cfg.n_embd);

        let check = |name: &str, got: &[f32]| {
            let path = format!(r"D:\hipfire-models\ref-visits\tok0\{name}.txt");
            if let Ok(s) = std::fs::read_to_string(&path) {
                let want: Vec<f32> = s
                    .lines()
                    .map(|l| f32::from_bits(l.trim().parse().unwrap()))
                    .collect();
                assert_eq!(want.len(), got.len(), "{name} len");
                let maxd = want
                    .iter()
                    .zip(got)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                eprintln!("{name}: max|d|={maxd:.6}");
            } else {
                // dump for the reference agent to consume
                let mut s = String::new();
                for v in got {
                    s.push_str(&format!("{}\n", v.to_bits()));
                }
                std::fs::write(&path, s).unwrap();
                eprintln!("{name}: dumped {} values", got.len());
            }
        };
        check("x_in", &x);
        check("q_inp", &q_inp);
        check("q_raw", &q_raw);

        // K/V inputs (SWA layer 0: all equal x_in, shared attn_norm).
        let k_inp = rmsnorm_gain(&x, &an, cfg.rms_norm_eps);
        let v_inp = k_inp.clone();
        check("k_inp", &k_inp);
        check("v_inp", &v_inp);
        let wk = w.f32(&format!("{p}attn.wk_dv.weight")).unwrap().1;
        let wv = w.f32(&format!("{p}attn.wv_dv.weight")).unwrap().1;
        let k_raw = oracle::matvec(&wk, &k_inp, cfg.n_head_kv * cfg.head_dim, cfg.n_embd);
        let v_raw = oracle::matvec(&wv, &v_inp, cfg.n_head_kv * cfg.head_dim, cfg.n_embd);
        check("k_raw", &k_raw);
        check("v_raw", &v_raw);
        let wr = w.f32(&format!("{p}attn.wr_du.weight")).unwrap().1;
        let r_raw = oracle::matvec(&wr, &q_inp, cfg.n_head * cfg.n_rel, cfg.n_embd);
        check("r_raw", &r_raw);

        // K/V convs â€” compare the raw conv output (before clip).
        let ekv = cfg.n_head_kv * cfg.head_dim;
        let kc_w = load_conv(&w, &format!("{p}attn.k_sconv.weight"), cfg.conv_kernel).unwrap();
        let vc_w = load_conv(&w, &format!("{p}attn.v_sconv.weight"), cfg.conv_kernel).unwrap();
        let (k_conv, _) = oracle::convolve_static(&k_raw, &kc_w, &[], 1, ekv, cfg.conv_kernel);
        let (v_conv, _) = oracle::convolve_static(&v_raw, &vc_w, &[], 1, ekv, cfg.conv_kernel);
        check("k_conv", &k_conv);
        check("v_conv", &v_conv);

        let qscale = w.f32(&format!("{p}attn.q_l2_clip.scale")).ok().map(|x| x.1);
        let rscale = w
            .f32(&format!("{p}attn.rel_l2_clip.scale"))
            .ok()
            .map(|x| x.1);
        let qc = oracle::clip(&q_raw, qscale.as_deref(), cfg.qk_clip, cfg.head_dim);
        let kc = oracle::clip(&k_conv, None, cfg.qk_clip, cfg.head_dim);
        let rc = oracle::clip(&r_raw, rscale.as_deref(), cfg.rel_clip, cfg.n_rel);
        check("qc", &qc);
        check("kc", &kc);
        check("rc", &rc);

        // V RMSNorm per KV head, sinks, rel bias, attention (single key, sink=0).
        let vc: Vec<f32> = (0..cfg.n_head_kv)
            .flat_map(|h| {
                oracle::rmsnorm(
                    &v_conv[h * cfg.head_dim..(h + 1) * cfg.head_dim],
                    cfg.rms_norm_eps,
                )
            })
            .collect();
        check("vc", &vc);
        let sinks = oracle::matvec(
            &w.f32(&format!("{p}attn.attn_sink_proj.weight")).unwrap().1,
            &q_inp,
            cfg.n_head,
            cfg.n_embd,
        );
        check("sinks", &sinks);
        let rel_table_raw = w.f32(&format!("{p}attn.rel_logits_proj.proj")).unwrap().1;
        // rel_table is [n_rel, n_extent] from HFQ; distance 0 gather column 0.
        let mut rel_bias = Vec::with_capacity(cfg.n_head);
        let mut scores = Vec::with_capacity(cfg.n_head);
        let mut attn_heads = vec![0.0f32; cfg.n_head * cfg.head_dim];
        for h in 0..cfg.n_head {
            let rh = &rc[h * cfg.n_rel..(h + 1) * cfg.n_rel];
            let bias0: f32 = (0..cfg.n_rel)
                .map(|rr| rel_table_raw[rr * cfg.n_extent] * rh[rr])
                .sum::<f32>()
                - sinks[h];
            rel_bias.push(bias0);
            let g = h / (cfg.n_head / cfg.n_head_kv);
            let qh = &qc[h * cfg.head_dim..(h + 1) * cfg.head_dim];
            let kh = &kc[g * cfg.head_dim..(g + 1) * cfg.head_dim];
            let score = qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>()
                / (cfg.head_dim as f32).sqrt()
                + bias0;
            scores.push(score);
            let wv = oracle::attention_softmax_with_sink(
                &[score],
                0.0,
                &vc[g * cfg.head_dim..(g + 1) * cfg.head_dim],
                1,
                cfg.head_dim,
            );
            attn_heads[h * cfg.head_dim..(h + 1) * cfg.head_dim].copy_from_slice(&wv);
        }
        check("rel_bias", &rel_bias);
        check("attn_scores", &scores);
        check("attn_heads", &attn_heads);

        // output gate + wo + post-conv + AttnRes block bookkeeping.
        let gate_raw = oracle::matvec(
            &w.f32(&format!("{p}attn.gate_attn.weight")).unwrap().1,
            &q_inp,
            cfg.n_head * cfg.head_dim,
            cfg.n_embd,
        );
        check("attn_gate_raw", &gate_raw);
        let eattn = cfg.n_head * cfg.head_dim;
        let mut gated = attn_heads.clone();
        for i in 0..eattn {
            gated[i] *= oracle::sigmoid(gate_raw[i]);
        }
        check("attn_gated", &gated);
        let wo = oracle::matvec(
            &w.f32(&format!("{p}attn.wo_ud.weight")).unwrap().1,
            &gated,
            cfg.n_embd,
            eattn,
        );
        check("wo_out", &wo);
        let wp = load_conv(&w, &format!("{p}attn_sconv.weight"), cfg.conv_kernel).unwrap();
        let (attn_res, _) = oracle::convolve_static(&wo, &wp, &[], 1, cfg.n_embd, cfg.conv_kernel);
        check("attn_conv", &attn_res);
        check("cur_attn", &attn_res); // visit 0: visit%6==0 -> cur = attn_res

        // MoE: FFN mix over blocks (block 0 = x pushed at visit 0), control n/a,
        // norm, router, top-8, SiTU experts, post-conv, residual.
        let cur = attn_res.clone();
        let ffn_q = w.f32("model.llm.attn_res.mlp_queries.0").unwrap().1;
        let ffn_mix = oracle::mix(&[x.clone()], &cur, &ffn_q, cfg.n_embd, 1);
        check("ffn_mix", &ffn_mix);
        let fnorm = rmsnorm_gain(
            &ffn_mix,
            &w.f32(&format!("{p}mlp_norm.weight")).unwrap().1,
            cfg.rms_norm_eps,
        );
        check("ffn_normed", &fnorm);
        let router = w.f32(&format!("{p}mlp.gate.weight")).unwrap().1;
        let rbias = w.f32(&format!("{p}mlp.gate.bias")).unwrap().1;
        let rlogits = oracle::matvec(&router, &fnorm, cfg.n_experts, cfg.n_embd);
        check("router_logits", &rlogits);
        let sel = oracle::sigmoid_topk(&rlogits, Some(&rbias), cfg.n_expert_used, cfg.route_scale);
        let sel_ids: Vec<u32> = sel.iter().map(|x| x.0 as u32).collect();
        let sel_weights: Vec<f32> = sel.iter().map(|x| x.1).collect();
        check(
            "sel_ids",
            &sel_ids.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        );
        check("sel_weights", &sel_weights);
        let moe = decode_moe(
            &fnorm,
            &router,
            Some(&rbias),
            &(0..cfg.n_experts)
                .map(|e| {
                    w.f32(&format!("{p}mlp.experts.{e}.gate_up_proj.weight"))
                        .unwrap()
                        .1
                })
                .collect::<Vec<_>>(),
            &(0..cfg.n_experts)
                .map(|e| {
                    w.f32(&format!("{p}mlp.experts.{e}.down_proj.weight"))
                        .unwrap()
                        .1
                })
                .collect::<Vec<_>>(),
            cfg.n_embd,
            cfg.inter,
            cfg.n_expert_used,
            cfg.route_scale,
            cfg.situ_beta,
            cfg.situ_linear_beta,
        );
        check("moe_raw", &moe);
        let wpost = load_conv(&w, &format!("{p}mlp_sconv.weight"), cfg.conv_kernel).unwrap();
        let (moe_conv, _) =
            oracle::convolve_static(&moe, &wpost, &[], 1, cfg.n_embd, cfg.conv_kernel);
        check("moe_conv", &moe_conv);
        let visit_out: Vec<f32> = cur.iter().zip(&moe_conv).map(|(a, b)| a + b).collect();
        check("visit_out", &visit_out);
    }

    #[test]
    #[ignore = "direct decode_visit(0) return vs reference visit_out"]
    fn decode_visit0_return_vs_reference() {
        use crate::weights::{GgufF32, WeightSource};
        let cfg = crate::config::Inkling2Config::from_metadata_json(crate::TEST_METADATA).unwrap();
        let w = GgufF32::open(std::path::Path::new(r"D:\hipfire-models\gguf-f32.bin")).unwrap();
        let emb = w.f32("model.llm.embed.weight").unwrap().1;
        let x = rmsnorm_gain(
            &emb[..cfg.n_embd],
            &w.f32("model.llm.embed_norm.weight").unwrap().1,
            cfg.rms_norm_eps,
        );
        let mut st = ConvState::default();
        let mut blocks = Vec::new();
        let mut dbg = std::collections::HashMap::new();
        let out = decode_visit(
            &w,
            &cfg,
            &mut st,
            &mut blocks,
            &x,
            0,
            0,
            None,
            Some(&mut dbg),
        )
        .unwrap();
        let cmp = |name: &str, got: &[f32]| {
            let p = format!(r"D:\hipfire-models\ref-visits\tok0\{name}.txt");
            if let Ok(s) = std::fs::read_to_string(&p) {
                let want: Vec<f32> = s
                    .lines()
                    .map(|l| f32::from_bits(l.trim().parse().unwrap()))
                    .collect();
                if want.len() == got.len() {
                    let m = want
                        .iter()
                        .zip(got)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    eprintln!("{name}: max|d|={m:.6}");
                }
            }
        };
        cmp("visit_out", &out);
        for k in [
            "x_in",
            "q_inp",
            "qc",
            "kc",
            "vc",
            "rc",
            "sinks",
            "attn_heads",
            "ffn_mix",
            "ffn_normed",
            "moe_raw",
            "moe_conv",
            "cur_attn",
        ] {
            if let Some(v) = dbg.get(k) {
                cmp(k, v);
            }
        }
    }

    #[test]
    #[ignore = "dump visit-5 (full layer) conv intermediates for parity"]
    fn visit5_conv_dump() {
        use crate::weights::{GgufF32, WeightSource};
        let cfg = crate::config::Inkling2Config::from_metadata_json(crate::TEST_METADATA).unwrap();
        let w = GgufF32::open(std::path::Path::new(r"D:\hipfire-models\gguf-f32.bin")).unwrap();
        let emb = w.f32("model.llm.embed.weight").unwrap().1;
        let x = rmsnorm_gain(
            &emb[..cfg.n_embd],
            &w.f32("model.llm.embed_norm.weight").unwrap().1,
            cfg.rms_norm_eps,
        );
        // Run visits 0..4 to build state, then capture debug on visit 5.
        let mut states: Vec<ConvState> = (0..48).map(|_| ConvState::default()).collect();
        let mut blocks = Vec::new();
        let mut cur = x;
        for v in 0..5 {
            cur = decode_visit(
                &w,
                &cfg,
                &mut states[v],
                &mut blocks,
                &cur,
                v,
                0,
                None,
                None,
            )
            .unwrap();
        }
        let mut dbg = std::collections::HashMap::new();
        let _ = decode_visit(
            &w,
            &cfg,
            &mut states[5],
            &mut blocks,
            &cur,
            5,
            0,
            None,
            Some(&mut dbg),
        )
        .unwrap();
        for k in [
            "x_in",
            "q_inp",
            "k_inp",
            "v_inp",
            "k_raw",
            "v_raw",
            "k_conv",
            "v_conv",
            "qc",
            "kc",
            "vc",
            "rc",
            "sinks",
            "rel_bias",
            "attn_heads",
            "wo_out",
            "cur_attn",
            "moe_raw",
            "visit_out",
        ] {
            if let Some(vals) = dbg.get(k) {
                let mut s = String::new();
                for v in vals {
                    s.push_str(&format!("{}\n", v.to_bits()));
                }
                std::fs::write(
                    format!(r"D:\hipfire-models\ref-visits\tok0\rust_visit5_{k}.txt"),
                    s,
                )
                .unwrap();
                eprintln!("dumped rust_visit5_{k}: {} vals", vals.len());
            }
        }
    }

    #[test]
    #[ignore = "dump visit-18 (first pass-1 visit: cross-pass messages applied)"]
    fn visit18_debug_dump() {
        use crate::weights::{GgufF32, WeightSource};
        let cfg = crate::config::Inkling2Config::from_metadata_json(crate::TEST_METADATA).unwrap();
        let w = GgufF32::open(std::path::Path::new(r"D:\hipfire-models\gguf-f32.bin")).unwrap();
        let emb = w.f32("model.llm.embed.weight").unwrap().1;
        let x = emb[..cfg.n_embd].to_vec();
        let mut cap = (18usize, std::collections::HashMap::new());
        let mut trace = Vec::new();
        let _ = forward_token(&w, &cfg, &x, 0, Some(&mut trace), Some(&mut cap)).unwrap();
        for k in [
            "x_in",
            "base_q",
            "q_ctrl",
            "q_inp",
            "k_inp",
            "v_inp",
            "fwd_msg0",
            "fwd_msg1",
            "fwd_msg2",
            "fwd_msg3",
            "fwd_prev_out",
            "fwd_bank_len",
            "k_raw",
            "v_raw",
            "k_conv",
            "v_conv",
            "qc",
            "kc",
            "vc",
            "rc",
            "sinks",
            "rel_bias",
            "attn_heads",
            "wo_out",
            "cur_attn",
            "ffn_mix",
            "ffn_normed",
            "moe_raw",
            "moe_conv",
            "visit_out",
        ] {
            if let Some(vals) = cap.1.get(k) {
                let mut s = String::new();
                for v in vals {
                    s.push_str(&format!("{}\n", v.to_bits()));
                }
                std::fs::write(
                    format!(r"D:\hipfire-models\ref-visits\tok0\rust_visit18_{k}.txt"),
                    s,
                )
                .unwrap();
                eprintln!("dumped rust_visit18_{k}: {} vals", vals.len());
            }
        }
    }

    #[test]
    #[ignore = "dump the layer-6/pass-1 cross-pass messages from the existing rust trace"]
    fn visit18_message_dump() {
        use crate::weights::{GgufF32, WeightSource};
        let cfg = crate::config::Inkling2Config::from_metadata_json(crate::TEST_METADATA).unwrap();
        let w = GgufF32::open(std::path::Path::new(r"D:\hipfire-models\gguf-f32.bin")).unwrap();
        let read = |v: usize| -> Vec<f32> {
            let s = std::fs::read_to_string(format!(
                r"D:\hipfire-models\rust-visits-tok0\visit{v:02}.txt"
            ))
            .unwrap();
            s.lines()
                .map(|l| f32::from_bits(l.trim().parse::<u32>().unwrap()))
                .collect()
        };
        // pass 0 output and the bank row pushed at visit 12 (output of visit 11)
        let prev_out = read(17);
        let bank = vec![read(11)];
        let mq = pack_message_queries(&w, &cfg).unwrap();
        for j in 0..4 {
            let msg = oracle::mix(&bank, &prev_out, &mq[0][j], cfg.n_embd, 1);
            let mut s = String::new();
            for v in &msg {
                s.push_str(&format!("{}\n", v.to_bits()));
            }
            std::fs::write(
                format!(r"D:\hipfire-models\ref-visits\tok0\rust_visit18_msg{j}.txt"),
                s,
            )
            .unwrap();
            eprintln!("dumped rust msg{j} head {:?}", &msg[..4]);
        }
    }

    #[test]
    #[ignore = "dump layer-6/pass-1 control intermediates from the existing rust trace"]
    fn visit18_control_dump() {
        use crate::weights::{GgufF32, WeightSource};
        let cfg = crate::config::Inkling2Config::from_metadata_json(crate::TEST_METADATA).unwrap();
        let w = GgufF32::open(std::path::Path::new(r"D:\hipfire-models\gguf-f32.bin")).unwrap();
        let read = |v: usize| -> Vec<f32> {
            let s = std::fs::read_to_string(format!(
                r"D:\hipfire-models\rust-visits-tok0\visit{v:02}.txt"
            ))
            .unwrap();
            s.lines()
                .map(|l| f32::from_bits(l.trim().parse::<u32>().unwrap()))
                .collect()
        };
        let prev_out = read(17);
        let bank = vec![read(11)];
        let mq = pack_message_queries(&w, &cfg).unwrap();
        let msg = oracle::mix(&bank, &prev_out, &mq[0][0], cfg.n_embd, 1);
        // base_q for visit 18 == base_q for visit 6: both start from the
        // prelude output with the prelude block list reset.
        let (_, aq) = w.f32("model.llm.attn_res.attn_queries.5").unwrap();
        let emb = w.f32("model.llm.embed.weight").unwrap().1;
        let tok_norm = w.f32("model.llm.embed_norm.weight").unwrap().1;
        let prelude_blocks = vec![rmsnorm_gain(
            &emb[..cfg.n_embd],
            &tok_norm,
            cfg.rms_norm_eps,
        )];
        let base = oracle::mix(&prelude_blocks, &read(5), &aq, cfg.n_embd, 1);
        let ctrls = load_controllers(&w, 6, &cfg).unwrap();
        let out = oracle::control(
            &base,
            &msg,
            &ctrls[0].down,
            &ctrls[0].gain,
            &ctrls[0].forget,
            cfg.n_embd,
            cfg.loop_cfg.gate_rank,
        );
        let write = |name: &str, v: &[f32]| {
            let mut s = String::new();
            for x in v {
                s.push_str(&format!("{}\n", x.to_bits()));
            }
            std::fs::write(
                format!(r"D:\hipfire-models\ref-visits\tok0\rust_visit18_{name}.txt"),
                s,
            )
            .unwrap();
        };
        write("base_q", &base);
        write("ctrl_out", &out);
        write("ctrl_down", &ctrls[0].down);
        write("ctrl_gain", &ctrls[0].gain);
        write("ctrl_forget", &ctrls[0].forget);
        eprintln!(
            "down {} gain {} forget {} base_head {:?} out_head {:?}",
            ctrls[0].down.len(),
            ctrls[0].gain.len(),
            ctrls[0].forget.len(),
            &base[..4],
            &out[..4]
        );
    }

    #[test]
    fn decode_moe_eager_vs_lazy_identical() {
        let n = 4;
        let inter = 3;
        let router = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let bias = vec![0.0; 4];
        let x = vec![0.5; n];
        let gu: Vec<Vec<f32>> = (0..4)
            .map(|e| vec![0.1 * (e + 1) as f32; 2 * inter * n])
            .collect();
        let dn: Vec<Vec<f32>> = (0..4)
            .map(|e| vec![0.05 * (e + 1) as f32; n * inter])
            .collect();
        let eager = decode_moe(
            &x,
            &router,
            Some(&bias),
            &gu,
            &dn,
            n,
            inter,
            2,
            8.0,
            4.0,
            25.0,
        );
        let lazy = decode_moe_lazy(
            &x,
            &router,
            Some(&bias),
            n,
            inter,
            2,
            8.0,
            4.0,
            25.0,
            &mut |e| (gu[e].clone(), dn[e].clone()),
        );
        assert_eq!(eager, lazy, "eager vs lazy MoE must be bit-identical");
    }

    fn oracle_close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }
}
