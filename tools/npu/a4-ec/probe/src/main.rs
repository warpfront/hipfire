// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
// A4-EC capture probe: runs real batched prefill of dense Qwen3.8-27B (mq4-xts)
// on gfx1201 and captures, for selected layers, the pre-quant rotated f32 GEMM
// inputs ("xrot") and the device block_i4_128 A4 bytes for the four input
// families (attn, o, mlp, down). See local://a4ec-contract.md.
//
// usage: a4-ec-probe MODEL PROMPT OUTDIR [--layers 2,21,...] [--t 4096]
//                    [--model-sha256 HEX] [--card E] [--card-uuid UUID]
//                    [--commit HASH]
use hip_bridge::DeviceBuffer;
use hipfire_arch_qwen35::qwen35::{
    self, DeltaNetState, DflashFusionCtx, LayerType, LayerWeights, PrefillBatchScratch,
    Qwen35Config, Qwen35Scratch, Qwen35Weights, StateQuant,
};
use hipfire_runtime::{hfq::HfqFile, llama::KvCache, tokenizer::Tokenizer};
use rdna_compute::gemv::SigmoidGate;
use rdna_compute::norm::GdnScanOut;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::{error::Error, fs, path::Path, time::Instant};

type Res<T> = Result<T, Box<dyn Error>>;

const DEFAULT_LAYERS: [usize; 8] = [2, 11, 21, 31, 42, 51, 61, 63];

struct Cap {
    xrot: Vec<f32>,
    a4: Vec<u8>,
}

fn dl_bytes(gpu: &Gpu, t: &GpuTensor, bytes: usize) -> Res<Vec<u8>> {
    assert!(bytes <= t.buf.size());
    let mut v = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh(&mut v, &t.buf)?;
    Ok(v)
}

fn dl_f32(gpu: &Gpu, t: &GpuTensor, n: usize) -> Res<Vec<f32>> {
    assert!(n * 4 <= t.buf.size());
    let mut v = vec![0f32; n];
    let bytes = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, n * 4) };
    gpu.hip.memcpy_dtoh(bytes, &t.buf)?;
    Ok(v)
}

fn dl_raw_ptr(gpu: &Gpu, ptr: *mut std::ffi::c_void, bytes: usize) -> Res<Vec<u8>> {
    let buf = unsafe { DeviceBuffer::from_raw(ptr, bytes) };
    let mut v = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh(&mut v, &buf)?;
    Ok(v)
}

fn all_zero(gpu: &Gpu, t: &GpuTensor, n: usize) -> Res<bool> {
    let v = dl_f32(gpu, t, n)?;
    Ok(v.iter().all(|x| *x == 0.0))
}

fn md5_hex(bytes: &[u8]) -> String {
    format!("{:x}", md5::compute(bytes))
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

/// Self-check of (xrot, a4): per 128-block, r = x' - d*q.
fn selfcheck(xrot: &[f32], a4: &[u8], t: usize, k: usize) -> Value {
    assert_eq!(xrot.len(), t * k);
    assert_eq!(a4.len(), (k / 128) * t * 72);
    let finite = xrot.iter().filter(|v| v.is_finite()).count();
    let mut sum_abs = 0f64;
    for v in xrot {
        if v.is_finite() {
            sum_abs += v.abs() as f64;
        }
    }
    let mean_abs = sum_abs / finite.max(1) as f64;
    let mut max_ratio_unsat = 0f64; // q not in {-8, 7}
    let mut max_ratio_all = 0f64;
    let mut max_ratio_sat = 0f64;
    let mut n_sat = 0u64;
    let mut n_elem = 0u64;
    let mut s_mismatch = 0u64;
    let mut d_bad = 0u64;
    let mut d_zero_blocks = 0u64;
    let mut ratio_sum = 0f64;
    let mut ratio_n = 0u64;
    let mut d_sum = 0f64;
    for e in 0..k / 128 {
        for tok in 0..t {
            let blk = (e * t + tok) * 72;
            let d = f32::from_le_bytes(a4[blk..blk + 4].try_into().unwrap());
            let s = i32::from_le_bytes(a4[blk + 4..blk + 8].try_into().unwrap());
            if !d.is_finite() || d < 0.0 {
                d_bad += 1;
                continue;
            }
            if d == 0.0 {
                d_zero_blocks += 1;
            }
            d_sum += d as f64;
            let qs = &a4[blk + 8..blk + 72];
            let xb = &xrot[tok * k + 128 * e..tok * k + 128 * e + 128];
            let mut ssum = 0i32;
            for j in 0..64 {
                let b = qs[j];
                let lo = (((b << 4) as i8) >> 4) as i32;
                let hi = ((b as i8) >> 4) as i32;
                ssum += lo + hi;
                for (q, x) in [(lo, xb[2 * j]), (hi, xb[2 * j + 1])] {
                    n_elem += 1;
                    if d == 0.0 || !x.is_finite() {
                        continue;
                    }
                    let r = ((x - d * q as f32).abs() / d) as f64;
                    if r > max_ratio_all {
                        max_ratio_all = r;
                    }
                    if q == -8 || q == 7 {
                        n_sat += 1;
                        if r > max_ratio_sat {
                            max_ratio_sat = r;
                        }
                    } else {
                        if r > max_ratio_unsat {
                            max_ratio_unsat = r;
                        }
                        ratio_sum += r;
                        ratio_n += 1;
                    }
                }
            }
            if ssum != s {
                s_mismatch += 1;
            }
        }
    }
    json!({
        "xrot_elems": xrot.len(),
        "xrot_finite": finite,
        "mean_abs_xrot": mean_abs,
        "mean_d": d_sum / ((k / 128) * t) as f64,
        "max_over_tokens_abs_resid_over_d_unsaturated": max_ratio_unsat,
        "max_abs_resid_over_d_all": max_ratio_all,
        "max_abs_resid_over_d_saturated": max_ratio_sat,
        "mean_abs_resid_over_d_unsaturated": ratio_sum / ratio_n.max(1) as f64,
        "saturated_elems": n_sat,
        "total_elems": n_elem,
        "s_field_mismatch_blocks": s_mismatch,
        "bad_d_blocks": d_bad,
        "zero_d_blocks": d_zero_blocks,
    })
}

struct Ctx {
    gpu: Gpu,
    weights: Qwen35Weights,
    config: Qwen35Config,
    kv: KvCache,
    scratch: Qwen35Scratch,
    pbs: PrefillBatchScratch,
    tokens: Vec<u32>,
    t: usize,
    xrot_a: GpuTensor,
    xrot_b: GpuTensor,
}

impl Ctx {
    fn run(&mut self, max_layer: usize) -> Res<()> {
        let mut dn = DeltaNetState::new_with_quant(&mut self.gpu, &self.config, StateQuant::Q8)?;
        let r = qwen35::forward_prefill_batch_with_pbs_opts(
            &mut self.gpu,
            &self.weights,
            &self.config,
            &self.tokens,
            0,
            &mut self.kv,
            &mut dn,
            &self.scratch,
            None,
            None,
            None,
            None,
            Some(&self.pbs),
            None,
            Some(max_layer),
            false,
            DflashFusionCtx::Off,
        );
        self.gpu.hip.device_synchronize()?;
        let _ = dn.free_gpu(&mut self.gpu);
        r?;
        Ok(())
    }
}

/// Run producer closure `f(gpu, reservation, xrot_buf)` with a fresh int4
/// reservation, then download (a4 bytes, xrot f32). `which` picks xrot_a/xrot_b.
fn grab<F>(
    gpu: &mut Gpu,
    xa: &GpuTensor,
    xb: &GpuTensor,
    n: usize,
    k: usize,
    which: u8,
    f: F,
) -> Res<Cap>
where
    F: FnOnce(&mut Gpu, rdna_compute::Int4MmqReservation, &GpuTensor) -> Res<()>,
{
    let res = gpu.reserve_int4_mmq(k, n)?;
    let ptr = res.ptr();
    let xr = if which == 0 { xa } else { xb };
    // zero xrot so a producer that skips the f32 store is detected
    gpu.hip.memset(&xr.buf, 0, xr.buf.size())?;
    f(gpu, res, xr)?;
    gpu.hip.device_synchronize()?;
    let a4 = dl_raw_ptr(gpu, ptr, (k / 128) * n * 72)?;
    let xrot = dl_f32(gpu, xr, n * k)?;
    Ok(Cap { xrot, a4 })
}

macro_rules! grab {
    ($cx:ident, $k:expr, $w:expr, $f:expr) => {
        grab(&mut $cx.gpu, &$cx.xrot_a, &$cx.xrot_b, $cx.t, $k, $w, $f)
    };
}

struct LayerInfo {
    is_fa: bool,
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        return Err("usage: a4-ec-probe MODEL PROMPT OUTDIR [--layers a,b,c] [--t N] [--model-sha256 H] [--card X] [--card-uuid U] [--commit H]".into());
    }
    let model_path = args[1].clone();
    let prompt_path = args[2].clone();
    let outdir = args[3].clone();
    let mut layers: Vec<usize> = DEFAULT_LAYERS.to_vec();
    let mut t: usize = 4096;
    let mut model_sha = String::new();
    let mut card = String::new();
    let mut card_uuid = String::new();
    let mut commit = String::new();
    let mut i = 4;
    while i < args.len() {
        match args[i].as_str() {
            "--layers" => {
                layers = args[i + 1].split(',').map(|s| s.parse().unwrap()).collect();
                i += 2;
            }
            "--t" => {
                t = args[i + 1].parse()?;
                i += 2;
            }
            "--model-sha256" => {
                model_sha = args[i + 1].clone();
                i += 2;
            }
            "--card" => {
                card = args[i + 1].clone();
                i += 2;
            }
            "--card-uuid" => {
                card_uuid = args[i + 1].clone();
                i += 2;
            }
            "--commit" => {
                commit = args[i + 1].clone();
                i += 2;
            }
            o => return Err(format!("unknown arg {o}").into()),
        }
    }
    let t0 = Instant::now();
    fs::create_dir_all(&outdir)?;
    let out = Path::new(&outdir);

    let mut hfq = HfqFile::open(Path::new(&model_path))?;
    let config = qwen35::config_from_hfq(&hfq)?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)?;
    let prompt = fs::read(&prompt_path)?;
    let prompt_md5 = md5_hex(&prompt);
    let all_tokens = tokenizer.encode(&String::from_utf8(prompt)?);
    eprintln!("prompt tokens: {}", all_tokens.len());
    if all_tokens.len() < t {
        return Err(format!("prompt has only {} tokens < T={t}", all_tokens.len()).into());
    }
    let tokens: Vec<u32> = all_tokens[..t].to_vec();
    let tok_bytes: Vec<u8> = tokens.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tokens_md5 = md5_hex(&tok_bytes);
    fs::write(out.join("tokens.u32"), &tok_bytes)?;

    let mut gpu = Gpu::init()?;
    let gcn_arch = gpu.hip.get_arch(gpu.device_id)?;
    eprintln!("gpu arch={} gcnArchName={gcn_arch}", gpu.arch);
    let mut source = qwen35::HfqSource::new(&mut hfq, &config);
    let weights = qwen35::load_weights(
        &mut source,
        std::slice::from_mut(&mut gpu),
        &qwen35::Layout::single(config.n_layers),
    )?;
    drop(source);
    eprintln!("weights loaded: {:.1}s", t0.elapsed().as_secs_f64());

    let mask: Vec<bool> = config
        .layer_types
        .iter()
        .map(|ty| *ty == LayerType::FullAttention)
        .collect();
    let cap = t + 256;
    let kv = KvCache::new_gpu_fp8_vmm_capped_filtered(
        &mut gpu,
        &mask,
        config.n_kv_heads,
        config.head_dim,
        cap,
        cap,
    )?;
    eprintln!("kv quant_fp8={}", kv.quant_fp8);
    let scratch = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 64, cap)?;
    let pbs = PrefillBatchScratch::new_opt(&mut gpu, &config, t, false)?;
    let max_k = config.hidden_dim.max(config.dim).max(6144);
    let xrot_a = gpu.alloc_tensor(&[t * max_k], DType::F32)?;
    let xrot_b = gpu.alloc_tensor(&[t * max_k], DType::F32)?;

    gpu.ensure_mq_signs()?;
    let signs1 = gpu.download_f32(gpu.scratch.mq_signs1.as_ref().unwrap())?;
    let signs2 = gpu.download_f32(gpu.scratch.mq_signs2.as_ref().unwrap())?;
    fs::write(out.join("signs1.f32"), f32_bytes(&signs1))?;
    fs::write(out.join("signs2.f32"), f32_bytes(&signs2))?;

    let mut cx = Ctx {
        gpu,
        weights,
        config,
        kv,
        scratch,
        pbs,
        tokens,
        t,
        xrot_a,
        xrot_b,
    };

    let dim = cx.config.dim;
    let hidden = cx.config.hidden_dim;
    let eps = cx.config.norm_eps;
    let n_v_heads = cx.config.linear_num_value_heads;
    let vhd = cx.config.linear_value_head_dim;
    let q_dim = cx.config.n_heads * cx.config.head_dim;
    let mut files: Vec<Value> = vec![];
    let mut tag_entries: Vec<Value> = vec![];
    let mut layer_types_out: Vec<Value> = vec![];

    let write_cap =
        |files: &mut Vec<Value>, l: usize, tag: &str, k: usize, cap: &Cap| -> Res<Value> {
            let xp = format!("L{l}.{tag}.xrot.f32");
            let ap = format!("L{l}.{tag}.a4.bin");
            let xb = f32_bytes(&cap.xrot);
            fs::write(out.join(&xp), xb)?;
            fs::write(out.join(&ap), &cap.a4)?;
            let (xm, am) = (md5_hex(xb), md5_hex(&cap.a4));
            files.push(json!({"file": xp, "bytes": xb.len(), "md5": xm}));
            files.push(json!({"file": ap, "bytes": cap.a4.len(), "md5": am}));
            let sc = selfcheck(&cap.xrot, &cap.a4, t, k);
            eprintln!("selfcheck L{l} {tag} K={k}: {sc}");
            Ok(json!({"xrot_md5": xm, "a4_md5": am, "selfcheck": sc}))
        };

    for &l in &layers {
        let lt0 = Instant::now();
        let is_fa = cx.config.layer_types[l] == LayerType::FullAttention;
        layer_types_out.push(
            json!({"layer": l, "type": if is_fa {"full_attention"} else {"linear_attention"}}),
        );
        let pre = format!("model.language_model.layers.{l}");

        // ---- Run A: layers 0..L -> pbs.x_batch = layer L input residual. ----
        cx.run(l)?;
        let x_in_finite = {
            let v = dl_f32(&cx.gpu, &cx.pbs.x_batch, t * dim)?;
            v.iter().filter(|x| x.is_finite()).count()
        };
        // attn tag
        let (attn_cap, attn_awq_present) = {
            let (norm, awq) = match &cx.weights.layers[l] {
                LayerWeights::DeltaNet(lw) => (&lw.attn_norm, lw.wqkv.awq_scale.as_ref()),
                LayerWeights::FullAttn(lw) => (&lw.attn_norm, lw.wq.awq_scale.as_ref()),
                _ => return Err("MoE layer unexpected".into()),
            };
            let xb = &cx.pbs.x_batch;
            let c = grab!(cx, dim, 0, |gpu, res, xr| {
                gpu.fused_rmsnorm_rotate_mq_i4_gfx12_batched(
                    xb,
                    norm,
                    awq,
                    Some(xr),
                    res,
                    dim,
                    eps,
                    t,
                )?;
                Ok(())
            })?;
            (c, awq.is_some())
        };
        let attn_meta = write_cap(&mut files, l, "attn", dim, &attn_cap)?;
        drop(attn_cap);

        // ---- Run B: layers 0..=L with w_down(L) zeroed -> x_batch = x_mid. ----
        let (wd_ptr, wd_size) = match &cx.weights.layers[l] {
            LayerWeights::DeltaNet(lw) => (lw.w_down.buf.buf.as_ptr(), lw.w_down.buf.buf.size()),
            LayerWeights::FullAttn(lw) => (lw.w_down.buf.buf.as_ptr(), lw.w_down.buf.buf.size()),
            _ => unreachable!(),
        };
        let wd_buf = unsafe { DeviceBuffer::from_raw(wd_ptr, wd_size) };
        let mut wd_backup = vec![0u8; wd_size];
        cx.gpu.hip.memcpy_dtoh(&mut wd_backup, &wd_buf)?;
        // sentinels: detect F1-lite h route (up_batch never written) and FA gate-in-place
        cx.gpu.hip.memset(&cx.pbs.up_batch.buf, 0, t * hidden * 4)?;
        cx.gpu
            .hip
            .memset(&cx.pbs.fa_gate_batch.buf, 0, t * q_dim * 4)?;
        cx.gpu.hip.memset(&wd_buf, 0, wd_size)?;
        cx.gpu.hip.device_synchronize()?;
        let run_b = cx.run(l + 1);
        // restore w_down regardless of run result
        cx.gpu.hip.memcpy_htod(&wd_buf, &wd_backup)?;
        cx.gpu.hip.device_synchronize()?;
        run_b?;
        let x_mid_finite = {
            let v = dl_f32(&cx.gpu, &cx.pbs.x_batch, t * dim)?;
            v.iter().filter(|x| x.is_finite()).count()
        };
        let up_untouched = all_zero(&cx.gpu, &cx.pbs.up_batch, t * hidden)?;
        let gate_untouched = if is_fa {
            all_zero(&cx.gpu, &cx.pbs.fa_gate_batch, t * q_dim)?
        } else {
            false
        };
        eprintln!(
            "L{l} runB: x_in_finite={x_in_finite}/{} x_mid_finite={x_mid_finite}/{} up_untouched(h-route)={up_untouched} fa_gate_in_place={gate_untouched}",
            t * dim,
            t * dim
        );

        // o tag
        let (o_k, o_cap, o_prod, o_awq_present) = match &cx.weights.layers[l] {
            LayerWeights::DeltaNet(lw) => {
                let k = lw.wo.k;
                let awq = lw.wo.awq_scale.as_ref();
                let pbs = &cx.pbs;
                let c = grab!(cx, k, 0, |gpu, res, xr| {
                    gpu.gated_norm_rotate_mq_i4_gfx12_batched(
                        &pbs.dn_attn_out_batch,
                        GdnScanOut::F32,
                        &pbs.dn_z_batch,
                        &lw.norm_weight,
                        awq,
                        Some(xr),
                        res,
                        n_v_heads,
                        vhd,
                        eps,
                        k,
                        t,
                    )?;
                    Ok(())
                })?;
                (
                    k,
                    c,
                    "gated_norm_rotate_mq_i4_gfx12_batched(x_fmt=F32)",
                    awq.is_some(),
                )
            }
            LayerWeights::FullAttn(lw) => {
                let k = lw.wo.k;
                let awq = lw.wo.awq_scale.as_ref().ok_or("FA wo has no AWQ scale")?;
                let pbs = &cx.pbs;
                let gate = if gate_untouched {
                    SigmoidGate::QGateInterleaved(&pbs.fa_q_full_batch)
                } else {
                    SigmoidGate::Rows(&pbs.fa_gate_batch)
                };
                let c = grab!(cx, k, 0, |gpu, res, xr| {
                    gpu.sigmoid_mul_rotate_x_mq_awq_i4_gfx12_batched(
                        &pbs.fa_attn_out_batch,
                        gate,
                        awq,
                        Some(xr),
                        res,
                        k,
                        t,
                    )?;
                    Ok(())
                })?;
                let name = if gate_untouched {
                    "sigmoid_mul_rotate_x_mq_awq_i4_gfx12_batched(SigmoidGate::QGateInterleaved)"
                } else {
                    "sigmoid_mul_rotate_x_mq_awq_i4_gfx12_batched(SigmoidGate::Rows)"
                };
                (k, c, name, true)
            }
            _ => unreachable!(),
        };
        let o_meta = write_cap(&mut files, l, "o", o_k, &o_cap)?;
        drop(o_cap);

        // mlp tag
        let (mlp_cap, mlp_awq_present) = {
            let (norm, awq) = match &cx.weights.layers[l] {
                LayerWeights::DeltaNet(lw) => (&lw.ffn_norm, lw.w_gate.awq_scale.as_ref()),
                LayerWeights::FullAttn(lw) => (&lw.ffn_norm, lw.w_gate.awq_scale.as_ref()),
                _ => unreachable!(),
            };
            let xb = &cx.pbs.x_batch;
            let c = grab!(cx, dim, 0, |gpu, res, xr| {
                gpu.fused_rmsnorm_rotate_mq_i4_gfx12_batched(
                    xb,
                    norm,
                    awq,
                    Some(xr),
                    res,
                    dim,
                    eps,
                    t,
                )?;
                Ok(())
            })?;
            (c, awq.is_some())
        };
        let mlp_meta = write_cap(&mut files, l, "mlp", dim, &mlp_cap)?;
        drop(mlp_cap);

        // down tag
        let (down_cap, down_prod_x, down_prod_a, down_extra, down_awq_present) = {
            let w_down = match &cx.weights.layers[l] {
                LayerWeights::DeltaNet(lw) => &lw.w_down,
                LayerWeights::FullAttn(lw) => &lw.w_down,
                _ => unreachable!(),
            };
            let pbs = &cx.pbs;
            let awq = w_down.awq_scale.as_ref();
            let k = w_down.k;
            assert_eq!(k, hidden);
            if up_untouched {
                // F1-lite route: gate_ffn_batch holds h = silu(gate)*up (f32).
                let h = &pbs.gate_ffn_batch;
                // a4 bytes: the production producer (silu_hin) — no f32 store.
                let a4_only = grab!(cx, k, 1, |gpu, res, _xr| {
                    gpu.fused_silu_hin_rotate_mq_i4_batched(
                        h,
                        awq.ok_or("w_down has no AWQ")?,
                        res,
                        k,
                        t,
                    )?;
                    Ok(())
                })?;
                // xrot: rotate_x producer on h (AWQ, FWHT), emits f32 + a4.
                let xc = grab!(cx, k, 0, |gpu, res, xr| {
                    gpu.rotate_x_mq_i4_gfx12_batched(h, awq, Some(xr), res, k, t)?;
                    Ok(())
                })?;
                let same = a4_only.a4 == xc.a4;
                let mism = if same {
                    0
                } else {
                    a4_only
                        .a4
                        .iter()
                        .zip(xc.a4.iter())
                        .filter(|(a, b)| a != b)
                        .count()
                };
                eprintln!("L{l} down: silu_hin a4 vs rotate_x(h) a4 identical={same} (differing bytes={mism})");
                (
                    Cap { xrot: xc.xrot, a4: a4_only.a4 },
                    "rotate_x_mq_i4_gfx12_batched(x=h=gate_ffn_batch; emits f32)",
                    "fused_silu_hin_rotate_mq_i4_batched(h=gate_ffn_batch) [production F1-lite route]",
                    json!({"route": "f1lite_h", "a4_equal_rotate_x_of_h": same, "a4_differing_bytes": mism}),
                    awq.is_some(),
                )
            } else {
                let c = grab!(cx, k, 0, |gpu, res, xr| {
                    gpu.fused_silu_mul_rotate_mq_i4_gfx12_batched(
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        awq,
                        Some(xr),
                        res,
                        k,
                        t,
                    )?;
                    Ok(())
                })?;
                (
                    c,
                    "fused_silu_mul_rotate_mq_i4_gfx12_batched (same call emits f32 + a4)",
                    "fused_silu_mul_rotate_mq_i4_gfx12_batched",
                    json!({"route": "separate_gate_up"}),
                    awq.is_some(),
                )
            }
        };
        let down_meta = write_cap(&mut files, l, "down", hidden, &down_cap)?;
        drop(down_cap);

        let (attn_consumers, attn_awq_name, o_consumers, o_awq_name): (
            Vec<String>,
            String,
            Vec<String>,
            String,
        ) = if is_fa {
            (
                ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                    .iter()
                    .map(|s| format!("{pre}.{s}.weight"))
                    .collect(),
                format!("{pre}.self_attn.q_proj.awq_scale.weight"),
                vec![format!("{pre}.self_attn.o_proj.weight")],
                format!("{pre}.self_attn.o_proj.awq_scale.weight"),
            )
        } else {
            (
                [
                    "linear_attn.in_proj_qkv",
                    "linear_attn.in_proj_z",
                    "linear_attn.in_proj_b",
                    "linear_attn.in_proj_a",
                ]
                .iter()
                .map(|s| format!("{pre}.{s}.weight"))
                .collect(),
                format!("{pre}.linear_attn.in_proj_qkv.awq_scale.weight"),
                vec![format!("{pre}.linear_attn.out_proj.weight")],
                format!("{pre}.linear_attn.out_proj.awq_scale.weight"),
            )
        };
        let attn_prod = "fused_rmsnorm_rotate_mq_i4_gfx12_batched (norm=input_layernorm; one call emits f32 xrot + a4; production producer: try_gfx12_rmsnorm_quant_fused_prepared)";
        let mlp_prod = "fused_rmsnorm_rotate_mq_i4_gfx12_batched (norm=post_attention_layernorm, x=x_mid; one call emits f32 xrot + a4; production producer: try_gfx12_rmsnorm_quant_fused_prepared)";
        tag_entries.push(json!({
            "layer": l,
            "type": if is_fa {"full_attention"} else {"linear_attention"},
            "x_in_finite": x_in_finite,
            "x_mid_finite": x_mid_finite,
            "route_flags": {"f1lite_h_route": up_untouched, "fa_gate_in_place": gate_untouched},
            "tags": {
                "attn": {
                    "K": dim,
                    "consumers": attn_consumers,
                    "awq_tensor": attn_awq_name,
                    "awq_present_on_device": attn_awq_present,
                    "producer_xrot": attn_prod,
                    "producer_a4": attn_prod,
                    "producer_a4_is_production_route": true,
                    "producer_xrot_is_production_route": true,
                    "files": attn_meta,
                },
                "o": {
                    "K": o_k,
                    "consumers": o_consumers,
                    "awq_tensor": o_awq_name,
                    "awq_present_on_device": o_awq_present,
                    "producer_xrot": o_prod,
                    "producer_a4": o_prod,
                    "producer_a4_is_production_route": true,
                    "producer_xrot_is_production_route": true,
                    "input_buffers": if is_fa { "fa_attn_out_batch (f32 attention output) x sigmoid(gate)" } else { "dn_attn_out_batch (f32 sequential GDN output) + dn_z_batch" },
                    "files": o_meta,
                },
                "mlp": {
                    "K": dim,
                    "consumers": [format!("{pre}.mlp.gate_proj.weight"), format!("{pre}.mlp.up_proj.weight")],
                    "awq_tensor": format!("{pre}.mlp.gate_proj.awq_scale.weight"),
                    "awq_present_on_device": mlp_awq_present,
                    "producer_xrot": mlp_prod,
                    "producer_a4": mlp_prod,
                    "producer_a4_is_production_route": true,
                    "producer_xrot_is_production_route": true,
                    "files": mlp_meta,
                },
                "down": {
                    "K": hidden,
                    "consumers": [format!("{pre}.mlp.down_proj.weight")],
                    "awq_tensor": format!("{pre}.mlp.down_proj.awq_scale.weight"),
                    "awq_present_on_device": down_awq_present,
                    "producer_xrot": down_prod_x,
                    "producer_a4": down_prod_a,
                    "producer_a4_is_production_route": true,
                    "producer_xrot_is_production_route": !up_untouched,
                    "route_detail": down_extra,
                    "files": down_meta,
                },
            },
            "wall_s": lt0.elapsed().as_secs_f64(),
        }));
        eprintln!(
            "layer {l} done in {:.1}s (total {:.1}s)",
            lt0.elapsed().as_secs_f64(),
            t0.elapsed().as_secs_f64()
        );
    }

    // sign / token files
    for f in ["signs1.f32", "signs2.f32", "tokens.u32"] {
        let b = fs::read(out.join(f))?;
        files.push(json!({"file": f, "bytes": b.len(), "md5": md5_hex(&b)}));
    }

    let model_bytes = fs::metadata(&model_path)?.len();
    let mut env: Vec<(String, String)> = std::env::vars()
        .filter(|(k, _)| {
            k.starts_with("HIPFIRE_")
                || k.starts_with("ROCR_")
                || k.starts_with("HIP_")
                || k.starts_with("HSA_")
                || k == "CARGO_TARGET_DIR"
        })
        .collect();
    env.sort();
    let manifest = json!({
        "model": {"path": model_path, "bytes": model_bytes, "sha256": model_sha},
        "prompt": {"path": prompt_path, "md5": prompt_md5, "tokens_total": all_tokens.len()},
        "T": t,
        "tokens_md5": tokens_md5,
        "n_layers": cx.config.n_layers,
        "layer_types_captured": layer_types_out,
        "gcnArchName": gcn_arch,
        "gpu_arch": cx.gpu.arch,
        "card": {"letter": card, "uuid": card_uuid},
        "hipfire_worktree_commit": commit,
        "env": env.into_iter().map(|(k, v)| json!({k: v})).collect::<Vec<_>>(),
        "signs_len": {"signs1": signs1.len(), "signs2": signs2.len()},
        "dims": {"dim": dim, "hidden_dim": hidden, "norm_eps": eps},
        "kv": "native fp8 (KvCache::new_gpu_fp8_vmm_capped_filtered), as production on exact gfx1201",
        "hipcc_extra_flags_runtime": cx.gpu.flags.hipcc_extra_flags,
        "no_device_compiler": "HIPFIRE_NO_DEVICE_COMPILER NOT set: with it set, the wt-land-041m runtime rejects the installed ~/.hipfire_kernels/gfx1201 plain-named packs (embedding_q8_batched: expected key mismatch) so nothing runs; unset, the runtime loads its toolchain-keyed hot-cache entries (no compile was triggered in the capture run)",
        "dn_state": "Q8",
        "layers": tag_entries,
        "files": files,
        "notes": [
            "Prefill driver: forward_prefill_batch_with_pbs_opts(max_layer=Some(..)), one chunk of T rows (HIPFIRE_PREFILL_MAX_BATCH>=T, pbs.max_batch=T). max_layer disables the GDN chunk-scan path (prefill.rs:13001), so GDN layers use the sequential gated_delta_net_q8_batch_seq (f32 plane) instead of the production chunk-scan/bf16 plane; values differ at rounding level only.",
            "attn tag: Run A (max_layer=L) leaves layer L's input residual in pbs.x_batch; the probe then calls the production rmsnorm+AWQ+FWHT+A4 producer on it with the next linear's (wqkv/wq) AWQ.",
            "o/mlp/down tags: Run B (max_layer=L+1) with w_down(L) device bytes temporarily zeroed (restored afterwards), so the final down residual-GEMM adds exactly 0 and pbs.x_batch == x_mid = x_in + o_proj(...) exactly as the production fused rmsnorm producer sees it (on gfx1201 the o_proj GEMM ADD-epilogue writes x_batch in place; the residual-fold route is gfx1151-only).",
            "o buffers are layer L's own: GDN dn_attn_out_batch/dn_z_batch and FA fa_attn_out_batch/fa_q_full_batch(gate)/fa_gate_batch are written only by layer L's attention half and not touched by layer L's FFN (which uses x_rot_batch/gate_ffn_batch/up_batch/ffn_hidden_batch).",
            "down: F1-lite h route is detected by an up_batch sentinel (memset 0 before Run B; still all-zero => gate/up epilogue wrote h into gate_ffn_batch). route_flags/route_detail record which route ran.",
            "HIPFIRE_A4_SLAB=0 so producers use the non-slab token-order block layout (block index e*T+t). Production uses the slab twins on gfx1201; values identical, layout differs.",
            "xrot and a4 for each tag come from the SAME producer call on the same f32 source buffers (x_rot Some(..) emit_f32), except down/F1-lite where a4 comes from silu_hin and xrot from rotate_x(h); equality of the two a4 outputs is recorded in route_detail.",
            "Self-check per file: max_over_tokens_abs_resid_over_d_unsaturated = max |x'-d*q|/d over elements with q not in {-8,7}; saturated (clipped) elements reported separately."
        ],
    });
    fs::write(
        out.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    eprintln!(
        "manifest written; total wall {:.1}s",
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}
