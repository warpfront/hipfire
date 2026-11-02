// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// PeaceMaker FP8 verify KLD: logits of 16-row verify windows with the FP8
// verify trunk and/or the FP8 verify lm_head vs the exact F16 baseline, on
// identical inputs and identical prior state.
//
// Per sequence: the prompt is prefilled on the F16 route (scope off); the
// continuation is processed in 16-token windows. Per window: snapshot the
// DeltaNet state, run the window with `gpu.fp8_verify_scope = true` (the arch
// slice makes forwards inside the scope use SingletonWmma math + the FP8
// verify GEMMs), then compute the window's logits from the captured post-norm
// hidden rows through the verify head: MQ rotate
// (`rotate_x_mq_batched_for`, AWQ-aware) -> `lmhead_verify_fp8` (FP8 head) and
// -> `lmhead_verify_exact` with the scope off (F16 head). The snapshot is then
// restored and the same window rerun on the F16 trunk (scope off, F16 head
// `gemm_mq4g256v2_lmhead_verify_exact`); the F16 run advances the state so
// every window starts from the F16 trajectory (KV slots of the window are
// rewritten by the F16 rerun).
//
// Logit sets per window (all over the same 16 rows):
//   f16       F16 trunk, F16 head              (reference)
//   fp8trunk  FP8 trunk, F16 head              (trunk contribution)
//   fp8head   F16 trunk, FP8 head              (head contribution, extra arm)
//   fp8       FP8 trunk, FP8 head              (full PM FP8 verify)
// Pairs reported (KL(first || second), top-1 agreement, max |dlogit|):
//   f16->fp8, f16->fp8trunk, f16->fp8head, fp8trunk->fp8 (head given FP8 trunk).
//
// The scope is set directly on the public field (no `HIPFIRE_CB_VERIFY_FP8`
// needed; `Gpu::fp8_verify_enter` is not used), restored on every path.
//
// Input JSON: [{"prompt": str, "text": str}, ...] (prompt content and a model
// continuation, tokenized raw). Output: <out>/kld_rows.csv (long format: one
// line per row and pair), <out>/summary.txt.
//
// Build: cargo build -p hipfire-arch-qwen35 --features lab,deltanet --release \
//          --example pm_fp8_verify_kld -j12
// Run:   pm_fp8_verify_kld --out <abs dir> [--model P] [--seqs P]
//          [--windows-per-seq 8] [--max-seqs N]

use hipfire_arch_qwen35::load_qwen35_bundle;
use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::speculative::DeltaNetSnapshot;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv_backend::KvBackend;
use hipfire_runtime::llama;
use hipfire_runtime::loader_api::{CaskConfig, LoadCtx, ModelSource, SequenceHint, SpecLoadCfg};
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::fmt::Write as _;

type R<T> = Result<T, Box<dyn std::error::Error>>;

const W: usize = 16;
const DEFAULT_MODEL: &str =
    "/home/kaden/qcal/release-0.4.0/fp8-tp-ep-geo/models/qwen3.8-27b.mq4-xts";
const DEFAULT_SEQS: &str = "/home/kaden/qcal/release-0.4.2/cb/fp8verify/kld_seqs.json";

#[derive(serde::Deserialize)]
struct Seq {
    prompt: String,
    text: String,
}

struct RowStat {
    kld: f64,
    top1_same: bool,
    max_abs: f64,
}

/// KL(P_ref || P_cand) over one logit row, plus argmax agreement and max |Δlogit|.
fn row_stats(r: &[f32], c: &[f32]) -> RowStat {
    let lse = |v: &[f32]| {
        let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        m + v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
    };
    let (lr, lc) = (lse(r), lse(c));
    let mut kld = 0f64;
    let mut max_abs = 0f64;
    for (&a, &b) in r.iter().zip(c) {
        let la = a as f64 - lr;
        let lb = b as f64 - lc;
        kld += la.exp() * (la - lb);
        max_abs = max_abs.max((a as f64 - b as f64).abs());
    }
    let am = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |acc, (i, &x)| {
                if x > acc.1 {
                    (i, x)
                } else {
                    acc
                }
            })
            .0
    };
    RowStat {
        kld: kld.max(0.0),
        top1_same: am(r) == am(c),
        max_abs,
    }
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p).round() as usize]
}

/// Logit set indices.
const F16: usize = 0;
const FP8_TRUNK: usize = 1;
const FP8_HEAD: usize = 2;
const FP8: usize = 3;
const SETS: [&str; 4] = ["f16", "fp8trunk", "fp8head", "fp8"];
/// (reference set, candidate set, label).
const PAIRS: [(usize, usize, &str); 4] = [
    (F16, FP8, "f16->fp8"),
    (F16, FP8_TRUNK, "f16->fp8trunk"),
    (F16, FP8_HEAD, "f16->fp8head"),
    (FP8_TRUNK, FP8, "fp8trunk->fp8"),
];

#[derive(Default)]
struct PairAcc {
    klds: Vec<f64>,
    maxabs: Vec<f64>,
    win_mean: Vec<f64>,
    top1_same: usize,
}

/// Verify head over the captured post-norm hidden rows: MQ rotate, then the
/// FP8 verify lm_head (`fp8`) or the F16 exact verify lm_head with the scope
/// off. Returns `rows * m` logits (row stride = `w_out.m`).
fn head_logits(
    gpu: &mut Gpu,
    w_out: &llama::WeightTensor,
    hidden: &GpuTensor,
    rot: &GpuTensor,
    logits: &GpuTensor,
    rows: usize,
    fp8: bool,
) -> R<Vec<f32>> {
    let (k, m) = (w_out.k, w_out.m);
    llama::rotate_x_mq_batched_for(gpu, w_out, hidden, rot, k, rows)?;
    let prev = gpu.fp8_verify_scope;
    gpu.fp8_verify_scope = false;
    let r = if fp8 {
        gpu.gemm_mq4g256v2_lmhead_verify_fp8(&w_out.buf, rot, logits, m, k, rows)
    } else {
        gpu.gemm_mq4g256v2_lmhead_verify_exact(&w_out.buf, rot, logits, m, k, rows)
    };
    gpu.fp8_verify_scope = prev;
    r?;
    gpu.hip.device_synchronize()?;
    let mut raw = vec![0u8; rows * m * 4];
    gpu.hip.memcpy_dtoh_at(&mut raw, &logits.buf, 0)?;
    Ok(raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("pm_fp8_verify_kld: {e}");
        std::process::exit(2);
    }
}

fn run() -> R<()> {
    let mut model = DEFAULT_MODEL.to_string();
    let mut out = String::new();
    let mut seqs_path = DEFAULT_SEQS.to_string();
    let mut wps = 8usize;
    let mut max_seqs = usize::MAX;
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let v = it.next().ok_or_else(|| format!("{k}: missing value"))?;
        match k.as_str() {
            "--model" => model = v,
            "--out" => out = v,
            "--seqs" => seqs_path = v,
            "--windows-per-seq" => wps = v.parse()?,
            "--max-seqs" => max_seqs = v.parse()?,
            _ => return Err(format!("unknown argument {k}").into()),
        }
    }
    if !out.starts_with('/') {
        return Err("absolute --out <dir> is required".into());
    }
    std::fs::create_dir_all(&out)?;
    let seqs: Vec<Seq> = serde_json::from_str(&std::fs::read_to_string(&seqs_path)?)?;
    let hfq = HfqFile::open(std::path::Path::new(&model))?;
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer: {e:?}"))?;
    drop(hfq);

    let mut gpu = Gpu::init()?;
    if gpu.arch != "gfx1201" {
        return Err(format!("gfx1201 required, got {}", gpu.arch).into());
    }
    let src = ModelSource::from_path(&model)?;
    let cask = CaskConfig::default();
    let mut lc = LoadCtx {
        path: &model,
        max_seq: 4096,
        sequence: Some(SequenceHint {
            model_ctx: 4096,
            automatic: true,
            card_cap: 0,
            reserve_bytes: 0,
        }),
        deepseek4_compute_placement: Default::default(),
        deepseek4_experts_per_token: None,
        draft_path: None,
        vision_path: None,
        mtp_path: None,
        vision_mode: "off".into(),
        kv_mode_override: None,
        kv_k_override: None,
        kv_v_override: None,
        qwen_default_q8: true,
        kv_backend: KvBackend::default(),
        kv_adaptive_override: None,
        state_quant_override: None,
        cask: &cask,
        pp: 1,
        pp_layer_counts: None,
        admitted_mesh: None,
        band_placement: None,
        spec: SpecLoadCfg::default(),
        gpu: &mut gpu,
        gemma4_drafter_path: None,
        gemma4_draft_len: 3,
        xdna: None,
        batch_request: Default::default(),
    };
    let mut b = load_qwen35_bundle(src, &mut lc).map_err(|e| e.to_string())?;
    if !gpu.mq4v2_symmetric || b.weights.output.gpu_dtype != DType::MQ4G256V2 {
        return Err("model needs symmetric MQ4G256V2 weights incl. the lm_head (FP8 verify head requirement)".into());
    }
    let (dim, vocab_cfg) = (b.config.dim, b.config.vocab_size);
    let m_out = b.weights.output.m;
    let vocab = vocab_cfg.min(m_out);
    let hidden = gpu.zeros(&[W * dim], DType::F32)?;
    let rot = gpu.zeros(&[W * dim], DType::F32)?;
    let logits = gpu.zeros(&[W * m_out], DType::F32)?;
    let mut snap = DeltaNetSnapshot::new_for(&mut gpu, &b.dn_state)?;

    let mut csv = String::from("seq,window,row,pos,pair,kld,top1_same,max_abs_dlogit\n");
    let mut acc: Vec<PairAcc> = PAIRS.iter().map(|_| PairAcc::default()).collect();
    let (mut rows_total, mut windows) = (0usize, 0usize);
    for (si, s) in seqs.iter().take(max_seqs).enumerate() {
        let p = tok.encode(&s.prompt);
        let c = tok.encode(&s.text);
        let nw = (c.len() / W).min(wps);
        if p.is_empty() || nw == 0 {
            continue;
        }
        // Fresh sequence: reset KV/state by re-prefilling from position 0 (F16 route).
        gpu.fp8_verify_scope = false;
        b.dn_state.reset(&mut gpu)?;
        qwen35::forward_prefill_batch(
            &mut gpu,
            &b.weights,
            &b.config,
            &p,
            0,
            &mut b.kv_cache,
            &mut b.dn_state,
            &b.scratch,
            None,
            None,
            None,
            None,
        )?;
        let mut pos = p.len();
        for w in 0..nw {
            let toks = &c[w * W..(w + 1) * W];
            snap.save_from(&b.dn_state, &mut gpu)?;
            // FP8 trunk (scope on), then both heads on its hidden rows.
            gpu.fp8_verify_scope = true;
            let r = qwen35::forward_prefill_batch_capture_hidden(
                &mut gpu,
                &b.weights,
                &b.config,
                toks,
                pos,
                &mut b.kv_cache,
                &mut b.dn_state,
                &b.scratch,
                &hidden,
            );
            gpu.fp8_verify_scope = false;
            r?;
            let l_fp8 = head_logits(&mut gpu, &b.weights.output, &hidden, &rot, &logits, W, true)?;
            let l_fp8trunk = head_logits(
                &mut gpu,
                &b.weights.output,
                &hidden,
                &rot,
                &logits,
                W,
                false,
            )?;
            // F16 trunk (restored state, scope off), F16 and FP8 heads.
            snap.restore_to(&mut b.dn_state, &mut gpu)?;
            qwen35::forward_prefill_batch_capture_hidden(
                &mut gpu,
                &b.weights,
                &b.config,
                toks,
                pos,
                &mut b.kv_cache,
                &mut b.dn_state,
                &b.scratch,
                &hidden,
            )?;
            let l_f16 = head_logits(
                &mut gpu,
                &b.weights.output,
                &hidden,
                &rot,
                &logits,
                W,
                false,
            )?;
            let l_fp8head =
                head_logits(&mut gpu, &b.weights.output, &hidden, &rot, &logits, W, true)?;
            let sets: [&Vec<f32>; 4] = [&l_f16, &l_fp8trunk, &l_fp8head, &l_fp8];
            for (pi, &(ra, ca, label)) in PAIRS.iter().enumerate() {
                let mut wsum = 0f64;
                for row in 0..W {
                    let rr = &sets[ra][row * m_out..row * m_out + vocab];
                    let cc = &sets[ca][row * m_out..row * m_out + vocab];
                    let st = row_stats(rr, cc);
                    let _ = writeln!(
                        csv,
                        "{si},{w},{row},{},{label},{:.6e},{},{:.4}",
                        pos + row,
                        st.kld,
                        st.top1_same as u8,
                        st.max_abs
                    );
                    wsum += st.kld;
                    acc[pi].klds.push(st.kld);
                    acc[pi].maxabs.push(st.max_abs);
                    acc[pi].top1_same += st.top1_same as usize;
                }
                acc[pi].win_mean.push(wsum / W as f64);
            }
            rows_total += W;
            windows += 1;
            pos += W;
        }
        let k0 = &acc[0].klds;
        eprintln!(
            "seq {si}: prompt {} tok, {nw} windows; running mean KLD f16->fp8 {:.4e}",
            p.len(),
            k0.iter().sum::<f64>() / k0.len() as f64
        );
    }
    if rows_total == 0 {
        return Err("no windows evaluated".into());
    }
    let mut s = String::new();
    let _ = writeln!(s, "model {model}");
    let _ = writeln!(s, "arch {} sets {SETS:?}", gpu.arch);
    let _ = writeln!(s, "windows {windows} rows {rows_total} (W={W}, <= {wps} windows/seq, scored over {vocab} logits)");
    for (pi, &(_, _, label)) in PAIRS.iter().enumerate() {
        let a = &mut acc[pi];
        let mean = a.klds.iter().sum::<f64>() / rows_total as f64;
        let _ = writeln!(
            s,
            "{label} KLD per row: mean {mean:.4e} median {:.4e} p90 {:.4e} p99 {:.4e} max {:.4e}",
            pct(&mut a.klds.clone(), 0.5),
            pct(&mut a.klds.clone(), 0.9),
            pct(&mut a.klds.clone(), 0.99),
            pct(&mut a.klds, 1.0)
        );
        let _ = writeln!(
            s,
            "{label} per-window mean KLD: median {:.4e} p99 {:.4e} max {:.4e}",
            pct(&mut a.win_mean.clone(), 0.5),
            pct(&mut a.win_mean.clone(), 0.99),
            pct(&mut a.win_mean, 1.0)
        );
        let _ = writeln!(
            s,
            "{label} top-1 agreement {}/{rows_total} = {:.4}%",
            a.top1_same,
            100.0 * a.top1_same as f64 / rows_total as f64
        );
        let _ = writeln!(
            s,
            "{label} max |dlogit|: median {:.4} p99 {:.4} max {:.4}",
            pct(&mut a.maxabs.clone(), 0.5),
            pct(&mut a.maxabs.clone(), 0.99),
            pct(&mut a.maxabs, 1.0)
        );
    }
    print!("{s}");
    std::fs::write(format!("{out}/kld_rows.csv"), csv)?;
    std::fs::write(format!("{out}/summary.txt"), s)?;
    Ok(())
}
