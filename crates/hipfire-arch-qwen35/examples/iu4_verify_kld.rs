// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// LAB diagnostic (IU4-VERIFY-CHECK §KLD): KLD of 16-row verify-window logits, IU4 (A4 slab producer + _b1s)
// trunk GEMMs (`Gpu::iu4_verify_scope`) vs the exact F16 trunk, on identical
// inputs and identical prior state.
//
// Per sequence: the prompt is prefilled on the F16 route; the continuation is
// then processed in 16-token windows. Per window: snapshot the DeltaNet
// state, run the window with the IU4 scope on (per-row post-norm hidden ->
// lm_head GEMV per row, F16 head on both arms), restore the snapshot, rerun
// the window on F16 (KV slots of the window are rewritten), and compare the
// two logit sets per row. The F16 run is the one that advances the state, so
// every window's comparison starts from the F16 trajectory.
//
// Input JSON: [{"prompt": str, "text": str}, ...] (prompt content and a model
// continuation, tokenized raw). Output: <out>/kld_rows.csv, <out>/summary.txt.
//
// Build: cargo build -p hipfire-arch-qwen35 --features lab,deltanet --release \
//          --example iu4_verify_kld -j12

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

fn main() {
    if let Err(e) = run() {
        eprintln!("iu4_verify_kld: {e}");
        std::process::exit(2);
    }
}

fn run() -> R<()> {
    let mut model = String::new();
    let mut out = String::new();
    let mut seqs_path = String::new();
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
    if model.is_empty() || seqs_path.is_empty() || !out.starts_with('/') {
        return Err("--model, --seqs and absolute --out are required".into());
    }
    std::fs::create_dir_all(&out)?;
    let seqs: Vec<Seq> = serde_json::from_str(&std::fs::read_to_string(&seqs_path)?)?;
    let hfq = HfqFile::open(std::path::Path::new(&model))?;
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer: {e:?}"))?;
    drop(hfq);

    let mut gpu = Gpu::init()?;
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
    let (dim, vocab) = (b.config.dim, b.config.vocab_size);
    let hidden = gpu.zeros(&[W * dim], DType::F32)?;
    let logits = gpu.zeros(&[W * vocab], DType::F32)?;
    let mut snap = DeltaNetSnapshot::new_for(&mut gpu, &b.dn_state)?;
    let head_rows = |gpu: &mut Gpu, w_out: &llama::WeightTensor, rows: usize| -> R<Vec<f32>> {
        for r in 0..rows {
            let h: GpuTensor = hidden.sub_offset(r * dim, dim);
            let l: GpuTensor = logits.sub_offset(r * vocab, vocab);
            llama::weight_gemv(gpu, w_out, &h, &l)?;
        }
        gpu.hip.device_synchronize()?;
        let mut raw = vec![0u8; rows * vocab * 4];
        gpu.hip.memcpy_dtoh_at(&mut raw, &logits.buf, 0)?;
        Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    };

    let mut csv = String::from("seq,window,row,pos,kld,top1_same,max_abs_dlogit\n");
    let mut klds = Vec::new();
    let mut maxabs = Vec::new();
    let (mut rows_total, mut top1_same, mut windows) = (0usize, 0usize, 0usize);
    let mut win_kld_mean = Vec::new();
    for (si, s) in seqs.iter().take(max_seqs).enumerate() {
        let p = tok.encode(&s.prompt);
        let c = tok.encode(&s.text);
        let nw = (c.len() / W).min(wps);
        if p.is_empty() || nw == 0 {
            continue;
        }
        // Fresh sequence: reset KV/state by re-prefilling from position 0.
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
            let prev = gpu.iu4_verify_enter(true);
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
            gpu.iu4_verify_exit(prev);
            r?;
            let lf = head_rows(&mut gpu, &b.weights.output, W)?;
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
            let lr = head_rows(&mut gpu, &b.weights.output, W)?;
            let mut wsum = 0f64;
            for row in 0..W {
                let st = row_stats(
                    &lr[row * vocab..(row + 1) * vocab],
                    &lf[row * vocab..(row + 1) * vocab],
                );
                let _ = writeln!(
                    csv,
                    "{si},{w},{row},{},{:.6e},{},{:.4}",
                    pos + row,
                    st.kld,
                    st.top1_same as u8,
                    st.max_abs
                );
                wsum += st.kld;
                klds.push(st.kld);
                maxabs.push(st.max_abs);
                rows_total += 1;
                top1_same += st.top1_same as usize;
            }
            win_kld_mean.push(wsum / W as f64);
            windows += 1;
            pos += W;
        }
        eprintln!(
            "seq {si}: prompt {} tok, {nw} windows; running mean KLD {:.4e}",
            p.len(),
            klds.iter().sum::<f64>() / klds.len() as f64
        );
    }
    if rows_total == 0 {
        return Err("no windows evaluated".into());
    }
    let mean = klds.iter().sum::<f64>() / rows_total as f64;
    let mut s = String::new();
    let _ = writeln!(s, "model {model}");
    let _ = writeln!(
        s,
        "windows {windows} rows {rows_total} (W={W}, <= {wps} windows/seq)"
    );
    let _ = writeln!(
        s,
        "KLD(F16 || IU4) per row: mean {mean:.4e} median {:.4e} p90 {:.4e} p99 {:.4e} max {:.4e}",
        pct(&mut klds.clone(), 0.5),
        pct(&mut klds.clone(), 0.9),
        pct(&mut klds.clone(), 0.99),
        pct(&mut klds, 1.0)
    );
    let _ = writeln!(
        s,
        "per-window mean KLD: median {:.4e} p99 {:.4e} max {:.4e}",
        pct(&mut win_kld_mean.clone(), 0.5),
        pct(&mut win_kld_mean.clone(), 0.99),
        pct(&mut win_kld_mean, 1.0)
    );
    let _ = writeln!(
        s,
        "top-1 agreement {}/{} = {:.4}%",
        top1_same,
        rows_total,
        100.0 * top1_same as f64 / rows_total as f64
    );
    let _ = writeln!(
        s,
        "max |dlogit|: median {:.4} p99 {:.4} max {:.4}",
        pct(&mut maxabs.clone(), 0.5),
        pct(&mut maxabs.clone(), 0.99),
        pct(&mut maxabs, 1.0)
    );
    print!("{s}");
    std::fs::write(format!("{out}/kld_rows.csv"), csv)?;
    std::fs::write(format!("{out}/summary.txt"), s)?;
    Ok(())
}
