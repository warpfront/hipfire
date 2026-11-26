// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! inkling_kld — paired KLD evaluator for Inkling-2-Mini (arch 24).
//!
//! Teacher: the published source checkpoint (`thinkingmachines/Inkling-2-Mini`,
//! NVFP4/MXFP8 routed experts, BF16/F32 everything else), dequantized to F32
//! by the quantizer's strict decoders (`hipfire-quantize/src/dequant.rs`,
//! the same values the shipped artifacts were quantized from) and run through
//! a token-batched CPU forward that reorganizes `forward::forward_token_cached`
//! visit-major without changing any per-element arithmetic: every product sum
//! runs in the same order from the same `-0.0` start, and every nonlinear step
//! calls the same `oracle` function. `fidelity` measures that claim against
//! the unmodified per-token reference.
//!
//! Candidate: the product loader (`hipfire_loader::load_model`, the Inkling
//! carrier) and the product AR route (`reset_session_state`, then one
//! `forward_token_gpu` per position from 0, as `generate_inkling` prefills).
//!
//! Formats: HFKLDR v3 reference (saddle-quant, top-k 256 plus residual, the
//! legacy back-half window), HFKSEQ v2 per-chunk `(mean, p99, mean NLL)` with
//! eval_hipfire's per-chunk statistics, HFKTOK v1 per-token KLDs.
//!
//! Usage:
//!   inkling_kld ref      --source MODEL.safetensors --model ARTIFACT.hfq --slice TEXT
//!                        --output REF.kldref [--n-ctx 2048] [--max-chunks 24] [--top-k 256]
//!                        [--batch-chunks N]
//!   inkling_kld fidelity --source MODEL.safetensors --model ARTIFACT.hfq --slice TEXT
//!                        --output OUT.json [--n-ctx 64] [--chunks 4]
//!                        [--sliding-window W] [--log-floor F]
//!   inkling_kld eval     --model CANDIDATE.hfq --ref REF.kldref --output OUT.kldseq
//!                        [--max-chunks N] [--chunks-per-process 1]
//!   inkling_kld pair     --base BASE.kldseq --cand CAND.kldseq [--output OUT.json]
//!                        [--boot 20000] [--seed 1]
//!
//! `ref` tokenizes the slice with the artifact's own tokenizer (the product
//! tokenizer; the artifact also supplies the config), cuts plain `n_ctx`
//! windows (no BOS insertion, like eval_hipfire), runs the teacher over
//! positions `[0, n_ctx-1)` and keeps the scored rows `[n_ctx/2, n_ctx-1)`.
//! `--batch-chunks` bounds host RAM (about 0.3 MiB per batched position plus
//! ~6.4 GiB of dense F32 weights and one layer's decoded experts).
//!
//! `fidelity` runs the same chunks through `forward_token_cached` +
//! `lm_head_logits` (the per-token CPU reference, on the same source weights)
//! and through the batched teacher, and reports bit identity of every final
//! hidden state and logit row. `--sliding-window`/`--log-floor` shrink the SWA
//! window and the log-scaling floor in both forwards, so a short run exercises
//! SWA eviction and full-attention tau != 1.
//!
//! `eval` scores each chunk in a fresh process (`--chunks-per-process`
//! chunks per process; the arch-24 GPU forward does not return its per-token
//! controller scratch, so one 2048-token chunk is what a single R9700 process
//! holds today). It writes `OUT.kldseq`, `OUT.kldseq.hfktok` (per-token KLD)
//! and `OUT.kldseq.json` (receipt: model/ref/binary SHA-256, route, per-chunk
//! scores and logits digests). Run with `HIPFIRE_GRAPH=0
//! HIPFIRE_NORMALIZE_PROMPT=0` and a fresh HOME, as the Qwen pins do.
//!
//! `pair` pairs chunk i of BASE with chunk i of CAND (both receipts must name
//! the same reference) and reports per-chunk deltas (cand − base; negative is
//! better), the paired-t 95% CI and a SplitMix64 percentile-bootstrap 95% CI,
//! for mean KLD, p99 KLD and NLL. Same JSON schema as the gfx1151 A4 receipts'
//! `paired.py`.

#[allow(dead_code)]
#[path = "../../hipfire-quantize/src/dequant.rs"]
mod dequant;

use hipfire_arch_inkling2::config::Inkling2Config;
use hipfire_arch_inkling2::forward::{self, ControllerWeights, CpuDecodeState};
use hipfire_arch_inkling2::oracle;
use hipfire_arch_inkling2::weights::WeightSource;
use hipfire_quantize::float16::{bf16_to_f32, f16_to_f32};
use hipfire_quantize::safetensors_file::SafetensorsFile;
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::eval_common::{seq_stats, write_hfkseq, write_hfktok};
use hipfire_runtime::hfq::HfqFile;
use rayon::prelude::*;
use saddle_quant::eval::estimator::{kld_from_block, nll_of, topk_block, TopKBlock};
use saddle_quant::eval::teacher::{evaluate, TeacherGate};
use saddle_quant::format::kldref::{KldRef, RefHeader, RefWriter};
use saddle_quant::format::kldseq;
use saddle_quant::{ArtifactId, Estimator, OracleStats, TeacherVerdict, WindowSpec};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

const USAGE: &str = "usage:
  inkling_kld ref      --source MODEL.safetensors --model ARTIFACT.hfq --slice TEXT --output REF.kldref
                       [--n-ctx 2048] [--max-chunks 24] [--top-k 256] [--batch-chunks N]
  inkling_kld fidelity --source MODEL.safetensors --model ARTIFACT.hfq --slice TEXT --output OUT.json
                       [--n-ctx 64] [--chunks 4] [--sliding-window W] [--log-floor F]
  inkling_kld eval     --model CANDIDATE.hfq --ref REF.kldref --output OUT.kldseq
                       [--max-chunks N] [--chunks-per-process 1]
  inkling_kld pair     --base BASE.kldseq --cand CAND.kldseq [--output OUT.json] [--boot 20000] [--seed 1]";

/// Receipt schema written beside every `.kldseq`.
const RECEIPT_SCHEMA: &str = "hipfire.inkling2.kldseq_receipt.v1";

// ---------------------------------------------------------------------------
// Source-checkpoint weights (the teacher's F32 values)
// ---------------------------------------------------------------------------

/// The pinned source safetensors: every non-expert `model.llm.*` tensor
/// converted once to F32 (BF16/F16 widen exactly), routed experts decoded on
/// demand per `(layer, expert)` by the quantizer's strict NVFP4/MXFP8 decoders
/// with the quantizer's w13 `[g0,u0,g1,u1,..] -> [g..,u..]` regroup. Names are
/// the checkpoint names, which are the HFQ names for everything but experts.
struct SourceWeights {
    st: SafetensorsFile,
    dense: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

fn to_f32(dtype: &str, bytes: &[u8]) -> Result<Vec<f32>, String> {
    match dtype {
        "F32" => Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        "BF16" => Ok(bytes
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        "F16" => Ok(bytes
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        other => Err(format!("unsupported dense dtype {other}")),
    }
}

impl SourceWeights {
    fn open(path: &Path) -> Result<Self, String> {
        let st =
            SafetensorsFile::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let names: Vec<String> = st
            .tensors
            .keys()
            .filter(|n| n.starts_with("model.llm.") && !n.contains(".mlp.experts."))
            .cloned()
            .collect();
        let dense = names
            .par_iter()
            .map(|n| {
                let (meta, bytes) = st.tensor_data(n).expect("indexed tensor");
                let v = to_f32(&meta.dtype, bytes).map_err(|e| format!("{n}: {e}"))?;
                if v.len() != meta.shape.iter().product::<usize>() {
                    return Err(format!("{n}: element count mismatch"));
                }
                Ok((n.clone(), (meta.shape.clone(), v)))
            })
            .collect::<Result<HashMap<_, _>, String>>()?;
        Ok(Self { st, dense })
    }

    fn dense(&self, name: &str) -> Result<&[f32], String> {
        self.dense
            .get(name)
            .map(|(_, v)| v.as_slice())
            .ok_or_else(|| format!("source: tensor not found: {name}"))
    }

    fn dense_shaped(&self, name: &str) -> Result<(&[usize], &[f32]), String> {
        self.dense
            .get(name)
            .map(|(s, v)| (s.as_slice(), v.as_slice()))
            .ok_or_else(|| format!("source: tensor not found: {name}"))
    }

    /// One expert of a stacked `w13_weight`/`w2_weight`: `(rows, cols, F32)`.
    fn decode_stacked(&self, stem: &str, e: usize) -> Result<(usize, usize, Vec<f32>), String> {
        let (wm, wb) = self
            .st
            .tensor_data(stem)
            .ok_or_else(|| format!("source: missing {stem}"))?;
        let scale = format!("{stem}_scale");
        let (sm, sb) = self
            .st
            .tensor_data(&scale)
            .ok_or_else(|| format!("source: missing {scale}"))?;
        if wm.shape.len() != 3 || sm.shape.len() != 3 || e >= wm.shape[0] {
            return Err(format!("{stem}: bad shape {:?} for expert {e}", wm.shape));
        }
        let rows = wm.shape[1];
        let amax = self.st.tensor_data(&format!("{stem}_amax"));
        let denom = self
            .st
            .tensor_data(&format!("{stem}_outer_scale_denominator"));
        match (amax, denom) {
            (Some((_, ab)), Some((_, db))) => {
                let cp = wm.shape[2];
                let blocks = sm.shape[2];
                let v = dequant::dequantize_nvfp4_amax_denominator(
                    &wb[e * rows * cp..(e + 1) * rows * cp],
                    &[1, rows, cp],
                    &sb[e * rows * blocks..(e + 1) * rows * blocks],
                    &[1, rows, blocks],
                    &ab[e * 4..e * 4 + 4],
                    db,
                )
                .map_err(|m| format!("{stem}[{e}] NVFP4: {m}"))?;
                Ok((rows, cp * 2, v))
            }
            (None, None) => {
                let cols = wm.shape[2];
                let sc = sm.shape[2];
                let v = dequant::dequantize_mxfp8_e8m0_strict(
                    &wb[e * rows * cols..(e + 1) * rows * cols],
                    &[rows, cols],
                    &sb[e * rows * sc..(e + 1) * rows * sc],
                    &[rows, sc],
                    32,
                )
                .map_err(|m| format!("{stem}[{e}] MXFP8: {m}"))?;
                Ok((rows, cols, v))
            }
            _ => Err(format!("{stem}: incomplete NVFP4 companion set")),
        }
    }

    /// `(gate_up [2*inter, n_embd] grouped, down [n_embd, inter])` of one expert.
    fn expert(&self, phys: usize, e: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
        Ok((self.gate_up(phys, e)?.2, self.down(phys, e)?.2))
    }

    fn gate_up(&self, phys: usize, e: usize) -> Result<(usize, usize, Vec<f32>), String> {
        let (rows, cols, f) = self.decode_stacked(
            &format!("model.llm.layers.{phys}.mlp.experts.w13_weight"),
            e,
        )?;
        // Converter stores w13 rows as [gate0, up0, gate1, up1, ...]; the
        // artifact (and the forward) use [gate..., up...] (pipeline.rs).
        let inter = rows / 2;
        let mut grouped = vec![0.0f32; rows * cols];
        for j in 0..inter {
            grouped[j * cols..(j + 1) * cols]
                .copy_from_slice(&f[(2 * j) * cols..(2 * j + 1) * cols]);
            grouped[(inter + j) * cols..(inter + j + 1) * cols]
                .copy_from_slice(&f[(2 * j + 1) * cols..(2 * j + 2) * cols]);
        }
        Ok((rows, cols, grouped))
    }

    fn down(&self, phys: usize, e: usize) -> Result<(usize, usize, Vec<f32>), String> {
        self.decode_stacked(&format!("model.llm.layers.{phys}.mlp.experts.w2_weight"), e)
    }
}

/// Parse `model.llm.layers.{phys}.mlp.experts.{e}.{gate_up|down}_proj.weight`.
fn parse_expert_name(name: &str) -> Option<(usize, usize, bool)> {
    let rest = name.strip_prefix("model.llm.layers.")?;
    let (phys, rest) = rest.split_once(".mlp.experts.")?;
    let (e, leaf) = rest.split_once('.')?;
    let gu = match leaf {
        "gate_up_proj.weight" => true,
        "down_proj.weight" => false,
        _ => return None,
    };
    Some((phys.parse().ok()?, e.parse().ok()?, gu))
}

/// The per-token reference reads weights through this (fidelity only).
impl WeightSource for SourceWeights {
    fn f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        if let Some((phys, e, gu)) = parse_expert_name(name) {
            let (rows, cols, v) = if gu {
                self.gate_up(phys, e)?
            } else {
                self.down(phys, e)?
            };
            return Ok((vec![rows, cols], v));
        }
        self.dense
            .get(name)
            .cloned()
            .ok_or_else(|| format!("source: tensor not found: {name}"))
    }
}

// ---------------------------------------------------------------------------
// Order-preserving batched GEMM
// ---------------------------------------------------------------------------

/// Tokens per micro tile (one 512-bit F32 vector).
const TN: usize = 16;
/// Weight rows per micro tile.
const TM: usize = 8;
/// Tokens per super block (its transposed tile set stays cache resident).
const SB: usize = 512;

#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

/// `R` weight rows against one 16-token transposed tile. Each lane is
/// `acc = acc + w[k] * x[k]` over `k` in order from `-0.0`: exactly the fold
/// `oracle::matvec`'s `Sum` performs (Rust never contracts to FMA), so every
/// output is bit-identical to the per-row reference whatever the ISA.
#[inline(always)]
unsafe fn micro_body<const R: usize>(w: *const f32, k: usize, xt: *const f32) -> [[f32; TN]; R] {
    let mut acc = [[-0.0f32; TN]; R];
    for kk in 0..k {
        let xv = xt.add(kk * TN);
        for (r, a) in acc.iter_mut().enumerate() {
            let wv = *w.add(r * k + kk);
            for (l, al) in a.iter_mut().enumerate() {
                *al = *al + wv * *xv.add(l);
            }
        }
    }
    acc
}

#[target_feature(enable = "avx512f")]
unsafe fn micro_avx512<const R: usize>(w: *const f32, k: usize, xt: *const f32) -> [[f32; TN]; R] {
    micro_body::<R>(w, k, xt)
}

#[target_feature(enable = "avx2")]
unsafe fn micro_avx2<const R: usize>(w: *const f32, k: usize, xt: *const f32) -> [[f32; TN]; R] {
    micro_body::<R>(w, k, xt)
}

fn micro<const R: usize>(w: &[f32], k: usize, xt: &[f32]) -> [[f32; TN]; R] {
    assert!(w.len() >= R * k && xt.len() >= k * TN);
    // SAFETY: bounds asserted above; feature presence checked at runtime.
    unsafe {
        if std::arch::is_x86_feature_detected!("avx512f") {
            micro_avx512::<R>(w.as_ptr(), k, xt.as_ptr())
        } else if std::arch::is_x86_feature_detected!("avx2") {
            micro_avx2::<R>(w.as_ptr(), k, xt.as_ptr())
        } else {
            micro_body::<R>(w.as_ptr(), k, xt.as_ptr())
        }
    }
}

/// `out[n][m] = Σ_k w[m][k] * x[n][k]`, each output summed exactly like
/// `oracle::matvec(w, x_n, m, k)[m]`.
fn gemm(x: &[f32], n: usize, k: usize, w: &[f32], m: usize) -> Vec<f32> {
    assert_eq!(x.len(), n * k, "gemm x shape");
    assert_eq!(w.len(), m * k, "gemm w shape");
    let mut out = vec![0.0f32; n * m];
    if n == 0 || m == 0 {
        return out;
    }
    let threads = rayon::current_num_threads().max(1);
    // Row blocks: enough tasks to fill the pool, a multiple of TM.
    let rb = (m.div_ceil(threads * 2)).div_ceil(TM).clamp(1, 8) * TM;
    let optr = SendPtr(out.as_mut_ptr());
    for sb0 in (0..n).step_by(SB) {
        let sbn = (n - sb0).min(SB);
        let tiles = sbn.div_ceil(TN);
        let mut xt = vec![0.0f32; tiles * k * TN];
        xt.par_chunks_mut(k * TN).enumerate().for_each(|(ti, dst)| {
            for l in 0..TN {
                let tok = ti * TN + l;
                if tok >= sbn {
                    break;
                }
                let src = &x[(sb0 + tok) * k..(sb0 + tok + 1) * k];
                for (kk, &v) in src.iter().enumerate() {
                    dst[kk * TN + l] = v;
                }
            }
        });
        (0..m.div_ceil(rb)).into_par_iter().for_each(|bi| {
            let optr = optr;
            let r0 = bi * rb;
            let rows = rb.min(m - r0);
            for ti in 0..tiles {
                let xtile = &xt[ti * k * TN..(ti + 1) * k * TN];
                let live = TN.min(sbn - ti * TN);
                let mut r = 0;
                while r + TM <= rows {
                    let acc = micro::<TM>(&w[(r0 + r) * k..(r0 + r + TM) * k], k, xtile);
                    for l in 0..live {
                        let base = (sb0 + ti * TN + l) * m + r0 + r;
                        for (rr, a) in acc.iter().enumerate() {
                            // SAFETY: each task owns rows [r0, r0+rows) of every token row.
                            unsafe { *optr.0.add(base + rr) = a[l] };
                        }
                    }
                    r += TM;
                }
                while r < rows {
                    let acc = micro::<1>(&w[(r0 + r) * k..(r0 + r + 1) * k], k, xtile);
                    for l in 0..live {
                        // SAFETY: as above.
                        unsafe { *optr.0.add((sb0 + ti * TN + l) * m + r0 + r) = acc[0][l] };
                    }
                    r += 1;
                }
            }
        });
    }
    out
}

fn flat(rows: &[Vec<f32>]) -> Vec<f32> {
    let mut v = Vec::with_capacity(rows.iter().map(Vec::len).sum());
    for r in rows {
        v.extend_from_slice(r);
    }
    v
}

// ---------------------------------------------------------------------------
// Private helpers of the reference forward, copied verbatim (forward.rs).
// `fidelity` proves the batched forward that uses them bit-identical.
// ---------------------------------------------------------------------------

fn rmsnorm_gain(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = rmsnorm_l2(x, eps);
    n.iter().zip(weight).map(|(v, g)| v * g).collect()
}

fn rmsnorm_l2(x: &[f32], eps: f32) -> Vec<f32> {
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (mean_sq + eps).sqrt();
    x.iter().map(|v| v * s).collect()
}

fn load_conv(shape: &[usize], v: &[f32], name: &str, kernel: usize) -> Vec<f32> {
    let n_ch = v.len() / kernel;
    let mut out = vec![0.0f32; v.len()];
    if shape.len() == 3 {
        for c in 0..n_ch {
            for k in 0..kernel {
                out[k * n_ch + c] = v[c * kernel + k];
            }
        }
    } else if name.ends_with(".static_w") {
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
    out
}

fn torch_conv(shape: &[usize], v: &[f32], kk: usize) -> Vec<f32> {
    let ch = v.len() / kk;
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
}

#[inline]
fn rel_bias_for_dist(
    table: &[f32],
    rh: &[f32],
    n_rel: usize,
    n_extent: usize,
    dist: usize,
    sink_h: f32,
) -> f32 {
    let dot = if dist < n_extent {
        (0..n_rel)
            .map(|rr| table[rr * n_extent + dist] * rh[rr])
            .sum()
    } else {
        0.0
    };
    dot - sink_h
}

fn controllers(w: &SourceWeights, phys: usize) -> Result<[ControllerWeights; 4], String> {
    let p = format!("model.llm.layers.{phys}.controllers.");
    let load = |name: &str| -> Result<ControllerWeights, String> {
        Ok(ControllerWeights {
            down: w.dense(&format!("{p}{name}.down_weight"))?.to_vec(),
            gain: w.dense(&format!("{p}{name}.gain_weight"))?.to_vec(),
            forget: w.dense(&format!("{p}{name}.forget_weight"))?.to_vec(),
        })
    };
    Ok([load("attn")?, load("k")?, load("v")?, load("mlp")?])
}

// ---------------------------------------------------------------------------
// Batched teacher forward
// ---------------------------------------------------------------------------

/// Cross-visit state of one token (what `forward_token_cached` keeps local).
#[derive(Clone, Default)]
struct Tok {
    x: Vec<f32>,
    blocks: Vec<Vec<f32>>,
}

struct Teacher<'a> {
    w: &'a SourceWeights,
    cfg: &'a Inkling2Config,
}

impl Teacher<'_> {
    /// Final normalized hidden state of every position of `seqs` (all the
    /// same length), token-major `[s * len + t]`.
    fn forward(&self, seqs: &[&[u32]]) -> Result<Vec<Vec<f32>>, String> {
        let cfg = self.cfg;
        let e = cfg.n_embd;
        let len = seqs[0].len();
        if seqs.iter().any(|s| s.len() != len) {
            return Err("teacher: unequal sequence lengths".into());
        }
        let embed = self.w.dense("model.llm.embed.weight")?;
        let tok_norm = self.w.dense("model.llm.embed_norm.weight")?;
        let ids: Vec<u32> = seqs.iter().flat_map(|s| s.iter().copied()).collect();
        let mut toks: Vec<Tok> = ids
            .par_iter()
            .map(|&id| {
                let row = &embed[id as usize * e..(id as usize + 1) * e];
                Tok {
                    x: rmsnorm_gain(row, tok_norm, cfg.rms_norm_eps),
                    blocks: Vec::new(),
                }
            })
            .collect();
        let started = Instant::now();
        let mut visit = 0usize;
        let lc = &cfg.loop_cfg;
        for _ in 0..lc.core_start {
            self.visit(visit, len, &mut toks, None)?;
            visit += 1;
            eprintln!(
                "  visit {visit}/{}  {:.0}s",
                lc.n_slot(),
                started.elapsed().as_secs_f64()
            );
        }
        let prelude: Vec<Tok> = toks.clone();
        let msg_queries = forward::pack_message_queries(self.w, cfg)?;
        let core_len = lc.core_end - lc.core_start;
        let mut prev: Vec<Tok> = prelude.clone();
        for pass in 0..lc.num_passes {
            toks.clone_from(&prelude);
            // Bank: the previous pass's blocks after the prelude and first core block.
            let banks: Vec<Vec<Vec<f32>>> = if pass > 0 {
                prev.par_iter()
                    .zip(&prelude)
                    .map(|(p, pre)| {
                        p.blocks
                            .iter()
                            .skip(pre.blocks.len() + 1)
                            .cloned()
                            .collect()
                    })
                    .collect()
            } else {
                Vec::new()
            };
            for i in 0..core_len {
                let phys = lc.core_start + i;
                let messages = if pass > 0 {
                    let msgs: Vec<[Vec<f32>; 4]> = banks
                        .par_iter()
                        .zip(&prev)
                        .map(|(bank, p)| {
                            let m: Vec<Vec<f32>> = msg_queries[i]
                                .iter()
                                .map(|q| oracle::mix(bank, &p.x, q, e, 1))
                                .collect();
                            [m[0].clone(), m[1].clone(), m[2].clone(), m[3].clone()]
                        })
                        .collect();
                    Some((msgs, controllers(self.w, phys)?))
                } else {
                    None
                };
                self.visit(visit, len, &mut toks, messages.as_ref())?;
                visit += 1;
                eprintln!(
                    "  visit {visit}/{}  {:.0}s",
                    lc.n_slot(),
                    started.elapsed().as_secs_f64()
                );
            }
            prev.clone_from(&toks);
        }
        for _ in lc.core_end..cfg.n_phys {
            self.visit(visit, len, &mut toks, None)?;
            visit += 1;
            eprintln!(
                "  visit {visit}/{}  {:.0}s",
                lc.n_slot(),
                started.elapsed().as_secs_f64()
            );
        }
        let out_q = self.w.dense("model.llm.attn_res.output_query")?;
        let final_norm = self.w.dense("model.llm.norm.weight")?;
        Ok(toks
            .par_iter()
            .map(|t| {
                let x = oracle::mix(&t.blocks, &t.x, out_q, e, 1);
                rmsnorm_gain(&x, final_norm, cfg.rms_norm_eps)
            })
            .collect())
    }

    /// One visit for every token: `decode_visit_cached`, visit-major.
    #[allow(clippy::type_complexity)]
    fn visit(
        &self,
        visit: usize,
        len: usize,
        toks: &mut [Tok],
        messages: Option<&(Vec<[Vec<f32>; 4]>, [ControllerWeights; 4])>,
    ) -> Result<(), String> {
        let cfg = self.cfg;
        let w = self.w;
        let n = toks.len();
        let n_seq = n / len;
        let ne = cfg.n_embd;
        let phys = cfg.loop_cfg.layer_phys(visit);
        let p = format!("model.llm.layers.{phys}.");
        let swa = cfg.swa_layers[phys];
        let g = |s: &str| w.dense(&format!("{p}{s}"));
        let eps = cfg.rms_norm_eps;
        let rank = cfg.loop_cfg.gate_rank;

        // --- attention inputs: AttnRes mix, controllers, input norms ---
        let attn_q = if phys == 0 {
            None
        } else {
            Some(w.dense_shaped(&format!("model.llm.attn_res.attn_queries.{}", phys - 1))?)
        };
        let (nq, nk, nv) = if swa {
            ("attn_norm.weight", "attn_norm.weight", "attn_norm.weight")
        } else {
            (
                "q_input_norm.weight",
                "k_input_norm.weight",
                "v_input_norm.weight",
            )
        };
        let (norm_q, norm_k, norm_v) = (g(nq)?, g(nk)?, g(nv)?);
        let inputs: Vec<[Vec<f32>; 3]> = toks
            .par_iter()
            .enumerate()
            .map(|(i, t)| {
                let (bq, bk, bv) = match attn_q {
                    None => (t.x.clone(), t.x.clone(), t.x.clone()),
                    Some((shape, q)) => {
                        let query_rows = if shape.len() == 1 { 1 } else { shape[0] };
                        let mixed = oracle::mix(&t.blocks, &t.x, q, ne, query_rows);
                        let row = |j: usize| mixed[j * ne..(j + 1) * ne].to_vec();
                        if query_rows >= 3 {
                            (row(0), row(1), row(2))
                        } else {
                            let r = row(0);
                            (r.clone(), r.clone(), r)
                        }
                    }
                };
                let (qc, kc, vc) = match messages {
                    Some((msgs, ctrls)) => {
                        let apply = |base: &[f32], j: usize, c: &ControllerWeights| {
                            oracle::control(
                                base,
                                &msgs[i][j],
                                &c.down,
                                &c.gain,
                                &c.forget,
                                ne,
                                rank,
                            )
                        };
                        (
                            apply(&bq, 0, &ctrls[0]),
                            apply(&bk, 1, &ctrls[1]),
                            apply(&bv, 2, &ctrls[2]),
                        )
                    }
                    None => (bq, bk, bv),
                };
                [
                    rmsnorm_gain(&qc, norm_q, eps),
                    rmsnorm_gain(&kc, norm_k, eps),
                    rmsnorm_gain(&vc, norm_v, eps),
                ]
            })
            .collect();

        let d = cfg.head_dim;
        let nh = cfg.n_head;
        let nkv = cfg.n_head_kv;
        let ekv = nkv * d;
        let eattn = nh * d;
        let column = |j: usize| {
            let mut v = Vec::with_capacity(n * ne);
            for r in &inputs {
                v.extend_from_slice(&r[j]);
            }
            v
        };
        let (q_inp, k_inp, v_inp) = (column(0), column(1), column(2));
        drop(inputs);
        let q = gemm(&q_inp, n, ne, g("attn.wq_du.weight")?, eattn);
        let k_raw = gemm(&k_inp, n, ne, g("attn.wk_dv.weight")?, ekv);
        let v_raw = gemm(&v_inp, n, ne, g("attn.wv_dv.weight")?, ekv);
        let r = gemm(&q_inp, n, ne, g("attn.wr_du.weight")?, nh * cfg.n_rel);
        let sinks = gemm(&q_inp, n, ne, g("attn.attn_sink_proj.weight")?, nh);
        let gate = gemm(&q_inp, n, ne, g("attn.gate_attn.weight")?, eattn);
        drop(q_inp);

        // --- K/V short convs, sequential within each sequence ---
        let kernel = cfg.conv_kernel;
        let conv_kv: Vec<(Vec<f32>, Vec<f32>)> = if !swa {
            let (ks, kw) = w.dense_shaped(&format!("{p}attn.k_sconv.static_w"))?;
            let (vs, vw) = w.dense_shaped(&format!("{p}attn.v_sconv.static_w"))?;
            let wkt = torch_conv(ks, kw, kernel);
            let wvt = torch_conv(vs, vw, kernel);
            let kproj = g("attn.k_sconv.weight_proj")?;
            let vproj = g("attn.v_sconv.weight_proj")?;
            (0..n_seq)
                .into_par_iter()
                .flat_map_iter(|s| {
                    let (mut kp, mut vp) = (Vec::new(), Vec::new());
                    let mut out = Vec::with_capacity(len);
                    for t in s * len..(s + 1) * len {
                        let kin = &k_inp[t * ne..(t + 1) * ne];
                        let vin = &v_inp[t * ne..(t + 1) * ne];
                        let (kc2, kp2) = oracle::convolve_dynamic(
                            &k_raw[t * ekv..(t + 1) * ekv],
                            &wkt,
                            kproj,
                            kin,
                            &kp,
                            nkv,
                            d,
                            kernel,
                            ne,
                        );
                        let (vc2, vp2) = oracle::convolve_dynamic(
                            &v_raw[t * ekv..(t + 1) * ekv],
                            &wvt,
                            vproj,
                            vin,
                            &vp,
                            nkv,
                            d,
                            kernel,
                            ne,
                        );
                        kp = kp2;
                        vp = vp2;
                        out.push((kc2, vc2));
                    }
                    out
                })
                .collect()
        } else {
            let conv_name = |which: &str| {
                let plain = format!("{p}attn.{which}_sconv.weight");
                if w.dense.contains_key(&plain) {
                    plain
                } else {
                    format!("{p}attn.{which}_sconv.static_w")
                }
            };
            let (kn, vn) = (conv_name("k"), conv_name("v"));
            let (ks, kw) = w.dense_shaped(&kn)?;
            let (vs, vw) = w.dense_shaped(&vn)?;
            let k_conv_w = load_conv(ks, kw, &kn, kernel);
            let v_conv_w = load_conv(vs, vw, &vn, kernel);
            (0..n_seq)
                .into_par_iter()
                .flat_map_iter(|s| {
                    let (mut kp, mut vp) = (Vec::new(), Vec::new());
                    let mut out = Vec::with_capacity(len);
                    for t in s * len..(s + 1) * len {
                        let (kc2, kp2) = oracle::convolve_static(
                            &k_raw[t * ekv..(t + 1) * ekv],
                            &k_conv_w,
                            &kp,
                            1,
                            ekv,
                            kernel,
                        );
                        let (vc2, vp2) = oracle::convolve_static(
                            &v_raw[t * ekv..(t + 1) * ekv],
                            &v_conv_w,
                            &vp,
                            1,
                            ekv,
                            kernel,
                        );
                        kp = kp2;
                        vp = vp2;
                        out.push((kc2, vc2));
                    }
                    out
                })
                .collect()
        };
        drop((k_inp, v_inp, k_raw, v_raw));

        // --- clips, V norm, tau ---
        let q_clip = g("attn.q_l2_clip.scale")?;
        let rel_clip = g("attn.rel_l2_clip.scale")?;
        let rel_table = g("attn.rel_logits_proj.proj")?;
        // (qc, kc, vc, rc) per token.
        let clipped: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> = conv_kv
            .par_iter()
            .enumerate()
            .map(|(i, (k_conv, v_conv))| {
                let pos = i % len;
                let qc = oracle::clip(&q[i * eattn..(i + 1) * eattn], Some(q_clip), cfg.qk_clip, d);
                let kc = oracle::clip(k_conv, None, cfg.qk_clip, d);
                let vc: Vec<f32> = (0..nkv)
                    .flat_map(|h| oracle::rmsnorm(&v_conv[h * d..(h + 1) * d], eps))
                    .collect();
                let rr = &r[i * nh * cfg.n_rel..(i + 1) * nh * cfg.n_rel];
                let mut rc = oracle::clip(rr, Some(rel_clip), cfg.rel_clip, cfg.n_rel);
                if !swa {
                    let t = oracle::tau(pos, cfg.log_floor, cfg.log_alpha);
                    for v in rc.iter_mut() {
                        *v *= t;
                    }
                }
                (qc, kc, vc, rc)
            })
            .collect();
        drop((conv_kv, q, r));

        // Per-sequence, per-KV-head contiguous K/V rows (the reference cache layout).
        let kv_rows: Vec<(Vec<Vec<f32>>, Vec<Vec<f32>>)> = (0..n_seq)
            .into_par_iter()
            .map(|s| {
                let mut kh = vec![Vec::with_capacity(len * d); nkv];
                let mut vh = vec![Vec::with_capacity(len * d); nkv];
                for (_, kc, vc, _) in &clipped[s * len..(s + 1) * len] {
                    for h in 0..nkv {
                        kh[h].extend_from_slice(&kc[h * d..(h + 1) * d]);
                        vh[h].extend_from_slice(&vc[h * d..(h + 1) * d]);
                    }
                }
                (kh, vh)
            })
            .collect();

        // --- attention core: identical per-query math to the reference ---
        let attn_out: Vec<f32> = clipped
            .par_iter()
            .enumerate()
            .flat_map_iter(|(i, (qc, _, _, rc))| {
                let (s, pos) = (i / len, i % len);
                let (kh_all, vh_all) = &kv_rows[s];
                let lo = if swa {
                    pos.saturating_sub(cfg.sliding_window.saturating_sub(1))
                } else {
                    0
                };
                let m = pos + 1 - lo;
                let mut out = vec![0.0f32; eattn];
                for h in 0..nh {
                    let rh = &rc[h * cfg.n_rel..(h + 1) * cfg.n_rel];
                    let mut qh = qc[h * d..(h + 1) * d].to_vec();
                    if !swa {
                        let t = oracle::tau(pos, cfg.log_floor, cfg.log_alpha);
                        for v in &mut qh {
                            *v *= t;
                        }
                    }
                    let kv = h / (nh / nkv);
                    let kcache = &kh_all[kv];
                    let sink_h = sinks[i * nh + h];
                    let mut scores = Vec::with_capacity(m);
                    for j in 0..m {
                        let key = lo + j;
                        let dist = pos - key;
                        let bias =
                            rel_bias_for_dist(rel_table, rh, cfg.n_rel, cfg.n_extent, dist, sink_h);
                        let kr = &kcache[key * d..(key + 1) * d];
                        scores.push(
                            qh.iter().zip(kr).map(|(a, b)| a * b).sum::<f32>() / (d as f32).sqrt()
                                + bias,
                        );
                    }
                    let wv = oracle::attention_softmax_with_sink(
                        &scores,
                        0.0,
                        &vh_all[kv][lo * d..],
                        m,
                        d,
                    );
                    out[h * d..(h + 1) * d].copy_from_slice(&wv);
                }
                for (o, gt) in out.iter_mut().zip(&gate[i * eattn..(i + 1) * eattn]) {
                    *o *= oracle::sigmoid(*gt);
                }
                out
            })
            .collect();
        drop((clipped, kv_rows, gate, sinks));
        let proj = gemm(&attn_out, n, eattn, g("attn.wo_ud.weight")?, ne);
        drop(attn_out);
        let ac_name = format!("{p}attn_sconv.weight");
        let (acs, acw) = w.dense_shaped(&ac_name)?;
        let attn_conv_w = load_conv(acs, acw, &ac_name, kernel);
        let attn_res: Vec<Vec<f32>> = (0..n_seq)
            .into_par_iter()
            .flat_map_iter(|s| {
                let mut past = Vec::new();
                let mut out = Vec::with_capacity(len);
                for t in s * len..(s + 1) * len {
                    let (o, np) = oracle::convolve_static(
                        &proj[t * ne..(t + 1) * ne],
                        &attn_conv_w,
                        &past,
                        1,
                        ne,
                        kernel,
                    );
                    past = np;
                    out.push(o);
                }
                out
            })
            .collect();
        drop(proj);

        // --- AttnRes block bookkeeping, FFN mix/controller/norm ---
        let fq = w.dense(&format!("model.llm.attn_res.mlp_queries.{phys}"))?;
        let mlp_norm = g("mlp_norm.weight")?;
        let push_block = phys % cfg.attn_res_block == 0;
        let curs_and_normed: Vec<(Vec<f32>, Vec<f32>)> = toks
            .par_iter_mut()
            .zip(attn_res)
            .enumerate()
            .map(|(i, (t, ar))| {
                let cur = if push_block {
                    t.blocks.push(t.x.clone());
                    ar
                } else {
                    t.x.iter().zip(&ar).map(|(a, b)| a + b).collect()
                };
                let mut ffn_in = oracle::mix(&t.blocks, &cur, fq, ne, 1);
                if let Some((msgs, ctrls)) = messages {
                    ffn_in = oracle::control(
                        &ffn_in,
                        &msgs[i][3],
                        &ctrls[3].down,
                        &ctrls[3].gain,
                        &ctrls[3].forget,
                        ne,
                        rank,
                    );
                }
                let normed = rmsnorm_gain(&ffn_in, mlp_norm, eps);
                (cur, normed)
            })
            .collect();
        let (curs, normed): (Vec<Vec<f32>>, Vec<Vec<f32>>) = curs_and_normed.into_iter().unzip();

        // --- routed MoE ---
        let router = g("mlp.gate.weight")?;
        let bias = g("mlp.gate.bias").ok();
        let n_experts = router.len() / ne;
        let x_ffn = flat(&normed);
        drop(normed);
        let logits = gemm(&x_ffn, n, ne, router, n_experts);
        let top_k = cfg.n_expert_used;
        let selected: Vec<Vec<(usize, f32)>> = logits
            .par_chunks_exact(n_experts)
            .map(|l| oracle::sigmoid_topk(l, bias, top_k, cfg.route_scale))
            .collect();
        drop(logits);
        // by_expert[ex] lists (token, rank); slot[token * top_k + rank] is the
        // row of that pair in its expert's output.
        let mut by_expert: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_experts];
        let mut slot = vec![0usize; n * top_k];
        for (i, sel) in selected.iter().enumerate() {
            for (rk, &(ex, _)) in sel.iter().enumerate() {
                slot[i * top_k + rk] = by_expert[ex].len();
                by_expert[ex].push((i, rk));
            }
        }
        let inter = cfg.inter;
        let per_expert: HashMap<usize, Vec<f32>> = (0..n_experts)
            .into_par_iter()
            .filter(|&ex| !by_expert[ex].is_empty())
            .map(|ex| -> Result<(usize, Vec<f32>), String> {
                let (gu, dn) = w.expert(phys, ex)?;
                let rows = &by_expert[ex];
                let mut xe = Vec::with_capacity(rows.len() * ne);
                for &(i, _) in rows {
                    xe.extend_from_slice(&x_ffn[i * ne..(i + 1) * ne]);
                }
                let gu_out = gemm(&xe, rows.len(), ne, &gu, 2 * inter);
                drop(gu);
                let act: Vec<f32> = gu_out
                    .chunks_exact(2 * inter)
                    .flat_map(|row| {
                        (0..inter).map(move |c| {
                            oracle::situ(
                                row[c],
                                row[inter + c],
                                cfg.situ_beta,
                                cfg.situ_linear_beta,
                            )
                        })
                    })
                    .collect();
                Ok((ex, gemm(&act, rows.len(), inter, &dn, ne)))
            })
            .collect::<Result<_, String>>()?;
        drop(x_ffn);
        let moe: Vec<Vec<f32>> = selected
            .par_iter()
            .enumerate()
            .map(|(i, sel)| {
                let mut out = vec![0.0f32; ne];
                for (rk, &(ex, weight)) in sel.iter().enumerate() {
                    let j = slot[i * top_k + rk];
                    let dv = &per_expert[&ex][j * ne..(j + 1) * ne];
                    for c in 0..ne {
                        out[c] += weight * dv[c];
                    }
                }
                out
            })
            .collect();
        drop(per_expert);
        let mc_name = format!("{p}mlp_sconv.weight");
        let (mcs, mcw) = w.dense_shaped(&mc_name)?;
        let mlp_conv_w = load_conv(mcs, mcw, &mc_name, kernel);
        let moe_conv: Vec<Vec<f32>> = (0..n_seq)
            .into_par_iter()
            .flat_map_iter(|s| {
                let mut past = Vec::new();
                let mut out = Vec::with_capacity(len);
                for mv in &moe[s * len..(s + 1) * len] {
                    let (o, np) = oracle::convolve_static(mv, &mlp_conv_w, &past, 1, ne, kernel);
                    past = np;
                    out.push(o);
                }
                out
            })
            .collect();
        toks.par_iter_mut()
            .zip(curs)
            .zip(moe_conv)
            .for_each(|((t, cur), mc)| {
                t.x = cur.iter().zip(&mc).map(|(a, b)| a + b).collect();
            });
        Ok(())
    }

    /// Softcapped, unpadded logits of `hidden` rows (`lm_head_logits`).
    fn head(&self, hidden: &[Vec<f32>]) -> Result<Vec<f32>, String> {
        let cfg = self.cfg;
        let head = self.w.dense("model.llm.unembed.weight")?;
        let n = hidden.len();
        let raw = gemm(&flat(hidden), n, cfg.n_embd, head, cfg.vocab_size);
        let uv = cfg.unpadded_vocab_size;
        let mut out = vec![0.0f32; n * uv];
        out.par_chunks_exact_mut(uv)
            .zip(raw.par_chunks_exact(cfg.vocab_size))
            .for_each(|(o, r)| {
                for (ov, &rv) in o.iter_mut().zip(&r[..uv]) {
                    *ov = oracle::softcap(rv, cfg.softcap);
                }
            });
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

struct Args(HashMap<String, String>);

impl Args {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut map = HashMap::new();
        let mut it = args.peekable();
        while let Some(a) = it.next() {
            let key = a
                .strip_prefix("--")
                .ok_or_else(|| format!("unexpected argument {a}\n{USAGE}"))?
                .to_string();
            let val = it.next().ok_or_else(|| format!("--{key} needs a value"))?;
            map.insert(key, val);
        }
        Ok(Self(map))
    }
    fn path(&self, k: &str) -> Result<PathBuf, String> {
        self.0
            .get(k)
            .map(PathBuf::from)
            .ok_or_else(|| format!("missing --{k}\n{USAGE}"))
    }
    fn num(&self, k: &str, default: usize) -> Result<usize, String> {
        self.0.get(k).map_or(Ok(default), |v| {
            v.parse().map_err(|_| format!("--{k}: not a number: {v}"))
        })
    }
    fn opt_num(&self, k: &str) -> Result<Option<usize>, String> {
        self.0
            .get(k)
            .map(|v| v.parse().map_err(|_| format!("--{k}: not a number: {v}")))
            .transpose()
    }
}

fn artifact_id(path: &Path) -> Result<ArtifactId, String> {
    use std::io::Read;
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buf = vec![0u8; 16 << 20];
    let mut bytes = 0u64;
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        digest.update(&buf[..n]);
        bytes += n as u64;
    }
    Ok(ArtifactId {
        path: path.display().to_string(),
        sha256: format!("{:x}", digest.finalize()),
        bytes,
    })
}

/// `git rev-parse HEAD` of the working directory (as qwen4_kld records it).
fn engine_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

/// Config + product tokenizer of an artifact, and the tokenized slice windows.
fn corpus_windows(
    model: &Path,
    slice: &Path,
    n_ctx: usize,
    n_chunk: usize,
) -> Result<(Inkling2Config, Vec<u32>), String> {
    let hfq = HfqFile::open(model).map_err(|e| format!("open {}: {e}", model.display()))?;
    let cfg = Inkling2Config::from_metadata_json(&hfq.metadata_json)?;
    let tok = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer: {e}"))?;
    let text =
        std::fs::read_to_string(slice).map_err(|e| format!("read {}: {e}", slice.display()))?;
    let stream = tok.encode(&text);
    let have = stream.len() / n_ctx;
    if have < n_chunk {
        return Err(format!(
            "{} tokenizes to {} tokens: {have} chunks of {n_ctx}, {n_chunk} requested",
            slice.display(),
            stream.len()
        ));
    }
    eprintln!(
        "inkling_kld: {} -> {} tokens, using {n_chunk} x {n_ctx} (first ids {:?})",
        slice.display(),
        stream.len(),
        &stream[..8.min(stream.len())]
    );
    Ok((cfg, stream[..n_chunk * n_ctx].to_vec()))
}

// ---------------------------------------------------------------------------
// ref
// ---------------------------------------------------------------------------

fn build_ref(a: &Args) -> Result<(), String> {
    let source = a.path("source")?;
    let model = a.path("model")?;
    let slice = a.path("slice")?;
    let output = a.path("output")?;
    let n_ctx = a.num("n-ctx", 2048)?;
    let n_chunk = a.num("max-chunks", 24)?;
    let top_k = a.num("top-k", 256)?;
    let batch = a.num("batch-chunks", n_chunk)?.max(1);
    let (cfg, tokens) = corpus_windows(&model, &slice, n_ctx, n_chunk)?;
    let window = WindowSpec::legacy_half(n_ctx);
    let vocab = cfg.unpadded_vocab_size;
    eprintln!("inkling_kld ref: hashing teacher and corpus");
    let teacher_id = artifact_id(&source)?;
    let corpus_id = artifact_id(&slice)?;
    let t0 = Instant::now();
    let w = SourceWeights::open(&source)?;
    eprintln!(
        "inkling_kld ref: source dense F32 ready ({} tensors) in {:.0}s",
        w.dense.len(),
        t0.elapsed().as_secs_f64()
    );
    let teacher = Teacher { w: &w, cfg: &cfg };
    let mut blocks: Vec<TopKBlock> = Vec::with_capacity(n_chunk * window.scored_per_chunk());
    let mut nll_sum = 0.0f64;
    // Rows up to score_to are all the scored rows need.
    let run_len = window.score_to;
    for b0 in (0..n_chunk).step_by(batch) {
        let b1 = (b0 + batch).min(n_chunk);
        let seqs: Vec<&[u32]> = (b0..b1)
            .map(|c| &tokens[c * n_ctx..c * n_ctx + run_len])
            .collect();
        let tb = Instant::now();
        eprintln!(
            "inkling_kld ref: chunks {b0}..{b1} ({} positions)",
            seqs.len() * run_len
        );
        let hidden = teacher.forward(&seqs)?;
        for (si, c) in (b0..b1).enumerate() {
            let rows = &hidden[si * run_len + window.score_from..si * run_len + window.score_to];
            let logits = teacher.head(rows)?;
            let chunk = &tokens[c * n_ctx..(c + 1) * n_ctx];
            let scored: Vec<(TopKBlock, f64)> = logits
                .par_chunks_exact(vocab)
                .zip(window.score_from..window.score_to)
                .map(|(row, pos)| -> Result<(TopKBlock, f64), String> {
                    Ok((
                        topk_block(row, top_k).map_err(|e| e.to_string())?,
                        nll_of(row, chunk[pos + 1] as usize).map_err(|e| e.to_string())?,
                    ))
                })
                .collect::<Result<_, String>>()?;
            let chunk_nll: f64 = scored.iter().map(|(_, l)| l).sum();
            nll_sum += chunk_nll;
            blocks.extend(scored.into_iter().map(|(b, _)| b));
            eprintln!(
                "  chunk {}/{n_chunk}  NLL {:.4}  running PPL {:.4}",
                c + 1,
                chunk_nll / window.scored_per_chunk() as f64,
                (nll_sum / blocks.len() as f64).exp()
            );
        }
        eprintln!(
            "inkling_kld ref: batch {b0}..{b1} done in {:.0}s (total {:.0}s)",
            tb.elapsed().as_secs_f64(),
            t0.elapsed().as_secs_f64()
        );
    }
    let mean_nll = nll_sum / blocks.len() as f64;
    let oracle_stats = OracleStats {
        mean_nll,
        ppl: mean_nll.exp(),
        n_scored: blocks.len(),
    };
    if let TeacherVerdict::Implausible { reason, .. } =
        evaluate(&TeacherGate::for_vocab(vocab), oracle_stats)
    {
        return Err(format!("teacher implausible: {reason}"));
    }
    let header = RefHeader {
        version: 3,
        arch_id: hipfire_arch_inkling2::ARCH_ID,
        n_ctx,
        n_chunk,
        n_vocab: vocab,
        estimator: Estimator::TopK {
            k: top_k as u32,
            bias_vs_full: None,
        },
        window,
        teacher: teacher_id,
        corpus: corpus_id,
        oracle: oracle_stats,
        engine_commit: engine_commit(),
        arch: "cpu-f32-source-dequant".into(),
    };
    let mut writer = RefWriter::create(&output, header).map_err(|e| e.to_string())?;
    writer.push_tokens(&tokens).map_err(|e| e.to_string())?;
    for b in &blocks {
        writer
            .push_block(b.residual_logprob, &b.top)
            .map_err(|e| e.to_string())?;
    }
    writer.finish().map_err(|e| e.to_string())?;
    eprintln!(
        "inkling_kld ref: wrote {}  teacher PPL {:.4} (mean NLL {mean_nll:.6}, {} rows) in {:.0}s",
        output.display(),
        oracle_stats.ppl,
        oracle_stats.n_scored,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// fidelity
// ---------------------------------------------------------------------------

fn fidelity(a: &Args) -> Result<(), String> {
    let source = a.path("source")?;
    let model = a.path("model")?;
    let slice = a.path("slice")?;
    let output = a.path("output")?;
    let n_ctx = a.num("n-ctx", 64)?;
    let n_chunk = a.num("chunks", 4)?;
    let (mut cfg, tokens) = corpus_windows(&model, &slice, n_ctx, n_chunk)?;
    let sliding_window = a.opt_num("sliding-window")?;
    let log_floor = a.opt_num("log-floor")?;
    if let Some(sw) = sliding_window {
        cfg.sliding_window = sw;
    }
    if let Some(lf) = log_floor {
        cfg.log_floor = lf;
    }
    let w = SourceWeights::open(&source)?;
    let vocab = cfg.unpadded_vocab_size;
    let teacher = Teacher { w: &w, cfg: &cfg };
    let seqs: Vec<&[u32]> = tokens.chunks_exact(n_ctx).collect();
    let t0 = Instant::now();
    let hidden = teacher.forward(&seqs)?;
    let logits = teacher.head(&hidden)?;
    let teacher_s = t0.elapsed().as_secs_f64();
    eprintln!(
        "inkling_kld fidelity: batched teacher {teacher_s:.1}s; running the per-token reference"
    );
    let embed = w.dense("model.llm.embed.weight")?;
    let t1 = Instant::now();
    let reference: Vec<Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), String>> = std::thread::scope(|sc| {
        let handles: Vec<_> = seqs
            .iter()
            .map(|seq| {
                let (w, cfg) = (&w, &cfg);
                sc.spawn(move || -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), String> {
                    let mut st = CpuDecodeState::new(cfg, n_ctx)?;
                    let (mut hs, mut ls) = (Vec::new(), Vec::new());
                    for (pos, &id) in seq.iter().enumerate() {
                        let row = &embed[id as usize * cfg.n_embd..(id as usize + 1) * cfg.n_embd];
                        let h = forward::forward_token_cached(w, cfg, &mut st, row, pos)?;
                        ls.push(forward::lm_head_logits(w, cfg, &h)?);
                        hs.push(h);
                    }
                    Ok((hs, ls))
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("reference thread"))
            .collect()
    });
    let reference_s = t1.elapsed().as_secs_f64();
    let mut rows = 0usize;
    let (mut hidden_equal, mut logits_equal) = (0usize, 0usize);
    let (mut hidden_max, mut logits_max) = (0.0f64, 0.0f64);
    let mut argmax_equal = 0usize;
    let mut first_diff: Option<(usize, usize)> = None;
    for (s, r) in reference.into_iter().enumerate() {
        let (hs, ls) = r?;
        for pos in 0..n_ctx {
            let i = s * n_ctx + pos;
            let th = &hidden[i];
            let tl = &logits[i * vocab..(i + 1) * vocab];
            let rh = &hs[pos];
            let rl = &ls[pos];
            if rl.len() != vocab || rh.len() != th.len() {
                return Err("fidelity: reference shape mismatch".into());
            }
            let hd = th
                .iter()
                .zip(rh)
                .map(|(a, b)| (a - b).abs() as f64)
                .fold(0.0, f64::max);
            let ld = tl
                .iter()
                .zip(rl)
                .map(|(a, b)| (a - b).abs() as f64)
                .fold(0.0, f64::max);
            let heq = th.iter().zip(rh).all(|(a, b)| a.to_bits() == b.to_bits());
            let leq = tl
                .iter()
                .zip(rl.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits());
            if !(heq && leq) && first_diff.is_none() {
                first_diff = Some((s, pos));
            }
            hidden_equal += usize::from(heq);
            logits_equal += usize::from(leq);
            hidden_max = hidden_max.max(hd);
            logits_max = logits_max.max(ld);
            let am = |v: &[f32]| {
                v.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
            };
            argmax_equal += usize::from(am(tl) == am(rl));
            rows += 1;
        }
    }
    let receipt = serde_json::json!({
        "schema": "hipfire.inkling2.teacher_fidelity.v1",
        "source": artifact_id(&source)?,
        "model_for_config_and_tokenizer": model.display().to_string(),
        "slice": slice.display().to_string(),
        "n_ctx": n_ctx, "chunks": n_chunk,
        "sliding_window": cfg.sliding_window, "log_floor": cfg.log_floor,
        "overrides": {"sliding_window": sliding_window, "log_floor": log_floor},
        "rows": rows,
        "hidden_bit_identical_rows": hidden_equal,
        "logits_bit_identical_rows": logits_equal,
        "argmax_equal_rows": argmax_equal,
        "hidden_max_abs_diff": hidden_max,
        "logits_max_abs_diff": logits_max,
        "first_differing_row": first_diff,
        "teacher_seconds": teacher_s,
        "reference_seconds": reference_s,
        "engine_commit": engine_commit(),
    });
    let text = serde_json::to_string_pretty(&receipt).map_err(|e| e.to_string())?;
    std::fs::write(&output, text.clone() + "\n").map_err(|e| e.to_string())?;
    println!("{text}");
    if hidden_equal != rows || logits_equal != rows {
        return Err(format!(
            "fidelity: {hidden_equal}/{rows} hidden and {logits_equal}/{rows} logit rows bit-identical"
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// eval (GPU candidate, product loader + forward_token_gpu)
// ---------------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize)]
struct ChunkResult {
    chunk: usize,
    mean_kld: f64,
    p99_kld: f64,
    mean_nll: f64,
    top1_hits: usize,
    scored: usize,
    logits_sha256: String,
    klds: Vec<f64>,
    seconds: f64,
}

fn open_ref(path: &Path) -> Result<(KldRef, memmap2::Mmap), String> {
    let reference = KldRef::open(path).map_err(|e| e.to_string())?;
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    // SAFETY: read-only map of a file this process does not modify.
    let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| e.to_string())?;
    Ok((reference, mmap))
}

/// Score `chunks` in this process and write their results to `shard`.
fn eval_worker(a: &Args) -> Result<(), String> {
    let model = a.path("model")?;
    let ref_path = a.path("ref")?;
    let shard = a.path("shard")?;
    let c0 = a.num("chunk-start", 0)?;
    let c1 = a.num("chunk-end", 0)?;
    let (reference, mmap) = open_ref(&ref_path)?;
    let h = &reference.header;
    let tokens = reference.tokens(&mmap).map_err(|e| e.to_string())?;
    let (from, to) = (h.window.score_from, h.window.score_to);
    let mut gpu = rdna_compute::Gpu::init().map_err(|e| format!("gpu init: {e:?}"))?;
    let max_seq = hipfire_arch_inkling2::INKLING2_MAX_CONTEXT;
    let mut m = hipfire_loader::load_model(
        &model.display().to_string(),
        max_seq,
        None,
        None,
        None,
        None,
        1,
        hipfire_runtime::loader_api::SpecLoadCfg::default(),
        &mut gpu,
    )?;
    if m.arch_id != hipfire_arch_inkling2::ARCH_ID {
        return Err(format!(
            "model arch {} is not Inkling-2 ({})",
            m.arch_id,
            hipfire_arch_inkling2::ARCH_ID
        ));
    }
    let mut results = Vec::new();
    for c in c0..c1 {
        let t0 = Instant::now();
        let chunk = &tokens[c * h.n_ctx..(c + 1) * h.n_ctx];
        let b = m
            .inkling2_mut()
            .ok_or("loaded model has no Inkling-2 bundle")?;
        b.reset_session_state(&mut gpu)?;
        let mut digest = Sha256::new();
        let (mut klds, mut nll_sum, mut top1) = (Vec::with_capacity(to - from), 0.0f64, 0usize);
        for (pos, &id) in chunk[..to].iter().enumerate() {
            let row = hipfire_arch_inkling2::forward_token_gpu(b, &mut gpu, id, pos)?;
            if pos < from {
                continue;
            }
            if row.len() != h.n_vocab {
                return Err(format!(
                    "candidate vocab {} != reference vocab {}",
                    row.len(),
                    h.n_vocab
                ));
            }
            for v in &row {
                digest.update(v.to_le_bytes());
            }
            let blk = reference
                .block(&mmap, c, pos - from)
                .map_err(|e| e.to_string())?;
            let blk = TopKBlock {
                residual_logprob: blk.residual_logprob,
                top: blk.top.to_vec(),
            };
            klds.push(kld_from_block(&blk, &row).map_err(|e| e.to_string())?);
            nll_sum += nll_of(&row, chunk[pos + 1] as usize).map_err(|e| e.to_string())?;
            let argmax = row
                .iter()
                .enumerate()
                .max_by(|x, y| x.1.total_cmp(y.1))
                .map(|(i, _)| i as u32);
            top1 += usize::from(argmax == Some(blk.top[0].0));
        }
        let (mean_kld, p99_kld, mean_nll) = seq_stats(&klds, nll_sum, klds.len());
        eprintln!(
            "  chunk {}/{}  KLD {mean_kld:.6}  p99 {p99_kld:.4}  NLL {mean_nll:.4}  {:.0}s",
            c + 1,
            h.n_chunk,
            t0.elapsed().as_secs_f64()
        );
        results.push(ChunkResult {
            chunk: c,
            mean_kld,
            p99_kld,
            mean_nll,
            top1_hits: top1,
            scored: klds.len(),
            logits_sha256: format!("{:x}", digest.finalize()),
            klds,
            seconds: t0.elapsed().as_secs_f64(),
        });
    }
    let text = serde_json::to_string(&results).map_err(|e| e.to_string())?;
    std::fs::write(&shard, text).map_err(|e| format!("write {}: {e}", shard.display()))?;
    Ok(())
}

fn with_suffix(p: &Path, suffix: &str) -> PathBuf {
    PathBuf::from(format!("{}{suffix}", p.display()))
}

fn eval(a: &Args) -> Result<(), String> {
    let model = a.path("model")?;
    let ref_path = a.path("ref")?;
    let output = a.path("output")?;
    let per_process = a.num("chunks-per-process", 1)?.max(1);
    let (reference, _mmap) = open_ref(&ref_path)?;
    let h = reference.header.clone();
    let n_chunk = h.n_chunk.min(a.num("max-chunks", h.n_chunk)?);
    if h.arch_id != hipfire_arch_inkling2::ARCH_ID {
        return Err(format!("reference arch {} is not Inkling-2", h.arch_id));
    }
    eprintln!(
        "inkling_kld eval: {n_chunk} chunks x {} (scored [{}, {})), teacher PPL {:.4}; hashing model",
        h.n_ctx, h.window.score_from, h.window.score_to, h.oracle.ppl
    );
    let model_id = artifact_id(&model)?;
    let ref_id = artifact_id(&ref_path)?;
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let exe_id = artifact_id(&exe)?;
    let shard_dir = with_suffix(&output, ".shards");
    std::fs::create_dir_all(&shard_dir).map_err(|e| e.to_string())?;
    let started = Instant::now();
    let mut results: Vec<ChunkResult> = Vec::with_capacity(n_chunk);
    for c0 in (0..n_chunk).step_by(per_process) {
        let c1 = (c0 + per_process).min(n_chunk);
        let shard = shard_dir.join(format!("chunks-{c0}-{c1}.json"));
        let status = std::process::Command::new(&exe)
            .arg("eval-worker")
            .args(["--model", &model.display().to_string()])
            .args(["--ref", &ref_path.display().to_string()])
            .args(["--shard", &shard.display().to_string()])
            .args([
                "--chunk-start",
                &c0.to_string(),
                "--chunk-end",
                &c1.to_string(),
            ])
            .status()
            .map_err(|e| format!("spawn worker: {e}"))?;
        if !status.success() {
            return Err(format!("worker for chunks {c0}..{c1} failed: {status}"));
        }
        let text = std::fs::read_to_string(&shard).map_err(|e| e.to_string())?;
        let part: Vec<ChunkResult> = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        if part.len() != c1 - c0 || part.iter().zip(c0..c1).any(|(r, c)| r.chunk != c) {
            return Err(format!(
                "shard {} does not hold chunks {c0}..{c1}",
                shard.display()
            ));
        }
        results.extend(part);
        let done: Vec<f64> = results.iter().map(|r| r.mean_kld).collect();
        eprintln!(
            "inkling_kld eval: {}/{n_chunk} chunks, running KLD {:.6}, {:.0}s",
            results.len(),
            done.iter().sum::<f64>() / done.len() as f64,
            started.elapsed().as_secs_f64()
        );
    }
    let means: Vec<f64> = results.iter().map(|r| r.mean_kld).collect();
    let p99s: Vec<f64> = results.iter().map(|r| r.p99_kld).collect();
    let nlls: Vec<f64> = results.iter().map(|r| r.mean_nll).collect();
    let mut f = std::fs::File::create(&output).map_err(|e| e.to_string())?;
    write_hfkseq(&mut f, n_chunk, &means, &p99s, &nlls).map_err(|e| e.to_string())?;
    let tok_path = with_suffix(&output, ".hfktok");
    let all_klds: Vec<f64> = results
        .iter()
        .flat_map(|r| r.klds.iter().copied())
        .collect();
    let mut tf = std::fs::File::create(&tok_path).map_err(|e| e.to_string())?;
    write_hfktok(&mut tf, n_chunk, h.window.scored_per_chunk(), &all_klds)
        .map_err(|e| e.to_string())?;
    let scored: usize = results.iter().map(|r| r.scored).sum();
    let mean_kld = all_klds.iter().sum::<f64>() / scored as f64;
    let mean_nll = results
        .iter()
        .map(|r| r.mean_nll * r.scored as f64)
        .sum::<f64>()
        / scored as f64;
    let top1 = results.iter().map(|r| r.top1_hits).sum::<usize>() as f64 / scored as f64;
    let mut all_digest = Sha256::new();
    for r in &results {
        all_digest.update(r.logits_sha256.as_bytes());
    }
    let env: HashMap<String, String> = std::env::vars()
        .filter(|(k, _)| k.starts_with("HIPFIRE_"))
        .collect();
    let receipt = serde_json::json!({
        "schema": RECEIPT_SCHEMA,
        "kldseq": artifact_id(&output)?,
        "hfktok": artifact_id(&tok_path)?,
        "model": model_id,
        "reference": ref_id,
        "reference_header": {
            "teacher": h.teacher, "corpus": h.corpus, "n_ctx": h.n_ctx, "n_chunk": h.n_chunk,
            "window": h.window, "oracle": h.oracle, "arch": h.arch, "engine_commit": h.engine_commit,
        },
        "evaluator": {"binary": exe_id, "engine_commit": engine_commit()},
        "route": "hipfire_loader::load_model + reset_session_state + forward_token_gpu per position (product AR n=1, F16 KV)",
        "kv_dtype": format!("{:?}", hipfire_arch_inkling2::INKLING2_KV_DTYPE),
        "max_seq": hipfire_arch_inkling2::INKLING2_MAX_CONTEXT,
        "chunks_per_process": per_process,
        "env": env,
        "n_chunk": n_chunk,
        "scored": scored,
        "mean_kld": mean_kld,
        "mean_of_chunk_means": means.iter().sum::<f64>() / n_chunk as f64,
        "mean_nll": mean_nll,
        "ppl": mean_nll.exp(),
        "top1": top1,
        "logits_sha256": format!("{:x}", all_digest.finalize()),
        "chunks": results.iter().map(|r| serde_json::json!({
            "chunk": r.chunk, "mean_kld": r.mean_kld, "p99_kld": r.p99_kld, "mean_nll": r.mean_nll,
            "top1_hits": r.top1_hits, "scored": r.scored, "logits_sha256": r.logits_sha256,
            "seconds": r.seconds,
        })).collect::<Vec<_>>(),
        "seconds": started.elapsed().as_secs_f64(),
    });
    std::fs::write(
        with_suffix(&output, ".json"),
        serde_json::to_string_pretty(&receipt).map_err(|e| e.to_string())? + "\n",
    )
    .map_err(|e| e.to_string())?;
    std::fs::remove_dir_all(&shard_dir).map_err(|e| e.to_string())?;
    eprintln!(
        "inkling_kld eval: mean KLD = {mean_kld:.6}  mean NLL = {mean_nll:.6}  PPL = {:.4}  top1 = {top1:.4}  ({scored} tokens) -> {}",
        mean_nll.exp(),
        output.display()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// pair
// ---------------------------------------------------------------------------

/// Two-sided 95% Student-t quantile (Cornish-Fisher, |error| < 1e-5 at df >= 10).
fn t975(df: f64) -> f64 {
    let z = 1.959_963_984_540_054f64;
    let (z3, z5, z7) = (z.powi(3), z.powi(5), z.powi(7));
    z + (z3 + z) / (4.0 * df)
        + (5.0 * z5 + 16.0 * z3 + 3.0 * z) / (96.0 * df * df)
        + (3.0 * z7 + 19.0 * z5 + 17.0 * z3 - 15.0 * z) / (384.0 * df.powi(3))
}

fn paired_stats(base: &[f64], cand: &[f64], boot: usize, seed: u64) -> serde_json::Value {
    let n = base.len();
    let d: Vec<f64> = base.iter().zip(cand).map(|(b, c)| c - b).collect();
    let mean = d.iter().sum::<f64>() / n as f64;
    let sd = (d.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt();
    let se = sd / (n as f64).sqrt();
    let tc = t975(n as f64 - 1.0);
    let (_, blo, bhi) = kldseq::bootstrap_mean_ci(&d, boot, seed);
    serde_json::json!({
        "base_mean": base.iter().sum::<f64>() / n as f64,
        "cand_mean": cand.iter().sum::<f64>() / n as f64,
        "delta_mean": mean, "delta_sd": sd, "delta_se": se,
        "t_stat": if se > 0.0 { Some(mean / se) } else { None },
        "ci95_paired_t": [mean - tc * se, mean + tc * se],
        "ci95_bootstrap": [blo, bhi],
        "chunks_cand_lower": d.iter().filter(|x| **x < 0.0).count(),
        "chunks_cand_higher": d.iter().filter(|x| **x > 0.0).count(),
        "chunks_equal": d.iter().filter(|x| **x == 0.0).count(),
        "per_chunk_delta": d,
    })
}

fn pair(a: &Args) -> Result<(), String> {
    let base = a.path("base")?;
    let cand = a.path("cand")?;
    let boot = a.num("boot", 20000)?;
    let seed = a.num("seed", 1)? as u64;
    let receipt = |p: &Path| -> Result<serde_json::Value, String> {
        let r = with_suffix(p, ".json");
        let text =
            std::fs::read_to_string(&r).map_err(|e| format!("receipt {}: {e}", r.display()))?;
        let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        if v["schema"] != RECEIPT_SCHEMA {
            return Err(format!("{}: not a {RECEIPT_SCHEMA} receipt", r.display()));
        }
        Ok(v)
    };
    let (rb, rc) = (receipt(&base)?, receipt(&cand)?);
    if rb["reference"]["sha256"] != rc["reference"]["sha256"] {
        return Err("base and candidate were scored against different references".into());
    }
    let sb = kldseq::open(&base).map_err(|e| e.to_string())?;
    let sc = kldseq::open(&cand).map_err(|e| e.to_string())?;
    if sb.chunks.len() != sc.chunks.len() || sb.chunks.len() < 2 {
        return Err(format!(
            "chunk counts {} vs {}",
            sb.chunks.len(),
            sc.chunks.len()
        ));
    }
    let hb = artifact_id(&base)?.sha256;
    let hc = artifact_id(&cand)?.sha256;
    let col = |s: &kldseq::KldSeq, f: fn(&kldseq::ChunkScore) -> f64| {
        s.chunks.iter().map(f).collect::<Vec<_>>()
    };
    let out = serde_json::json!({
        "base": base.display().to_string(), "cand": cand.display().to_string(),
        "base_sha256": hb, "cand_sha256": hc, "byte_identical": hb == hc,
        "reference_sha256": rb["reference"]["sha256"],
        "base_model_sha256": rb["model"]["sha256"], "cand_model_sha256": rc["model"]["sha256"],
        "n_chunks": sb.chunks.len(),
        "bootstrap": {"resamples": boot, "seed": seed, "rng": "SplitMix64 (saddle_quant::format::kldseq::bootstrap_mean_ci)"},
        "kld": paired_stats(&col(&sb, |c| c.mean_kld), &col(&sc, |c| c.mean_kld), boot, seed),
        "p99_kld": paired_stats(&col(&sb, |c| c.p99_kld), &col(&sc, |c| c.p99_kld), boot, seed),
        "nll": paired_stats(&col(&sb, |c| c.mean_nll), &col(&sc, |c| c.mean_nll), boot, seed),
    });
    let text = serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?;
    if let Some(o) = a.0.get("output") {
        std::fs::write(o, text.clone() + "\n").map_err(|e| e.to_string())?;
    }
    println!("{text}");
    Ok(())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let cmd = args.next().unwrap_or_default();
    let result = Args::parse(args).and_then(|a| match cmd.as_str() {
        "ref" => build_ref(&a),
        "fidelity" => fidelity(&a),
        "eval" => eval(&a),
        "eval-worker" => eval_worker(&a),
        "pair" => pair(&a),
        _ => Err(USAGE.to_string()),
    });
    if let Err(e) = result {
        eprintln!("inkling_kld: {e}");
        std::process::exit(1);
    }
}
