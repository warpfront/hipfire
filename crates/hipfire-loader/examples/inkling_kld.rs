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
//! `eval` scores `--chunks-per-process` chunks (default 1) per fresh process.
//! The arch-24 GPU forward does not return its per-token controller scratch,
//! so VRAM grows with every position a process runs; two 2048-token chunks
//! per process fit an R9700 and give the same logits as one. It writes
//! `OUT.kldseq`, `OUT.kldseq.hfktok` (per-token KLD)
//! and `OUT.kldseq.json` (receipt: model/ref/binary SHA-256, route, per-chunk
//! scores and logits digests). Run with `HIPFIRE_GRAPH=0
//! HIPFIRE_NORMALIZE_PROMPT=0` and a fresh HOME, as the Qwen pins do.
//!
//! `pair` pairs chunk i of BASE with chunk i of CAND (both receipts must name
//! the same reference) and reports per-chunk deltas (cand − base; negative is
//! better), the paired-t 95% CI and a SplitMix64 percentile-bootstrap 95% CI,
//! for mean KLD, p99 KLD and NLL. Same JSON schema as the gfx1151 A4 receipts'
//! `paired.py`.
//!
//! `calib` runs the teacher over the spans of a calibration manifest (role
//! `calib`, or `check` for the held-out rows; every span an independent causal
//! sequence, nothing truncated) with a developer-only `CalibObserver` tapped
//! into `Teacher::visit`. Role `calib` accumulates `G = Σ x xᵀ` per HFHS name
//! (attention Q/K/V/AttnOut rows, per-expert selected gate/up rows and post-SiTU
//! down rows) in F64 on disk per layer, then writes `hfhs/L{LL}.hfhs` (F32
//! `H = G/N`) plus `L{LL}.json` (N, N per visit, payload hashes, route stats).
//! Role `check` dumps the raw F32 rows instead. `srcdump` writes the source F32
//! tensors of chosen layers as the quantizer ingests them; `captest` proves the
//! observer leaves the teacher's logits bit-identical and the Grams equal an
//! independent recomputation. The observer is a shared `&dyn` (the expert loop
//! is parallel); with `None` every output is bit-identical to the unobserved
//! teacher.

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
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

const USAGE: &str = "usage:
  inkling_kld ref      --source MODEL.safetensors --model ARTIFACT.hfq --slice TEXT --output REF.kldref
                       [--n-ctx 2048] [--max-chunks 24] [--top-k 256] [--batch-chunks N]
  inkling_kld fidelity --source MODEL.safetensors --model ARTIFACT.hfq --slice TEXT --output OUT.json
                       [--n-ctx 64] [--chunks 4] [--sliding-window W] [--log-floor F]
  inkling_kld calib    --source MODEL.safetensors --model ARTIFACT.hfq --manifest MANIFEST.json
                       --ids CALIB.u32 --out-dir DIR [--layers 5,6,20] [--batch-spans 8]
                       [--role calib|check]
  inkling_kld srcdump  --source MODEL.safetensors --model ARTIFACT.hfq --layers 5,20 --out-dir DIR
                       [--verify 1]
  inkling_kld captest  --source MODEL.safetensors --model ARTIFACT.hfq --slice TEXT --out-dir DIR
                       [--n-ctx 64] [--chunks 2] [--layers 5,6] [--sliding-window W] [--log-floor F]
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
    fn forward(
        &self,
        seqs: &[&[u32]],
        obs: Option<&dyn CalibObserver>,
    ) -> Result<Vec<Vec<f32>>, String> {
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
            self.visit(visit, len, &mut toks, None, obs)?;
            visit += 1;
            if obs.is_some_and(|o| o.finished_after(visit - 1)) {
                return Ok(Vec::new());
            }
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
                self.visit(visit, len, &mut toks, messages.as_ref(), obs)?;
                visit += 1;
                if obs.is_some_and(|o| o.finished_after(visit - 1)) {
                    return Ok(Vec::new());
                }
                eprintln!(
                    "  visit {visit}/{}  {:.0}s",
                    lc.n_slot(),
                    started.elapsed().as_secs_f64()
                );
            }
            prev.clone_from(&toks);
        }
        for _ in lc.core_end..cfg.n_phys {
            self.visit(visit, len, &mut toks, None, obs)?;
            visit += 1;
            if obs.is_some_and(|o| o.finished_after(visit - 1)) {
                return Ok(Vec::new());
            }
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
        obs: Option<&dyn CalibObserver>,
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
        // Capture taps: only layers the observer wants; rows are final when handed out.
        let watch = obs.filter(|o| o.wants(phys));
        let tok_meta: Vec<RowMeta> = if watch.is_some() {
            (0..n)
                .map(|i| RowMeta {
                    seq: (i / len) as u32,
                    pos: (i % len) as u32,
                    rank: 0,
                    weight: f32::NAN,
                    margin: f32::NAN,
                })
                .collect()
        } else {
            Vec::new()
        };

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
        if let Some(o) = watch {
            o.observe(phys, visit, CalibSite::Q, None, &q_inp, ne, &tok_meta)?;
            o.observe(phys, visit, CalibSite::K, None, &k_inp, ne, &tok_meta)?;
            o.observe(phys, visit, CalibSite::V, None, &v_inp, ne, &tok_meta)?;
        }
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
        if let Some(o) = watch {
            o.observe(
                phys,
                visit,
                CalibSite::AttnOut,
                None,
                &attn_out,
                eattn,
                &tok_meta,
            )?;
        }
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
        let margins: Vec<(f32, usize)> = if watch.is_some() {
            logits
                .par_chunks_exact(n_experts)
                .zip(selected.par_iter())
                .map(|(l, sel)| route_margin(l, bias, top_k, sel))
                .collect::<Result<_, String>>()?
        } else {
            Vec::new()
        };
        drop(logits);
        // by_expert[ex] lists (token, rank); slot[token * top_k + rank] is the
        // row of that pair in its expert's output.
        let (by_expert, slot) = route_tables(&selected, n_experts, top_k);
        if let Some(o) = watch {
            o.route(
                phys,
                visit,
                &RouteInfo {
                    top_k,
                    n_experts,
                    selected: &selected,
                    margins: &margins,
                },
            )?;
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
                let meta: Vec<RowMeta> = if watch.is_some() {
                    rows.iter()
                        .map(|&(i, rk)| RowMeta {
                            seq: (i / len) as u32,
                            pos: (i % len) as u32,
                            rank: rk as u8,
                            weight: selected[i][rk].1,
                            margin: margins[i].0,
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                if let Some(o) = watch {
                    o.observe(
                        phys,
                        visit,
                        CalibSite::ExpertGateUp,
                        Some(ex),
                        &xe,
                        ne,
                        &meta,
                    )?;
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
                if let Some(o) = watch {
                    o.observe(
                        phys,
                        visit,
                        CalibSite::ExpertDown,
                        Some(ex),
                        &act,
                        inter,
                        &meta,
                    )?;
                }
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
        if let Some(o) = watch {
            o.end_visit(phys, visit)?;
        }
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
// Calibration capture: observer API, F64 Gram accumulation, HFHS output
// ---------------------------------------------------------------------------

/// Where in a visit a row set is observed. `Q` is the normalized Q-stream row
/// that feeds `wq_du`, `wr_du` and `gate_attn` (one producer); `K`/`V` are the
/// separately controlled and normalized rows of `wk_dv`/`wv_dv`; `AttnOut` is
/// the gated attention output (the `wo_ud` input, K 3072); the expert sites
/// see only the tokens that selected that expert (never router-weighted).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum CalibSite {
    Q,
    K,
    V,
    AttnOut,
    ExpertGateUp,
    ExpertDown,
}

impl CalibSite {
    fn file_name(self) -> &'static str {
        match self {
            CalibSite::Q => "q",
            CalibSite::K => "k",
            CalibSite::V => "v",
            CalibSite::AttnOut => "attn_out",
            CalibSite::ExpertGateUp => "gate_up",
            CalibSite::ExpertDown => "down",
        }
    }
}

/// Provenance of one observed row. `seq`/`pos` are the sequence index within
/// the `forward` call and the token position; `rank`, `weight` and `margin`
/// are the route rank (0 = best), the renormalized route weight and the
/// 8th-vs-9th selection-score margin of that token at that visit. Attention
/// rows carry `rank 0` and NaN `weight`/`margin`.
#[derive(Clone, Copy, Debug)]
struct RowMeta {
    seq: u32,
    pos: u32,
    rank: u8,
    weight: f32,
    margin: f32,
}

/// One visit's routing decision for every token (token-major `seq * len + pos`).
struct RouteInfo<'a> {
    top_k: usize,
    n_experts: usize,
    selected: &'a [Vec<(usize, f32)>],
    /// `(selection-score margin between the top_k-th and the next expert,
    /// that next expert's id)` per token.
    margins: &'a [(f32, usize)],
}

/// Developer-only tap on the batched teacher. Capture never changes
/// selection, rows or state: the teacher hands out shared borrows after the
/// values are final. The expert sites are called from the teacher's
/// per-expert parallel loop (one call per expert and site per visit), so an
/// observer must be `Sync`; `end_visit` runs serially after the loop.
trait CalibObserver: Sync {
    fn wants(&self, phys: usize) -> bool;
    /// True when no later visit is needed (the teacher then returns early).
    fn finished_after(&self, _visit: usize) -> bool {
        false
    }
    fn route(&self, phys: usize, visit: usize, info: &RouteInfo) -> Result<(), String>;
    #[allow(clippy::too_many_arguments)]
    fn observe(
        &self,
        phys: usize,
        visit: usize,
        site: CalibSite,
        expert: Option<usize>,
        rows: &[f32],
        k: usize,
        meta: &[RowMeta],
    ) -> Result<(), String>;
    fn end_visit(&self, phys: usize, visit: usize) -> Result<(), String>;
}

/// Fans one teacher run out to several observers.
struct Tee<'a>(Vec<&'a dyn CalibObserver>);

impl CalibObserver for Tee<'_> {
    fn wants(&self, phys: usize) -> bool {
        self.0.iter().any(|o| o.wants(phys))
    }
    fn finished_after(&self, visit: usize) -> bool {
        self.0.iter().all(|o| o.finished_after(visit))
    }
    fn route(&self, phys: usize, visit: usize, info: &RouteInfo) -> Result<(), String> {
        for o in self.0.iter().filter(|o| o.wants(phys)) {
            o.route(phys, visit, info)?;
        }
        Ok(())
    }
    fn observe(
        &self,
        phys: usize,
        visit: usize,
        site: CalibSite,
        expert: Option<usize>,
        rows: &[f32],
        k: usize,
        meta: &[RowMeta],
    ) -> Result<(), String> {
        for o in self.0.iter().filter(|o| o.wants(phys)) {
            o.observe(phys, visit, site, expert, rows, k, meta)?;
        }
        Ok(())
    }
    fn end_visit(&self, phys: usize, visit: usize) -> Result<(), String> {
        for o in self.0.iter().filter(|o| o.wants(phys)) {
            o.end_visit(phys, visit)?;
        }
        Ok(())
    }
}

/// `by_expert[e]` lists `(token, rank)` and `slot[token * top_k + rank]` is the
/// row of that pair in its expert's output.
type RouteTables = (Vec<Vec<(usize, usize)>>, Vec<usize>);

fn route_tables(selected: &[Vec<(usize, f32)>], n_experts: usize, top_k: usize) -> RouteTables {
    let mut by_expert: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_experts];
    let mut slot = vec![0usize; selected.len() * top_k];
    for (i, sel) in selected.iter().enumerate() {
        for (rk, &(ex, _)) in sel.iter().enumerate() {
            slot[i * top_k + rk] = by_expert[ex].len();
            by_expert[ex].push((i, rk));
        }
    }
    (by_expert, slot)
}

/// Selection-score margin between the `k`-th and `(k+1)`-th expert of one
/// token and the `(k+1)`-th expert's id: the same scores and stable sort as
/// `oracle::sigmoid_topk`, and its top-`k` ids must equal `selected`.
fn route_margin(
    logits: &[f32],
    bias: Option<&[f32]>,
    k: usize,
    selected: &[(usize, f32)],
) -> Result<(f32, usize), String> {
    let mut sel: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &s)| (i, oracle::sigmoid(s) + bias.map_or(0.0, |v| v[i])))
        .collect();
    sel.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if sel.len() <= k || selected.len() != k {
        return Err("route margin: fewer than top_k + 1 experts".into());
    }
    if sel[..k]
        .iter()
        .map(|s| s.0)
        .ne(selected.iter().map(|s| s.0))
    {
        return Err(
            "route margin: recomputed top-k ids differ from the teacher's selection".into(),
        );
    }
    Ok((sel[k - 1].1 - sel[k].1, sel[k].0))
}

/// Rows per F32 Gram chunk. Within a chunk every product sum runs in `r`
/// order in F32 (no FMA contraction), each chunk's tile is then added to the
/// F64 accumulator in chunk order: bitwise independent of thread count and of
/// the vector ISA.
const GRAM_CHUNK: usize = 256;
/// Feature panel width: K must be a multiple of it (2304, 3072, 1152 are).
const PB: usize = 16;

/// Elements of the block-upper-triangular F64 accumulator of a `k x k` Gram:
/// block row `ib` stores its 16 rows from column `16 * ib` to `k`.
fn tri_len(k: usize) -> usize {
    let nb = k / PB;
    PB * (k * nb - PB * nb * nb.saturating_sub(1) / 2)
}

fn tri_row_off(k: usize, ib: usize) -> usize {
    PB * (k * ib - PB * ib * ib.saturating_sub(1) / 2)
}

#[inline(always)]
unsafe fn tile_body(pi: *const f32, pj: *const f32, rr: usize) -> [[f32; PB]; PB] {
    let mut acc = [[0.0f32; PB]; PB];
    for r in 0..rr {
        let xv = pj.add(r * PB);
        for (ii, a) in acc.iter_mut().enumerate() {
            let s = *pi.add(r * PB + ii);
            for (l, al) in a.iter_mut().enumerate() {
                *al = *al + s * *xv.add(l);
            }
        }
    }
    acc
}

#[target_feature(enable = "avx512f")]
unsafe fn tile_avx512(pi: *const f32, pj: *const f32, rr: usize) -> [[f32; PB]; PB] {
    tile_body(pi, pj, rr)
}

#[target_feature(enable = "avx2")]
unsafe fn tile_avx2(pi: *const f32, pj: *const f32, rr: usize) -> [[f32; PB]; PB] {
    tile_body(pi, pj, rr)
}

/// `acc[i][j] = Σ_r pi[r][i] * pj[r][j]` for one 16 x 16 tile of packed panels.
fn tile16(pi: &[f32], pj: &[f32], rr: usize) -> [[f32; PB]; PB] {
    assert!(pi.len() >= rr * PB && pj.len() >= rr * PB);
    // SAFETY: bounds asserted above; feature presence checked at runtime.
    unsafe {
        if std::arch::is_x86_feature_detected!("avx512f") {
            tile_avx512(pi.as_ptr(), pj.as_ptr(), rr)
        } else if std::arch::is_x86_feature_detected!("avx2") {
            tile_avx2(pi.as_ptr(), pj.as_ptr(), rr)
        } else {
            tile_body(pi.as_ptr(), pj.as_ptr(), rr)
        }
    }
}

/// `acc += Σ_rows x xᵀ` (block-upper triangle) for `n` rows of width `k`.
/// Deterministic: every accumulator element is owned by one task and is
/// updated in fixed chunk/row order.
fn gram_add(acc: &mut [f64], x: &[f32], n: usize, k: usize) {
    assert!(
        k % PB == 0 && k > 0,
        "gram: K {k} is not a multiple of {PB}"
    );
    assert_eq!(acc.len(), tri_len(k), "gram: accumulator size");
    assert_eq!(x.len(), n * k, "gram: row shape");
    let nb = k / PB;
    let mut blocks: Vec<&mut [f64]> = Vec::with_capacity(nb);
    let mut rest = acc;
    for ib in 0..nb {
        let (head, tail) = rest.split_at_mut(PB * (k - PB * ib));
        blocks.push(head);
        rest = tail;
    }
    let mut panels = vec![0.0f32; nb * GRAM_CHUNK.min(n.max(1)) * PB];
    for r0 in (0..n).step_by(GRAM_CHUNK) {
        let rr = (n - r0).min(GRAM_CHUNK);
        panels[..nb * rr * PB]
            .par_chunks_mut(rr * PB)
            .enumerate()
            .for_each(|(jb, dst)| {
                for r in 0..rr {
                    let src = (r0 + r) * k + jb * PB;
                    dst[r * PB..(r + 1) * PB].copy_from_slice(&x[src..src + PB]);
                }
            });
        let pan = &panels[..nb * rr * PB];
        blocks.par_iter_mut().enumerate().for_each(|(ib, blk)| {
            let pi = &pan[ib * rr * PB..(ib + 1) * rr * PB];
            let width = k - ib * PB;
            for jb in ib..nb {
                let pj = &pan[jb * rr * PB..(jb + 1) * rr * PB];
                let tile = tile16(pi, pj, rr);
                for (ii, trow) in tile.iter().enumerate() {
                    let base = ii * width + (jb - ib) * PB;
                    for (l, &v) in trow.iter().enumerate() {
                        blk[base + l] += v as f64;
                    }
                }
            }
        });
    }
}

/// Full row-major `k x k` F32 `H = G / n` from the block-upper accumulator.
fn hessian_from_tri(tri: &[f64], k: usize, n: u64) -> Vec<f32> {
    assert_eq!(tri.len(), tri_len(k));
    let nf = n as f64;
    let mut out = vec![0.0f32; k * k];
    out.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
        let (ib, ii) = (i / PB, i % PB);
        for (j, o) in row.iter_mut().enumerate() {
            let g = if j >= ib * PB {
                tri[tri_row_off(k, ib) + ii * (k - ib * PB) + (j - ib * PB)]
            } else {
                let (jb, jj) = (j / PB, j % PB);
                tri[tri_row_off(k, jb) + jj * (k - jb * PB) + (i - jb * PB)]
            };
            *o = (g / nf) as f32;
        }
    });
    out
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    #[cfg(not(target_endian = "little"))]
    compile_error!("HFHS payloads are written as raw little-endian floats");
    // SAFETY: any f32 slice is valid as bytes; little-endian asserted above.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

fn f64_bytes_mut(v: &mut [f64]) -> &mut [u8] {
    // SAFETY: any bit pattern is a valid f64; the slice covers exactly the vector.
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

/// One accumulator: tensor key (without `.weight` and layer prefix), K, and
/// where its block-upper F64 triangle lives in the layer's accumulator file.
struct Slot {
    k: usize,
    off: u64,
    elems: usize,
}

struct Layout {
    slots: Vec<Slot>,
    index: HashMap<String, usize>,
    bytes: u64,
}

impl Layout {
    fn new(n_experts: usize, ne: usize, eattn: usize, inter: usize) -> Self {
        let mut keys: Vec<(String, usize)> = vec![
            ("attn.wq_du".into(), ne),
            ("attn.wk_dv".into(), ne),
            ("attn.wv_dv".into(), ne),
            ("attn.wo_ud".into(), eattn),
        ];
        for e in 0..n_experts {
            keys.push((format!("mlp.experts.{e}.gate_up_proj"), ne));
            keys.push((format!("mlp.experts.{e}.down_proj"), inter));
        }
        let (mut slots, mut index, mut off) = (Vec::new(), HashMap::new(), 0u64);
        for (key, k) in keys {
            let elems = tri_len(k);
            index.insert(key.clone(), slots.len());
            slots.push(Slot { k, off, elems });
            off += elems as u64 * 8;
        }
        Self {
            slots,
            index,
            bytes: off,
        }
    }

    fn slot(&self, key: &str) -> Result<&Slot, String> {
        self.index
            .get(key)
            .map(|&i| &self.slots[i])
            .ok_or_else(|| format!("no accumulator slot {key}"))
    }
}

/// Slot key of an observed `(site, expert)`.
fn site_key(site: CalibSite, expert: Option<usize>) -> Result<String, String> {
    Ok(match (site, expert) {
        (CalibSite::Q, None) => "attn.wq_du".into(),
        (CalibSite::K, None) => "attn.wk_dv".into(),
        (CalibSite::V, None) => "attn.wv_dv".into(),
        (CalibSite::AttnOut, None) => "attn.wo_ud".into(),
        (CalibSite::ExpertGateUp, Some(e)) => format!("mlp.experts.{e}.gate_up_proj"),
        (CalibSite::ExpertDown, Some(e)) => format!("mlp.experts.{e}.down_proj"),
        (s, e) => return Err(format!("site {s:?} with expert {e:?}")),
    })
}

/// Pass index of a visit (0 outside the looped core).
fn pass_of(cfg: &Inkling2Config, visit: usize) -> usize {
    let lc = &cfg.loop_cfg;
    let core_len = lc.core_end - lc.core_start;
    if visit >= lc.core_start && visit < lc.core_start + lc.num_passes * core_len {
        (visit - lc.core_start) / core_len
    } else {
        0
    }
}

fn parse_layers(spec: Option<&String>, n_phys: usize) -> Result<Vec<usize>, String> {
    let Some(spec) = spec else {
        return Ok((0..n_phys).collect());
    };
    let mut out = BTreeSet::new();
    for part in spec.split(',').filter(|p| !p.is_empty()) {
        let l: usize = part
            .parse()
            .map_err(|_| format!("--layers: not a layer index: {part}"))?;
        if l >= n_phys {
            return Err(format!("--layers: {l} >= {n_phys} physical layers"));
        }
        out.insert(l);
    }
    if out.is_empty() {
        return Err("--layers: empty".into());
    }
    Ok(out.into_iter().collect())
}

/// Last visit index at which any of `layers` runs.
fn last_visit_of(cfg: &Inkling2Config, layers: &BTreeSet<usize>) -> usize {
    (0..cfg.loop_cfg.n_slot())
        .filter(|&v| layers.contains(&cfg.loop_cfg.layer_phys(v)))
        .max()
        .unwrap_or(0)
}

/// Per-expert route statistics of one physical layer, over every observed visit.
struct ExpertRoute {
    n: u64,
    by_visit: BTreeMap<usize, u64>,
    rank_hist: Vec<u64>,
    w_sum: f64,
    w_min: f32,
    /// Times this expert was the last selected one (rank top_k-1) of a token whose
    /// top_k-th vs (top_k+1)-th margin was below 1e-6.
    near_as_last: u64,
    /// Times this expert was the first unselected one of such a token.
    near_as_next: u64,
}

struct LayerRoute {
    top_k: usize,
    tokens_by_visit: BTreeMap<usize, u64>,
    experts: Vec<ExpertRoute>,
    near_tie_token_visits: u64,
    margin_min: f32,
}

/// F64 Gram accumulators on disk (one block-upper-triangle slot per name),
/// per-name/visit counts and route statistics.
struct GramObserver {
    cfg: Inkling2Config,
    layers: BTreeSet<usize>,
    early_stop: bool,
    last_visit: usize,
    layout: Layout,
    dir: PathBuf,
    files: HashMap<usize, std::fs::File>,
    counts: Mutex<BTreeMap<(usize, String), BTreeMap<usize, u64>>>,
    route: Mutex<BTreeMap<usize, LayerRoute>>,
    expert_gram_ns: AtomicU64,
    dense_gram_ns: AtomicU64,
    io_ns: AtomicU64,
    rows_seen: AtomicU64,
}

impl GramObserver {
    fn acc_path(dir: &Path, phys: usize) -> PathBuf {
        dir.join(format!("L{phys:02}.f64acc"))
    }

    fn new(
        cfg: &Inkling2Config,
        layers: &[usize],
        dir: &Path,
        n_experts: usize,
        early_stop: bool,
    ) -> Result<Self, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
        let layout = Layout::new(n_experts, cfg.n_embd, cfg.n_head * cfg.head_dim, cfg.inter);
        let mut files = HashMap::new();
        for &l in layers {
            let p = Self::acc_path(dir, l);
            let f = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&p)
                .map_err(|e| format!("create {}: {e}", p.display()))?;
            f.set_len(layout.bytes)
                .map_err(|e| format!("size {}: {e}", p.display()))?;
            files.insert(l, f);
        }
        let set: BTreeSet<usize> = layers.iter().copied().collect();
        Ok(Self {
            cfg: cfg.clone(),
            last_visit: last_visit_of(cfg, &set),
            layers: set,
            early_stop,
            layout,
            dir: dir.to_path_buf(),
            files,
            counts: Mutex::new(BTreeMap::new()),
            route: Mutex::new(BTreeMap::new()),
            expert_gram_ns: AtomicU64::new(0),
            dense_gram_ns: AtomicU64::new(0),
            io_ns: AtomicU64::new(0),
            rows_seen: AtomicU64::new(0),
        })
    }

    fn n_by_visit(&self, phys: usize, key: &str) -> BTreeMap<usize, u64> {
        self.counts
            .lock()
            .expect("counts lock")
            .get(&(phys, key.to_string()))
            .cloned()
            .unwrap_or_default()
    }

    /// `(N_by_visit, N)` of one accumulator name.
    fn n_of(&self, phys: usize, key: &str) -> (BTreeMap<usize, u64>, u64) {
        let by = self.n_by_visit(phys, key);
        let n = by.values().sum();
        (by, n)
    }

    /// Writes `hfhs/L{LL}.hfhs` + `L{LL}.json` for one layer from its accumulator
    /// file and deletes the accumulator. Returns the layer JSON.
    fn finalize_layer(
        &self,
        phys: usize,
        hfhs_dir: &Path,
        run: &serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        use std::io::Write as _;
        let file = self
            .files
            .get(&phys)
            .ok_or("finalize: layer not accumulated")?;
        let route = self.route.lock().expect("route lock");
        let lr = route
            .get(&phys)
            .ok_or_else(|| format!("layer {phys}: no routing observed"))?;
        let pfx = format!("model.llm.layers.{phys}.");
        let top_k = lr.top_k;
        let visits: Vec<usize> = lr.tokens_by_visit.keys().copied().collect();
        let tokens_visits: u64 = lr.tokens_by_visit.values().sum();
        // Σ_e N_e == top_k · tokens · visits, for both expert projections, and the
        // route statistics agree with the row counts of the gathered inputs.
        let mut sums = [0u64; 2];
        let mut expert_json = Vec::new();
        let total_n: u64 = lr.experts.iter().map(|e| e.n).sum();
        let total_w: f64 = lr.experts.iter().map(|e| e.w_sum).sum();
        for (e, er) in lr.experts.iter().enumerate() {
            let (gu_by, gu_n) = self.n_of(phys, &format!("mlp.experts.{e}.gate_up_proj"));
            let (dn_by, dn_n) = self.n_of(phys, &format!("mlp.experts.{e}.down_proj"));
            if gu_n != er.n || dn_n != er.n || gu_by != er.by_visit || dn_by != er.by_visit {
                return Err(format!(
                    "layer {phys} expert {e}: route count {} vs gate_up {gu_n} / down {dn_n} rows",
                    er.n
                ));
            }
            sums[0] += gu_n;
            sums[1] += dn_n;
            let mut per_pass = BTreeMap::new();
            for (&v, &c) in &er.by_visit {
                *per_pass.entry(pass_of(&self.cfg, v)).or_insert(0u64) += c;
            }
            expert_json.push(serde_json::json!({
                "expert": e, "N": er.n,
                "count_by_pass": per_pass,
                "count_by_visit": er.by_visit,
                "traffic_share": if total_n > 0 { er.n as f64 / total_n as f64 } else { 0.0 },
                "weight_share": if total_w > 0.0 { er.w_sum / total_w } else { 0.0 },
                "route_weight_min": if er.n > 0 { Some(er.w_min) } else { None },
                "route_weight_mean": if er.n > 0 { Some(er.w_sum / er.n as f64) } else { None },
                "rank_hist": er.rank_hist,
                "near_tie_as_last_selected": er.near_as_last,
                "near_tie_as_first_unselected": er.near_as_next,
            }));
        }
        let expected = top_k as u64 * tokens_visits;
        if sums[0] != expected || sums[1] != expected || total_n != expected {
            return Err(format!(
                "layer {phys}: Σ_e N_e gate_up {} down {} route {total_n} != top_k·tokens·visits {expected}",
                sums[0], sums[1]
            ));
        }

        // Record list: (full name, accumulator key, K).
        let mut records: Vec<(String, String, usize)> = Vec::new();
        let q_name = format!("{pfx}attn.wq_du");
        for (leaf, key) in [
            ("wq_du", "attn.wq_du"),
            ("wr_du", "attn.wq_du"),
            ("gate_attn", "attn.wq_du"),
            ("wk_dv", "attn.wk_dv"),
            ("wv_dv", "attn.wv_dv"),
            ("wo_ud", "attn.wo_ud"),
        ] {
            let k = self.layout.slot(key)?.k;
            records.push((format!("{pfx}attn.{leaf}"), key.into(), k));
        }
        for e in 0..lr.experts.len() {
            for leaf in ["gate_up_proj", "down_proj"] {
                let key = format!("mlp.experts.{e}.{leaf}");
                let k = self.layout.slot(&key)?.k;
                records.push((format!("{pfx}{key}"), key, k));
            }
        }
        let present: Vec<&(String, String, usize)> = records
            .iter()
            .filter(|(_, key, _)| self.n_of(phys, key).1 > 0)
            .collect();
        let missing: Vec<&str> = records
            .iter()
            .filter(|(_, key, _)| self.n_of(phys, key).1 == 0)
            .map(|(n, _, _)| n.as_str())
            .collect();

        std::fs::create_dir_all(hfhs_dir).map_err(|e| e.to_string())?;
        let path = hfhs_dir.join(format!("L{phys:02}.hfhs"));
        let mut w = std::io::BufWriter::with_capacity(
            8 << 20,
            std::fs::File::create(&path).map_err(|e| format!("create {}: {e}", path.display()))?,
        );
        let mut file_hash = Sha256::new();
        let mut put = |w: &mut std::io::BufWriter<std::fs::File>, b: &[u8]| -> Result<(), String> {
            file_hash.update(b);
            w.write_all(b).map_err(|e| e.to_string())
        };
        put(&mut w, b"HFHS")?;
        put(&mut w, &1u32.to_le_bytes())?;
        put(&mut w, &(present.len() as u64).to_le_bytes())?;
        put(&mut w, &0u64.to_le_bytes())?;
        let mut names_json = serde_json::Map::new();
        let mut cached: Option<(String, Vec<f32>, String)> = None; // (key, H, sha) of the last dense Q
        let mut io_s = 0.0f64;
        for (name, key, k) in &records {
            let (by, n) = self.n_of(phys, key);
            if n == 0 {
                names_json.insert(
                    name.clone(),
                    serde_json::json!({"K": k, "N": 0, "N_by_visit": by, "missing": true}),
                );
                continue;
            }
            let cache_hit = matches!(&cached, Some((ck, _, _)) if ck == key);
            let mut fresh: Option<(Vec<f32>, String)> = None;
            if !cache_hit {
                let slot = self.layout.slot(key)?;
                let t = Instant::now();
                let mut tri = vec![0.0f64; slot.elems];
                file.read_exact_at(f64_bytes_mut(&mut tri), slot.off)
                    .map_err(|e| format!("read accumulator {key}: {e}"))?;
                io_s += t.elapsed().as_secs_f64();
                let h = hessian_from_tri(&tri, *k, n);
                drop(tri);
                if let Some(i) = h.iter().position(|v| !v.is_finite()) {
                    return Err(format!("{name}: non-finite H entry at {i}"));
                }
                if let Some(i) = (0..*k).find(|&i| h[i * k + i] < 0.0) {
                    return Err(format!("{name}: negative diagonal at {i}"));
                }
                let sha = format!("{:x}", Sha256::digest(f32_bytes(&h)));
                if key.starts_with("attn.") {
                    cached = Some((key.clone(), h, sha));
                } else {
                    fresh = Some((h, sha));
                }
            }
            // Dense records may alias the Q producer's cached matrix.
            let (h, sha): (&[f32], &str) = match (&fresh, &cached) {
                (Some((h, s)), _) => (h.as_slice(), s.as_str()),
                (None, Some((_, h, s))) => (h.as_slice(), s.as_str()),
                (None, None) => return Err("finalize: no matrix".into()),
            };
            put_record(&mut put, &mut w, name, *k, h)?;
            let diag: Vec<f64> = (0..*k).map(|i| h[i * k + i] as f64).collect();
            let alias =
                (key == "attn.wq_du" && !name.ends_with(".wq_du")).then_some(q_name.as_str());
            names_json.insert(name.clone(), record_json(*k, n, &by, sha, &diag, alias));
        }
        w.flush().map_err(|e| e.to_string())?;
        let file_sha = format!("{:x}", file_hash.finalize());
        let bytes = std::fs::metadata(&path).map_err(|e| e.to_string())?.len();
        let mut producer_map = serde_json::Map::new();
        for leaf in ["wr_du", "gate_attn"] {
            producer_map.insert(
                format!("{pfx}attn.{leaf}"),
                serde_json::Value::String(q_name.clone()),
            );
        }
        let json = serde_json::json!({
            "schema": "hipfire.inkling2.hfhs_layer.v1",
            "layer": phys,
            "hfhs": path.file_name().map(|s| s.to_string_lossy().to_string()),
            "hfhs_bytes": bytes, "hfhs_sha256": file_sha,
            "hfhs_format": "HFHS v1, F32 K x K of H = G/N, expert_idx 0, name = tensor name minus .weight",
            "n_records": present.len(),
            "missing_records": missing,
            "accumulation": format!("G = sum x x^T: F32 within {GRAM_CHUNK}-row chunks (no FMA, r order), chunks summed in F64 in chunk order; H = G/N in F64 then rounded to F32"),
            "run": run,
            "visits": visits,
            "producer_map": producer_map,
            "names": names_json,
            "route": {
                "top_k": top_k,
                "tokens_by_visit": lr.tokens_by_visit,
                "token_visits": tokens_visits,
                "sum_N_expected_top_k_tokens_visits": expected,
                "sum_N_gate_up": sums[0], "sum_N_down": sums[1],
                "near_tie_threshold": 1e-6,
                "near_tie_token_visits": lr.near_tie_token_visits,
                "margin_min": lr.margin_min,
                "experts": expert_json,
            },
            "finalize_read_seconds": io_s,
        });
        std::fs::write(
            hfhs_dir.join(format!("L{phys:02}.json")),
            serde_json::to_string_pretty(&json).map_err(|e| e.to_string())? + "\n",
        )
        .map_err(|e| e.to_string())?;
        std::fs::remove_file(Self::acc_path(&self.dir, phys)).ok();
        Ok(json)
    }
}

fn put_record(
    put: &mut impl FnMut(&mut std::io::BufWriter<std::fs::File>, &[u8]) -> Result<(), String>,
    w: &mut std::io::BufWriter<std::fs::File>,
    name: &str,
    k: usize,
    h: &[f32],
) -> Result<(), String> {
    put(w, &(name.len() as u32).to_le_bytes())?;
    put(w, name.as_bytes())?;
    put(w, &0u32.to_le_bytes())?;
    put(w, &(k as u32).to_le_bytes())?;
    put(w, &1u32.to_le_bytes())?;
    put(w, f32_bytes(h))
}

fn record_json(
    k: usize,
    n: u64,
    by: &BTreeMap<usize, u64>,
    sha: &str,
    diag: &[f64],
    alias_of: Option<&str>,
) -> serde_json::Value {
    let mean = diag.iter().sum::<f64>() / diag.len() as f64;
    let min = diag.iter().copied().fold(f64::INFINITY, f64::min);
    serde_json::json!({
        "K": k, "N": n, "N_by_visit": by, "payload_sha256": sha,
        "diag_mean": mean, "diag_min": min, "alias_of": alias_of,
    })
}

impl CalibObserver for GramObserver {
    fn wants(&self, phys: usize) -> bool {
        self.layers.contains(&phys)
    }

    fn finished_after(&self, visit: usize) -> bool {
        self.early_stop && visit >= self.last_visit
    }

    fn route(&self, phys: usize, visit: usize, info: &RouteInfo) -> Result<(), String> {
        let mut map = self.route.lock().expect("route lock");
        let lr = map.entry(phys).or_insert_with(|| LayerRoute {
            top_k: info.top_k,
            tokens_by_visit: BTreeMap::new(),
            experts: (0..info.n_experts)
                .map(|_| ExpertRoute {
                    n: 0,
                    by_visit: BTreeMap::new(),
                    rank_hist: vec![0; info.top_k],
                    w_sum: 0.0,
                    w_min: f32::INFINITY,
                    near_as_last: 0,
                    near_as_next: 0,
                })
                .collect(),
            near_tie_token_visits: 0,
            margin_min: f32::INFINITY,
        });
        if lr.top_k != info.top_k || lr.experts.len() != info.n_experts {
            return Err("route: top_k or expert count changed between visits".into());
        }
        *lr.tokens_by_visit.entry(visit).or_insert(0) += info.selected.len() as u64;
        for (i, sel) in info.selected.iter().enumerate() {
            let (margin, next) = info.margins[i];
            let tie = margin < 1e-6;
            lr.margin_min = lr.margin_min.min(margin);
            if tie {
                lr.near_tie_token_visits += 1;
                if next < info.n_experts {
                    lr.experts[next].near_as_next += 1;
                }
            }
            for (rk, &(ex, w)) in sel.iter().enumerate() {
                let er = &mut lr.experts[ex];
                er.n += 1;
                *er.by_visit.entry(visit).or_insert(0) += 1;
                er.rank_hist[rk] += 1;
                er.w_sum += w as f64;
                er.w_min = er.w_min.min(w);
                if tie && rk == info.top_k - 1 {
                    er.near_as_last += 1;
                }
            }
        }
        Ok(())
    }

    fn observe(
        &self,
        phys: usize,
        visit: usize,
        site: CalibSite,
        expert: Option<usize>,
        rows: &[f32],
        k: usize,
        meta: &[RowMeta],
    ) -> Result<(), String> {
        let file = self
            .files
            .get(&phys)
            .ok_or_else(|| format!("observe: layer {phys} is not accumulated"))?;
        let key = site_key(site, expert)?;
        let slot = self.layout.slot(&key)?;
        let n = rows.len() / k;
        if slot.k != k || rows.len() != n * k || meta.len() != n {
            return Err(format!(
                "{key}: K {k} / rows {} / meta {} do not match the slot",
                rows.len(),
                meta.len()
            ));
        }
        if n == 0 {
            return Ok(());
        }
        let t0 = Instant::now();
        let mut buf = vec![0.0f64; slot.elems];
        file.read_exact_at(f64_bytes_mut(&mut buf), slot.off)
            .map_err(|e| format!("read accumulator {key}: {e}"))?;
        let t1 = Instant::now();
        gram_add(&mut buf, rows, n, k);
        let t2 = Instant::now();
        file.write_all_at(f64_bytes_mut(&mut buf), slot.off)
            .map_err(|e| format!("write accumulator {key}: {e}"))?;
        let t3 = Instant::now();
        let gram = if expert.is_some() {
            &self.expert_gram_ns
        } else {
            &self.dense_gram_ns
        };
        gram.fetch_add((t2 - t1).as_nanos() as u64, Ordering::Relaxed);
        self.io_ns
            .fetch_add(((t1 - t0) + (t3 - t2)).as_nanos() as u64, Ordering::Relaxed);
        self.rows_seen.fetch_add(n as u64, Ordering::Relaxed);
        *self
            .counts
            .lock()
            .expect("counts lock")
            .entry((phys, key))
            .or_default()
            .entry(visit)
            .or_insert(0) += n as u64;
        Ok(())
    }

    fn end_visit(&self, _phys: usize, _visit: usize) -> Result<(), String> {
        Ok(())
    }
}

/// Raw F32 rows per layer/site for the held-out reconstruction metric.
struct CheckObserver {
    layers: BTreeSet<usize>,
    last_visit: usize,
    early_stop: bool,
    cfg: Inkling2Config,
    dir: PathBuf,
    span_ids: Mutex<Vec<String>>,
    pending: Mutex<Vec<CheckRows>>,
    state: Mutex<BTreeMap<(usize, CalibSite), CheckFile>>,
}

struct CheckRows {
    phys: usize,
    visit: usize,
    site: CalibSite,
    expert: Option<usize>,
    k: usize,
    rows: Vec<f32>,
    meta: Vec<RowMeta>,
    spans: Vec<String>,
}

struct CheckFile {
    file: std::io::BufWriter<std::fs::File>,
    k: usize,
    rows: u64,
    segments: Vec<serde_json::Value>,
}

impl CheckObserver {
    fn new(
        cfg: &Inkling2Config,
        layers: &[usize],
        dir: &Path,
        early_stop: bool,
    ) -> Result<Self, String> {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let set: BTreeSet<usize> = layers.iter().copied().collect();
        Ok(Self {
            last_visit: last_visit_of(cfg, &set),
            layers: set,
            early_stop,
            cfg: cfg.clone(),
            dir: dir.to_path_buf(),
            span_ids: Mutex::new(Vec::new()),
            pending: Mutex::new(Vec::new()),
            state: Mutex::new(BTreeMap::new()),
        })
    }

    /// Names (manifest ids) of the sequences of the next `forward` call.
    fn set_batch(&self, ids: Vec<String>) {
        *self.span_ids.lock().expect("span lock") = ids;
    }

    /// Writes the per-layer index JSON files; returns `(path, sha256)` of each.
    fn finish(&self) -> Result<Vec<(PathBuf, String)>, String> {
        use std::io::Write as _;
        let mut state = self.state.lock().expect("state lock");
        let mut by_layer: BTreeMap<usize, serde_json::Map<String, serde_json::Value>> =
            BTreeMap::new();
        for ((phys, site), cf) in state.iter_mut() {
            cf.file.flush().map_err(|e| e.to_string())?;
            by_layer.entry(*phys).or_default().insert(
                site.file_name().to_string(),
                serde_json::json!({
                    "file": format!("L{phys:02}.{}.f32", site.file_name()),
                    "K": cf.k, "rows": cf.rows, "segments": cf.segments,
                }),
            );
        }
        let mut out = Vec::new();
        for (phys, sites) in by_layer {
            let path = self.dir.join(format!("L{phys:02}.index.json"));
            let body = serde_json::to_string_pretty(&serde_json::json!({
                "schema": "hipfire.inkling2.check_rows.v1",
                "layer": phys,
                "dtype": "F32 little-endian, row-major, rows x K",
                "sites": sites,
            }))
            .map_err(|e| e.to_string())?
                + "\n";
            std::fs::write(&path, &body).map_err(|e| e.to_string())?;
            out.push((path, format!("{:x}", Sha256::digest(body.as_bytes()))));
        }
        Ok(out)
    }
}

impl CalibObserver for CheckObserver {
    fn wants(&self, phys: usize) -> bool {
        self.layers.contains(&phys)
    }

    fn finished_after(&self, visit: usize) -> bool {
        self.early_stop && visit >= self.last_visit
    }

    fn route(&self, _phys: usize, _visit: usize, _info: &RouteInfo) -> Result<(), String> {
        Ok(())
    }

    fn observe(
        &self,
        phys: usize,
        visit: usize,
        site: CalibSite,
        expert: Option<usize>,
        rows: &[f32],
        k: usize,
        meta: &[RowMeta],
    ) -> Result<(), String> {
        if rows.len() != meta.len() * k {
            return Err("check observe: rows/meta mismatch".into());
        }
        let spans = self.span_ids.lock().expect("span lock").clone();
        self.pending.lock().expect("pending lock").push(CheckRows {
            phys,
            visit,
            site,
            expert,
            k,
            rows: rows.to_vec(),
            meta: meta.to_vec(),
            spans,
        });
        Ok(())
    }

    fn end_visit(&self, phys: usize, visit: usize) -> Result<(), String> {
        use std::io::Write as _;
        let mut mine: Vec<CheckRows> = {
            let mut p = self.pending.lock().expect("pending lock");
            let (mine, rest): (Vec<_>, Vec<_>) = p
                .drain(..)
                .partition(|c| c.phys == phys && c.visit == visit);
            *p = rest;
            mine
        };
        mine.sort_by_key(|c| (c.site, c.expert));
        let mut state = self.state.lock().expect("state lock");
        for c in mine {
            if !state.contains_key(&(phys, c.site)) {
                let p = self
                    .dir
                    .join(format!("L{phys:02}.{}.f32", c.site.file_name()));
                let f = std::fs::File::create(&p)
                    .map_err(|e| format!("create {}: {e}", p.display()))?;
                state.insert(
                    (phys, c.site),
                    CheckFile {
                        file: std::io::BufWriter::with_capacity(8 << 20, f),
                        k: c.k,
                        rows: 0,
                        segments: Vec::new(),
                    },
                );
            }
            let cf = state.get_mut(&(phys, c.site)).expect("inserted");
            if cf.k != c.k {
                return Err("check observe: K changed within a site".into());
            }
            cf.file
                .write_all(f32_bytes(&c.rows))
                .map_err(|e| e.to_string())?;
            let n = c.meta.len();
            cf.segments.push(serde_json::json!({
                "visit": c.visit, "pass": pass_of(&self.cfg, c.visit), "expert": c.expert,
                "row0": cf.rows, "rows": n,
                "span": c.meta.iter().map(|m| c.spans.get(m.seq as usize).cloned().unwrap_or_default()).collect::<Vec<_>>(),
                "pos": c.meta.iter().map(|m| m.pos).collect::<Vec<_>>(),
                "rank": c.expert.map(|_| c.meta.iter().map(|m| m.rank).collect::<Vec<_>>()),
                "weight": c.expert.map(|_| c.meta.iter().map(|m| m.weight).collect::<Vec<_>>()),
                "margin": c.expert.map(|_| c.meta.iter().map(|m| m.margin).collect::<Vec<_>>()),
            }));
            cf.rows += n as u64;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Run bookkeeping
// ---------------------------------------------------------------------------

/// `(VmRSS, VmHWM)` of this process in KiB.
fn rss_kib() -> (u64, u64) {
    let mut out = (0, 0);
    if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
        for line in s.lines() {
            let kib = |l: &str| {
                l.split_whitespace()
                    .nth(1)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0)
            };
            if line.starts_with("VmRSS:") {
                out.0 = kib(line);
            } else if line.starts_with("VmHWM:") {
                out.1 = kib(line);
            }
        }
    }
    out
}

/// Bytes actually allocated under `dir` (sparse files count their blocks).
fn disk_bytes(dir: &Path) -> u64 {
    use std::os::unix::fs::MetadataExt;
    let mut total = 0u64;
    let Ok(rd) = std::fs::read_dir(dir) else {
        return 0;
    };
    for ent in rd.flatten() {
        let Ok(md) = ent.metadata() else { continue };
        total += if md.is_dir() {
            disk_bytes(&ent.path())
        } else {
            md.blocks() * 512
        };
    }
    total
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

/// One calibration-manifest span: tokens `ids[offset .. offset + len]`.
struct Span {
    id: String,
    source: String,
    offset: usize,
    len: usize,
    role: String,
}

/// Parses `calib-manifest.json` (an array, or `{"spans": [...]}`, of
/// `{id, source, offset, len, role}`; `offset` counts u32 tokens into the ids
/// file) and returns the spans of `role` in manifest order plus the ids.
fn load_spans(
    manifest: &Path,
    ids_path: &Path,
    role: &str,
    vocab: usize,
) -> Result<(Vec<Span>, Vec<u32>), String> {
    let text = std::fs::read_to_string(manifest)
        .map_err(|e| format!("read {}: {e}", manifest.display()))?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("manifest: {e}"))?;
    let arr = match &v {
        serde_json::Value::Array(a) => a,
        serde_json::Value::Object(o) => o
            .get("spans")
            .and_then(|s| s.as_array())
            .ok_or("manifest: no `spans` array")?,
        _ => return Err("manifest: expected an array or {spans: []}".into()),
    };
    let bytes = std::fs::read(ids_path).map_err(|e| format!("read {}: {e}", ids_path.display()))?;
    if bytes.len() % 4 != 0 {
        return Err("ids file is not a whole number of u32".into());
    }
    let ids: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut spans = Vec::new();
    for (i, s) in arr.iter().enumerate() {
        let get_str = |k: &str| match s.get(k) {
            Some(serde_json::Value::String(x)) => x.clone(),
            Some(x) => x.to_string(),
            None => String::new(),
        };
        let num = |k: &str| {
            s.get(k)
                .and_then(|x| x.as_u64())
                .ok_or_else(|| format!("manifest span {i}: missing numeric `{k}`"))
        };
        let (offset, len) = (num("offset")? as usize, num("len")? as usize);
        if get_str("role") != role {
            continue;
        }
        if len == 0 || offset + len > ids.len() {
            return Err(format!(
                "manifest span {i}: [{offset}, +{len}) outside the {} ids",
                ids.len()
            ));
        }
        if let Some(bad) = ids[offset..offset + len]
            .iter()
            .find(|&&t| t as usize >= vocab)
        {
            return Err(format!(
                "manifest span {i}: token id {bad} >= vocab {vocab}"
            ));
        }
        spans.push(Span {
            id: get_str("id"),
            source: get_str("source"),
            offset,
            len,
            role: role.to_string(),
        });
    }
    if spans.is_empty() {
        return Err(format!("manifest: no spans with role `{role}`"));
    }
    Ok((spans, ids))
}

/// Batches of span indices: equal length within a batch (the teacher's rule),
/// ascending length, manifest order within a length, at most `batch` spans.
fn plan_batches(spans: &[Span], batch: usize) -> Vec<Vec<usize>> {
    let mut by_len: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, s) in spans.iter().enumerate() {
        by_len.entry(s.len).or_default().push(i);
    }
    by_len
        .into_values()
        .flat_map(|v| v.chunks(batch).map(|c| c.to_vec()).collect::<Vec<_>>())
        .collect()
}

fn config_of(model: &Path) -> Result<Inkling2Config, String> {
    let hfq = HfqFile::open(model).map_err(|e| format!("open {}: {e}", model.display()))?;
    Inkling2Config::from_metadata_json(&hfq.metadata_json)
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
        let hidden = teacher.forward(&seqs, None)?;
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
    let hidden = teacher.forward(&seqs, None)?;
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

// ---------------------------------------------------------------------------
// calib (Hessian / check-row capture over the calibration manifest)
// ---------------------------------------------------------------------------

fn calib(a: &Args) -> Result<(), String> {
    let t_all = Instant::now();
    let source = a.path("source")?;
    let model = a.path("model")?;
    let manifest = a.path("manifest")?;
    let ids_path = a.path("ids")?;
    let out_dir = a.path("out-dir")?;
    let role =
        a.0.get("role")
            .cloned()
            .unwrap_or_else(|| "calib".to_string());
    if role != "calib" && role != "check" {
        return Err(format!("--role: `calib` or `check`, not {role}"));
    }
    let batch_spans = a.num("batch-spans", 8)?.max(1);
    let cfg = config_of(&model)?;
    let layers = parse_layers(a.0.get("layers"), cfg.n_phys)?;
    let (spans, ids) = load_spans(&manifest, &ids_path, &role, cfg.vocab_size)?;
    let batches = plan_batches(&spans, batch_spans);
    let total_tokens: usize = spans.iter().map(|s| s.len).sum();
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("mkdir {}: {e}", out_dir.display()))?;
    eprintln!(
        "inkling_kld calib: role {role}, {} spans / {total_tokens} tokens in {} batches (<= {batch_spans} spans), layers {layers:?}",
        spans.len(),
        batches.len()
    );
    let t0 = Instant::now();
    let w = SourceWeights::open(&source)?;
    let load_s = t0.elapsed().as_secs_f64();
    let (rss0, _) = rss_kib();
    eprintln!(
        "inkling_kld calib: source dense F32 ready ({} tensors) in {load_s:.0}s, RSS {:.1} GiB",
        w.dense.len(),
        rss0 as f64 / (1u64 << 20) as f64
    );
    let teacher = Teacher { w: &w, cfg: &cfg };
    let n_experts = w
        .dense(&format!("model.llm.layers.{}.mlp.gate.weight", layers[0]))?
        .len()
        / cfg.n_embd;
    let run = serde_json::json!({
        "role": role,
        "manifest": artifact_id(&manifest)?,
        "ids": artifact_id(&ids_path)?,
        "source": {"path": source.display().to_string(),
                   "bytes": std::fs::metadata(&source).map_err(|e| e.to_string())?.len()},
        "model_for_config_and_tokenizer": model.display().to_string(),
        "binary": artifact_id(&std::env::current_exe().map_err(|e| e.to_string())?)?,
        "n_spans": spans.len(), "tokens": total_tokens, "batch_spans": batch_spans,
        "spans": spans.iter().map(|s| serde_json::json!({
            "id": s.id, "source": s.source, "offset": s.offset, "len": s.len, "role": s.role,
        })).collect::<Vec<_>>(),
        "layers": layers,
        "engine_commit": engine_commit(),
    });
    let mut disk_peak = 0u64;
    let mut batch_log = Vec::new();
    let t_fwd = Instant::now();
    let visits_run;
    let gram_obs;
    let check_obs;
    match role.as_str() {
        "calib" => {
            gram_obs = Some(GramObserver::new(
                &cfg,
                &layers,
                &out_dir.join("acc"),
                n_experts,
                true,
            )?);
            check_obs = None;
            visits_run = gram_obs.as_ref().map_or(0, |o| o.last_visit + 1);
        }
        _ => {
            gram_obs = None;
            check_obs = Some(CheckObserver::new(
                &cfg,
                &layers,
                &out_dir.join("check"),
                true,
            )?);
            visits_run = check_obs.as_ref().map_or(0, |o| o.last_visit + 1);
        }
    }
    let obs: &dyn CalibObserver = match (&gram_obs, &check_obs) {
        (Some(g), _) => g,
        (_, Some(c)) => c,
        _ => unreachable!("one observer"),
    };
    for (bi, b) in batches.iter().enumerate() {
        let seqs: Vec<&[u32]> = b
            .iter()
            .map(|&i| &ids[spans[i].offset..spans[i].offset + spans[i].len])
            .collect();
        if let Some(c) = &check_obs {
            c.set_batch(b.iter().map(|&i| spans[i].id.clone()).collect());
        }
        let tb = Instant::now();
        teacher.forward(&seqs, Some(obs))?;
        let (rss, hwm) = rss_kib();
        let disk = disk_bytes(&out_dir);
        disk_peak = disk_peak.max(disk);
        let secs = tb.elapsed().as_secs_f64();
        let toks: usize = b.iter().map(|&i| spans[i].len).sum();
        eprintln!(
            "inkling_kld calib: batch {}/{} ({} spans, {toks} tokens) {secs:.0}s ({:.1} s/1K tokens), RSS {:.1} GiB (peak {:.1}), disk {:.1} GiB",
            bi + 1,
            batches.len(),
            b.len(),
            secs / (toks as f64 / 1000.0),
            rss as f64 / (1u64 << 20) as f64,
            hwm as f64 / (1u64 << 20) as f64,
            gib(disk)
        );
        batch_log.push(serde_json::json!({
            "batch": bi, "spans": b.len(), "tokens": toks, "seconds": secs,
            "seconds_per_1k_tokens": secs / (toks as f64 / 1000.0),
            "rss_kib": rss, "rss_hwm_kib": hwm, "disk_bytes": disk,
        }));
    }
    let forward_s = t_fwd.elapsed().as_secs_f64();

    let t_fin = Instant::now();
    let mut outputs = Vec::new();
    if let Some(g) = &gram_obs {
        let hfhs_dir = out_dir.join("hfhs");
        for &l in &layers {
            let tl = Instant::now();
            let j = g.finalize_layer(l, &hfhs_dir, &run)?;
            let disk = disk_bytes(&out_dir);
            disk_peak = disk_peak.max(disk);
            let missing = j["missing_records"].as_array().map_or(0, Vec::len);
            eprintln!(
                "inkling_kld calib: layer {l} -> {} ({:.2} GiB, sha256 {}), {missing} records without data, {:.0}s, disk {:.1} GiB",
                j["hfhs"].as_str().unwrap_or("?"),
                gib(j["hfhs_bytes"].as_u64().unwrap_or(0)),
                j["hfhs_sha256"].as_str().unwrap_or("?"),
                tl.elapsed().as_secs_f64(),
                gib(disk)
            );
            outputs.push(serde_json::json!({
                "layer": l, "hfhs_bytes": j["hfhs_bytes"], "hfhs_sha256": j["hfhs_sha256"],
                "missing_records": missing,
            }));
        }
        std::fs::remove_dir_all(out_dir.join("acc")).ok();
    }
    if let Some(c) = &check_obs {
        for (p, sha) in c.finish()? {
            outputs.push(serde_json::json!({"index": p.display().to_string(), "sha256": sha}));
        }
        disk_peak = disk_peak.max(disk_bytes(&out_dir));
    }
    let finalize_s = t_fin.elapsed().as_secs_f64();
    let (_, hwm) = rss_kib();
    let ns = |x: &AtomicU64| x.load(Ordering::Relaxed) as f64 / 1e9;
    let summary = serde_json::json!({
        "schema": "hipfire.inkling2.calib_run.v1",
        "run": run,
        "visits_run": visits_run,
        "phases": {
            "load_source_s": load_s,
            "forward_wall_s": forward_s,
            "finalize_s": finalize_s,
            "total_s": t_all.elapsed().as_secs_f64(),
            "gram_expert_thread_s": gram_obs.as_ref().map(|g| ns(&g.expert_gram_ns)),
            "gram_dense_wall_s": gram_obs.as_ref().map(|g| ns(&g.dense_gram_ns)),
            "accumulator_io_thread_s": gram_obs.as_ref().map(|g| ns(&g.io_ns)),
            "rows_accumulated": gram_obs.as_ref().map(|g| g.rows_seen.load(Ordering::Relaxed)),
            "threads": rayon::current_num_threads(),
        },
        "forward_s_per_1k_tokens": forward_s / (total_tokens as f64 / 1000.0),
        "rss_hwm_kib": hwm,
        "disk_peak_bytes": disk_peak,
        "batches": batch_log,
        "outputs": outputs,
    });
    std::fs::write(
        out_dir.join(format!("calib-run-{role}.json")),
        serde_json::to_string_pretty(&summary).map_err(|e| e.to_string())? + "\n",
    )
    .map_err(|e| e.to_string())?;
    eprintln!(
        "inkling_kld calib: done in {:.0}s (forward {forward_s:.0}s, finalize {finalize_s:.0}s), peak RSS {:.1} GiB, peak disk {:.1} GiB",
        t_all.elapsed().as_secs_f64(),
        hwm as f64 / (1u64 << 20) as f64,
        gib(disk_peak)
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// srcdump (the quantizer's source F32 tensors, for the offline solver)
// ---------------------------------------------------------------------------

fn srcdump(a: &Args) -> Result<(), String> {
    let source = a.path("source")?;
    let model = a.path("model")?;
    let out_dir = a.path("out-dir")?;
    let verify = a.num("verify", 1)? != 0;
    let cfg = config_of(&model)?;
    let layers = parse_layers(a.0.get("layers"), cfg.n_phys)?;
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("mkdir {}: {e}", out_dir.display()))?;
    let t0 = Instant::now();
    let w = SourceWeights::open(&source)?;
    eprintln!(
        "inkling_kld srcdump: source ready in {:.0}s",
        t0.elapsed().as_secs_f64()
    );
    let n_experts = w
        .dense(&format!("model.llm.layers.{}.mlp.gate.weight", layers[0]))?
        .len()
        / cfg.n_embd;
    for &l in &layers {
        use std::io::Write as _;
        let tl = Instant::now();
        let pfx = format!("model.llm.layers.{l}.");
        let path = out_dir.join(format!("L{l:02}.f32"));
        let mut bw = std::io::BufWriter::with_capacity(
            8 << 20,
            std::fs::File::create(&path).map_err(|e| format!("create {}: {e}", path.display()))?,
        );
        let mut hash = Sha256::new();
        let mut off = 0u64;
        let mut tensors: Vec<serde_json::Value> = Vec::new();
        let mut emit = |bw: &mut std::io::BufWriter<std::fs::File>,
                        name: String,
                        shape: Vec<usize>,
                        v: &[f32]|
         -> Result<(), String> {
            if shape.iter().product::<usize>() != v.len() {
                return Err(format!("{name}: shape {shape:?} vs {} elements", v.len()));
            }
            let b = f32_bytes(v);
            hash.update(b);
            bw.write_all(b).map_err(|e| e.to_string())?;
            tensors.push(serde_json::json!({
                "name": name, "offset_bytes": off, "shape": shape, "numel": v.len(),
            }));
            off += b.len() as u64;
            Ok(())
        };
        for leaf in ["wq_du", "wk_dv", "wv_dv", "wr_du", "gate_attn", "wo_ud"] {
            let name = format!("{pfx}attn.{leaf}.weight");
            let (shape, v) = w.dense_shaped(&name)?;
            emit(&mut bw, name, shape.to_vec(), v)?;
        }
        for leaf in ["mlp.gate.weight", "mlp.gate.bias"] {
            let name = format!("{pfx}{leaf}");
            if let Ok((shape, v)) = w.dense_shaped(&name) {
                emit(&mut bw, name, shape.to_vec(), v)?;
            } else if leaf.ends_with("weight") {
                return Err(format!("source: tensor not found: {name}"));
            }
        }
        for e in 0..n_experts {
            let (rows, cols, gu) = w.gate_up(l, e)?;
            emit(
                &mut bw,
                format!("{pfx}mlp.experts.{e}.gate_up_proj.weight"),
                vec![rows, cols],
                &gu,
            )?;
            let (rows, cols, dn) = w.down(l, e)?;
            emit(
                &mut bw,
                format!("{pfx}mlp.experts.{e}.down_proj.weight"),
                vec![rows, cols],
                &dn,
            )?;
        }
        bw.flush().map_err(|e| e.to_string())?;
        drop(bw);
        let sha = format!("{:x}", hash.finalize());
        let kind = |stem: &str| {
            if w.st
                .tensor_data(&format!("{pfx}mlp.experts.{stem}_amax"))
                .is_some()
            {
                "nvfp4"
            } else {
                "mxfp8"
            }
        };
        let mut verify_json = serde_json::Value::Null;
        if verify {
            let file = std::fs::File::open(&path).map_err(|e| e.to_string())?;
            // SAFETY: read-only map of a file this process just wrote and closed.
            let mm = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| e.to_string())?;
            let by_name: HashMap<&str, (u64, usize)> = tensors
                .iter()
                .map(|t| {
                    (
                        t["name"].as_str().expect("name"),
                        (
                            t["offset_bytes"].as_u64().expect("offset"),
                            t["numel"].as_u64().expect("numel") as usize,
                        ),
                    )
                })
                .collect();
            let equal = |name: &str, want: &[f32]| -> Result<bool, String> {
                let &(o, n) = by_name
                    .get(name)
                    .ok_or_else(|| format!("{name} not dumped"))?;
                if n != want.len() {
                    return Ok(false);
                }
                let got = &mm[o as usize..o as usize + n * 4];
                Ok(got
                    .chunks_exact(4)
                    .zip(want)
                    .all(|(c, &x)| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) == x.to_bits()))
            };
            // (1) every expert tensor equals the per-token reference's own read
            //     (`WeightSource::f32`, the path `forward_token_cached` uses);
            // (2) the regroup indices re-derived independently from the raw
            //     decoder output: grouped gate j = raw row 2j, up j = raw row 2j+1.
            let results: Vec<Result<(bool, bool), String>> = (0..n_experts)
                .into_par_iter()
                .map(|e| {
                    let gu_name = format!("{pfx}mlp.experts.{e}.gate_up_proj.weight");
                    let dn_name = format!("{pfx}mlp.experts.{e}.down_proj.weight");
                    let (gs, gv) = w.f32(&gu_name)?;
                    let (ds, dv) = w.f32(&dn_name)?;
                    let same = gs == vec![2 * cfg.inter, cfg.n_embd]
                        && ds == vec![cfg.n_embd, cfg.inter]
                        && equal(&gu_name, &gv)?
                        && equal(&dn_name, &dv)?;
                    let (rows, cols, raw) =
                        w.decode_stacked(&format!("{pfx}mlp.experts.w13_weight"), e)?;
                    let inter = rows / 2;
                    let mut regroup = true;
                    let &(o, n) = by_name.get(gu_name.as_str()).ok_or("gate_up not dumped")?;
                    let got = &mm[o as usize..o as usize + n * 4];
                    let at = |row: usize, c: usize| {
                        let i = (row * cols + c) * 4;
                        u32::from_le_bytes([got[i], got[i + 1], got[i + 2], got[i + 3]])
                    };
                    for j in 0..inter {
                        for c in 0..cols {
                            regroup &= at(j, c) == raw[(2 * j) * cols + c].to_bits()
                                && at(inter + j, c) == raw[(2 * j + 1) * cols + c].to_bits();
                        }
                    }
                    Ok((same, regroup))
                })
                .collect();
            let (mut same, mut regroup) = (0usize, 0usize);
            for r in results {
                let (s, g) = r?;
                same += usize::from(s);
                regroup += usize::from(g);
            }
            // Attention/router tensors, re-read through the reference's path.
            let mut dense_same = 0usize;
            let mut dense_total = 0usize;
            for t in &tensors {
                let name = t["name"].as_str().expect("name");
                if name.contains(".mlp.experts.") {
                    continue;
                }
                dense_total += 1;
                let (_, v) = w.f32(name)?;
                dense_same += usize::from(equal(name, &v)?);
            }
            verify_json = serde_json::json!({
                "experts": n_experts,
                "experts_bit_identical_to_reference_read": same,
                "experts_regroup_independently_verified": regroup,
                "dense_tensors": dense_total, "dense_bit_identical": dense_same,
            });
            if same != n_experts || regroup != n_experts || dense_same != dense_total {
                return Err(format!(
                    "layer {l}: srcdump verification failed: {verify_json}"
                ));
            }
        }
        let json = serde_json::json!({
            "schema": "hipfire.inkling2.srcdump.v1",
            "layer": l,
            "file": path.file_name().map(|s| s.to_string_lossy().to_string()),
            "dtype": "F32 little-endian, row-major",
            "bytes": off, "sha256": sha,
            "formats": {"w13": kind("w13_weight"), "w2": kind("w2_weight")},
            "decode": "hipfire-quantize dequant.rs strict NVFP4 (amax/outer_scale_denominator) / MXFP8 E8M0; w13 [g0,u0,g1,u1..] regrouped to [gate..., up...]",
            "tensors": tensors,
            "verify": verify_json,
            "engine_commit": engine_commit(),
        });
        std::fs::write(
            out_dir.join(format!("L{l:02}.json")),
            serde_json::to_string_pretty(&json).map_err(|e| e.to_string())? + "\n",
        )
        .map_err(|e| e.to_string())?;
        eprintln!(
            "inkling_kld srcdump: layer {l} ({} / {}) {:.2} GiB sha256 {sha} in {:.0}s, verify {verify_json}",
            kind("w13_weight"),
            kind("w2_weight"),
            gib(off),
            tl.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// captest (observer on vs off, count identity, Gram vs raw rows)
// ---------------------------------------------------------------------------

/// `Σ_r x xᵀ` in F64 of the rows of one check site (all visits, or one expert).
fn naive_gram_from_check(
    dir: &Path,
    phys: usize,
    site: &str,
    expert: Option<usize>,
) -> Result<(usize, u64, Vec<f64>), String> {
    let idx: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join(format!("L{phys:02}.index.json")))
            .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let s = &idx["sites"][site];
    let k = s["K"].as_u64().ok_or("check index: K")? as usize;
    let bytes = std::fs::read(dir.join(s["file"].as_str().ok_or("check index: file")?))
        .map_err(|e| e.to_string())?;
    let mut g = vec![0.0f64; k * k];
    let mut n = 0u64;
    for seg in s["segments"].as_array().ok_or("check index: segments")? {
        if expert.is_some() && seg["expert"].as_u64().map(|e| e as usize) != expert {
            continue;
        }
        let (row0, rows) = (
            seg["row0"].as_u64().ok_or("row0")? as usize,
            seg["rows"].as_u64().ok_or("rows")? as usize,
        );
        for r in row0..row0 + rows {
            let x: Vec<f64> = bytes[r * k * 4..(r + 1) * k * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
                .collect();
            for i in 0..k {
                for j in 0..k {
                    g[i * k + j] += x[i] * x[j];
                }
            }
            n += 1;
        }
    }
    Ok((k, n, g))
}

fn captest(a: &Args) -> Result<(), String> {
    let source = a.path("source")?;
    let model = a.path("model")?;
    let slice = a.path("slice")?;
    let out_dir = a.path("out-dir")?;
    let n_ctx = a.num("n-ctx", 64)?;
    let n_chunk = a.num("chunks", 2)?;
    let (mut cfg, tokens) = corpus_windows(&model, &slice, n_ctx, n_chunk)?;
    if let Some(sw) = a.opt_num("sliding-window")? {
        cfg.sliding_window = sw;
    }
    if let Some(lf) = a.opt_num("log-floor")? {
        cfg.log_floor = lf;
    }
    let default_layers = "5,6".to_string();
    let layers = parse_layers(a.0.get("layers").or(Some(&default_layers)), cfg.n_phys)?;
    std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
    let w = SourceWeights::open(&source)?;
    let teacher = Teacher { w: &w, cfg: &cfg };
    let n_experts = w
        .dense(&format!("model.llm.layers.{}.mlp.gate.weight", layers[0]))?
        .len()
        / cfg.n_embd;
    let seqs: Vec<&[u32]> = tokens.chunks_exact(n_ctx).collect();
    let n_tok = seqs.len() * n_ctx;

    let t0 = Instant::now();
    let hidden_off = teacher.forward(&seqs, None)?;
    let logits_off = teacher.head(&hidden_off)?;
    let off_s = t0.elapsed().as_secs_f64();

    let gram = GramObserver::new(&cfg, &layers, &out_dir.join("acc"), n_experts, false)?;
    let check = CheckObserver::new(&cfg, &layers, &out_dir.join("check"), false)?;
    check.set_batch((0..seqs.len()).map(|i| format!("chunk{i}")).collect());
    let tee = Tee(vec![&gram, &check]);
    let t1 = Instant::now();
    let hidden_on = teacher.forward(&seqs, Some(&tee))?;
    let logits_on = teacher.head(&hidden_on)?;
    let on_s = t1.elapsed().as_secs_f64();
    let hidden_equal = hidden_off.len() == hidden_on.len()
        && hidden_off.iter().zip(&hidden_on).all(|(a, b)| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        });
    let logits_equal = logits_off.len() == logits_on.len()
        && logits_off
            .iter()
            .zip(&logits_on)
            .all(|(x, y)| x.to_bits() == y.to_bits());
    let index = check.finish()?;

    // Finalize (also enforces Σ_e N_e == top_k·tokens·visits for gate_up and down).
    let run = serde_json::json!({"role": "captest", "tokens": n_tok});
    let hfhs_dir = out_dir.join("hfhs");
    let mut layer_receipts = Vec::new();
    let mut worst_rel = 0.0f64;
    for &l in &layers {
        let j = gram.finalize_layer(l, &hfhs_dir, &run)?;
        let sc = hipfire_quantize::hessian_io::HessianSidecar::open(
            &hfhs_dir.join(format!("L{l:02}.hfhs")),
        )
        .map_err(|e| format!("reader: {e}"))?;
        let pfx = format!("model.llm.layers.{l}.");
        // Gram vs an independent F64 recomputation from the raw check rows.
        let mut probes: Vec<(String, &str, Option<usize>)> = vec![
            (format!("{pfx}attn.wq_du"), "q", None),
            (format!("{pfx}attn.wk_dv"), "k", None),
            (format!("{pfx}attn.wo_ud"), "attn_out", None),
        ];
        let busiest = j["route"]["experts"]
            .as_array()
            .and_then(|v| v.iter().max_by_key(|e| e["N"].as_u64().unwrap_or(0)))
            .and_then(|e| e["expert"].as_u64())
            .map(|e| e as usize)
            .ok_or("no experts")?;
        probes.push((
            format!("{pfx}mlp.experts.{busiest}.gate_up_proj"),
            "gate_up",
            Some(busiest),
        ));
        probes.push((
            format!("{pfx}mlp.experts.{busiest}.down_proj"),
            "down",
            Some(busiest),
        ));
        let mut probe_json = Vec::new();
        for (name, site, ex) in probes {
            let (k, n, g) = naive_gram_from_check(&out_dir.join("check"), l, site, ex)?;
            let h = sc
                .get(&name, 0)
                .ok_or_else(|| format!("HFHS has no record {name}"))?;
            let n_json = j["names"][&name]["N"].as_u64().unwrap_or(0);
            if h.k != k || n_json != n {
                return Err(format!("{name}: K {} vs {k}, N {n_json} vs rows {n}", h.k));
            }
            let gmax = g.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let mut worst = 0.0f64;
            for i in 0..k {
                for jx in 0..k {
                    let got = h.at(i, jx) * n as f64;
                    worst = worst.max((got - g[i * k + jx]).abs() / gmax);
                }
            }
            worst_rel = worst_rel.max(worst);
            probe_json.push(serde_json::json!({"name": name, "K": k, "N": n, "max_rel_err_vs_f64_naive": worst}));
        }
        layer_receipts.push(serde_json::json!({
            "layer": l,
            "sum_N_gate_up": j["route"]["sum_N_gate_up"], "sum_N_down": j["route"]["sum_N_down"],
            "sum_N_expected": j["route"]["sum_N_expected_top_k_tokens_visits"],
            "visits": j["visits"], "records": j["n_records"], "probes": probe_json,
        }));
    }
    let receipt = serde_json::json!({
        "schema": "hipfire.inkling2.captest.v1",
        "n_ctx": n_ctx, "chunks": n_chunk, "tokens": n_tok,
        "sliding_window": cfg.sliding_window, "log_floor": cfg.log_floor,
        "observer_off_seconds": off_s, "observer_on_seconds": on_s,
        "hidden_bit_identical_on_vs_off": hidden_equal,
        "logits_bit_identical_on_vs_off": logits_equal,
        "worst_gram_rel_err": worst_rel,
        "layers": layer_receipts,
        "check_index": index.iter().map(|(p, s)| serde_json::json!({"path": p.display().to_string(), "sha256": s})).collect::<Vec<_>>(),
        "engine_commit": engine_commit(),
    });
    let text = serde_json::to_string_pretty(&receipt).map_err(|e| e.to_string())?;
    std::fs::write(out_dir.join("captest.json"), text.clone() + "\n").map_err(|e| e.to_string())?;
    println!("{text}");
    std::fs::remove_dir_all(out_dir.join("acc")).ok();
    if !hidden_equal || !logits_equal {
        return Err("captest: observer-on teacher differs from observer-off".into());
    }
    if worst_rel > 1e-4 {
        return Err(format!(
            "captest: Gram differs from the naive F64 recomputation (rel {worst_rel:e})"
        ));
    }
    Ok(())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let cmd = args.next().unwrap_or_default();
    let result = Args::parse(args).and_then(|a| match cmd.as_str() {
        "ref" => build_ref(&a),
        "fidelity" => fidelity(&a),
        "calib" => calib(&a),
        "srcdump" => srcdump(&a),
        "captest" => captest(&a),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Small-integer test rows: every product and sum is exact in F32 and F64,
    /// so Gram comparisons are bitwise.
    fn rows_of(n: usize, k: usize, salt: usize) -> Vec<f32> {
        (0..n * k)
            .map(|i| (((i * 7 + salt * 13 + i / k * 3) % 7) as f32) - 3.0)
            .collect()
    }

    fn naive_expert_grams(
        visits: &[(Vec<Vec<(usize, f32)>>, Vec<f32>)],
        k: usize,
        n_experts: usize,
    ) -> Vec<(Vec<f64>, u64)> {
        let mut out = vec![(vec![0.0f64; k * k], 0u64); n_experts];
        for (selected, x) in visits {
            for (t, sel) in selected.iter().enumerate() {
                for &(e, _) in sel {
                    let row = &x[t * k..(t + 1) * k];
                    for i in 0..k {
                        for j in 0..k {
                            out[e].0[i * k + j] += row[i] as f64 * row[j] as f64;
                        }
                    }
                    out[e].1 += 1;
                }
            }
        }
        out
    }

    /// What `Teacher::visit` does to the gathered rows of every expert, one
    /// accumulator per expert, visits in order.
    fn observed_expert_grams(
        visits: &[(Vec<Vec<(usize, f32)>>, Vec<f32>)],
        k: usize,
        n_experts: usize,
        top_k: usize,
    ) -> Vec<(Vec<f64>, u64)> {
        let mut acc: Vec<(Vec<f64>, u64)> = (0..n_experts)
            .map(|_| (vec![0.0f64; tri_len(k)], 0u64))
            .collect();
        for (selected, x) in visits {
            let (by_expert, _) = route_tables(selected, n_experts, top_k);
            for (e, rows) in by_expert.iter().enumerate() {
                let mut xe = Vec::new();
                for &(i, _) in rows {
                    xe.extend_from_slice(&x[i * k..(i + 1) * k]);
                }
                gram_add(&mut acc[e].0, &xe, rows.len(), k);
                acc[e].1 += rows.len() as u64;
            }
        }
        acc.into_iter()
            .map(|(tri, n)| {
                (
                    hessian_from_tri(&tri, k, 1)
                        .iter()
                        .map(|&v| v as f64)
                        .collect(),
                    n,
                )
            })
            .collect()
    }

    /// The count / moment identity the capture must satisfy.
    fn compare(want: &[(Vec<f64>, u64)], got: &[(Vec<f64>, u64)]) -> Result<(), String> {
        for (e, (w, g)) in want.iter().zip(got).enumerate() {
            if w.1 != g.1 {
                return Err(format!("expert {e}: N {} vs {}", w.1, g.1));
            }
            if w.0 != g.0 {
                return Err(format!("expert {e}: Gram differs"));
            }
        }
        Ok(())
    }

    fn scenario() -> (usize, usize, usize, Vec<(Vec<Vec<(usize, f32)>>, Vec<f32>)>) {
        let (k, n_experts, top_k, tokens) = (32usize, 5usize, 2usize, 12usize);
        // Three passes over the same tokens; expert 1 is selected by every token
        // of every pass (a repeated 3-pass expert), the second slot alternates.
        let visits = (0..3)
            .map(|v| {
                let selected: Vec<Vec<(usize, f32)>> = (0..tokens)
                    .map(|t| vec![(1, 0.6), (2 + (t + v) % 2, 0.4)])
                    .collect();
                (selected, rows_of(tokens, k, v + 1))
            })
            .collect();
        (k, n_experts, top_k, visits)
    }

    #[test]
    fn hand_gram_repeated_three_pass_expert() {
        let (k, n_experts, top_k, visits) = scenario();
        let want = naive_expert_grams(&visits, k, n_experts);
        let got = observed_expert_grams(&visits, k, n_experts, top_k);
        compare(&want, &got).expect("observer Gram equals the hand Gram");
        // Expert 1: 12 tokens x 3 passes; expert 0 and 4 never selected.
        assert_eq!(got[1].1, 36);
        assert_eq!(got[0].1, 0);
        assert_eq!(got[4].1, 0);
        // Σ_e N_e == top_k · tokens · visits.
        assert_eq!(got.iter().map(|g| g.1).sum::<u64>(), (2 * 12 * 3) as u64);
        // Expert 1 accumulates all three passes: hand-check one entry.
        let h: f64 = visits
            .iter()
            .map(|(_, x)| {
                (0..12)
                    .map(|t| x[t * k + 3] as f64 * x[t * k + 17] as f64)
                    .sum::<f64>()
            })
            .sum();
        assert_eq!(got[1].0[3 * k + 17], h);
        assert_eq!(got[1].0[17 * k + 3], h, "symmetric");
    }

    #[test]
    fn planted_selected_id_permutation_fails_the_check() {
        let (k, n_experts, top_k, visits) = scenario();
        let want = naive_expert_grams(&visits, k, n_experts);
        // Plant a permutation e -> (e + 1) % E of the selected ids in the gather.
        let permuted: Vec<(Vec<Vec<(usize, f32)>>, Vec<f32>)> = visits
            .iter()
            .map(|(sel, x)| {
                (
                    sel.iter()
                        .map(|s| s.iter().map(|&(e, w)| ((e + 1) % n_experts, w)).collect())
                        .collect(),
                    x.clone(),
                )
            })
            .collect();
        let got = observed_expert_grams(&permuted, k, n_experts, top_k);
        let err = compare(&want, &got).expect_err("a permuted expert id must be caught");
        assert!(err.contains("expert"), "{err}");
        // A swap of two ids in a single visit is caught too.
        let mut swapped = visits.clone();
        for s in &mut swapped[1].0 {
            for p in s.iter_mut() {
                if p.0 == 2 {
                    p.0 = 3;
                } else if p.0 == 3 {
                    p.0 = 2;
                }
            }
        }
        let got = observed_expert_grams(&swapped, k, n_experts, top_k);
        assert!(compare(&want, &got).is_err());
    }

    #[test]
    fn gram_chunks_and_threads_are_exact() {
        // More rows than one chunk, a K with several panels, ragged last chunk.
        let (n, k) = (GRAM_CHUNK * 2 + 37, 48usize);
        let x = rows_of(n, k, 5);
        let mut tri = vec![0.0f64; tri_len(k)];
        gram_add(&mut tri, &x, n, k);
        let h = hessian_from_tri(&tri, k, 1);
        for i in 0..k {
            for j in 0..k {
                let want: f64 = (0..n)
                    .map(|r| x[r * k + i] as f64 * x[r * k + j] as f64)
                    .sum();
                assert_eq!(h[i * k + j] as f64, want, "G[{i}][{j}]");
            }
        }
        // Splitting the rows across calls (batches) gives the same bits.
        let mut tri2 = vec![0.0f64; tri_len(k)];
        gram_add(&mut tri2, &x[..100 * k], 100, k);
        gram_add(&mut tri2, &x[100 * k..], n - 100, k);
        assert_eq!(tri, tri2);
        // Thread count does not matter.
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("pool");
        let mut tri3 = vec![0.0f64; tri_len(k)];
        one.install(|| gram_add(&mut tri3, &x, n, k));
        assert_eq!(tri, tri3);
    }

    #[test]
    fn gram_on_real_valued_rows_matches_f64_within_f32_chunk_error() {
        let (n, k) = (700usize, 32usize);
        let x: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32 * 0.37).sin() * 1.7 + (i as f32 * 0.011).cos()) * 0.5)
            .collect();
        let mut tri = vec![0.0f64; tri_len(k)];
        gram_add(&mut tri, &x, n, k);
        let h = hessian_from_tri(&tri, k, 1);
        let mut gmax = 0.0f64;
        let mut worst = 0.0f64;
        for i in 0..k {
            for j in 0..k {
                let want: f64 = (0..n)
                    .map(|r| x[r * k + i] as f64 * x[r * k + j] as f64)
                    .sum();
                gmax = gmax.max(want.abs());
                worst = worst.max((h[i * k + j] as f64 - want).abs());
            }
        }
        assert!(worst / gmax < 1e-5, "rel err {}", worst / gmax);
    }

    #[test]
    fn tri_layout_counts() {
        // K = 32: block rows of 16 x 32 and 16 x 16.
        assert_eq!(tri_len(32), 16 * 32 + 16 * 16);
        assert_eq!(tri_row_off(32, 1), 16 * 32);
        let k = 2304;
        let nb = k / PB;
        assert_eq!(
            tri_len(k),
            (0..nb).map(|ib| PB * (k - PB * ib)).sum::<usize>()
        );
        assert_eq!(
            tri_row_off(k, 5),
            (0..5).map(|ib| PB * (k - PB * ib)).sum::<usize>()
        );
    }

    #[test]
    fn route_margin_matches_selection_and_names_the_ninth() {
        let logits = [0.0f32, 3.0, -1.0, 2.0, 1.0, 2.5];
        let sel = oracle::sigmoid_topk(&logits, None, 3, 1.0);
        let (margin, next) = route_margin(&logits, None, 3, &sel).expect("margin");
        assert_eq!(sel.iter().map(|s| s.0).collect::<Vec<_>>(), vec![1, 5, 3]);
        assert_eq!(next, 4);
        let want = oracle::sigmoid(2.0) - oracle::sigmoid(1.0);
        assert!((margin - want).abs() < 1e-7);
        // A planted selection that is not the top-k is rejected.
        let mut bad = sel.clone();
        bad.swap(0, 2);
        assert!(route_margin(&logits, None, 3, &bad).is_err());
    }

    #[test]
    fn hfhs_records_round_trip_through_the_quantizer_reader() {
        use std::io::Write as _;
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        let (k, n_experts, top_k, visits) = scenario();
        let got = observed_expert_grams(&visits, k, n_experts, top_k);
        let mut w = std::io::BufWriter::new(tmp.reopen().expect("reopen"));
        let mut put = |w: &mut std::io::BufWriter<std::fs::File>, b: &[u8]| -> Result<(), String> {
            w.write_all(b).map_err(|e| e.to_string())
        };
        put(&mut w, b"HFHS").expect("w");
        put(&mut w, &1u32.to_le_bytes()).expect("w");
        put(&mut w, &1u64.to_le_bytes()).expect("w");
        put(&mut w, &0u64.to_le_bytes()).expect("w");
        let h: Vec<f32> = got[1].0.iter().map(|&v| v as f32).collect();
        put_record(
            &mut put,
            &mut w,
            "model.llm.layers.0.mlp.experts.1.down_proj",
            k,
            &h,
        )
        .expect("record");
        w.flush().expect("flush");
        drop(w);
        let sc = hipfire_quantize::hessian_io::HessianSidecar::open(tmp.path()).expect("open");
        let r = sc
            .get("model.llm.layers.0.mlp.experts.1.down_proj", 0)
            .expect("record present");
        assert_eq!(r.k, k);
        assert_eq!(r.at(3, 17), got[1].0[3 * k + 17]);
        assert!(sc
            .get("model.llm.layers.0.mlp.experts.0.down_proj", 0)
            .is_none());
    }
}
