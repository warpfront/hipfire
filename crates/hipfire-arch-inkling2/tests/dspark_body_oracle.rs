// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU oracle for the Inkling-2-Mini DSpark draft body (`MTP-PLAN.md` §5,
//! slice C), without the target forward seam: the body is fed taps recorded by
//! the CPU acceptance gate (`inkling_kld mtp-gpu-gen`, branch
//! `inkling-mtp-cpu`) and compared against the CPU reference
//! (`examples/inkling_mtp/reference.rs`).
//!
//! 1. Golden windows (`mtp-golden` fixtures): body hidden vs the reference
//!    within pinned tolerances, identical proposals, the `f_rnn` state oracle
//!    (row `P - 1`, value vs a host recomputation) and the planted
//!    `f_rnn`-from-anchor-row negative.
//! 2. Every moving window of the gate's `correct` arm (17 prompts, first 128
//!    continuation positions) through the shared `DsparkDrafter` (committed
//!    binding) and the scalar `InklingSpecTarget` over a replay of the recorded
//!    target: identical proposals, accepted prefixes and emitted tokens.
//! 3. State behaviour: reset between prompts, cold start, forced advance,
//!    `k = 0`, terminal-prefix repair (and its negatives), all bitwise against
//!    a cold path.
//!
//! ```sh
//! W=/home/kaden/qcal/release-0.5.0/inkling/mtp-work T=/home/kaden/qcal/release-0.5.0/inkling/teacher/source
//! INKLING_DSPARK_TRUNK=~/.hipfire/models/inkling2/inkling2-mini-mq4.hfq \
//! INKLING_DSPARK_SIDECAR=/home/kaden/qcal/release-0.5.0/inkling/dspark-art/inkling2-mini-mq4-dspark.hfq \
//! INKLING_DSPARK_SOURCE=$T/model-00001-of-00001.safetensors \
//! INKLING_DSPARK_FOLD=$W/fold.bin INKLING_DSPARK_GOLDEN=$W/runs/golden INKLING_DSPARK_GEN=$W/runs/gpu/gen \
//! INKLING_DSPARK_ACCEPT=$W/runs/accept/gpu136.json INKLING_DSPARK_RECEIPT=out.json \
//!   cargo test --release -p hipfire-arch-inkling2 --test dspark_body_oracle -- --ignored --nocapture
//! ```

use hipfire_arch_inkling2::dspark_body::{
    bf16_bits, Bf16TargetHead, InklingDsparkBody, InklingFold, BLOCK, EPS, HIDDEN, K_MAX, N_TAP,
    RANK, TAP_ROW, VOCAB,
};
use hipfire_arch_inkling2::dspark_weights::{
    InklingDsparkSource, InklingDsparkTargetIdentity, InklingDsparkWeights,
};
use hipfire_arch_inkling2::spec_target::{InklingSpecTarget, InklingTapForward};
use hipfire_runtime::dspark_core::{new_dspark_committed_drafter, DsparkCommittedBody};
use hipfire_runtime::spec::{
    MtpDrafter, PrefillOutcome, SpecAdvance, SpecScratch, SpecTapSink, SpecTarget,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// `<|content_model_end_sampling|>`.
const EOS: u32 = 200_006;
/// Gate tap slots in a generation directory (`TAP_VISITS` of `inkling_mtp/mod.rs`
/// plus the final slot): inputs to visits 4,5,6,11,17,22,23,24,40,41,42, final.
const GEN_SLOTS: usize = 12;
/// Slots of visits 5, 23, 41 and the final normalized state.
const CORRECT_SLOTS: [usize; N_TAP] = [1, 6, 9, 11];
const BUDGET: usize = 128;
/// Pinned tolerances of body hidden vs the CPU reference (golden.py vs the
/// reference measured 0.99998 / 0.0050 on the same windows).
const MIN_ROW_COS: f64 = 0.9999;
const MAX_RMS_REL: f64 = 0.01;
/// `f_rnn` vs the host recomputation: at most one BF16 step per element.
const MAX_FRNN_STEPS: u32 = 1;
/// Frozen tolerance: a proposal may differ from the reference only where the
/// pick gap, measured on the body's own softcapped BF16 logits, is at most one
/// BF16 step at the softcap top (|logit| in [16, 30]): 0.125. The draft only
/// proposes and verify keeps the output exact, so such a flip costs τ, never
/// correctness (gate arm: τ 2.715 vs 2.725). The body's hidden differs from
/// the reference by ~0.45% RMS (accumulation order), which moves logits by
/// about this much.
const MAX_FLIP_LOGIT: f32 = 0.125;

fn env(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name}")))
}

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bf16(x: f32) -> f32 {
    f32::from_bits((bf16_bits(x) as u32) << 16)
}

fn widen(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

/// BF16 steps between two BF16-valued floats (ordered bit distance).
fn bf16_steps(a: f32, b: f32) -> u32 {
    let key = |x: f32| {
        let b = (x.to_bits() >> 16) as i32;
        if b & 0x8000 != 0 {
            -(b & 0x7FFF)
        } else {
            b
        }
    };
    (key(a) - key(b)).unsigned_abs()
}

fn upload(gpu: &mut Gpu, v: &[f32]) -> GpuTensor {
    gpu.upload_f32(v, &[v.len()]).expect("upload")
}

/// Host `mix_rmsnorm` of the reference: `Σ_t x_t·α_t` (F32, tap order),
/// RMSNorm in F32, BF16 out.
fn host_mix_norm(feat: &[f32], alpha: &[f32; N_TAP], w: &[f32]) -> Vec<f32> {
    let mut m = vec![0.0f32; HIDDEN];
    for (t, a) in alpha.iter().enumerate() {
        for (mv, x) in m.iter_mut().zip(&feat[t * HIDDEN..(t + 1) * HIDDEN]) {
            *mv += x * a;
        }
    }
    let ms = m.iter().map(|v| v * v).sum::<f32>() / HIDDEN as f32;
    let inv = 1.0 / (ms + EPS).sqrt();
    m.iter().zip(w).map(|(v, g)| bf16(v * inv * g)).collect()
}

#[derive(Clone)]
struct GenSeq {
    name: String,
    prompt_len: usize,
    ids: Vec<u32>,
}

/// The recorded target of one sequence: its IDs and its rows' four taps.
struct Recorded {
    seq: GenSeq,
    /// `[rows][N_TAP][HIDDEN]`, F32 as recorded (the body rounds).
    taps: Vec<f32>,
    rows: usize,
}

fn load_recorded(dir: &Path, name: &str) -> Recorded {
    let v: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join(format!("{name}.json"))).expect("gen json"),
    )
    .expect("gen json");
    let seq = GenSeq {
        name: v["name"].as_str().expect("name").to_string(),
        prompt_len: v["prompt_len"].as_u64().expect("prompt_len") as usize,
        ids: serde_json::from_value(v["ids"].clone()).expect("ids"),
    };
    let all = read_f32(&dir.join(format!("{name}.taps")));
    let rows = seq.ids.len() - 1;
    assert_eq!(all.len(), rows * GEN_SLOTS * HIDDEN, "{name}: taps size");
    let mut taps = Vec::with_capacity(rows * TAP_ROW);
    for r in 0..rows {
        for &s in &CORRECT_SLOTS {
            let o = (r * GEN_SLOTS + s) * HIDDEN;
            taps.extend_from_slice(&all[o..o + HIDDEN]);
        }
    }
    Recorded { seq, taps, rows }
}

/// Replays the recorded target through the scalar tap seam: row `p` must be
/// the recorded `ids[p]`; its pick is `ids[p + 1]`.
struct Replay {
    rec: std::rc::Rc<Recorded>,
    restorable: bool,
    forwarded: usize,
}

impl InklingTapForward for Replay {
    fn forward_row_taps(
        &mut self,
        gpu: &mut Gpu,
        token: u32,
        position: usize,
        taps: &GpuTensor,
    ) -> Result<u32, String> {
        let ids = &self.rec.seq.ids;
        if ids.get(position) != Some(&token) {
            return Err(format!(
                "replay {}: row {position} is {token}, recorded {:?}",
                self.rec.seq.name,
                ids.get(position)
            ));
        }
        // The last ID is never forwarded by the recording; an accepted EOS row
        // gets zero taps (nothing reads them after the terminal window).
        let row = if position < self.rec.rows {
            self.rec.taps[position * TAP_ROW..(position + 1) * TAP_ROW].to_vec()
        } else {
            vec![0.0; TAP_ROW]
        };
        let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
        gpu.memcpy_htod_auto(&taps.buf, &bytes)
            .map_err(|e| format!("replay upload: {e:?}"))?;
        self.forwarded += 1;
        Ok(ids.get(position + 1).copied().unwrap_or(u32::MAX))
    }

    fn reset_state(&mut self, _gpu: &mut Gpu) -> Result<(), String> {
        Ok(())
    }

    fn restore_window(
        &mut self,
        _gpu: &mut Gpu,
        window_start: usize,
        replay: &[u32],
    ) -> Result<bool, String> {
        if !self.restorable {
            return Ok(false);
        }
        let ids = &self.rec.seq.ids;
        if ids.get(window_start..window_start + replay.len()) != Some(replay) {
            return Err("replay: restore replays unrecorded rows".into());
        }
        Ok(true)
    }

    fn eos_token(&self) -> u32 {
        EOS
    }

    fn ctx_capacity(&self) -> usize {
        1 << 16
    }
}

/// Records every verify block the driver hands the target.
struct Recorder {
    inner: InklingSpecTarget<Replay>,
    blocks: Vec<Vec<u32>>,
}

impl SpecTarget for Recorder {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn reset_recurrent(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.inner.reset_recurrent(gpu)
    }
    fn new_spec_scratch(
        &mut self,
        gpu: &mut Gpu,
        block_size: usize,
    ) -> Result<Box<dyn SpecScratch>, String> {
        self.inner.new_spec_scratch(gpu, block_size)
    }
    fn spec_advance(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start_pos: usize,
        reset: bool,
        abort: &dyn Fn() -> bool,
        hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<SpecAdvance, String> {
        self.inner
            .spec_advance(gpu, tokens, start_pos, reset, abort, hidden_out)
    }
    fn spec_advance_taps_gpu(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start_pos: usize,
        abort: &dyn Fn() -> bool,
        sink: &mut dyn SpecTapSink,
    ) -> Result<SpecAdvance, String> {
        self.inner
            .spec_advance_taps_gpu(gpu, tokens, start_pos, abort, sink)
    }
    fn verify_block(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        scratch: &mut dyn SpecScratch,
        hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<u32>, String> {
        self.blocks.push(block.to_vec());
        self.inner
            .verify_block(gpu, block, position, scratch, hidden_out)
    }
    fn verify_block_capture_gpu(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        scratch: &mut dyn SpecScratch,
        hidden_gpu: &GpuTensor,
    ) -> Result<(Vec<u32>, bool), String> {
        self.blocks.push(block.to_vec());
        self.inner
            .verify_block_capture_gpu(gpu, block, position, scratch, hidden_gpu)
    }
    fn commit_prefix(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        accept_len: usize,
        position: usize,
        scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        self.inner
            .commit_prefix(gpu, block, accept_len, position, scratch)
    }
    fn abandon_verify(
        &mut self,
        gpu: &mut Gpu,
        scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        self.inner.abandon_verify(gpu, scratch)
    }
    fn repair_committed_prefix(
        &mut self,
        gpu: &mut Gpu,
        scratch: &mut dyn SpecScratch,
        position: usize,
        replay: &[u32],
    ) -> Result<bool, String> {
        self.inner
            .repair_committed_prefix(gpu, scratch, position, replay)
    }
    fn eos_token(&self) -> u32 {
        self.inner.eos_token()
    }
    fn ctx_capacity(&self) -> usize {
        self.inner.ctx_capacity()
    }
    fn dflash_extract_layers(&self) -> Option<&[usize]> {
        self.inner.dflash_extract_layers()
    }
}

fn target(rec: &std::rc::Rc<Recorded>) -> Recorder {
    Recorder {
        inner: InklingSpecTarget::new(Replay {
            rec: rec.clone(),
            restorable: true,
            forwarded: 0,
        }),
        blocks: Vec::new(),
    }
}

/// One driver window record.
#[derive(Clone, Debug, PartialEq)]
struct Win {
    pos: usize,
    k: usize,
    drafts: Vec<u32>,
    accepted: usize,
    committed: Vec<u32>,
}

fn step(
    d: &mut dyn MtpDrafter,
    gpu: &mut Gpu,
    t: &mut Recorder,
    pos: usize,
    seed: u32,
    k: usize,
) -> Result<Win, String> {
    t.blocks.clear();
    let w = d.mtp_step(gpu, t, pos, seed, &[], k, EOS, None)?;
    let block = t.blocks.last().ok_or("no verify block")?;
    Ok(Win {
        pos,
        k,
        drafts: block[1..].to_vec(),
        accepted: w.accepted,
        committed: w.committed,
    })
}

fn prefill(d: &mut dyn MtpDrafter, gpu: &mut Gpu, t: &mut Recorder, ids: &[u32]) -> u32 {
    let never = || false;
    match d
        .mtp_prefill(gpu, t, ids, ids, 0, false, &never)
        .expect("prefill")
    {
        PrefillOutcome::Ready { first_token } => first_token,
        _ => panic!("prefill aborted"),
    }
}

/// The gate's moving window over one sequence (`mtp-accept` `simulate`).
fn run_sequence(d: &mut dyn MtpDrafter, gpu: &mut Gpu, rec: &std::rc::Rc<Recorded>) -> Vec<Win> {
    let s = &rec.seq;
    let mut t = target(rec);
    d.mtp_reset(gpu).expect("reset");
    let mut seed = prefill(d, gpu, &mut t, &s.ids[..s.prompt_len]);
    let last = s.ids.len() - 1;
    let mut p = s.prompt_len;
    let mut out = Vec::new();
    while p + 1 < s.ids.len() && p < s.prompt_len + BUDGET {
        assert_eq!(seed, s.ids[p], "{}: anchor at {p}", s.name);
        let k = K_MAX.min(last - p);
        let w = step(d, gpu, &mut t, p, seed, k).unwrap_or_else(|e| panic!("{} @{p}: {e}", s.name));
        p += w.accepted + 1;
        seed = *w.committed.last().expect("committed");
        out.push(w);
    }
    out
}

fn stats(gpu_h: &[f32], cpu_h: &[f32]) -> Value {
    let mut cos = Vec::new();
    let (mut num, mut den) = (0.0f64, 0.0f64);
    let mut max_abs = 0.0f32;
    let mut hist = [0usize; 4];
    for r in 0..BLOCK {
        let a = &gpu_h[r * HIDDEN..(r + 1) * HIDDEN];
        let b = &cpu_h[r * HIDDEN..(r + 1) * HIDDEN];
        let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            ab += x as f64 * y as f64;
            aa += x as f64 * x as f64;
            bb += y as f64 * y as f64;
            num += ((x - y) as f64).powi(2);
            den += (y as f64).powi(2);
            max_abs = max_abs.max((x - y).abs());
            hist[(bf16_steps(x, y) as usize).min(3)] += 1;
        }
        cos.push(ab / (aa.sqrt() * bb.sqrt()));
    }
    json!({
        "rows_cos": cos,
        "min_row_cos": cos.iter().cloned().fold(1.0, f64::min),
        "rms_rel_diff": (num / den).sqrt(),
        "max_abs_diff": max_abs,
        "bf16_step_hist": {"0": hist[0], "1": hist[1], "2": hist[2], ">2": hist[3]},
    })
}

#[test]
#[ignore = "needs the pinned draft, the source teacher, the gate fixtures and a GPU"]
fn dspark_body_oracle() {
    let trunk = env("INKLING_DSPARK_TRUNK");
    let sidecar = env("INKLING_DSPARK_SIDECAR");
    let source = env("INKLING_DSPARK_SOURCE");
    let fold_path = env("INKLING_DSPARK_FOLD");
    let golden_dir = env("INKLING_DSPARK_GOLDEN");
    let gen_dir = env("INKLING_DSPARK_GEN");
    let accept = env("INKLING_DSPARK_ACCEPT");
    let receipt_path = env("INKLING_DSPARK_RECEIPT");
    let t0 = std::time::Instant::now();
    let mut gpu = Gpu::init().expect("gpu init");
    let gpu = &mut gpu;

    // --- weights: raw draft, source teacher embed/unembed, cached fold ---
    // The draft comes from the packed sidecar paired with the shipped trunk;
    // the target head and fold stay the source teacher's, which is what the
    // CPU reference binds.
    let trunk_hfq = hipfire_runtime::hfq::HfqFile::open(&trunk).expect("trunk");
    let identity = InklingDsparkTargetIdentity::from_trunk(&trunk_hfq, InklingDsparkSource::mini())
        .expect("trunk identity");
    let side_hfq = hipfire_runtime::hfq::HfqFile::open(&sidecar).expect("sidecar");
    let draft = InklingDsparkWeights::load(&side_hfq, &identity).expect("sidecar load");
    let dt = |n: &str| -> &[u8] { &draft.tensors.iter().find(|(m, _)| m == n).expect(n).1.data };
    let src_file = std::fs::File::open(&source).expect("source");
    let src_map = unsafe { memmap2::Mmap::map(&src_file) }.expect("mmap source");
    let st = safetensors::SafeTensors::deserialize(&src_map).expect("source header");
    let tbytes = |n: &str| st.tensor(n).expect(n).data();
    let unembed = tbytes("model.llm.unembed.weight");
    let head = Bf16TargetHead::upload(
        gpu,
        tbytes("model.llm.embed.weight"),
        tbytes("model.llm.embed_norm.weight"),
        unembed,
    )
    .expect("target head");
    let fold_raw = std::fs::read(&fold_path).expect("fold cache");
    let key_len = fold_raw.len() - 4 * VOCAB * (RANK + 1);
    let key = String::from_utf8(fold_raw[..key_len].to_vec()).expect("fold key");
    let fvals: Vec<f32> = fold_raw[key_len..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let fold = InklingFold {
        w_bf16: fvals[..VOCAB * RANK]
            .iter()
            .map(|&v| bf16_bits(v))
            .collect(),
        b: fvals[VOCAB * RANK..].to_vec(),
    };
    // The cache is the CPU reference's fold; spot-check InklingFold::compute
    // reproduces it bit for bit on sampled rows.
    let ho_w = widen(dt("markov_head.hidden_out.weight"));
    let ho_b = widen(dt("markov_head.hidden_out.bias"));
    let sample: Vec<usize> = (0..VOCAB).step_by(4099).collect();
    let mut fold_rows_equal = 0usize;
    for &v in &sample {
        let u = widen(&unembed[v * HIDDEN * 2..(v + 1) * HIDDEN * 2]);
        let (w, b) = InklingFold::row(&u, &ho_w, &ho_b);
        if w[..] == fold.w_bf16[v * RANK..(v + 1) * RANK] && b.to_bits() == fold.b[v].to_bits() {
            fold_rows_equal += 1;
        }
    }
    let mut body =
        InklingDsparkBody::load(gpu, &draft, Box::new(head), &fold, 1 << 16).expect("body load");
    body.set_trace(true);
    let alpha = *body.alpha();
    let markov_norm = widen(dt("markov_head.dflare_norm.weight"));
    eprintln!("oracle: loaded in {:.0}s", t0.elapsed().as_secs_f64());

    // --- 1. golden windows ---
    let mut golden = Vec::new();
    let mut golden_ok = true;
    let mut names: Vec<PathBuf> = std::fs::read_dir(&golden_dir)
        .expect("golden dir")
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("meta.json").exists())
        .collect();
    names.sort();
    let x_head = gpu.alloc_tensor(&[BLOCK * HIDDEN], DType::F32).unwrap();
    for dir in &names {
        let meta: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
        let pos = meta["pos"].as_u64().unwrap() as usize;
        let anchor = meta["anchor"].as_u64().unwrap() as u32;
        let want: Vec<u32> = serde_json::from_value(meta["tokens"].clone()).unwrap();
        let feats = read_f32(&dir.join("feats.f32"));
        let cpu_h = read_f32(&dir.join("hidden.f32"));
        assert_eq!(feats.len(), pos * TAP_ROW);
        let dev = upload(gpu, &feats);
        body.reset(gpu).unwrap();
        body.ingest_committed(gpu, &dev, 0, pos).expect("ingest");
        body.draft_block(gpu, anchor, pos, &x_head).expect("draft");
        let gpu_h = gpu.download_f32(&x_head).unwrap();
        let got = body
            .score_block(gpu, &x_head, anchor, K_MAX)
            .expect("score")
            .tokens;
        let tr = body.trace().clone();
        // f_rnn state oracle: row P - 1, value = host recomputation.
        let want_frnn = host_mix_norm(
            &feats[(pos - 1) * TAP_ROW..pos * TAP_ROW],
            &alpha[3],
            &markov_norm,
        );
        let frnn_steps = tr
            .frnn
            .iter()
            .zip(&want_frnn)
            .map(|(&a, &b)| bf16_steps(a, b))
            .max()
            .unwrap();
        // Planted negative: f_rnn from the anchor's own row P. Its taps only
        // exist once row P is committed, so the body must refuse to draft the
        // anchor at P then; and the f_rnn value oracle separates the two rows.
        let name = meta["name"].as_str().unwrap().to_string();
        let rec = load_recorded(&gen_dir, &name);
        let row_p: Vec<f32> = rec.taps[pos * TAP_ROW..(pos + 1) * TAP_ROW]
            .iter()
            .map(|&v| bf16(v))
            .collect();
        let leak_frnn = host_mix_norm(&row_p, &alpha[3], &markov_norm);
        let leak_steps = tr
            .frnn
            .iter()
            .zip(&leak_frnn)
            .map(|(&a, &b)| bf16_steps(a, b))
            .max()
            .unwrap();
        let p_dev = upload(gpu, &rec.taps[pos * TAP_ROW..(pos + 1) * TAP_ROW]);
        body.ingest_committed(gpu, &p_dev, pos, 1)
            .expect("ingest anchor row");
        let leak_refused = body.draft_block(gpu, anchor, pos, &x_head).is_err();
        let _ = gpu.free_tensor(p_dev);
        let _ = gpu.free_tensor(dev);
        let st = stats(&gpu_h, &cpu_h);
        let ok = got == want
            && st["min_row_cos"].as_f64().unwrap() >= MIN_ROW_COS
            && st["rms_rel_diff"].as_f64().unwrap() <= MAX_RMS_REL
            && tr.frnn_row == Some(pos - 1)
            && frnn_steps <= MAX_FRNN_STEPS
            && leak_steps > MAX_FRNN_STEPS
            && leak_refused;
        golden_ok &= ok;
        eprintln!(
            "golden {}: tokens_equal={} cos={:.6} rms_rel={:.4} frnn_steps={frnn_steps} leak_steps={leak_steps} leak_refused={leak_refused}",
            dir.display(),
            got == want,
            st["min_row_cos"].as_f64().unwrap(),
            st["rms_rel_diff"].as_f64().unwrap()
        );
        golden.push(json!({
            "fixture": dir.file_name().unwrap().to_string_lossy(), "pos": pos, "anchor": anchor,
            "cpu_tokens": want, "gpu_tokens": got, "tokens_equal": got == want,
            "hidden_vs_cpu": st,
            "frnn_row": tr.frnn_row, "frnn_max_bf16_steps_vs_host_row_p_minus_1": frnn_steps,
            "planted_frnn_from_anchor_row": {
                "max_bf16_steps_vs_row_p": leak_steps,
                "draft_refused_with_anchor_row_committed": leak_refused,
            },
            "ok": ok,
        }));
    }
    let _ = gpu.free_tensor(x_head);

    // --- 2. the gate's `correct` arm: 17 prompts, every window ---
    let accept_json: Value =
        serde_json::from_str(&std::fs::read_to_string(&accept).unwrap()).unwrap();
    let correct = accept_json["results"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["config"] == "correct")
        .expect("correct arm");
    assert_eq!(accept_json["budget_continuation_positions"], json!(BUDGET));
    let cpu_seqs = correct["sequences"].as_array().unwrap().clone();
    let seqs: Vec<std::rc::Rc<Recorded>> = cpu_seqs
        .iter()
        .map(|s| std::rc::Rc::new(load_recorded(&gen_dir, s["name"].as_str().unwrap())))
        .collect();

    // 2a. Each CPU window on its own (body level, no cascade): the committed
    // prefix `[0, P)` of the recorded taps, the recorded anchor, the CPU's k.
    let x_head = gpu.alloc_tensor(&[BLOCK * HIDDEN], DType::F32).unwrap();
    let (mut windows, mut proposals, mut prop_equal, mut win_equal) =
        (0usize, 0usize, 0usize, 0usize);
    let mut near_ties = Vec::new();
    let mut unexplained = Vec::new();
    let mut frnn_worst = 0u32;
    for (s, rec) in cpu_seqs.iter().zip(&seqs) {
        let name = s["name"].as_str().unwrap().to_string();
        body.reset(gpu).unwrap();
        for (i, c) in s["trace"].as_array().unwrap().iter().enumerate() {
            let p = c["pos"].as_u64().unwrap() as usize;
            let k = c["k"].as_u64().unwrap() as usize;
            let cd: Vec<u32> = serde_json::from_value(c["draft"].clone()).unwrap();
            let have = body.committed_len();
            let dev = upload(gpu, &rec.taps[have * TAP_ROW..p * TAP_ROW]);
            body.ingest_committed(gpu, &dev, have, p - have)
                .expect("ingest");
            let _ = gpu.free_tensor(dev);
            let anchor = rec.seq.ids[p];
            body.draft_block(gpu, anchor, p, &x_head).expect("draft");
            let got = body
                .score_block(gpu, &x_head, anchor, k)
                .expect("score")
                .tokens;
            let tr = body.trace();
            let feat: Vec<f32> = rec.taps[(p - 1) * TAP_ROW..p * TAP_ROW]
                .iter()
                .map(|&v| bf16(v))
                .collect();
            let want_frnn = host_mix_norm(&feat, &alpha[3], &markov_norm);
            let fs = tr
                .frnn
                .iter()
                .zip(&want_frnn)
                .map(|(&a, &b)| bf16_steps(a, b))
                .max()
                .unwrap();
            frnn_worst = frnn_worst.max(fs);
            assert_eq!(tr.frnn_row, Some(p - 1));
            windows += 1;
            proposals += k;
            prop_equal += got.iter().zip(&cd).filter(|(a, b)| a == b).count();
            if got == cd {
                win_equal += 1;
                continue;
            }
            // Only the first differing step is comparable (later steps follow
            // a different previous id). Near tie: on the GPU's own logits the
            // CPU's pick sits within MAX_FLIP_LOGIT of the max.
            let j = got.iter().zip(&cd).position(|(a, b)| a != b).unwrap();
            let lg = &tr.logits[j];
            let gap = lg[got[j] as usize] - lg[cd[j] as usize];
            let rec_j = json!({
                "seq": name, "window": i, "pos": p, "step": j, "gpu": got[j], "cpu": cd[j],
                "gpu_logit": lg[got[j] as usize], "gpu_logit_of_cpu_pick": lg[cd[j] as usize],
                "gap": gap, "cpu_accepted": c["accepted"],
            });
            if gap <= MAX_FLIP_LOGIT {
                near_ties.push(rec_j);
            } else {
                unexplained.push(rec_j);
            }
        }
    }
    let _ = gpu.free_tensor(x_head);
    eprintln!(
        "windows: {win_equal}/{windows} identical, proposals {prop_equal}/{proposals}, near ties {}, unexplained {}, f_rnn worst {frnn_worst} steps",
        near_ties.len(),
        unexplained.len()
    );

    // 2b. The moving window through the shared driver (committed binding) and
    // the scalar target over the recorded target. It follows the CPU
    // trajectory exactly until a window's proposals differ (a 2a near tie);
    // past an acceptance change the two trajectories are different decodes.
    let mut drafter = new_dspark_committed_drafter(Box::new(body), 1 << 16);
    let d: &mut dyn MtpDrafter = &mut drafter;
    let mut wide = Vec::new();
    let mut drive_ok = true;
    let (mut cpu_emit, mut cpu_win, mut gpu_emit, mut gpu_win) = (0usize, 0usize, 0usize, 0usize);
    let mut first_run: Option<(std::rc::Rc<Recorded>, Vec<Win>)> = None;
    for (s, rec) in cpu_seqs.iter().zip(&seqs) {
        let name = s["name"].as_str().unwrap();
        let got = run_sequence(d, gpu, rec);
        let cpu = s["trace"].as_array().unwrap();
        let ids = &rec.seq.ids;
        // Every emitted token is the recorded target's (exact greedy AR).
        let tokens_exact = got
            .iter()
            .all(|g| g.committed[..] == ids[g.pos + 1..g.pos + 1 + g.committed.len()]);
        let mut identical_prefix = 0;
        for (c, g) in cpu.iter().zip(&got) {
            let cd: Vec<u32> = serde_json::from_value(c["draft"].clone()).unwrap();
            let same = g.pos == c["pos"].as_u64().unwrap() as usize
                && g.drafts == cd
                && g.accepted == c["accepted"].as_u64().unwrap() as usize
                && g.committed.len() == c["emitted"].as_u64().unwrap() as usize;
            if !same {
                break;
            }
            identical_prefix += 1;
        }
        let all_same = identical_prefix == cpu.len() && got.len() == cpu.len();
        // A divergence must start at a window 2a classified as a near tie.
        let divergence_explained = all_same
            || near_ties
                .iter()
                .any(|t| t["seq"] == json!(name) && t["window"] == json!(identical_prefix));
        drive_ok &= tokens_exact && divergence_explained;
        let ce: usize = cpu
            .iter()
            .map(|c| c["emitted"].as_u64().unwrap() as usize)
            .sum();
        let ge: usize = got.iter().map(|g| g.committed.len()).sum();
        cpu_emit += ce;
        cpu_win += cpu.len();
        gpu_emit += ge;
        gpu_win += got.len();
        eprintln!(
            "drive {name}: {}/{} windows identical before divergence, tau gpu {:.3} cpu {:.3}",
            identical_prefix,
            cpu.len(),
            ge as f64 / got.len() as f64,
            ce as f64 / cpu.len() as f64
        );
        wide.push(json!({
            "name": name, "windows_gpu": got.len(), "windows_cpu": cpu.len(),
            "identical_windows_before_divergence": identical_prefix,
            "all_identical": all_same, "divergence_at_near_tie": divergence_explained,
            "emitted_tokens_equal_recorded_target": tokens_exact,
            "tau_gpu": ge as f64 / got.len() as f64, "tau_cpu": ce as f64 / cpu.len() as f64,
        }));
        if first_run.is_none() {
            first_run = Some((rec.clone(), got));
        }
    }
    let wide_ok = unexplained.is_empty() && frnn_worst <= MAX_FRNN_STEPS && drive_ok;

    // --- 3. state behaviour (bitwise against a cold path) ---
    let mut state = serde_json::Map::new();
    // Reset: the first sequence again after 16 others reproduces its windows.
    let (rec0, run0) = first_run.unwrap();
    let again = run_sequence(d, gpu, &rec0);
    state.insert("reset_rerun_identical".into(), json!(again == run0));
    // A sequence with a first window accepting >= 2 drafts.
    let rec = seqs
        .iter()
        .zip(correct["sequences"].as_array().unwrap())
        .find(|(_, s)| {
            s["trace"][0]["accepted"].as_u64().unwrap() >= 2 && s["trace"][0]["k"] == json!(7)
        })
        .map(|(r, _)| r.clone())
        .expect("a window with two accepted drafts");
    let ids = rec.seq.ids.clone();
    let p = rec.seq.prompt_len;
    let cold = |d: &mut dyn MtpDrafter, gpu: &mut Gpu, at: usize| -> Win {
        let mut t = target(&rec);
        d.mtp_reset(gpu).unwrap();
        let seed = prefill(d, gpu, &mut t, &ids[..at]);
        assert_eq!(seed, ids[at]);
        step(d, gpu, &mut t, at, seed, K_MAX).unwrap()
    };
    let cold_p = cold(d, gpu, p);
    // Forced advance: prefill a shorter prefix, force the rest, draft at P.
    {
        let m = p - 5;
        let mut t = target(&rec);
        d.mtp_reset(gpu).unwrap();
        let first = prefill(d, gpu, &mut t, &ids[..m]);
        let never = || false;
        let advanced = d
            .mtp_forced_advance(gpu, &mut t, &ids[m..p], m, &never)
            .unwrap();
        let w = step(d, gpu, &mut t, p, ids[p], K_MAX).unwrap();
        state.insert(
            "forced_advance_equals_cold".into(),
            json!(first == ids[m] && advanced && w == cold_p),
        );
        // A forced advance from the wrong position is refused, not re-bootstrapped.
        let gap = d.mtp_forced_advance(gpu, &mut t, &ids[p + 1..p + 2], p + 2, &never);
        state.insert("forced_advance_gap_refused".into(), json!(gap.is_err()));
    }
    // k = 0: verify the anchor only, still ingest its row.
    {
        let mut t = target(&rec);
        d.mtp_reset(gpu).unwrap();
        prefill(d, gpu, &mut t, &ids[..p]);
        let w0 = step(d, gpu, &mut t, p, ids[p], 0).unwrap();
        let w1 = step(d, gpu, &mut t, p + 1, ids[p + 1], K_MAX).unwrap();
        let cold_p1 = cold(d, gpu, p + 1);
        state.insert(
            "k0_ingests_anchor_row".into(),
            json!(w0.drafts.is_empty() && w0.committed == vec![ids[p + 1]] && w1 == cold_p1),
        );
    }
    // Terminal-prefix repair: keep two of the first window's tokens, then the
    // next window equals a cold path at P + 2.
    {
        let mut t = target(&rec);
        d.mtp_reset(gpu).unwrap();
        prefill(d, gpu, &mut t, &ids[..p]);
        let w = step(d, gpu, &mut t, p, ids[p], K_MAX).unwrap();
        assert!(w.accepted >= 2);
        let bad_seed = d.mtp_repair_terminal_prefix(gpu, &mut t, p, ids[p] + 1, &w.committed[..2]);
        // A mismatched window consumes the retained window; take another.
        let mut t = target(&rec);
        d.mtp_reset(gpu).unwrap();
        prefill(d, gpu, &mut t, &ids[..p]);
        step(d, gpu, &mut t, p, ids[p], K_MAX).unwrap();
        let repaired = d
            .mtp_repair_terminal_prefix(gpu, &mut t, p, ids[p], &w.committed[..2])
            .unwrap();
        let twice = d
            .mtp_repair_terminal_prefix(gpu, &mut t, p, ids[p], &w.committed[..2])
            .unwrap();
        let next = step(d, gpu, &mut t, p + 2, ids[p + 2], K_MAX).unwrap();
        let cold_p2 = cold(d, gpu, p + 2);
        // Without a restorable target the repair declines and leaves state alone.
        let mut t = target(&rec);
        t.inner.forward().restorable = false;
        d.mtp_reset(gpu).unwrap();
        prefill(d, gpu, &mut t, &ids[..p]);
        let w2 = step(d, gpu, &mut t, p, ids[p], K_MAX).unwrap();
        let declined = d
            .mtp_repair_terminal_prefix(gpu, &mut t, p, ids[p], &w2.committed[..2])
            .unwrap();
        let p_next = p + w2.accepted + 1;
        let cont = step(d, gpu, &mut t, p_next, *w2.committed.last().unwrap(), K_MAX);
        state.insert(
            "terminal_repair".into(),
            json!({
                "window_mismatch_refused": bad_seed.is_err(),
                "repaired": repaired,
                "second_repair_declined": !twice,
                "next_window_equals_cold": next == cold_p2,
                "unrestorable_declines": !declined,
                "unrestorable_state_intact": cont.is_ok(),
            }),
        );
    }
    // The drafter's own state oracle: an anchor off the committed rows fails.
    {
        let mut t = target(&rec);
        d.mtp_reset(gpu).unwrap();
        prefill(d, gpu, &mut t, &ids[..p]);
        let off = d.mtp_step(gpu, &mut t, p + 1, ids[p + 1], &[], K_MAX, EOS, None);
        state.insert(
            "anchor_off_committed_rows_refused".into(),
            json!(off.is_err()),
        );
    }
    let state_ok = state.values().all(|v| match v {
        Value::Bool(b) => *b,
        Value::Object(m) => m.values().all(|x| x.as_bool() == Some(true)),
        _ => false,
    });

    let receipt = json!({
        "schema": "hipfire.inkling2.dspark_body_oracle.v1",
        "draft": {"sidecar": sidecar.display().to_string(), "trunk": trunk.display().to_string(), "fold_key": key},
        "fold_compute_rows_equal": format!("{fold_rows_equal}/{}", sample.len()),
        "tolerances": {"min_row_cos": MIN_ROW_COS, "max_rms_rel": MAX_RMS_REL, "max_frnn_bf16_steps": MAX_FRNN_STEPS, "max_flip_logit_gap": MAX_FLIP_LOGIT},
        "golden": golden,
        "windows_independent": {
            "accept_receipt": accept.display().to_string(),
            "windows": windows, "windows_identical": win_equal,
            "proposals": proposals, "proposals_equal": prop_equal,
            "frnn_worst_bf16_steps": frnn_worst,
            "near_ties": near_ties, "unexplained": unexplained,
        },
        "driver_moving_window": {
            "windows_cpu": cpu_win, "windows_gpu": gpu_win,
            "tau_cpu": cpu_emit as f64 / cpu_win as f64, "tau_gpu": gpu_emit as f64 / gpu_win as f64,
            "sequences": wide,
        },
        "state": state,
        "seconds": t0.elapsed().as_secs_f64(),
        "ok": {"golden": golden_ok, "wide": wide_ok, "state": state_ok, "fold": fold_rows_equal == sample.len()},
    });
    std::fs::write(
        &receipt_path,
        serde_json::to_string_pretty(&receipt).unwrap() + "\n",
    )
    .expect("write receipt");
    eprintln!(
        "oracle: {} in {:.0}s",
        receipt_path.display(),
        t0.elapsed().as_secs_f64()
    );
    assert!(golden_ok, "golden windows");
    assert!(
        wide_ok,
        "windows: {win_equal}/{windows}, unexplained {unexplained:?}, driver {drive_ok}"
    );
    assert!(state_ok, "state behaviour: {state:?}");
    assert_eq!(fold_rows_equal, sample.len(), "fold rows");
}
