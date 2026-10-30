// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! raon_oracle `logits` subcommand (PLAN §6.2, slice I join).
//!
//! Loads the production bundle (`load_raon_bundle`) and exercises the integrated
//! forward (`hipfire_arch_raon::forward`) on a token list. Up to three checks run
//! in one process, in this order; `pass` is the AND of every check that ran:
//!
//! 1. `--teacher F` (logits mode): prefill all tokens (`--prefill serial|batched`,
//!    default batched) and compare the FINAL-row logits with the teacher's last
//!    logits row. Teacher file = `fixtures/logits.npz` written by
//!    `scripts/raon_reference.py fixtures` (uncompressed `np.savez`; arrays
//!    `tokens` int64 `[T]`, `logits` float32 `[T, vocab]`). Reported: top-k ids and
//!    values of both, rank of each side's top-1 in the other, top-k overlap,
//!    max/mean abs diff, relative L2, cosine, KL(teacher || ours) over the full
//!    vocab. Bring-up bands (NOT final KLD gates; stated in the JSON and
//!    overridable by `--min-cosine` / `--top1-in`): cosine >= 0.99 and the
//!    teacher top-1 inside our top-5.
//! 2. `--dump-layers DIR`: reset, run the tokens through `forward_rows_tapped` in
//!    pieces of <= scratch.rows rows, and write per-site stream states
//!    (`[T, 4, hidden]` F32 holding BF16 values) as safetensors, one open file per
//!    site, pieces appended in place (no full buffering). Layout mirrors the
//!    teacher's `cmd_layer_dump` so a diff tool can pair the directories:
//!    `embed.safetensors {stream}`, `layerNN.safetensors {stream_mid` (after the
//!    attention mHC site), `stream_out` (after the FFN site)`}`,
//!    `final.safetensors {logits [1, vocab]}`.
//! 3. `--kld REF.kldref.bin` (HFKLDR v1): per chunk, reset and feed the first
//!    `n_ctx - 1` tokens (NO BOS), scoring positions `[n_ctx/2, n_ctx-2]` with the
//!    exact top-K + residual-tail math of `eval_hipfire_llama.rs::score_position`.
//!    `--max-chunks N`, `--prefill serial|batched` (serial: `forward_token` per
//!    token). The KLD threshold is judged by the caller (`thresholds_applied:false`);
//!    `pass` only requires the run to complete with every value finite.
//!
//! Token file (`--tokens`, also used by `state`/`generate` helpers here): a
//! little-endian u32 binary array, OR a text file of decimal ids separated by
//! whitespace/commas (optional `[`/`]`). A file made only of digits, separators and
//! brackets is text; anything else must be a multiple of 4 bytes and is binary
//! (valid ids are < 2^24, so a binary file always contains a NUL byte).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

use hipfire_arch_raon::bundle::{PrefillMode, RaonBundle};
use hipfire_arch_raon::config::MHC_STREAMS;
use hipfire_arch_raon::forward::{
    forward_rows, forward_rows_tapped, forward_token, readout_logits, TapSite,
};
use hipfire_arch_raon::load_raon_bundle;
use hipfire_runtime::kv_backend::KvBackend;
use hipfire_runtime::loader_api::{CaskConfig, LoadCtx, ModelSource, SpecLoadCfg};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Map, Value};

/// Rows read back per `readout_logits` call in the KLD scorer.
const KLD_READOUT_ROWS: usize = 16;

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers (also used by state.rs)
// ─────────────────────────────────────────────────────────────────────────────

/// Read a token file: u32-LE binary, or whitespace/comma separated decimal text
/// (see the module doc for the disambiguation rule).
pub fn read_tokens(path: &str) -> Result<Vec<u32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read tokens {path}: {e}"))?;
    if bytes.is_empty() {
        return Err(format!("tokens {path}: empty file"));
    }
    let texty = bytes.iter().all(|b| {
        matches!(
            b,
            b'0'..=b'9' | b',' | b' ' | b'\t' | b'\r' | b'\n' | b'[' | b']'
        )
    });
    let ids: Vec<u32> = if texty {
        let s = std::str::from_utf8(&bytes).map_err(|e| format!("tokens {path}: {e}"))?;
        let mut v = Vec::new();
        for t in s
            .split(|c: char| c.is_whitespace() || matches!(c, ',' | '[' | ']'))
            .filter(|t| !t.is_empty())
        {
            v.push(
                t.parse::<u32>()
                    .map_err(|e| format!("tokens {path}: bad id {t:?}: {e}"))?,
            );
        }
        v
    } else if bytes.len() % 4 == 0 {
        bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    } else {
        return Err(format!(
            "tokens {path}: {} bytes is neither a u32le array (multiple of 4) nor a text id list",
            bytes.len()
        ));
    };
    if ids.is_empty() {
        return Err(format!("tokens {path}: no ids"));
    }
    Ok(ids)
}

/// Every id must index the vocabulary.
pub fn check_tokens(tokens: &[u32], vocab: usize, label: &str) -> Result<(), String> {
    if let Some((i, &t)) = tokens
        .iter()
        .enumerate()
        .find(|(_, &t)| t as usize >= vocab)
    {
        return Err(format!("{label}: token[{i}] = {t} >= vocab {vocab}"));
    }
    Ok(())
}

/// Production carrier load at an explicit `max_seq`, PP=1, no drafter/vision.
pub fn load_bundle(
    gpu: &mut Gpu,
    src: ModelSource,
    path: &str,
    max_seq: usize,
) -> Result<RaonBundle, String> {
    let cask = CaskConfig::default();
    let mut ctx = LoadCtx {
        path,
        max_seq,
        sequence: None,
        deepseek4_compute_placement: Default::default(),
        deepseek4_experts_per_token: None,
        draft_path: None,
        vision_path: None,
        mtp_path: None,
        vision_mode: "off".to_string(),
        kv_mode_override: None,
        kv_k_override: None,
        kv_v_override: None,
        qwen_default_q8: false,
        kv_backend: KvBackend::default(),
        kv_adaptive_override: None,
        state_quant_override: None,
        cask: &cask,
        pp: 1,
        pp_layer_counts: None,
        admitted_mesh: None,
        band_placement: None,
        spec: SpecLoadCfg::default(),
        gpu,
        gemma4_drafter_path: None,
        gemma4_draft_len: 3,
        xdna: None,
        batch_request: Default::default(),
    };
    load_raon_bundle(src, &mut ctx)
}

/// `--prefill serial|batched` (default batched).
pub fn parse_prefill(opts: &BTreeMap<String, String>) -> Result<PrefillMode, String> {
    match opts.get("prefill").map(String::as_str).unwrap_or("batched") {
        "batched" => Ok(PrefillMode::Batched),
        "serial" => Ok(PrefillMode::Serial),
        other => Err(format!("--prefill `{other}`: serial|batched")),
    }
}

fn mode_name(m: PrefillMode) -> &'static str {
    match m {
        PrefillMode::Serial => "serial",
        PrefillMode::Batched => "batched",
    }
}

fn hip_str<T>(r: hip_bridge::HipResult<T>, what: &str) -> Result<T, String> {
    r.map_err(|e| format!("{what}: {e:?}"))
}

fn sync(gpu: &Gpu) -> Result<(), String> {
    hip_str(gpu.hip.device_synchronize(), "device synchronize")
}

fn opt_usize(opts: &BTreeMap<String, String>, key: &str, default: usize) -> Result<usize, String> {
    match opts.get(key) {
        Some(s) => s
            .parse::<usize>()
            .map_err(|e| format!("--{key} `{s}`: {e}")),
        None => Ok(default),
    }
}

fn opt_f64(opts: &BTreeMap<String, String>, key: &str, default: f64) -> Result<f64, String> {
    match opts.get(key) {
        Some(s) => s.parse::<f64>().map_err(|e| format!("--{key} `{s}`: {e}")),
        None => Ok(default),
    }
}

fn resolve(fixtures: &Path, p: &str) -> PathBuf {
    let direct = PathBuf::from(p);
    if direct.exists() {
        direct
    } else {
        fixtures.join(p)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry
// ─────────────────────────────────────────────────────────────────────────────

pub fn run(fixtures: &Path, opts: &BTreeMap<String, String>) -> Result<Value, String> {
    let model = opts.get("model").ok_or("logits: missing --model")?.clone();
    let teacher = opts.get("teacher").cloned();
    let dump_dir = opts.get("dump-layers").cloned();
    let kld_path = opts.get("kld").cloned();
    if teacher.is_none() && dump_dir.is_none() && kld_path.is_none() {
        return Err(
            "logits: need at least one of --teacher F, --dump-layers DIR, --kld REF.kldref.bin"
                .into(),
        );
    }
    let mode = parse_prefill(opts)?;
    let topk = opt_usize(opts, "topk", 5)?.max(1);

    let tokens: Option<Vec<u32>> = if teacher.is_some() || dump_dir.is_some() {
        let p = opts
            .get("tokens")
            .ok_or("logits: --teacher/--dump-layers need --tokens F")?;
        Some(read_tokens(&resolve(fixtures, p).to_string_lossy())?)
    } else {
        None
    };
    let kref = match &kld_path {
        Some(p) => Some(open_kldref(&resolve(fixtures, p))?),
        None => None,
    };

    let mut max_seq = 512usize;
    if let Some(t) = &tokens {
        max_seq = max_seq.max(t.len() + 64);
    }
    if let Some(k) = &kref {
        max_seq = max_seq.max(k.n_ctx + 16);
    }

    let mut gpu = Gpu::init().map_err(|e| format!("logits: gpu init: {e:?}"))?;
    let model_path = resolve(fixtures, &model).to_string_lossy().into_owned();
    let src = ModelSource::from_path(&model_path)?;
    let mut bundle = load_bundle(&mut gpu, src, &model_path, max_seq)?;
    // `--mirror-budget-rows N`: force the decompressed-route mirror budget so
    // the beyond-budget (absorbed, HIPFIRE_RAON_MLA_ABSORB=1) route can be
    // scored on the KLD corpora.
    if let Some(v) = opts.get("mirror-budget-rows") {
        let rows = v
            .parse::<usize>()
            .map_err(|e| format!("--mirror-budget-rows `{v}`: {e}"))?;
        bundle.state.set_mirror_budget_rows(rows);
    }
    let result = run_checks(
        fixtures,
        &mut gpu,
        &mut bundle,
        opts,
        mode,
        topk,
        tokens.as_deref(),
        teacher.as_deref(),
        dump_dir.as_deref(),
        kref,
    );
    bundle.free_all(&mut gpu);
    let mut result = result?;
    result["model"] = json!(model_path);
    result["prefill"] = json!(mode_name(mode));
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn run_checks(
    fixtures: &Path,
    gpu: &mut Gpu,
    bundle: &mut RaonBundle,
    opts: &BTreeMap<String, String>,
    mode: PrefillMode,
    topk: usize,
    tokens: Option<&[u32]>,
    teacher: Option<&str>,
    dump_dir: Option<&str>,
    kref: Option<KldRef>,
) -> Result<Value, String> {
    let vocab = bundle.config.vocab;
    let mut out = json!({});
    let mut pass = true;
    if let Some(t) = tokens {
        check_tokens(t, vocab, "tokens")?;
        out["n_tokens"] = json!(t.len());
    }
    if let (Some(tok), Some(tp)) = (tokens, teacher) {
        let r = logits_check(
            fixtures,
            gpu,
            bundle,
            tok,
            &resolve(fixtures, tp),
            mode,
            topk,
            opts,
        )?;
        pass &= r["pass"].as_bool() == Some(true);
        out["logits"] = r;
    }
    if let (Some(tok), Some(dir)) = (tokens, dump_dir) {
        let r = dump_layers(gpu, bundle, tok, Path::new(dir))?;
        pass &= r["pass"].as_bool() == Some(true);
        out["dump_layers"] = r;
    }
    if let Some(k) = kref {
        let max_chunks = match opts.get("max-chunks") {
            Some(s) => Some(
                s.parse::<usize>()
                    .map_err(|e| format!("--max-chunks `{s}`: {e}"))?,
            ),
            None => None,
        };
        let r = kld_check(gpu, bundle, k, mode, max_chunks)?;
        pass &= r["pass"].as_bool() == Some(true);
        out["kld"] = r;
    }
    out["pass"] = json!(pass);
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// (1) teacher logits comparison
// ─────────────────────────────────────────────────────────────────────────────

fn rd_u16(b: &[u8], o: usize) -> Result<u16, String> {
    b.get(o..o + 2)
        .map(|s| u16::from_le_bytes([s[0], s[1]]))
        .ok_or_else(|| format!("zip: truncated read at {o}"))
}
fn rd_u32(b: &[u8], o: usize) -> Result<u32, String> {
    b.get(o..o + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        .ok_or_else(|| format!("zip: truncated read at {o}"))
}

/// Bytes of the STORED member `name` (e.g. `logits.npy`) of an uncompressed zip.
fn npz_member<'a>(zip: &'a [u8], name: &str) -> Result<&'a [u8], String> {
    if zip.len() < 22 {
        return Err("npz: file shorter than a zip end-of-central-directory record".into());
    }
    let lo = zip.len().saturating_sub(22 + 65535);
    let mut eocd = None;
    let mut i = zip.len() - 22;
    loop {
        if zip[i..i + 4] == [0x50, 0x4b, 0x05, 0x06] {
            eocd = Some(i);
            break;
        }
        if i == lo {
            break;
        }
        i -= 1;
    }
    let eocd = eocd.ok_or("npz: no zip end-of-central-directory record")?;
    let entries = rd_u16(zip, eocd + 10)? as usize;
    let cd_off = rd_u32(zip, eocd + 16)? as usize;
    if entries == 0xFFFF || cd_off == 0xFFFF_FFFF {
        return Err("npz: zip64 archives are not supported".into());
    }
    let mut p = cd_off;
    for _ in 0..entries {
        if zip.get(p..p + 4) != Some(&[0x50, 0x4b, 0x01, 0x02][..]) {
            return Err(format!("npz: bad central directory entry at {p}"));
        }
        let method = rd_u16(zip, p + 10)?;
        let csize = rd_u32(zip, p + 20)? as usize;
        let usize_ = rd_u32(zip, p + 24)? as usize;
        let nlen = rd_u16(zip, p + 28)? as usize;
        let elen = rd_u16(zip, p + 30)? as usize;
        let clen = rd_u16(zip, p + 32)? as usize;
        let lh = rd_u32(zip, p + 42)? as usize;
        let fname = zip
            .get(p + 46..p + 46 + nlen)
            .ok_or("npz: truncated central directory file name")?;
        if fname == name.as_bytes() {
            if method != 0 {
                return Err(format!(
                    "npz member {name}: compression method {method} (deflate) is unsupported; \
                     regenerate with np.savez (stored), not np.savez_compressed"
                ));
            }
            if csize != usize_ {
                return Err(format!(
                    "npz member {name}: stored size mismatch {csize} != {usize_}"
                ));
            }
            if zip.get(lh..lh + 4) != Some(&[0x50, 0x4b, 0x03, 0x04][..]) {
                return Err(format!("npz member {name}: bad local header at {lh}"));
            }
            let ln = rd_u16(zip, lh + 26)? as usize;
            let le = rd_u16(zip, lh + 28)? as usize;
            let start = lh + 30 + ln + le;
            return zip
                .get(start..start + csize)
                .ok_or_else(|| format!("npz member {name}: data runs past end of file"));
        }
        p += 46 + nlen + elen + clen;
    }
    Err(format!("npz: no member named {name}"))
}

struct Npy<'a> {
    descr: String,
    shape: Vec<usize>,
    data: &'a [u8],
}

fn parse_npy(raw: &[u8]) -> Result<Npy<'_>, String> {
    if raw.len() < 10 || &raw[..6] != b"\x93NUMPY" {
        return Err("npy: bad magic".into());
    }
    let major = raw[6];
    let (hlen, hstart) = match major {
        1 => (rd_u16(raw, 8)? as usize, 10usize),
        2 | 3 => (rd_u32(raw, 8)? as usize, 12usize),
        v => return Err(format!("npy: unsupported format version {v}")),
    };
    let header = raw
        .get(hstart..hstart + hlen)
        .ok_or("npy: truncated header")?;
    let header = String::from_utf8_lossy(header).into_owned();
    let key_pos = |key: &str| -> Result<usize, String> {
        header
            .find(key)
            .map(|i| i + key.len())
            .ok_or_else(|| format!("npy header lacks {key}: {header}"))
    };
    let d = key_pos("'descr':")?;
    let rest = header[d..].trim_start();
    let q = rest.chars().next().ok_or("npy: empty descr")?;
    if q != '\'' && q != '"' {
        return Err(format!("npy: unsupported descr in {header}"));
    }
    let end = rest[1..].find(q).ok_or("npy: unterminated descr")?;
    let descr = rest[1..1 + end].to_string();
    let f = key_pos("'fortran_order':")?;
    if header[f..].trim_start().starts_with("True") {
        return Err("npy: fortran_order arrays are unsupported".into());
    }
    let s = key_pos("'shape':")?;
    let rest = header[s..].trim_start();
    let open = rest.find('(').ok_or("npy: shape tuple missing")?;
    let close = rest.find(')').ok_or("npy: shape tuple unterminated")?;
    let mut shape = Vec::new();
    for t in rest[open + 1..close]
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
    {
        shape.push(
            t.parse::<usize>()
                .map_err(|e| format!("npy shape {t:?}: {e}"))?,
        );
    }
    Ok(Npy {
        descr,
        shape,
        data: &raw[hstart + hlen..],
    })
}

/// The teacher's last logits row (and the token list, when the file carries it).
fn load_teacher(path: &Path, vocab: usize) -> Result<(Vec<f32>, Option<Vec<i64>>, usize), String> {
    let zip = std::fs::read(path).map_err(|e| format!("read teacher {}: {e}", path.display()))?;
    let lg = parse_npy(npz_member(&zip, "logits.npy")?)?;
    if lg.descr != "<f4" {
        return Err(format!("teacher logits dtype {} != <f4", lg.descr));
    }
    if lg.shape.len() != 2 || lg.shape[1] != vocab || lg.shape[0] == 0 {
        return Err(format!(
            "teacher logits shape {:?}, want [T, {vocab}]",
            lg.shape
        ));
    }
    let rows = lg.shape[0];
    let row_bytes = vocab * 4;
    if lg.data.len() < rows * row_bytes {
        return Err(format!(
            "teacher logits data {} B < {} B for shape {:?}",
            lg.data.len(),
            rows * row_bytes,
            lg.shape
        ));
    }
    let last = &lg.data[(rows - 1) * row_bytes..rows * row_bytes];
    let row: Vec<f32> = last
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let toks = match npz_member(&zip, "tokens.npy") {
        Ok(raw) => {
            let t = parse_npy(raw)?;
            let w = match t.descr.as_str() {
                "<i8" => 8usize,
                "<i4" | "<u4" => 4,
                d => return Err(format!("teacher tokens dtype {d} unsupported")),
            };
            Some(
                t.data
                    .chunks_exact(w)
                    .map(|b| {
                        if w == 8 {
                            i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
                        } else {
                            u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64
                        }
                    })
                    .collect::<Vec<i64>>(),
            )
        }
        Err(_) => None,
    };
    Ok((row, toks, rows))
}

fn top_indices(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    let k = k.min(idx.len());
    let cmp = |&a: &usize, &b: &usize| v[b].total_cmp(&v[a]).then(a.cmp(&b));
    if k < idx.len() {
        idx.select_nth_unstable_by(k - 1, cmp);
        idx.truncate(k);
    }
    idx.sort_by(cmp);
    idx
}

/// 0-based rank of `id` in `v` (descending, lower index wins ties).
fn rank_of(v: &[f32], id: usize) -> usize {
    let x = v[id];
    v.iter()
        .enumerate()
        .filter(|&(j, &y)| y > x || (y == x && j < id))
        .count()
}

fn log_sum_exp(v: &[f32]) -> f64 {
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let s: f64 = v.iter().map(|&x| ((x - m) as f64).exp()).sum();
    m as f64 + s.ln()
}

fn top_json(v: &[f32], idx: &[usize]) -> Value {
    Value::Array(
        idx.iter()
            .map(|&i| json!({"id": i, "logit": v[i]}))
            .collect(),
    )
}

#[allow(clippy::too_many_arguments)]
fn logits_check(
    _fixtures: &Path,
    gpu: &mut Gpu,
    bundle: &mut RaonBundle,
    tokens: &[u32],
    teacher_path: &Path,
    mode: PrefillMode,
    topk: usize,
    opts: &BTreeMap<String, String>,
) -> Result<Value, String> {
    let vocab = bundle.config.vocab;
    let min_cos = opt_f64(opts, "min-cosine", 0.99)?;
    let top1_in = opt_usize(opts, "top1-in", 5)?.max(1);

    let (teacher, teacher_tokens, teacher_rows) = load_teacher(teacher_path, vocab)?;
    if let Some(tt) = &teacher_tokens {
        let same = tt.len() == tokens.len() && tt.iter().zip(tokens).all(|(&a, &b)| a == b as i64);
        if !same {
            return Err(format!(
                "teacher tokens ({} ids) differ from --tokens ({} ids): the teacher was computed on a different prompt",
                tt.len(),
                tokens.len()
            ));
        }
    }
    if teacher_rows != tokens.len() {
        return Err(format!(
            "teacher has {teacher_rows} logits rows, --tokens has {} ids",
            tokens.len()
        ));
    }

    bundle.set_prefill_mode(mode);
    bundle.reset(gpu)?;
    let t0 = Instant::now();
    bundle.prefill(gpu, tokens)?;
    sync(gpu)?;
    let secs = t0.elapsed().as_secs_f64();
    if bundle.n_tokens() != tokens.len() {
        return Err(format!(
            "state advanced to {} tokens, expected {}",
            bundle.n_tokens(),
            tokens.len()
        ));
    }
    let ours = hip_str(gpu.download_f32(bundle.logits()), "download logits")?;
    if ours.len() != vocab {
        return Err(format!(
            "device logits have {} elements, vocab is {vocab}",
            ours.len()
        ));
    }
    let nonfinite = ours.iter().filter(|x| !x.is_finite()).count();

    let mut max_abs = 0.0f64;
    let mut sum_abs = 0.0f64;
    let (mut dd, mut tt, mut oo, mut ot) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&o, &t) in ours.iter().zip(&teacher) {
        let (o, t) = (o as f64, t as f64);
        let d = (o - t).abs();
        max_abs = max_abs.max(d);
        sum_abs += d;
        dd += (o - t) * (o - t);
        tt += t * t;
        oo += o * o;
        ot += o * t;
    }
    let rel_l2 = (dd / tt.max(1e-300)).sqrt();
    let cosine = ot / (oo.sqrt() * tt.sqrt()).max(1e-300);
    let mean_abs = sum_abs / vocab as f64;

    let (lz_t, lz_o) = (log_sum_exp(&teacher), log_sum_exp(&ours));
    let mut kl = 0.0f64;
    for (&o, &t) in ours.iter().zip(&teacher) {
        let lp_t = t as f64 - lz_t;
        let lp_o = o as f64 - lz_o;
        kl += lp_t.exp() * (lp_t - lp_o);
    }

    let k = topk.min(vocab);
    let top_o = top_indices(&ours, k.max(top1_in));
    let top_t = top_indices(&teacher, k.max(top1_in));
    let t1 = top_t[0];
    let o1 = top_o[0];
    let teacher_top1_rank_in_ours = rank_of(&ours, t1);
    let our_top1_rank_in_teacher = rank_of(&teacher, o1);
    let overlap = top_o[..k].iter().filter(|i| top_t[..k].contains(i)).count();

    let cos_ok = cosine.is_finite() && cosine >= min_cos;
    let top1_ok = teacher_top1_rank_in_ours < top1_in;
    let pass = nonfinite == 0 && cos_ok && top1_ok;

    Ok(json!({
        "pass": pass,
        "bands": {
            "kind": "bring-up (not final KLD gates)",
            "min_cosine": min_cos,
            "teacher_top1_within_our_top": top1_in,
        },
        "checks": {"finite": nonfinite == 0, "cosine": cos_ok, "teacher_top1_in_our_topn": top1_ok},
        "teacher": teacher_path.display().to_string(),
        "compared_row": "last",
        "n_tokens": tokens.len(),
        "prefill_secs": secs,
        "nonfinite_logits": nonfinite,
        "cosine": cosine,
        "rel_l2": rel_l2,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "kl_teacher_to_ours": kl,
        "teacher_top1": t1,
        "our_top1": o1,
        "top1_equal": t1 == o1,
        "teacher_top1_rank_in_ours": teacher_top1_rank_in_ours,
        "our_top1_rank_in_teacher": our_top1_rank_in_teacher,
        "topk": k,
        "topk_overlap": overlap,
        "ours_topk": top_json(&ours, &top_o[..k]),
        "teacher_topk": top_json(&teacher, &top_t[..k]),
    }))
}

// ─────────────────────────────────────────────────────────────────────────────
// (2) layer dump (safetensors, appended in place)
// ─────────────────────────────────────────────────────────────────────────────

/// One safetensors F32 file whose header is written up front and whose tensor
/// rows are filled in place (`write_all_at`), so pieces never accumulate in RAM.
struct StFile {
    f: File,
    data_start: u64,
    /// Per tensor: (data offset, row bytes, rows).
    tensors: Vec<(u64, usize, usize)>,
    path: PathBuf,
}

impl StFile {
    /// `tensors`: (name, shape) in file order; row = product of trailing dims.
    fn create(
        path: &Path,
        tensors: &[(&str, Vec<usize>)],
        meta: &[(&str, String)],
    ) -> Result<Self, String> {
        let mut header = Map::new();
        let mut m = Map::new();
        for (k, v) in meta {
            m.insert((*k).to_string(), json!(v));
        }
        header.insert("__metadata__".into(), Value::Object(m));
        let mut off = 0u64;
        let mut ents = Vec::new();
        for (name, shape) in tensors {
            if shape.is_empty() || shape.contains(&0) {
                return Err(format!("{}: empty shape for {name}", path.display()));
            }
            let numel: usize = shape.iter().product();
            let bytes = (numel * 4) as u64;
            header.insert(
                (*name).to_string(),
                json!({"dtype": "F32", "shape": shape, "data_offsets": [off, off + bytes]}),
            );
            ents.push((off, numel / shape[0] * 4, shape[0]));
            off += bytes;
        }
        let mut hs = serde_json::to_string(&Value::Object(header)).map_err(|e| e.to_string())?;
        while (8 + hs.len()) % 8 != 0 {
            hs.push(' ');
        }
        let data_start = 8 + hs.len() as u64;
        let f = File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
        f.write_all_at(&(hs.len() as u64).to_le_bytes(), 0)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        f.write_all_at(hs.as_bytes(), 8)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        f.set_len(data_start + off)
            .map_err(|e| format!("{}: size: {e}", path.display()))?;
        Ok(StFile {
            f,
            data_start,
            tensors: ents,
            path: path.to_path_buf(),
        })
    }

    /// Write whole rows `row0..row0 + bytes/row_bytes` of tensor `t`.
    fn write_rows(&self, t: usize, row0: usize, bytes: &[u8]) -> Result<(), String> {
        let (off, row_bytes, rows) = self.tensors[t];
        if bytes.len() % row_bytes != 0 || row0 + bytes.len() / row_bytes > rows {
            return Err(format!(
                "{}: tensor {t} rows [{row0}, +{}) outside {rows}",
                self.path.display(),
                bytes.len() / row_bytes
            ));
        }
        self.f
            .write_all_at(bytes, self.data_start + off + (row0 * row_bytes) as u64)
            .map_err(|e| format!("{}: write: {e}", self.path.display()))
    }
}

fn dump_layers(
    gpu: &mut Gpu,
    bundle: &mut RaonBundle,
    tokens: &[u32],
    dir: &Path,
) -> Result<Value, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let n = tokens.len();
    let hidden = bundle.config.hidden;
    let n_layers = bundle.config.n_layers;
    let vocab = bundle.config.vocab;
    let row_bytes = MHC_STREAMS * hidden * 4;
    let piece = bundle.scratch.rows.min(512);
    let meta = [
        ("tokens", n.to_string()),
        ("kind", "layer-dump".to_string()),
        ("producer", "raon_oracle logits".to_string()),
    ];
    let shape = vec![n, MHC_STREAMS, hidden];

    let embed = StFile::create(
        &dir.join("embed.safetensors"),
        &[("stream", shape.clone())],
        &meta,
    )?;
    let mut layers = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        layers.push(StFile::create(
            &dir.join(format!("layer{l:02}.safetensors")),
            &[("stream_mid", shape.clone()), ("stream_out", shape.clone())],
            &meta,
        )?);
    }

    bundle.reset(gpu)?;
    bundle.state.ensure_capacity(gpu, n)?;
    let t0 = Instant::now();
    let mut host: Vec<u8> = Vec::new();
    let mut pos = 0usize;
    while pos < n {
        let len = (n - pos).min(piece);
        let last = pos + len == n;
        let row0 = pos;
        let mut tap =
            |g: &mut Gpu, site: TapSite, streams: &GpuTensor, rows: usize| -> Result<(), String> {
                if rows != len {
                    return Err(format!("tap {site:?}: {rows} rows, piece has {len}"));
                }
                let nbytes = rows * row_bytes;
                if streams.buf.size() < nbytes {
                    return Err(format!(
                        "tap {site:?}: stream tensor {} B < {nbytes} B",
                        streams.buf.size()
                    ));
                }
                hip_str(g.hip.device_synchronize(), "tap synchronize")?;
                host.resize(nbytes, 0);
                let view = streams.buf.byte_view(0, nbytes);
                hip_str(g.hip.memcpy_dtoh(&mut host, &view), "tap download")?;
                let (file, t) = match site {
                    TapSite::Embedding => (&embed, 0),
                    TapSite::Attn(l) => (
                        layers
                            .get(l)
                            .ok_or_else(|| format!("tap layer {l} out of range"))?,
                        0,
                    ),
                    TapSite::Ffn(l) => (
                        layers
                            .get(l)
                            .ok_or_else(|| format!("tap layer {l} out of range"))?,
                        1,
                    ),
                };
                file.write_rows(t, row0, &host)
            };
        forward_rows_tapped(
            &bundle.config,
            &bundle.weights,
            &mut bundle.state,
            &mut bundle.scratch,
            &mut bundle.work,
            gpu,
            &tokens[pos..pos + len],
            pos as u32,
            last,
            Some(&mut tap),
        )?;
        pos += len;
    }
    sync(gpu)?;
    let secs = t0.elapsed().as_secs_f64();
    let logits = hip_str(
        gpu.download_f32(&bundle.scratch.logits),
        "download final logits",
    )?;
    if logits.len() != vocab {
        return Err(format!(
            "final logits have {} elements, vocab is {vocab}",
            logits.len()
        ));
    }
    let finite = logits.iter().all(|x| x.is_finite());
    let fin = StFile::create(
        &dir.join("final.safetensors"),
        &[("logits", vec![1, vocab])],
        &meta,
    )?;
    let bytes: Vec<u8> = logits.iter().flat_map(|x| x.to_le_bytes()).collect();
    fin.write_rows(0, 0, &bytes)?;

    let mut files = vec!["embed.safetensors".to_string()];
    files.extend((0..n_layers).map(|l| format!("layer{l:02}.safetensors")));
    files.push("final.safetensors".into());
    Ok(json!({
        "pass": finite && bundle.n_tokens() == n,
        "dir": dir.display().to_string(),
        "n_tokens": n,
        "piece_rows": piece,
        "secs": secs,
        "final_logits_finite": finite,
        "files": files,
        "layout": "embed{stream}, layerNN{stream_mid,stream_out}, final{logits[1,vocab]}; streams [T,4,hidden] F32 (BF16 values)",
    }))
}

// ─────────────────────────────────────────────────────────────────────────────
// (3) HFKLDR v1 scoring
// ─────────────────────────────────────────────────────────────────────────────

struct KldRef {
    n_ctx: usize,
    n_vocab: usize,
    n_chunk: usize,
    top_k: usize,
    tokens: Vec<u32>,
    reader: BufReader<File>,
    path: PathBuf,
}

fn open_kldref(path: &Path) -> Result<KldRef, String> {
    let file = File::open(path).map_err(|e| format!("open kld ref {}: {e}", path.display()))?;
    let len = file.metadata().map_err(|e| e.to_string())?.len();
    let mut r = BufReader::with_capacity(8 << 20, file);
    let mut hdr = [0u8; 32];
    r.read_exact(&mut hdr)
        .map_err(|e| format!("{}: header: {e}", path.display()))?;
    if &hdr[..8] != b"HFKLDR\0\0" {
        return Err(format!("{}: bad magic {:?}", path.display(), &hdr[..8]));
    }
    let u32_at =
        |o: usize| u32::from_le_bytes([hdr[o], hdr[o + 1], hdr[o + 2], hdr[o + 3]]) as usize;
    let version = u32_at(8);
    if version != 1 {
        return Err(format!(
            "{}: unsupported HFKLDR version {version} (this reader is v1)",
            path.display()
        ));
    }
    let n_ctx = u32_at(12);
    let n_vocab = u32_at(16);
    let n_chunk = u32_at(20);
    let top_k = u16::from_le_bytes([hdr[24], hdr[25]]) as usize;
    if n_ctx < 4 || n_chunk == 0 || top_k == 0 {
        return Err(format!(
            "{}: degenerate header n_ctx={n_ctx} n_chunk={n_chunk} top_k={top_k}",
            path.display()
        ));
    }
    let scored = n_ctx - 1 - n_ctx / 2;
    let expect = 32 + (n_ctx * n_chunk * 4) as u64 + (n_chunk * scored * (8 + 8 * top_k)) as u64;
    if len != expect {
        return Err(format!(
            "{}: size {len} != expected {expect} for the declared header",
            path.display()
        ));
    }
    let mut raw = vec![0u8; n_ctx * n_chunk * 4];
    r.read_exact(&mut raw)
        .map_err(|e| format!("{}: tokens: {e}", path.display()))?;
    let tokens = raw
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(KldRef {
        n_ctx,
        n_vocab,
        n_chunk,
        top_k,
        tokens,
        reader: r,
        path: path.to_path_buf(),
    })
}

struct Block {
    idx: Vec<u32>,
    lp: Vec<f32>,
    resid: f32,
}

fn parse_block(buf: &[u8], k: usize) -> Block {
    let idx = (0..k)
        .map(|j| u32::from_le_bytes([buf[j * 4], buf[j * 4 + 1], buf[j * 4 + 2], buf[j * 4 + 3]]))
        .collect();
    let lo = k * 4;
    let lp = (0..k)
        .map(|j| {
            f32::from_le_bytes([
                buf[lo + j * 4],
                buf[lo + j * 4 + 1],
                buf[lo + j * 4 + 2],
                buf[lo + j * 4 + 3],
            ])
        })
        .collect();
    let ro = k * 8;
    let resid = f32::from_le_bytes([buf[ro], buf[ro + 1], buf[ro + 2], buf[ro + 3]]);
    Block { idx, lp, resid }
}

#[derive(Clone, Copy)]
struct Score {
    kld: f64,
    nll: Option<f64>,
    top1_agree: bool,
}

/// Same math as `eval_hipfire_llama.rs::score_position`: log-Z over the full
/// vocab in f64, top-K terms plus the residual-tail cross term, NLL of the
/// actual next token. `top1_agree`: our first-max argmax equals the ref's rank-0 id.
fn score_row(logits: &[f32], b: &Block, actual_next: usize) -> Score {
    let mut max_logit = f32::NEG_INFINITY;
    let mut argmax = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v > max_logit {
            max_logit = v;
            argmax = i;
        }
    }
    let mut sum_exp = 0.0f64;
    for &v in logits {
        sum_exp += ((v - max_logit) as f64).exp();
    }
    let log_z = (max_logit as f64) + sum_exp.ln();

    let mut kld = 0.0f64;
    let mut sum_p_cand = 0.0f64;
    for j in 0..b.idx.len() {
        let ri = b.idx[j] as usize;
        if ri >= logits.len() {
            continue;
        }
        let log_p_ref = b.lp[j] as f64;
        let log_p_cand = (logits[ri] as f64) - log_z;
        let p_ref = log_p_ref.exp();
        let p_cand = log_p_cand.exp();
        kld += p_ref * (log_p_ref - log_p_cand);
        sum_p_cand += p_cand;
    }
    let resid_ref = b.resid as f64;
    let resid_cand = (1.0 - sum_p_cand).max(0.0);
    if resid_ref > 1e-9 && resid_cand > 1e-9 {
        kld += resid_ref * (resid_ref.ln() - resid_cand.ln());
    }
    // Negative values beyond fp round-off are clamped exactly like the in-tree scorer.
    let kld = kld.max(0.0);
    let nll = if actual_next < logits.len() {
        Some(-((logits[actual_next] as f64) - log_z))
    } else {
        None
    };
    Score {
        kld,
        nll,
        top1_agree: b.idx.first().map(|&t| t as usize) == Some(argmax),
    }
}

fn score_batch(rows: &[f32], vocab: usize, blocks: &[Block], actual: &[usize]) -> Vec<Score> {
    let n = blocks.len();
    let threads = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1)
        .min(4)
        .min(n)
        .max(1);
    let per = n.div_ceil(threads);
    let mut out: Vec<Option<Score>> = vec![None; n];
    std::thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(per).enumerate() {
            let base = ci * per;
            s.spawn(move || {
                for (j, slot) in chunk.iter_mut().enumerate() {
                    let i = base + j;
                    *slot = Some(score_row(
                        &rows[i * vocab..(i + 1) * vocab],
                        &blocks[i],
                        actual[i],
                    ));
                }
            });
        }
    });
    out.into_iter().map(|s| s.expect("scored row")).collect()
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        f64::NAN
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn p99(sorted: &[f64]) -> f64 {
    let i = ((sorted.len() as f64 * 0.99) as usize).min(sorted.len() - 1);
    sorted[i]
}

fn kld_check(
    gpu: &mut Gpu,
    bundle: &mut RaonBundle,
    mut kref: KldRef,
    mode: PrefillMode,
    max_chunks: Option<usize>,
) -> Result<Value, String> {
    let vocab = bundle.config.vocab;
    if kref.n_vocab != vocab {
        return Err(format!(
            "{}: ref n_vocab {} != model vocab {vocab}",
            kref.path.display(),
            kref.n_vocab
        ));
    }
    check_tokens(&kref.tokens, vocab, "kld ref tokens")?;
    let n_ctx = kref.n_ctx;
    let n_feed = n_ctx - 1;
    let half = n_ctx / 2;
    let scored_per_chunk = n_ctx - 1 - half;
    let chunks = max_chunks.map_or(kref.n_chunk, |m| m.min(kref.n_chunk));
    if chunks == 0 {
        return Err("--max-chunks 0".into());
    }
    let rows_cap = bundle.scratch.rows;
    let block_bytes = 8 + 8 * kref.top_k;
    let mut block_buf = vec![0u8; block_bytes];
    let batched = mode == PrefillMode::Batched;
    let out_rows = KLD_READOUT_ROWS.min(rows_cap).max(1);
    let readout = if batched {
        Some(hip_str(
            gpu.alloc_tensor(&[out_rows, vocab], DType::F32),
            "alloc readout tensor",
        )?)
    } else {
        None
    };

    let run = (|| -> Result<Value, String> {
        let t0 = Instant::now();
        let mut all_kld: Vec<f64> = Vec::with_capacity(scored_per_chunk * chunks);
        let mut all_nll: Vec<f64> = Vec::with_capacity(scored_per_chunk * chunks);
        let mut top1_hits = 0usize;
        let mut nonfinite = 0usize;
        let mut per_chunk: Vec<Value> = Vec::with_capacity(chunks);

        for c in 0..chunks {
            let chunk = &kref.tokens[c * n_ctx..(c + 1) * n_ctx];
            let mut ck_kld: Vec<f64> = Vec::with_capacity(scored_per_chunk);
            let mut ck_nll: Vec<f64> = Vec::with_capacity(scored_per_chunk);
            let mut ck_top1 = 0usize;
            bundle.reset(gpu)?;
            bundle.state.ensure_capacity(gpu, n_feed)?;
            let tc = Instant::now();

            let mut take = |scores: Vec<Score>| {
                for s in scores {
                    if !s.kld.is_finite() || s.nll.is_some_and(|x| !x.is_finite()) {
                        nonfinite += 1;
                    }
                    ck_kld.push(s.kld);
                    if let Some(x) = s.nll {
                        ck_nll.push(x);
                    }
                    ck_top1 += s.top1_agree as usize;
                }
            };

            if batched {
                let out_t = readout.as_ref().expect("batched readout tensor");
                let mut pos = 0usize;
                while pos < n_feed {
                    let len = (n_feed - pos).min(rows_cap);
                    forward_rows(
                        &bundle.config,
                        &bundle.weights,
                        &mut bundle.state,
                        &mut bundle.scratch,
                        &mut bundle.work,
                        gpu,
                        &chunk[pos..pos + len],
                        pos as u32,
                        false,
                    )?;
                    let mut p = half.max(pos);
                    while p < pos + len {
                        let cnt = (pos + len - p).min(out_rows);
                        readout_logits(
                            &bundle.config,
                            &bundle.weights,
                            &mut bundle.scratch,
                            &mut bundle.work,
                            gpu,
                            p - pos,
                            cnt,
                            out_t,
                        )?;
                        let host = hip_str(gpu.download_f32(out_t), "download readout")?;
                        let mut blocks = Vec::with_capacity(cnt);
                        let mut actual = Vec::with_capacity(cnt);
                        for i in 0..cnt {
                            kref.reader
                                .read_exact(&mut block_buf)
                                .map_err(|e| format!("{}: ref block: {e}", kref.path.display()))?;
                            blocks.push(parse_block(&block_buf, kref.top_k));
                            actual.push(chunk[p + i + 1] as usize);
                        }
                        take(score_batch(&host[..cnt * vocab], vocab, &blocks, &actual));
                        p += cnt;
                    }
                    pos += len;
                }
            } else {
                for pos in 0..n_feed {
                    let scored = pos >= half;
                    forward_token(
                        &bundle.config,
                        &bundle.weights,
                        &mut bundle.state,
                        &mut bundle.scratch,
                        &mut bundle.work,
                        gpu,
                        chunk[pos],
                        pos as u32,
                        scored,
                    )?;
                    if !scored {
                        continue;
                    }
                    let host =
                        hip_str(gpu.download_f32(&bundle.scratch.logits), "download logits")?;
                    kref.reader
                        .read_exact(&mut block_buf)
                        .map_err(|e| format!("{}: ref block: {e}", kref.path.display()))?;
                    let blocks = [parse_block(&block_buf, kref.top_k)];
                    take(score_batch(
                        &host,
                        vocab,
                        &blocks,
                        &[chunk[pos + 1] as usize],
                    ));
                }
            }

            if ck_kld.len() != scored_per_chunk {
                return Err(format!(
                    "chunk {c}: scored {} positions, expected {scored_per_chunk}",
                    ck_kld.len()
                ));
            }
            let mut sorted = ck_kld.clone();
            sorted.sort_by(f64::total_cmp);
            let mean_kld = mean(&ck_kld);
            let mean_nll = mean(&ck_nll);
            eprintln!(
                "  chunk {:3}/{chunks}: mean KLD {mean_kld:.6} p99 {:.6} NLL {mean_nll:.4} ({:.1}s)",
                c + 1,
                p99(&sorted),
                tc.elapsed().as_secs_f64()
            );
            per_chunk.push(json!({
                "chunk": c,
                "scored": ck_kld.len(),
                "mean_kld": mean_kld,
                "p99_kld": p99(&sorted),
                "max_kld": sorted[sorted.len() - 1],
                "mean_nll": mean_nll,
                "top1_agree": ck_top1 as f64 / ck_kld.len() as f64,
            }));
            top1_hits += ck_top1;
            all_kld.extend_from_slice(&ck_kld);
            all_nll.extend_from_slice(&ck_nll);
        }
        sync(gpu)?;
        let secs = t0.elapsed().as_secs_f64();

        let mut sorted = all_kld.clone();
        sorted.sort_by(f64::total_cmp);
        let median = if sorted.len() % 2 == 1 {
            sorted[sorted.len() / 2]
        } else {
            0.5 * (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2])
        };
        let mean_nll = mean(&all_nll);
        let scored = all_kld.len();
        let all_finite = nonfinite == 0
            && mean(&all_kld).is_finite()
            && mean_nll.is_finite()
            && scored == scored_per_chunk * chunks;
        Ok(json!({
            "pass": all_finite,
            "thresholds_applied": false,
            "ref": kref.path.display().to_string(),
            "ref_top_k": kref.top_k,
            "kld_definition": "top-K + residual-tail KLD against the teacher reference (not full-vocab exact)",
            "n_ctx": n_ctx,
            "chunks_run": chunks,
            "chunks_in_ref": kref.n_chunk,
            "fed_tokens_per_chunk": n_feed,
            "scored_range": [half, n_ctx - 2],
            "scored": scored,
            "nonfinite_scores": nonfinite,
            "mean_kld": mean(&all_kld),
            "median_kld": median,
            "p99_kld": p99(&sorted),
            "max_kld": sorted[sorted.len() - 1],
            "top1_agreement": top1_hits as f64 / scored as f64,
            "mean_nll": mean_nll,
            "ppl": mean_nll.exp(),
            "secs": secs,
            "scored_tok_per_s": scored as f64 / secs.max(1e-9),
            "per_chunk": per_chunk,
        }))
    })();

    if let Some(t) = readout {
        if let Err(e) = gpu.free_tensor(t) {
            eprintln!("logits: free readout tensor: {e:?}");
        }
    }
    run
}
