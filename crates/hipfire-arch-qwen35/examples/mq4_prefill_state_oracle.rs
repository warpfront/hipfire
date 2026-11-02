// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! Test-only cross-process state oracle; not a serving/performance acceptance test.
//! dump MODEL.hfq PROMPT.txt ROWS CONTINUATION_STEPS NEW_OUTPUT_DIR
//! compare BASELINE_DIR CANDIDATE_DIR
//! --plan / --self-check (pure CPU: no model open, no GPU init)
//! Set chunk/experimental flags externally BEFORE each fresh process. Suggested
//! row counts: 513, 1025, 8193. Prompt tokens repeat/truncate to exact ROWS;
//! continuation is teacher-forced from the same token stream, never greedy.
//! Same-codec Q8/FP8 VMM KV selected explicitly via the existing production
//! config codec (q8/fp8 only; auto/other refused, no new key). DeltaNet stays
//! Q8 with EF as before. Full meaningful prefix only, no unused KV.
//! Raw dumps preserve codes/scales/EF. compare is STRICT: any changed raw
//! byte, truncation, missing tensor or nonfinite decoded value returns Err
//! (identical NaNs fail too). Only same-codec/same-capacity opposite
//! eager/graph pairs compare; never Q8-vs-FP8, never prefill-vs-AR.
//! Graph execution is NOT proven by flags: the manifest records observed
//! graph state (exec presence, binding, dirty/replay-enabled, blob count)
//! without inventing a replay counter; the native `[qwen-ar-graph]`
//! capture/replay trace gate stays Main-owned.

use hipfire_arch_qwen35::qwen35::{
    self, DeltaNetState, LayerType, Qwen35Config, Qwen35Scratch, StateQuant,
};
use hipfire_runtime::{
    config,
    hfq::HfqFile,
    kv_backend::KvBackend,
    kv_mode::KvMode,
    llama::{KvCache, KvCacheExt, KvDims, KvLayers, KvTarget, VMode},
    tokenizer::Tokenizer,
};
use rdna_compute::{Gpu, GpuTensor};
use serde::{Deserialize, Serialize};
use std::{error::Error, fs, path::Path};

type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;

// ---- Historical Q8 pad/floor (FP8 capacity is derived per-run, see oracle_capacity) ----
const Q8_CAP_PAD: usize = 16;
const Q8_CAP_MIN: usize = 512;

#[derive(Serialize, Deserialize)]
struct Entry {
    name: String,
    kind: String,
    file: String,
    bytes: usize,
}

/// Observed native graph state around one completed continuation call.
/// Observations only: there is deliberately NO replay counter here. A replay
/// count cannot be observed from these fields; Main's gate requires the
/// native `[qwen-ar-graph] capture/replay` trace at distinct advancing
/// positions instead. Flags or wrapper completion alone prove nothing.
#[derive(Serialize, Deserialize)]
struct GraphWitness {
    stage: String,
    has_graph_exec: bool,
    binding: u64,
    replay_enabled: bool,
    kernel_dirty: bool,
    blob_count: usize,
    capture_mode: bool,
}

/// The five production capture switches as seen by the dumping process.
#[derive(Serialize, Deserialize)]
struct GraphEnv {
    replay_backend: Option<String>,
    graph: Option<String>,
    ar_graph: Option<String>,
    graph_moe: Option<String>,
    ar_graph_trace: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    model: String,
    arch: String,
    tokens: Vec<u32>,
    prefill_rows: usize,
    admitted_ceiling: usize,
    state_head_dim: usize,
    entries: Vec<Entry>,
    // ---- Required codec/backend evidence (no inferred defaults) ----
    // Old manifests missing these fields fail to parse and fail closed.
    codec: String,             // "q8" | "fp8": ACTUAL cache codec
    backend: String,           // "vmm"
    k_stride: usize,           // actual K bytes per token row
    v_stride: usize,           // actual V bytes per token row
    n_kv_heads: usize,         // geometry
    head_dim: usize,           // geometry
    max_seq: usize,            // logical bound used at construction
    physical_cap: usize,       // physical token capacity used at construction
    compact_offset: usize,     // must be 0: compacted caches are refused
    prompt_text: String,       // exact prompt-file bytes (identity, not tokens)
    continuation_steps: usize, // teacher-forced continuation length
    requested_mode: String,    // "eager" | "graph": derived from GraphEnv
    graph_env: GraphEnv,
    graph_witnesses: Vec<GraphWitness>,
}

fn half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = (bits >> 10) & 31;
    let m = bits & 1023;
    match e {
        0 => sign * (m as f32) * 2f32.powi(-24),
        31 if m == 0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + m as f32 / 1024.0) * 2f32.powi(e as i32 - 15),
    }
}

/// OCP E4M3FN decode (bias 7, max finite 448, NaN 0x7F/0xFF, no infinities),
/// matching the saddle-core CPU ground truth. E=15/m!=7 decodes finite (the
/// saturating encoder never emits those codes for finite inputs).
fn e4m3_decode(code: u8) -> f32 {
    let s = (code & 0x80) != 0;
    let e = ((code >> 3) & 0x0f) as i32;
    let m = (code & 7) as f32;
    let v = if e == 0 {
        m * 2f32.powi(-9)
    } else if e == 15 {
        if m as u8 == 7 {
            f32::NAN
        } else {
            2f32.powi(8) * (1.0 + m / 8.0)
        }
    } else {
        2f32.powi(e - 7) * (1.0 + m / 8.0)
    };
    if s {
        -v
    } else {
        v
    }
}

/// Checked Q8_0 token stride: n_kv_heads * (head_dim/32) * 34.
/// At Hkv2/D256 this is 2*8*34 = 544.
fn q8_stride_checked(n_kv_heads: usize, head_dim: usize) -> Result<usize> {
    if n_kv_heads == 0 || head_dim == 0 || head_dim % 32 != 0 {
        return Err(format!(
            "q8 stride needs n_kv_heads>0 and head_dim nonzero multiple of 32 (got {n_kv_heads}/{head_dim})"
        )
        .into());
    }
    n_kv_heads
        .checked_mul(head_dim / 32)
        .and_then(|v| v.checked_mul(34))
        .ok_or_else(|| "q8 stride overflow".into())
}

/// Checked native-FP8 token stride per side: [Hkv*D codes][Hkv f16 scales].
/// At Hkv2/D256 this is 2*256 + 2*2 = 516. head_dim 256 is the only shape
/// this path admits.
fn fp8_stride_checked(n_kv_heads: usize, head_dim: usize) -> Result<usize> {
    if n_kv_heads == 0 {
        return Err("fp8 stride needs n_kv_heads>0".into());
    }
    if head_dim != 256 {
        return Err(format!("fp8 stride admits head_dim 256 only (got {head_dim})").into());
    }
    let codes = n_kv_heads
        .checked_mul(head_dim)
        .ok_or("fp8 codes bytes overflow")?;
    let scales = n_kv_heads
        .checked_mul(2)
        .ok_or("fp8 scales bytes overflow")?;
    codes
        .checked_add(scales)
        .ok_or_else(|| "fp8 row bytes overflow".into())
}

/// Checked addressed-prefix bytes. Rejects wrapping before any readback so a
/// huge rows*stride can never become a small trusted length.
fn checked_prefix_bytes(label: &str, rows: usize, stride: usize) -> Result<usize> {
    rows.checked_mul(stride)
        .ok_or_else(|| format!("{label} prefix size overflowed ({rows}*{stride})").into())
}

/// Checked per-run capacity from the requested rows+continuation total.
/// FP8 capacity is NOT a fixed ceiling: it is the derived `total` used as
/// both cap and max_seq, so the 16384+256=16640 fixture arises naturally
/// (and any other legitimate total is admitted). Q8 keeps the historical
/// total+16/floor-512 rule. Overflow refuses via checked addition.
fn oracle_capacity(mode: KvMode, rows: usize, continuation: usize) -> Result<(usize, usize)> {
    let total = rows.checked_add(continuation).ok_or("row overflow")?;
    match mode {
        KvMode::Q8 => {
            let c = total
                .checked_add(Q8_CAP_PAD)
                .ok_or("capacity overflow")?
                .max(Q8_CAP_MIN);
            Ok((c, c))
        }
        KvMode::Fp8 => Ok((total, total)),
        _ => Err("oracle qualification is q8/fp8 only".into()),
    }
}
/// Independently decode one packed native-FP8 row: head-major codes followed
/// by head-major little-endian f16 scales. Returns staged values
/// (scale * code). Any NaN code, nonfinite scale, wrong length or nonfinite
/// staged value is Err (never silent, never a placeholder).
fn decode_fp8_row(row: &[u8], n_kv_heads: usize, head_dim: usize) -> Result<Vec<f32>> {
    let codes_len = n_kv_heads
        .checked_mul(head_dim)
        .ok_or("fp8 codes extent overflow")?;
    let scales_len = n_kv_heads
        .checked_mul(2)
        .ok_or("fp8 scales extent overflow")?;
    let want = codes_len
        .checked_add(scales_len)
        .ok_or("fp8 row extent overflow")?;
    if row.len() != want {
        return Err(format!("fp8 row wrong length: got {}, want {want}", row.len()).into());
    }
    let (codes, scales) = row.split_at(codes_len);
    let mut out = Vec::with_capacity(codes_len);
    for h in 0..n_kv_heads {
        let sbits = u16::from_le_bytes([scales[2 * h], scales[2 * h + 1]]);
        let s = half(sbits);
        if !s.is_finite() {
            return Err(format!("fp8 corrupted scale at head {h} (bits {sbits:#06x})").into());
        }
        for d in 0..head_dim {
            let code = codes[h * head_dim + d];
            let v = e4m3_decode(code);
            if v.is_nan() {
                return Err(format!("fp8 NaN code {code:#04x} at head {h} dim {d}").into());
            }
            let staged = s * v;
            if !staged.is_finite() {
                return Err(format!(
                    "fp8 nonfinite staged value at head {h} dim {d} (scale {s}, code {code:#04x})"
                )
                .into());
            }
            out.push(staged);
        }
    }
    Ok(out)
}

/// Fail-closed decoder: wrong length, corrupted scale, NaN code and any
/// nonfinite value are Err. Packed FP8 rows are NEVER run through the f32
/// decoder; they need geometry for native per-row decoding.
fn decode_checked(kind: &str, bytes: &[u8], geom: Option<(usize, usize)>) -> Result<Vec<f32>> {
    match kind {
        "f32" => {
            if bytes.len() % 4 != 0 {
                return Err(format!("f32 wrong length {}", bytes.len()).into());
            }
            let v: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            if v.iter().any(|x| !x.is_finite()) {
                return Err("nonfinite f32 state/logits".into());
            }
            Ok(v)
        }
        "f16" => {
            if bytes.len() % 2 != 0 {
                return Err(format!("f16 wrong length {}", bytes.len()).into());
            }
            let v: Vec<f32> = bytes
                .chunks_exact(2)
                .map(|b| half(u16::from_le_bytes(b.try_into().unwrap())))
                .collect();
            if v.iter().any(|x| !x.is_finite()) {
                return Err("nonfinite f16 state".into());
            }
            Ok(v)
        }
        "i8" => Ok(bytes.iter().map(|&b| (b as i8) as f32).collect()),
        "q8_0" => {
            if bytes.len() % 34 != 0 {
                return Err(format!("q8_0 wrong length {}", bytes.len()).into());
            }
            let mut v = Vec::with_capacity(bytes.len() / 34 * 32);
            for b in bytes.chunks_exact(34) {
                let d = half(u16::from_le_bytes([b[0], b[1]]));
                if !d.is_finite() {
                    return Err("q8_0 corrupted scale".into());
                }
                for &q in &b[2..] {
                    let x = (q as i8) as f32 * d;
                    if !x.is_finite() {
                        return Err("q8_0 nonfinite staged value".into());
                    }
                    v.push(x);
                }
            }
            Ok(v)
        }
        "fp8_kv_row" => {
            let (heads, dim) = geom.ok_or("fp8_kv_row needs geometry")?;
            let stride = fp8_stride_checked(heads, dim)?;
            if bytes.is_empty() || bytes.len() % stride != 0 {
                return Err(
                    format!("fp8_kv_row wrong length {} (stride {stride})", bytes.len()).into(),
                );
            }
            let mut v = Vec::with_capacity(bytes.len() / stride * heads * dim);
            for row in bytes.chunks_exact(stride) {
                v.extend(decode_fp8_row(row, heads, dim)?);
            }
            Ok(v)
        }
        _ => Err(format!("unsupported dump encoding {kind}").into()),
    }
}

fn save(
    gpu: &Gpu,
    dir: &Path,
    entries: &mut Vec<Entry>,
    name: String,
    kind: &str,
    tensor: &GpuTensor,
    bytes: usize,
) -> Result<()> {
    if bytes > tensor.buf.size() {
        return Err(format!(
            "{name}: download exceeds allocation ({bytes} > {})",
            tensor.buf.size()
        )
        .into());
    }
    let mut raw = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh(&mut raw, &tensor.buf)?;
    let values = decode_checked(kind, &raw, None).map_err(|e| format!("{name}: {e}"))?;
    eprintln!("dump {name}: {} values, nonfinite=0", values.len());
    let file = format!("{:04}.bin", entries.len());
    fs::write(dir.join(&file), &raw)?;
    entries.push(Entry {
        name,
        kind: kind.into(),
        file,
        bytes,
    });
    Ok(())
}

/// Download exactly the addressed rows*stride prefix (never virtual reserve
/// or rounded tail) and validate packed KV rows natively. Wrong length,
/// corrupted scale, NaN code and nonfinite staged values return Err.
#[allow(clippy::too_many_arguments)]
fn save_packed_kv(
    gpu: &Gpu,
    dir: &Path,
    entries: &mut Vec<Entry>,
    name: String,
    codec: &str,
    tensor: &GpuTensor,
    rows: usize,
    n_kv_heads: usize,
    head_dim: usize,
    stride: usize,
) -> Result<()> {
    let bytes = checked_prefix_bytes(&name, rows, stride)?;
    if bytes > tensor.buf.size() {
        return Err(format!(
            "{name}: addressed prefix {bytes} exceeds mapped allocation {} (virtual reserve is not readable)",
            tensor.buf.size()
        )
        .into());
    }
    let mut raw = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh(&mut raw, &tensor.buf)?;
    let kind = match codec {
        "q8" => "q8_0",
        "fp8" => "fp8_kv_row",
        _ => return Err(format!("{name}: unsupported KV codec {codec:?}").into()),
    };
    let values = decode_checked(kind, &raw, Some((n_kv_heads, head_dim)))
        .map_err(|e| format!("{name}: {e}"))?;
    eprintln!(
        "dump {name}: {} staged values from {rows} packed rows",
        values.len()
    );
    let file = format!("{:04}.bin", entries.len());
    fs::write(dir.join(&file), &raw)?;
    entries.push(Entry {
        name,
        kind: kind.into(),
        file,
        bytes,
    });
    Ok(())
}

/// Acquire the resolved existing configuration through the same source the
/// production carrier uses. Qualification must explicitly select q8 or fp8;
/// auto/empty/other modes are refused rather than guessed. No new key.
fn requested_codec() -> Result<KvMode> {
    let raw = config::get().kv_mode.clone();
    match raw.trim() {
        "q8" => Ok(KvMode::Q8),
        "fp8" => Ok(KvMode::Fp8),
        other => Err(format!(
            "oracle qualification must explicitly select q8 or fp8 (kv_mode={other:?}); refusing to guess"
        )
        .into()),
    }
}

/// Resolve the ACTUAL codec identity and per-side strides from the live
/// cache. Validates VMM backend, uncompacted state and supported geometry.
/// Note: an internal neutral VMode::Q8 on an FP8 cache does NOT mean Q8 V;
/// the K-mode authority is current_kv_mode(). Any non-q8/non-fp8 V
/// representation is rejected.
fn cache_layout(kv: &KvCache) -> Result<(String, usize, usize)> {
    if !kv.uses_vmm_backend() {
        return Err("oracle requires a VMM-backed KV cache".into());
    }
    if kv.compact_offset != 0 {
        return Err(
            "oracle refuses a compacted KV cache (absolute addressing needs compact_offset == 0)"
                .into(),
        );
    }
    match kv.current_kv_mode()? {
        KvMode::Fp8 => {
            let stride = kv.fp8_bytes_per_token()?;
            let expect = fp8_stride_checked(kv.n_kv_heads, kv.head_dim)?;
            if stride != expect {
                return Err("live fp8 stride disagrees with checked layout".into());
            }
            Ok(("fp8".into(), stride, stride))
        }
        KvMode::Q8 => {
            if kv.v_mode != VMode::Q8 {
                return Err("q8 cache with non-Q8 V representation is out of scope".into());
            }
            let stride = q8_stride_checked(kv.n_kv_heads, kv.head_dim)?;
            Ok(("q8".into(), stride, stride))
        }
        other => Err(format!("KV mode {other:?} has no qualified oracle stride; refusing").into()),
    }
}

fn graph_env() -> GraphEnv {
    let v = |k: &str| std::env::var(k).ok();
    GraphEnv {
        replay_backend: v("HIPFIRE_REPLAY_BACKEND"),
        graph: v("HIPFIRE_GRAPH"),
        ar_graph: v("HIPFIRE_AR_GRAPH"),
        graph_moe: v("HIPFIRE_GRAPH_MOE"),
        ar_graph_trace: v("HIPFIRE_AR_GRAPH_TRACE"),
    }
}

/// "graph" only when the full production capture set is present; anything
/// else is honestly "eager". Partial flag sets never claim graph execution.
fn requested_mode(env: &GraphEnv) -> &'static str {
    if env.replay_backend.as_deref() == Some("hip")
        && env.graph.as_deref() == Some("1")
        && env.ar_graph.as_deref() == Some("1")
        && env.graph_moe.as_deref() == Some("1")
        && env.ar_graph_trace.as_deref() == Some("1")
    {
        "graph"
    } else {
        "eager"
    }
}

/// Record actual graph state (exec existence, binding, dirty/replay-enabled
/// state, blob count). This is an observation, NOT a replay counter: no
/// replay count is recorded or invented here.
fn graph_witness(gpu: &Gpu, stage: String) -> GraphWitness {
    GraphWitness {
        stage,
        has_graph_exec: gpu.graphs.graph_exec.is_some(),
        binding: gpu.graphs.ar_forward_binding,
        replay_enabled: gpu.graphs.ar_forward_replay_enabled,
        kernel_dirty: gpu.graphs.ar_forward_kernel_dirty,
        blob_count: gpu.graphs.ar_forward_blobs.len(),
        capture_mode: gpu.graphs.capture_mode,
    }
}

#[allow(clippy::too_many_arguments)]
fn snapshot(
    gpu: &Gpu,
    dir: &Path,
    entries: &mut Vec<Entry>,
    stage: &str,
    rows: usize,
    codec: &str,
    k_stride: usize,
    v_stride: usize,
    config: &Qwen35Config,
    dn: &DeltaNetState,
    kv: &KvCache,
    scratch: &Qwen35Scratch,
) -> Result<()> {
    gpu.hip.device_synchronize()?;
    // Actual physical mapped bounds, not virtual reserve.
    kv.require_mapped_capacity(rows)?;
    if rows > kv.physical_cap {
        return Err(format!(
            "{stage}: rows {rows} exceed physical_cap {}",
            kv.physical_cap
        )
        .into());
    }
    if let Some(mapped) = kv.mapped_token_capacity()? {
        if rows > mapped {
            return Err(format!(
                "{stage}: rows {rows} exceed mapped prefix {mapped} (reserve is not readable)"
            )
            .into());
        }
    }
    let k_bytes = checked_prefix_bytes("K", rows, k_stride)?;
    let v_bytes = checked_prefix_bytes("V", rows, v_stride)?;
    save(
        gpu,
        dir,
        entries,
        format!("{stage}/logits"),
        "f32",
        &scratch.logits,
        config.vocab_size * 4,
    )?;
    let (n_heads, head_dim) = (config.n_kv_heads, config.head_dim);
    let mut la = 0;
    for (layer, ty) in config.layer_types.iter().enumerate() {
        let prefix = format!("{stage}/layer_{layer:02}");
        if *ty == LayerType::LinearAttention {
            let n = config.linear_num_value_heads * config.linear_value_head_dim.pow(2);
            save(
                gpu,
                dir,
                entries,
                format!("{prefix}/state_codes"),
                "i8",
                &dn.s_matrices[la],
                n,
            )?;
            save(
                gpu,
                dir,
                entries,
                format!("{prefix}/state_scales"),
                "f32",
                &dn.s_scales[la],
                dn.s_scales[la].numel() * 4,
            )?;
            save(
                gpu,
                dir,
                entries,
                format!("{prefix}/ef"),
                "f16",
                &dn.s_ef_residual[la],
                n * 2,
            )?;
            save(
                gpu,
                dir,
                entries,
                format!("{prefix}/conv"),
                "f32",
                &dn.conv_states[la],
                dn.conv_states[la].numel() * 4,
            )?;
            la += 1;
        } else {
            if k_bytes > kv.k_gpu[layer].buf.size() {
                return Err(
                    format!("{prefix}/key: addressed prefix exceeds mapped K allocation").into(),
                );
            }
            if v_bytes > kv.v_gpu[layer].buf.size() {
                return Err(format!(
                    "{prefix}/value: addressed prefix exceeds mapped V allocation"
                )
                .into());
            }
            save_packed_kv(
                gpu,
                dir,
                entries,
                format!("{prefix}/key"),
                codec,
                &kv.k_gpu[layer],
                rows,
                n_heads,
                head_dim,
                k_stride,
            )?;
            save_packed_kv(
                gpu,
                dir,
                entries,
                format!("{prefix}/value"),
                codec,
                &kv.v_gpu[layer],
                rows,
                n_heads,
                head_dim,
                v_stride,
            )?;
        }
    }
    // Residual (pre-final-norm) hidden state in scratch.x. The normalized
    // hidden is scratch.tmp on AR (rmsnorm_f32(&s.x, &weights.output_norm,
    // &s.tmp, ...)); never mislabel residual hidden as normalized hidden.
    let x_bytes = scratch
        .x
        .numel()
        .checked_mul(4)
        .ok_or("residual hidden extent overflow")?;
    save(
        gpu,
        dir,
        entries,
        format!("{stage}/residual_hidden_x"),
        "f32",
        &scratch.x,
        x_bytes,
    )?;
    Ok(())
}

fn dump(args: &[String]) -> Result<()> {
    if args.len() != 5 {
        return Err("dump MODEL PROMPT.txt ROWS CONTINUATION_STEPS NEW_OUTPUT_DIR".into());
    }
    // Explicit q8/fp8 before any model payload is opened. No default.
    let mode = requested_codec()?;
    let rows: usize = args[2].parse()?;
    let continuation: usize = args[3].parse()?;
    if rows < 2 {
        return Err("ROWS must be >=2".into());
    }
    // Checked per-run capacity: Q8 historical total+16/floor-512; FP8 derives
    // cap=max_seq=total so the 16384+256=16640 fixture is a natural example,
    // never a fixed ceiling or arbitrary cap. No production limit or flag.
    let (cap, max_seq) = oracle_capacity(mode, rows, continuation)?;
    let total = rows.checked_add(continuation).ok_or("row overflow")?;
    let env = graph_env();
    let run_mode = requested_mode(&env);
    let dir = Path::new(&args[4]);
    fs::create_dir(dir)?; // fail rather than overwrite an existing run
    let prompt_text = fs::read_to_string(&args[1])?;
    let mut hfq = HfqFile::open(Path::new(&args[0]))?;
    let config = qwen35::config_from_hfq(&hfq)?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)?;
    let seed = tokenizer.encode(&prompt_text);
    if seed.is_empty() {
        return Err("prompt tokenized to empty sequence".into());
    }
    let tokens: Vec<u32> = seed.iter().copied().cycle().take(total).collect();
    let mut gpu = Gpu::init()?;
    let mut source = qwen35::HfqSource::new(&mut hfq, &config);
    let weights = qwen35::load_weights(
        &mut source,
        std::slice::from_mut(&mut gpu),
        &qwen35::Layout::single(config.n_layers),
    )?;
    let mask: Vec<bool> = config
        .layer_types
        .iter()
        .map(|t| *t == LayerType::FullAttention)
        .collect();
    let dims = KvDims {
        layers: KvLayers::Mask(mask),
        n_kv_heads: config.n_kv_heads,
        head_dim: config.head_dim,
        max_seq,
        physical_cap: Some(cap),
    };
    let mut kv = <KvCache as KvCacheExt>::from_mode_with_backend(
        mode,
        KvBackend::Vmm,
        KvTarget::Single(&mut gpu),
        &dims,
    )?;
    // Requested codec must equal the actual laid-out codec.
    if kv.current_kv_mode()? != mode {
        return Err("requested codec != actual cache codec".into());
    }
    let (codec_id, k_stride, v_stride) = cache_layout(&kv)?;
    if k_stride != v_stride {
        return Err(
            "K/V stride divergence: this oracle only qualifies indivisible same-stride codecs"
                .into(),
        );
    }
    // Cross-check the pure stride helper against the live allocator stride.
    let helper_stride = if codec_id == "fp8" {
        fp8_stride_checked(config.n_kv_heads, config.head_dim)?
    } else {
        q8_stride_checked(config.n_kv_heads, config.head_dim)?
    };
    if helper_stride != k_stride {
        return Err("live allocator stride disagrees with checked stride helper".into());
    }
    let mut dn = DeltaNetState::new_with_quant(&mut gpu, &config, StateQuant::Q8)?;
    if dn.s_ef_residual.len() != dn.s_matrices.len() {
        return Err("Q8 EF must be enabled".into());
    }
    let scratch = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 64, cap)?;
    // Final executed ceiling is also printed by the production prefill receipt;
    // this pre-call query may differ after KV mapping and is labeled as such.
    let ceiling = qwen35::ordinary_prefill_chunk_limit(&gpu, &weights, &config, &dn, &kv, None)?;
    let mut manifest = Manifest {
        model: args[0].clone(),
        arch: gpu.arch.clone(),
        tokens,
        prefill_rows: rows,
        admitted_ceiling: ceiling,
        state_head_dim: config.linear_value_head_dim,
        entries: vec![],
        codec: codec_id.clone(),
        backend: "vmm".into(),
        k_stride,
        v_stride,
        n_kv_heads: config.n_kv_heads,
        head_dim: config.head_dim,
        max_seq,
        physical_cap: cap,
        compact_offset: kv.compact_offset,
        prompt_text,
        continuation_steps: continuation,
        requested_mode: run_mode.into(),
        graph_env: env,
        graph_witnesses: vec![],
    };
    eprintln!("oracle pre-call admitted ceiling={ceiling}; consult executed prefill receipt");
    eprintln!(
        "oracle codec={codec_id} mode={run_mode} k_stride={k_stride} v_stride={v_stride} cap={cap}"
    );
    qwen35::forward_prefill_batch(
        &mut gpu,
        &weights,
        &config,
        &manifest.tokens[..rows],
        0,
        &mut kv,
        &mut dn,
        &scratch,
        None,
        None,
        None,
        None,
    )?;
    snapshot(
        &gpu,
        dir,
        &mut manifest.entries,
        "prefill",
        rows,
        &codec_id,
        k_stride,
        v_stride,
        &config,
        &dn,
        &kv,
        &scratch,
    )?;
    manifest
        .graph_witnesses
        .push(graph_witness(&gpu, "post_prefill".into()));
    for pos in rows..total {
        qwen35::forward_scratch(
            &mut gpu,
            &weights,
            &config,
            manifest.tokens[pos],
            pos,
            &mut kv,
            &mut dn,
            &scratch,
        )?;
        gpu.hip.device_synchronize()?;
        manifest
            .graph_witnesses
            .push(graph_witness(&gpu, format!("post_decode_{}", pos - rows)));
        save(
            &gpu,
            dir,
            &mut manifest.entries,
            format!("decode_{}/logits", pos - rows),
            "f32",
            &scratch.logits,
            config.vocab_size * 4,
        )?;
    }
    if continuation > 0 {
        snapshot(
            &gpu,
            dir,
            &mut manifest.entries,
            "continued",
            total,
            &codec_id,
            k_stride,
            v_stride,
            &config,
            &dn,
            &kv,
            &scratch,
        )?;
        manifest
            .graph_witnesses
            .push(graph_witness(&gpu, "post_continued".into()));
    }
    fs::write(
        dir.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    // Process isolation is intentional; OS teardown releases the single-run GPU.
    Ok(())
}

fn metrics(name: &str, a: &[f32], b: &[f32], raw_mismatches: Option<usize>) {
    assert_eq!(a.len(), b.len(), "{name}: element counts differ");
    let mut ss = 0f64;
    let mut reference = 0f64;
    let mut max = 0f64;
    let mut max_index = 0;
    let mut first = None;
    let mut finite_pairs = 0;
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        if x.to_bits() != y.to_bits() && first.is_none() {
            first = Some(i);
        }
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        finite_pairs += 1;
        let d = (x as f64 - y as f64).abs();
        ss += d * d;
        reference += (x as f64).powi(2);
        if d > max {
            max = d;
            max_index = i;
        }
    }
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .filter(|(_, x)| x.is_finite())
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
    };
    println!(
        "{}",
        serde_json::json!({"tensor": name, "elements": a.len(),
        "raw_byte_mismatches": raw_mismatches, "first_diff_index": first,
        "max_abs": max, "max_abs_index": max_index,
        "rmse": (ss / (finite_pairs.max(1) as f64)).sqrt(),
        "relative_l2": if reference > 0.0 { Some((ss/reference).sqrt()) } else { None },
        "nonfinite_a": a.iter().filter(|v| !v.is_finite()).count(),
        "nonfinite_b": b.iter().filter(|v| !v.is_finite()).count(),
        "argmax_a": if name.ends_with("logits") { argmax(a) } else { None },
        "argmax_b": if name.ends_with("logits") { argmax(b) } else { None }})
    );
}

/// Fail-closed pair validation: same actual codec/backend/strides/geometry/
/// capacity, same prompt bytes/tokens/rows/teacher-forced continuation, and
/// opposite eager/graph mode. Old manifests missing codec evidence are
/// refused, never defaulted. Tensor names/kinds/extents are checked per entry
/// in compare.
fn validate_manifest_pair(a: &Manifest, b: &Manifest) -> Result<()> {
    for (side, m) in [("baseline", a), ("candidate", b)] {
        if m.codec != "q8" && m.codec != "fp8" {
            return Err(format!(
                "{side}: manifest missing codec evidence (old dump?); refusing to guess"
            )
            .into());
        }
        if m.backend != "vmm" {
            return Err(
                format!("{side}: backend {} is not qualified (need vmm)", m.backend).into(),
            );
        }
        if m.k_stride == 0 || m.v_stride == 0 {
            return Err(format!("{side}: manifest missing stride evidence").into());
        }
        if m.physical_cap == 0 || m.max_seq == 0 {
            return Err(format!("{side}: manifest missing capacity evidence").into());
        }
        if m.compact_offset != 0 {
            return Err(format!(
                "{side}: compacted cache (offset {}) is refused",
                m.compact_offset
            )
            .into());
        }
        if m.requested_mode != "eager" && m.requested_mode != "graph" {
            return Err(format!("{side}: unknown execution mode {:?}", m.requested_mode).into());
        }
    }
    if a.codec != b.codec {
        return Err("cross-codec comparison refused: Q8 vs FP8 is never byte-identical".into());
    }
    if a.backend != b.backend
        || a.k_stride != b.k_stride
        || a.v_stride != b.v_stride
        || a.n_kv_heads != b.n_kv_heads
        || a.head_dim != b.head_dim
        || a.max_seq != b.max_seq
        || a.physical_cap != b.physical_cap
    {
        return Err("incompatible manifests (backend, strides, geometry, or capacity)".into());
    }
    if a.tokens != b.tokens
        || a.prompt_text != b.prompt_text
        || a.prefill_rows != b.prefill_rows
        || a.continuation_steps != b.continuation_steps
        || a.state_head_dim != b.state_head_dim
        || a.entries.len() != b.entries.len()
        || a.model != b.model
        || a.arch != b.arch
    {
        return Err(
            "incompatible manifests (model/path, arch, prompt bytes/tokens, rows, continuation, or state shape)"
                .into(),
        );
    }
    if a.requested_mode == b.requested_mode {
        return Err("qualification pair must be opposite eager/graph modes".into());
    }
    Ok(())
}

fn compare(args: &[String]) -> Result<()> {
    if args.len() != 2 {
        return Err("compare BASELINE_DIR CANDIDATE_DIR".into());
    }
    // Manifests missing codec evidence fail to parse here: fail closed.
    let load = |dir: &str| -> Result<Manifest> {
        Ok(serde_json::from_slice(&fs::read(
            Path::new(dir).join("manifest.json"),
        )?)?)
    };
    let a = load(&args[0])?;
    let b = load(&args[1])?;
    validate_manifest_pair(&a, &b)?;
    let geom = Some((a.n_kv_heads, a.head_dim));
    let mut failures: Vec<String> = vec![];
    for (x, y) in a.entries.iter().zip(&b.entries) {
        if x.name != y.name || x.kind != y.kind || x.bytes != y.bytes {
            return Err("tensor metadata mismatch".into());
        }
        let xb = fs::read(Path::new(&args[0]).join(&x.file))?;
        let yb = fs::read(Path::new(&args[1]).join(&y.file))?;
        if xb.len() != x.bytes || yb.len() != y.bytes {
            return Err(format!("truncated tensor dump: {}", x.name).into());
        }
        // Any nonfinite decoded value (including identical NaNs) is Err.
        let va =
            decode_checked(&x.kind, &xb, geom).map_err(|e| format!("baseline {}: {e}", x.name))?;
        let vb =
            decode_checked(&y.kind, &yb, geom).map_err(|e| format!("candidate {}: {e}", y.name))?;
        if va.len() != vb.len() {
            failures.push(format!("{}: decoded lengths differ", x.name));
            continue;
        }
        let raw_mm = xb.iter().zip(&yb).filter(|(u, v)| u != v).count();
        metrics(&x.name, &va, &vb, Some(raw_mm));
        if raw_mm != 0 {
            let first = xb
                .iter()
                .zip(yb.iter())
                .position(|(u, v)| u != v)
                .unwrap_or(0);
            failures.push(format!(
                "{}: {raw_mm} changed raw bytes, first at byte {first}",
                x.name
            ));
        }
        if va
            .iter()
            .zip(vb.iter())
            .any(|(p, q)| p.to_bits() != q.to_bits())
        {
            let first = va
                .iter()
                .zip(vb.iter())
                .position(|(p, q)| p.to_bits() != q.to_bits())
                .unwrap_or(0);
            failures.push(format!(
                "{}: staged values differ bitwise, first at index {first}",
                x.name
            ));
        }
        if let Some(prefix) = x.name.strip_suffix("/state_codes") {
            let reconstruct =
                |manifest: &Manifest, dir: &str, codes: &[u8]| -> Result<(Vec<f32>, Vec<f32>)> {
                    let component = |suffix: &str| -> Result<Vec<f32>> {
                        let name = format!("{prefix}/{suffix}");
                        let entry = manifest
                            .entries
                            .iter()
                            .find(|e| e.name == name)
                            .ok_or("missing state component")?;
                        let raw = fs::read(Path::new(dir).join(&entry.file))?;
                        if raw.len() != entry.bytes {
                            return Err("truncated state component".into());
                        }
                        Ok(decode_checked(&entry.kind, &raw, None)?)
                    };
                    let scales = component("state_scales")?;
                    let ef = component("ef")?;
                    if ef.len() != codes.len()
                        || scales.len() * manifest.state_head_dim != codes.len()
                    {
                        return Err("invalid rowwise Q8 state geometry".into());
                    }
                    let state: Vec<f32> = codes
                        .iter()
                        .enumerate()
                        .map(|(i, &q)| (q as i8) as f32 * scales[i / manifest.state_head_dim])
                        .collect();
                    let effective = state.iter().zip(ef).map(|(s, e)| s + e).collect();
                    Ok((state, effective))
                };
            let (sa, ea) = reconstruct(&a, &args[0], &xb)?;
            let (sb, eb) = reconstruct(&b, &args[1], &yb)?;
            metrics(&format!("{prefix}/state_dequantized"), &sa, &sb, None);
            metrics(&format!("{prefix}/state_plus_ef"), &ea, &eb, None);
        }
    }
    if !failures.is_empty() {
        for f in &failures {
            eprintln!("mismatch {f}");
        }
        return Err(format!(
            "compare FAILED: {} tensor(s) differ; first: {}",
            failures.len(),
            failures[0]
        )
        .into());
    }
    Ok(())
}

/// Diagnostic-only plan: prints strides, capacity arithmetic, snapshot
/// extents, required graph witnesses and incompatibility rules. Reads no
/// model payload and initializes no GPU.
fn plan() -> Result<()> {
    println!("mq4_prefill_state_oracle plan (diagnostic only, no model payload read):");
    println!(
        "codecs: explicit q8 or fp8 via existing config kv_mode; auto/other refused, no new key"
    );
    println!(
        "q8 stride at Hkv2/D256: 2*(256/32)*34 = {}",
        q8_stride_checked(2, 256)?
    );
    println!(
        "fp8 stride per side at Hkv2/D256: 2*256 + 2*2 = {}",
        fp8_stride_checked(2, 256)?
    );
    println!("capacity: Q8 historical total+{Q8_CAP_PAD} floor {Q8_CAP_MIN}; FP8 derives cap=max_seq=total per run");
    println!("  (checked addition; e.g. the 16384+256=16640 fixture is an example, NOT a limit)");
    println!("construction: KvCacheExt::from_mode_with_backend(mode, Vmm, Single, KvDims::Mask)");
    println!("snapshot extents per stage: logits vocab*4 f32; per linear-attention layer");
    println!("  state_codes/state_scales(state_head_dim)/ef/conv; per full-attention layer");
    println!("  key/value rows*stride addressed prefixes only (never virtual reserve);");
    println!("  residual_hidden_x = scratch.x numel*4 f32 (residual pre-final-norm hidden;");
    println!("  normalized hidden is scratch.tmp, never mislabeled)");
    println!("guards: require_mapped_capacity(rows), rows<=physical_cap, checked rows*stride,");
    println!(
        "  per-tensor mapped allocation, mapped prefix capacity; packed rows decoded natively"
    );
    println!("  (head-major codes + f16 scales, finite-checked); corrupted scale/NaN code/wrong");
    println!("  length/nonfinite state or logits return Err");
    println!("graph arm requires: HIPFIRE_REPLAY_BACKEND=hip HIPFIRE_GRAPH=1 HIPFIRE_AR_GRAPH=1");
    println!("  HIPFIRE_GRAPH_MOE=1 HIPFIRE_AR_GRAPH_TRACE=1; no CPU-executed weight placement;");
    println!("  Redline disabled (production guards own enforcement); witnesses record graph_exec");
    println!("  presence/binding/dirty/replay-enabled/blob count per continuation call WITHOUT a");
    println!("  replay counter; native proof additionally requires [qwen-ar-graph] capture/replay");
    println!("  trace lines at distinct advancing positions with successful calls and unchanged");
    println!("  binding (Main gate); flags or wrapper completion alone never qualify");
    println!("compare rules: same actual codec/backend/strides/geometry/capacity, same prompt");
    println!(
        "  bytes/tokens/prefill rows/teacher-forced continuation, same tensor names/kinds/extents,"
    );
    println!(
        "  opposite eager/graph mode; ANY changed raw byte/truncation/missing tensor/nonfinite"
    );
    println!(
        "  value (identical NaNs included) returns Err; never Q8-vs-FP8, never prefill-vs-AR;"
    );
    println!("  old manifests missing codec evidence are refused, never defaulted");
    Ok(())
}

fn fake_manifest(codec: &str, mode: &str) -> Manifest {
    Manifest {
        model: "m.hfq".into(),
        arch: "gfx1201".into(),
        tokens: vec![1, 2, 3],
        prefill_rows: 3,
        admitted_ceiling: 512,
        state_head_dim: 64,
        entries: vec![Entry {
            name: "prefill/logits".into(),
            kind: "f32".into(),
            file: "0000.bin".into(),
            bytes: 16,
        }],
        codec: codec.into(),
        backend: "vmm".into(),
        k_stride: 544,
        v_stride: 544,
        n_kv_heads: 2,
        head_dim: 256,
        max_seq: 512,
        physical_cap: 512,
        compact_offset: 0,
        prompt_text: "abc".into(),
        continuation_steps: 0,
        requested_mode: mode.into(),
        graph_env: GraphEnv {
            replay_backend: None,
            graph: None,
            ar_graph: None,
            graph_moe: None,
            ar_graph_trace: None,
        },
        graph_witnesses: vec![],
    }
}

fn write_fake_dump(dir: &Path, m: &Manifest, blobs: &[Vec<u8>]) -> Result<()> {
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir)?;
    for (i, b) in blobs.iter().enumerate() {
        fs::write(dir.join(format!("{i:04}.bin")), b)?;
    }
    fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(m)?)?;
    Ok(())
}

fn f32_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Diagnostic-only self-check: exercises pure decoder/layout/comparator
/// helpers, including every negative comparator, with no model open and no
/// GPU init. Existing Q8 decode behavior is preserved and re-covered here.
fn self_check() -> Result<()> {
    // --- E4M3FN known vectors ---
    for (code, want) in [
        (0x00u8, 0.0f32),
        (0x80, -0.0),
        (0x38, 1.0),
        (0x40, 2.0),
        (0x01, 2f32.powi(-9)),
        (0x7e, 448.0),
        (0x7c, 384.0),
    ] {
        let got = e4m3_decode(code);
        if got.to_bits() != want.to_bits() {
            return Err(format!("self-check: e4m3 {code:#04x} decoded {got}, want {want}").into());
        }
    }
    for code in [0x7fu8, 0xff] {
        if !e4m3_decode(code).is_nan() {
            return Err(format!("self-check: e4m3 {code:#04x} must decode NaN").into());
        }
    }
    println!("self-check ok: e4m3 vectors (incl 0x7f/0xff NaN, -0, 448)");

    // --- Packed FP8 row: ok case + rejections ---
    let mut row = vec![0x38u8; 8];
    row.extend([0x00u8, 0x3c, 0x00, 0x3c]); // two f16 1.0 scales
    let staged = decode_fp8_row(&row, 2, 4)?;
    if staged.len() != 8 || staged.iter().any(|&x| x.to_bits() != 1.0f32.to_bits()) {
        return Err("self-check: fp8 ok-row must stage eight 1.0 values".into());
    }
    let mut nan_row = row.clone();
    nan_row[3] = 0x7f;
    if decode_fp8_row(&nan_row, 2, 4).is_ok() {
        return Err("self-check: fp8 NaN code unexpectedly accepted".into());
    }
    let mut inf_scale = row.clone();
    inf_scale[8] = 0x00;
    inf_scale[9] = 0x7c; // +inf scale
    if decode_fp8_row(&inf_scale, 2, 4).is_ok() {
        return Err("self-check: fp8 corrupt (inf) scale unexpectedly accepted".into());
    }
    let mut nan_scale = row.clone();
    nan_scale[10] = 0x00;
    nan_scale[11] = 0x7e; // NaN scale
    if decode_fp8_row(&nan_scale, 2, 4).is_ok() {
        return Err("self-check: fp8 NaN scale unexpectedly accepted".into());
    }
    if decode_fp8_row(&row[..11], 2, 4).is_ok() {
        return Err("self-check: fp8 short row unexpectedly accepted".into());
    }
    // Largest finite staged product (65504 * 448) stays finite: there is no
    // finite-overflow hiding behind saturation; overflow rejection below is
    // about checked address arithmetic, which must never wrap.
    let big = decode_fp8_row(
        &[vec![0x7eu8; 512], vec![0xff, 0x7b, 0xff, 0x7b]].concat(),
        2,
        256,
    )?;
    if big.iter().any(|x| !x.is_finite()) {
        return Err("self-check: max-finite fp8 product must stay finite".into());
    }
    println!("self-check ok: fp8 row decode + NaN/corrupt-scale/length rejections");

    // --- Stride helpers + checked arithmetic overflow rejection ---
    if q8_stride_checked(2, 256)? != 544 {
        return Err("self-check: q8 stride at Hkv2/D256 must be 544".into());
    }
    if fp8_stride_checked(2, 256)? != 516 {
        return Err("self-check: fp8 stride at Hkv2/D256 must be 516".into());
    }
    if q8_stride_checked(2, 100).is_ok() {
        return Err("self-check: q8 stride with head_dim 100 unexpectedly accepted".into());
    }
    if fp8_stride_checked(2, 128).is_ok() {
        return Err("self-check: fp8 stride with head_dim 128 unexpectedly accepted".into());
    }
    if q8_stride_checked(usize::MAX, 32).is_ok() {
        return Err("self-check: q8 stride overflow unexpectedly accepted".into());
    }
    if fp8_stride_checked(usize::MAX, 256).is_ok() {
        return Err("self-check: fp8 stride overflow unexpectedly accepted".into());
    }
    if checked_prefix_bytes("K", usize::MAX, 544).is_ok() {
        return Err("self-check: prefix size overflow unexpectedly accepted".into());
    }
    // Derived capacity: the 16384+256 fixture must come out of the helper,
    // a larger legitimate total must also be admitted, and wrapping refuses.
    if oracle_capacity(KvMode::Fp8, 16384, 256)? != (16640, 16640) {
        return Err("self-check: fp8 16384+256 must derive 16640 via helper".into());
    }
    if oracle_capacity(KvMode::Fp8, 20000, 7)? != (20007, 20007) {
        return Err("self-check: fp8 larger legitimate total unexpectedly refused".into());
    }
    if oracle_capacity(KvMode::Fp8, usize::MAX, 1).is_ok() {
        return Err("self-check: capacity addition overflow unexpectedly accepted".into());
    }
    if oracle_capacity(KvMode::Q8, 100, 0)? != (Q8_CAP_MIN, Q8_CAP_MIN) {
        return Err("self-check: q8 historical floor changed".into());
    }
    if oracle_capacity(KvMode::Q8, 1000, 0)? != (1016, 1016) {
        return Err("self-check: q8 historical rule changed".into());
    }
    println!(
        "self-check ok: strides 544/516, geometry + overflow rejections, derived 16640 fixture"
    );

    // --- Comparator suite on synthetic dumps (host-only, no GPU) ---
    let root = std::env::temp_dir().join(format!("mq4_oracle_selfcheck_{}", std::process::id()));
    let a_dir = root.join("a");
    let b_dir = root.join("b");
    let pair = |ma: &Manifest, ba: &[Vec<u8>], mb: &Manifest, bb: &[Vec<u8>]| -> Result<()> {
        write_fake_dump(&a_dir, ma, ba)?;
        write_fake_dump(&b_dir, mb, bb)?;
        compare(&[
            a_dir.to_string_lossy().into_owned(),
            b_dir.to_string_lossy().into_owned(),
        ])
    };
    let good = f32_blob(&[1.0, 2.0, 3.0, 4.0]);
    let ma = fake_manifest("q8", "eager");
    let mb = fake_manifest("q8", "graph");
    pair(&ma, &[good.clone()], &mb, &[good.clone()])?;
    println!("self-check ok: identical eager/graph pair compares clean");

    let mut changed = good.clone();
    changed[0] ^= 1;
    if pair(&ma, &[good.clone()], &mb, &[changed]).is_ok() {
        return Err("self-check: changed byte unexpectedly passed".into());
    }
    println!("self-check ok: changed raw byte refused");

    let nan = f32_blob(&[f32::NAN]);
    let mut mnan_a = fake_manifest("q8", "eager");
    mnan_a.entries[0].bytes = 4;
    let mut mnan_b = fake_manifest("q8", "graph");
    mnan_b.entries[0].bytes = 4;
    if pair(&mnan_a, &[nan.clone()], &mnan_b, &[nan]).is_ok() {
        return Err("self-check: identical NaNs unexpectedly passed".into());
    }
    println!("self-check ok: identical NaNs refused");

    let mfp8 = fake_manifest("fp8", "graph");
    if pair(&ma, &[good.clone()], &mfp8, &[good.clone()]).is_ok() {
        return Err("self-check: cross-codec comparison unexpectedly passed".into());
    }
    println!("self-check ok: wrong codec refused");

    let mut mstride = fake_manifest("q8", "graph");
    mstride.k_stride = 545;
    if pair(&ma, &[good.clone()], &mstride, &[good.clone()]).is_ok() {
        return Err("self-check: wrong stride unexpectedly passed".into());
    }
    println!("self-check ok: wrong stride refused");

    let mut mcap = fake_manifest("q8", "graph");
    mcap.physical_cap = 513;
    if pair(&ma, &[good.clone()], &mcap, &[good.clone()]).is_ok() {
        return Err("self-check: wrong capacity unexpectedly passed".into());
    }
    println!("self-check ok: wrong capacity refused");

    if pair(&ma, &[good.clone()], &mb, &[good[..15].to_vec()]).is_ok() {
        return Err("self-check: truncated file unexpectedly passed".into());
    }
    println!("self-check ok: truncated file refused");

    write_fake_dump(&a_dir, &ma, &[good.clone()])?;
    write_fake_dump(&b_dir, &mb, &[good.clone()])?;
    fs::remove_file(b_dir.join("0000.bin"))?;
    if compare(&[
        a_dir.to_string_lossy().into_owned(),
        b_dir.to_string_lossy().into_owned(),
    ])
    .is_ok()
    {
        return Err("self-check: missing entry unexpectedly passed".into());
    }
    println!("self-check ok: missing entry refused");

    let meager = fake_manifest("q8", "eager");
    if pair(&ma, &[good.clone()], &meager, &[good.clone()]).is_ok() {
        return Err("self-check: same-mode pair unexpectedly passed".into());
    }
    println!("self-check ok: same-mode pair refused (need opposite eager/graph)");

    // Old manifest missing codec evidence fails closed, never defaulted.
    let old = br#"{"model":"m.hfq","arch":"gfx1201","tokens":[1],"prefill_rows":1,"admitted_ceiling":512,"state_head_dim":64,"entries":[]}"#;
    if serde_json::from_slice::<Manifest>(old).is_ok() {
        return Err("self-check: manifest without codec evidence unexpectedly parsed".into());
    }
    println!("self-check ok: old manifest without codec evidence refused");

    let _ = fs::remove_dir_all(&root);
    println!("self-check: all decoder/layout/comparator checks passed");
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("dump") => dump(&args[1..]),
        Some("compare") => compare(&args[1..]),
        // Pure CPU branches: before any model open or GPU init.
        Some("--plan") if args.len() == 1 => plan(),
        Some("--self-check") if args.len() == 1 => self_check(),
        _ => Err(
            "use dump MODEL PROMPT.txt ROWS CONTINUATION_STEPS NEW_OUTPUT_DIR | compare BASE CANDIDATE | --plan | --self-check"
                .into(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn half_specials_and_subnormals() {
        assert_eq!(half(0x3c00), 1.0);
        assert_eq!(half(0xbc00), -1.0);
        assert_eq!(half(1), 2f32.powi(-24));
        assert!(half(0x7c00).is_infinite());
        assert!(half(0x7e00).is_nan());
    }
    #[test]
    fn q8_block_decodes_signed_codes_and_scale() {
        let mut block = vec![0u8; 34];
        block[..2].copy_from_slice(&0x3800u16.to_le_bytes()); // 0.5
        block[2] = 127;
        block[3] = (-127i8) as u8;
        let decoded = decode_checked("q8_0", &block, None).unwrap();
        assert_eq!(decoded.len(), 32);
        assert_eq!(&decoded[..3], &[63.5, -63.5, 0.0]);
    }
    #[test]
    fn e4m3_vectors_match_oracle_semantics() {
        assert_eq!(e4m3_decode(0x00).to_bits(), 0.0f32.to_bits());
        assert_eq!(e4m3_decode(0x80).to_bits(), (-0.0f32).to_bits());
        assert_eq!(e4m3_decode(0x38).to_bits(), 1.0f32.to_bits());
        assert_eq!(e4m3_decode(0x7e).to_bits(), 448.0f32.to_bits());
        assert!(e4m3_decode(0x7f).is_nan());
        assert!(e4m3_decode(0xff).is_nan());
    }
    #[test]
    fn fp8_row_ok_and_rejections() {
        let mut row = vec![0x38u8; 8];
        row.extend([0x00u8, 0x3c, 0x00, 0x3c]);
        let staged = decode_fp8_row(&row, 2, 4).unwrap();
        assert!(staged.iter().all(|&x| x == 1.0));
        let mut bad = row.clone();
        bad[0] = 0x7f;
        assert!(decode_fp8_row(&bad, 2, 4).is_err());
        assert!(decode_fp8_row(&row[..11], 2, 4).is_err());
    }
    #[test]
    fn stride_helpers_match_qualified_geometry() {
        assert_eq!(q8_stride_checked(2, 256).unwrap(), 544);
        assert_eq!(fp8_stride_checked(2, 256).unwrap(), 516);
        assert!(q8_stride_checked(usize::MAX, 32).is_err());
        assert!(fp8_stride_checked(2, 128).is_err());
        assert!(checked_prefix_bytes("K", usize::MAX, 544).is_err());
    }
}
