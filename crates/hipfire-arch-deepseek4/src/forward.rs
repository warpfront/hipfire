// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! DeepSeek V4 forward pass — skeleton.
//!
//! Layout-only: the function signatures and per-layer call sequence
//! are locked in; the bodies are `unimplemented!` until each piece
//! gets wired. Reading this file gives a future implementer (or
//! reviewer) the entire decode-step flow at a glance.
//!
//! The seven GPU-validated kernels referenced here:
//!   - `gpu.hc_compute_control`       (Phase 3)
//!   - `gpu.hc_sinkhorn_4x4`          (Phase 3)
//!   - `gpu.hc_mix_4stream`           (Phase 3)
//!   - `gpu.indexer_compressed_k_score`  (Phase 2)
//!   - `gpu.indexer_top_k`            (Phase 2)
//!   - `gpu.indexer_kv_gather`        (Phase 2)
//!   - `gpu.rope_tail_halfsplit`      (Phase 4)
//!
//! Existing hipfire-runtime kernels reused (no DeepSeek V4-specific impl):
//!   - RMSNorm
//!   - Quantized GEMV (MQ-family) for Q-LoRA, KV, O-LoRA, experts
//!   - Embedding lookup, lm_head matmul, sampler

use crate::backend::Mq2rBackend;
use crate::config_cache;
use crate::deepseek4::{DeepseekV4HeterogeneousWeights, DeepseekV4RoutedWeights};
use crate::heterogeneous::DeepseekV4HeterogeneousExecution;
use crate::program::{self, LayerProgram, Mix};
use crate::rows_program::{self, Observed, RowSlots, RowsWindow};
use crate::{DeepseekV4Config, DeepseekV4State, DeepseekV4Weights};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::moe::MoeBiasAwareParams;
use hipfire_dispatch::pipeline::mla::attention::{
    execute_sparse_index, CompressedCache, IdentityGatherOp, IdentityGatherRoute, IndexTopK,
    OLoraOp, OLoraRoute, OloraA, SparseAttentionOp, SparseAttentionRoute, SparseIndexOp,
    SparseIndexRoute, SwaAttentionOp, SwaAttentionRoute, SwaRingWriteOp, SwaRingWriteRoute,
    TopkGather,
};
use hipfire_dispatch::pipeline::mla::compressor::{
    execute_compressor_project, execute_kv_compress, CompressorCommit, CompressorDebug,
    CompressorInput, CompressorProjectOp, CompressorProjectRoute, CompressorProjection,
    CompressorRope, KvCompressOp, KvCompressRoute,
};
use hipfire_dispatch::pipeline::mla::expert_ffn::{
    execute_ffn_norm, execute_moe_route, execute_routed_experts, execute_shared_expert,
    ExpertFfnOp, ExpertFfnRoute, ExpertFfnRouter, ExpertFfnSchedule, HashRoutedExpertsOp,
    HashSelection, HashToken, RoutedExpertsOp,
};
use hipfire_dispatch::pipeline::mla::hc::{
    execute_hc_post, execute_hc_pre, HcHeadOp, HcHeadRoute, HcMixBranch, HcPostOp, HcPostRoute,
    HcPreOp, HcPreRoute,
};
use hipfire_dispatch::pipeline::mla::project::{PreparedProjectionOp, PreparedProjectionRoute};
use hipfire_dispatch::pipeline::mla::query_kv::{
    execute_apply_tail_rope, execute_kv_joint, execute_q_lora_prepare, execute_q_lora_project,
    ApplyTailRopeOp, ApplyTailRopeRoute, KvJointOp, KvJointRoute, PackedProjection,
    PackedProjectionsOp, PackedProjectionsRoute, QLoraPrepareOp, QLoraPrepareRoute, QLoraProjectOp,
    QLoraProjectRoute,
};
use hipfire_dispatch::pipeline::superop::{
    self, ForwardBindings, OpBinding, OpFlavor, SuperOp, SuperOpKind,
};
use hipfire_dispatch::pipeline::{execute_steps, ClearOp, Step};
use hipfire_dispatch::types::DispatchError;
use rdna_compute::replay::ReplayController;
use rdna_compute::{DType, Gpu, GpuTensor};
use smallvec::{smallvec, SmallVec};

struct DenseActivationWriter {
    writer: std::io::BufWriter<std::fs::File>,
    k: usize,
    rows: u32,
}

struct DenseActivationDump {
    out_dir: std::path::PathBuf,
    writers: std::collections::BTreeMap<String, DenseActivationWriter>,
    finished: bool,
}

impl DenseActivationDump {
    fn new(out_dir: std::path::PathBuf) -> Result<Self, String> {
        std::fs::create_dir_all(&out_dir).map_err(|e| {
            format!(
                "create dense activation directory {}: {e}",
                out_dir.display()
            )
        })?;
        Ok(Self {
            out_dir,
            writers: std::collections::BTreeMap::new(),
            finished: false,
        })
    }

    fn record(&mut self, tensor_name: &str, k: usize, values: &[f32]) -> Result<(), String> {
        use std::io::Write;

        if self.finished {
            return Err("dense activation dump already finalized".to_string());
        }
        if k == 0 || values.len() % k != 0 {
            return Err(format!(
                "dense activation {tensor_name}: {} values are not whole K={k} rows",
                values.len()
            ));
        }
        let entry = if let Some(entry) = self.writers.get_mut(tensor_name) {
            entry
        } else {
            let key = tensor_name.replace(['/', '\\'], "_").replace("..", "_");
            let path = self.out_dir.join(format!("{key}.acts"));
            let file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(|e| format!("create dense activation dump {}: {e}", path.display()))?;
            let mut writer = std::io::BufWriter::new(file);
            writer
                .write_all(&0u32.to_le_bytes())
                .and_then(|_| writer.write_all(&(k as u32).to_le_bytes()))
                .map_err(|e| format!("write dense activation header {}: {e}", path.display()))?;
            self.writers.insert(
                tensor_name.to_string(),
                DenseActivationWriter { writer, k, rows: 0 },
            );
            self.writers
                .get_mut(tensor_name)
                .expect("dense activation writer inserted")
        };
        if entry.k != k {
            return Err(format!(
                "dense activation {tensor_name}: K changed from {} to {k}",
                entry.k
            ));
        }
        let rows: u32 = (values.len() / k)
            .try_into()
            .map_err(|_| format!("dense activation {tensor_name}: row count overflow"))?;
        entry.rows = entry
            .rows
            .checked_add(rows)
            .ok_or_else(|| format!("dense activation {tensor_name}: cumulative row overflow"))?;
        let bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        };
        entry
            .writer
            .write_all(bytes)
            .map_err(|e| format!("append dense activation {tensor_name}: {e}"))
    }

    fn finish(&mut self) -> Result<(usize, u64), String> {
        use std::io::{Seek, SeekFrom, Write};

        if self.finished {
            return Ok((
                self.writers.len(),
                self.writers.values().map(|entry| entry.rows as u64).sum(),
            ));
        }
        let mut total_rows = 0u64;
        for (name, entry) in &mut self.writers {
            entry
                .writer
                .flush()
                .and_then(|_| entry.writer.seek(SeekFrom::Start(0)).map(|_| ()))
                .and_then(|_| entry.writer.write_all(&entry.rows.to_le_bytes()))
                .and_then(|_| entry.writer.write_all(&(entry.k as u32).to_le_bytes()))
                .and_then(|_| entry.writer.flush())
                .map_err(|e| format!("finalize dense activation {name}: {e}"))?;
            total_rows += entry.rows as u64;
        }
        self.finished = true;
        Ok((self.writers.len(), total_rows))
    }
}

fn dense_activation_dump() -> Result<Option<&'static std::sync::Mutex<DenseActivationDump>>, String>
{
    use std::sync::{Mutex, OnceLock};

    static DUMP: OnceLock<Result<Option<Mutex<DenseActivationDump>>, String>> = OnceLock::new();
    match DUMP.get_or_init(|| {
        let Some(path) = hipfire_config::developer_var_os("HIPFIRE_DS4_DENSE_ACT_DIR") else {
            return Ok(None);
        };
        if path.is_empty() {
            return Err("HIPFIRE_DS4_DENSE_ACT_DIR must not be empty".to_string());
        }
        DenseActivationDump::new(path.into()).map(|dump| Some(Mutex::new(dump)))
    }) {
        Ok(dump) => Ok(dump.as_ref()),
        Err(error) => Err(error.clone()),
    }
}

pub(crate) fn dense_activation_dump_enabled() -> Result<bool, String> {
    Ok(dense_activation_dump()?.is_some())
}

pub(crate) fn dump_dense_activation_if_enabled(
    gpu: &Gpu,
    tensor_name: &str,
    input: &GpuTensor,
    k: usize,
) -> Result<(), String> {
    dump_dense_activations_if_enabled(gpu, &[tensor_name], input, k)
}

/// Record one logical activation for every weight that consumes it. Keeping
/// this fan-out here matters for P3: several projections share the same
/// pre-rotation input, and downloading that input once is far cheaper than one
/// device synchronization and D2H copy per tensor name.
pub(crate) fn dump_dense_activations_if_enabled<S: AsRef<str>>(
    gpu: &Gpu,
    tensor_names: &[S],
    input: &GpuTensor,
    k: usize,
) -> Result<(), String> {
    let Some(dump) = dense_activation_dump()? else {
        return Ok(());
    };
    if tensor_names.is_empty() {
        return Ok(());
    }
    let values = gpu
        .download_f32(input)
        .map_err(|e| format!("download dense activation fan-out: {e:?}"))?;
    let mut dump = dump
        .lock()
        .map_err(|_| "dense activation dump mutex poisoned".to_string())?;
    for tensor_name in tensor_names {
        dump.record(tensor_name.as_ref(), k, &values)?;
    }
    Ok(())
}

/// Finalize the env-gated P3 activation dump produced by the decode path.
///
/// Each file uses the `collect_e8_hessian` input contract:
/// `[u32 n_rows][u32 K][f32 rows...]`. The row count is patched only here so
/// interrupted captures cannot masquerade as complete calibration inputs.
pub fn finish_dense_activation_dump() -> Result<(), String> {
    let Some(dump) = dense_activation_dump()? else {
        return Ok(());
    };
    let mut dump = dump
        .lock()
        .map_err(|_| "dense activation dump mutex poisoned".to_string())?;
    let out_dir = dump.out_dir.clone();
    let (files, rows) = dump.finish()?;
    eprintln!(
        "Finalized DeepSeek P3 dense activation dump: {files} files, {rows} rows ({})",
        out_dir.display()
    );
    Ok(())
}

/// Effective MoE route scale for a DeepSeek V4 artifact, as the forward path
/// will actually apply it.
///
/// Exposed so capture tools and manifests report the value that runs instead of
/// re-deriving the precedence rule. A copy of that rule in
/// `examples/ds4_quant_plog.rs` drifted from the source once already — it still
/// claimed the `.mq2r` default was 2.0 after the measured optimum moved to 1.8,
/// which silently mislabelled captures. There must be exactly one place that
/// decides this.
///
/// See [`config_cache::resolve_route_scale`] for the precedence and the
/// measurements behind each default.
pub fn effective_route_scale(cfg_routed_scaling_factor: f32, mq2r: bool) -> f32 {
    config_cache::route_scale(cfg_routed_scaling_factor, mq2r)
}

pub(crate) fn config_cache_log_gfx942_a2_levers(arch: &str, gfx942_route_v1: bool) {
    config_cache::log_gfx942_a2_levers(arch, gfx942_route_v1);
}

/// DeepSeek V4 GEMV dispatch: switch kernel based on weight dtype.
///
/// - `DType::MQ4G256` (default DeepSeek V4 non-expert quant): consume FWHT-rotated
///   input via `gemv_mq4g256_prerotated`. This is the existing fast path.
/// - `DType::F32` (set by `--non-expert-f16` quantizer flag, F16 source
///   converted to F32 on upload): consume plain RMSNorm'd input (no FWHT)
///   via `gemv_f32`. Used to faithfully reproduce antirez/ds4's PROVEN
///   recipe of keeping compressor / indexer / attn projections at F16
///   precision.
///
/// Caller passes BOTH the FWHT-rotated and plain inputs; helper picks
/// whichever the weight needs. `m` and `k` are passed-through for the
/// MQ4 path only — gemv_f32 derives them from the weight's shape.
/// True if `gemv_auto` for this weight dtype will read the FWHT-rotated
/// input (`x_rotated` arg), false if it only reads the plain input.
/// DeepSeek V4's mq2lloyd-q8 build has F16/Q8 everywhere except the routed
/// MoE experts (which take a separate path) — meaning most decode-path
/// rotations into `ffn_x_rot` / `silu_rot` / `q_lat_rot` / etc. are
/// DEAD WORK (kernel runs, output never read). Use this to skip them.
#[inline]
pub(crate) fn weight_needs_fwht(weight: &GpuTensor) -> bool {
    hipfire_dispatch::types::dtype_needs_rotation(weight.dtype)
}

#[inline]
fn mfp_e8_row_bytes(dtype: DType, k: usize) -> usize {
    debug_assert_eq!(k % 256, 0);
    match dtype {
        DType::MFP4G32E8 | DType::MFP3G32E8 | DType::MFP4G32E8SOA => dtype
            .row_bytes(k)
            .expect("E8 row geometry is defined for K > 0"),
        _ => unreachable!("mfp_e8_row_bytes called for {dtype:?}"),
    }
}

pub(crate) fn gemv_auto(
    gpu: &mut Gpu,
    mq2r_backend: Mq2rBackend,
    weight: &GpuTensor,
    x_rotated: &GpuTensor,
    x_plain: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
) -> Result<(), String> {
    let e8_soa = e8_soa_kernel(&gpu.arch, mq2r_backend);
    hipfire_dispatch::pipeline::mla::project_prepared(
        gpu, weight, x_rotated, x_plain, y, m, k, e8_soa,
    )
    .map_err(|e| e.to_string())
}

/// The E8 SoA projection kernel of this model's MQ2R backend on `arch`.
pub(crate) fn e8_soa_kernel(
    arch: &str,
    mq2r_backend: Mq2rBackend,
) -> hipfire_dispatch::pipeline::mla::E8SoaKernel {
    use hipfire_dispatch::pipeline::mla::E8SoaKernel;
    if !config_cache::e8_u4_on(arch, mq2r_backend.is_gfx1151()) {
        E8SoaKernel::Prerotated
    } else if mq2r_backend.is_gfx1151() {
        E8SoaKernel::U4Gfx1151
    } else {
        E8SoaKernel::U4
    }
}

/// Batched twin of `gemv_auto` for Phase B2 chunk forward.
///
/// Same dispatch shape but each call processes `batch_size` inputs against
/// a single weight matrix. Output `y` is row-major `[batch_size, m]` —
/// matches what concatenating `batch_size` sequential gemv_auto outputs
/// would produce.
///
/// Inputs:
///   - `x_rotated_batch`: `[batch_size, k]` FWHT-rotated (consumed by the
///     MQ4 path only)
///   - `x_plain_batch`:   `[batch_size, k]` plain RMSNorm'd (consumed by
///     the F32 and Q8 paths)
///
/// Backed by the existing GEMM-batched kernels:
///   - F32  → `gemm_f32_batched` (M_kernel=batch, N_kernel=output_dim)
///   - Q8_0 → `gemm_q8_0_batched_chunked` (handles batch > 64 via internal
///            sub-batching; same MAX_BATCH=64 as the underlying kernel)
///   - Raw (MQ4G256) → `gemm_hfq4g256` (consumes pre-rotated x)
///
/// At batch_size == 1 each path reduces to the equivalent of one
/// sequential gemv_auto call against the same weight; per-row outputs
/// match within FMA-order ε.
#[allow(dead_code, clippy::too_many_arguments)]
fn gemv_auto_batched(
    gpu: &mut Gpu,
    mq2r_backend: Mq2rBackend,
    weight: &GpuTensor,
    x_rotated_batch: &GpuTensor,
    x_plain_batch: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    batch_size: usize,
) -> Result<(), String> {
    gemv_auto_batched_wmma(
        gpu,
        mq2r_backend,
        weight,
        x_rotated_batch,
        x_plain_batch,
        y,
        m,
        k,
        batch_size,
        /*x_f16_scratch=*/ None,
    )
}

/// `gemv_auto_batched` plus an opt-in WMMA path. When `x_f16_scratch`
/// Determinism-bisection helper: when `HIPFIRE_DEEPSEEK4_DUMP_STATE=<dir>` is
/// set, this writes the entire device buffer to `<dir>/<tag>.bin` after a
/// device-sync. Two same-seed runs with different output dirs can then be
/// compared with `cmp -l a/x.bin b/x.bin` to find the first byte that
/// differs — pinpointing which kernel introduces non-determinism.
pub(crate) fn dump_buf(gpu: &mut Gpu, tag: &str, buf: &rdna_compute::GpuTensor) {
    let dir = match hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_DUMP_STATE") {
        Ok(d) => d,
        Err(_) => return,
    };
    let _ = gpu.hip.device_synchronize();
    let n = buf.byte_size();
    let mut host = vec![0u8; n];
    if gpu.hip.memcpy_dtoh(&mut host, &buf.buf).is_ok() {
        let path = format!("{dir}/{tag}.bin");
        if let Err(e) = std::fs::write(&path, &host) {
            eprintln!("[dump_buf] write {path}: {e}");
        }
    }
}

/// Opt-in per-layer residual L2 dump for prod-vs-parent trajectory compare.
///
/// Set `HIPFIRE_DEEPSEEK4_LAYER_NORM=1`. Emits one line per layer after the
/// FFN HC mix:
/// `PROD_LAYER_NORM pos=<p> layer=<l> l2=<f64> nelems=<n>`.
/// Absolute norms are single-token (`hc_mult * hidden`); compare consecutive
/// layer *ratios* against the parent multi-row trajectory.
///
/// The most recent position's series is also retained in process memory for
/// [`take_layer_norm_trace`].
fn dump_residual_layer_norm(
    gpu: &mut Gpu,
    state: &DeepseekV4State,
    layer_idx: usize,
    position: u32,
) {
    use std::sync::LazyLock;
    static ON: LazyLock<bool> = LazyLock::new(|| {
        hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_LAYER_NORM")
            .ok()
            .as_deref()
            == Some("1")
    });
    if !*ON {
        return;
    }
    let Some(streams) = state.residual_streams.as_ref() else {
        return;
    };
    let _ = gpu.hip.device_synchronize();
    let Ok(v) = gpu.download_f32(streams) else {
        return;
    };
    let l2: f64 = v
        .iter()
        .map(|&x| {
            let x = x as f64;
            x * x
        })
        .sum::<f64>()
        .sqrt();
    eprintln!(
        "PROD_LAYER_NORM pos={position} layer={layer_idx} l2={l2:.9e} nelems={}",
        v.len()
    );
    // Retain last-seen position's full series (overwrite on position change).
    if let Ok(mut g) = LAYER_NORM_TRACE.lock() {
        if g.position != Some(position) {
            g.position = Some(position);
            g.l2.clear();
        }
        if g.l2.len() == layer_idx {
            g.l2.push(l2);
        } else if layer_idx < g.l2.len() {
            g.l2[layer_idx] = l2;
        } else {
            g.l2.resize(layer_idx + 1, f64::NAN);
            g.l2[layer_idx] = l2;
        }
    }
}

#[derive(Default, Clone, Debug)]
struct LayerNormTrace {
    position: Option<u32>,
    l2: Vec<f64>,
}

static LAYER_NORM_TRACE: std::sync::LazyLock<std::sync::Mutex<LayerNormTrace>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(LayerNormTrace::default()));

/// Drain the most recent `HIPFIRE_DEEPSEEK4_LAYER_NORM` series
/// `(position, per_layer_l2)`. Empty when the env gate is off or no layer
/// has run yet.
pub fn take_layer_norm_trace() -> Option<(u32, Vec<f64>)> {
    let mut g = LAYER_NORM_TRACE.lock().ok()?;
    let pos = g.position?;
    if g.l2.is_empty() {
        return None;
    }
    let l2 = std::mem::take(&mut g.l2);
    g.position = None;
    Some((pos, l2))
}

/// Opt-in per-stage residual/activation L2 for spike-layer localization.
/// `HIPFIRE_DEEPSEEK4_STAGE_NORM=1` plus optional
/// `HIPFIRE_DEEPSEEK4_STAGE_LAYERS=25,26,27` (default those three).
pub(crate) fn dump_stage_norm(
    gpu: &mut Gpu,
    tag: &str,
    buf: &GpuTensor,
    layer_idx: usize,
    position: u32,
) {
    use std::sync::LazyLock;
    static ON: LazyLock<bool> = LazyLock::new(|| {
        hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_STAGE_NORM")
            .ok()
            .as_deref()
            == Some("1")
    });
    if !*ON {
        return;
    }
    static LAYERS: LazyLock<Vec<usize>> = LazyLock::new(|| {
        hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_STAGE_LAYERS")
            .ok()
            .map(|s| {
                s.split(',')
                    .filter_map(|t| t.trim().parse().ok())
                    .collect::<Vec<_>>()
            })
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| vec![25, 26, 27])
    });
    if !LAYERS.contains(&layer_idx) {
        return;
    }
    let _ = gpu.hip.device_synchronize();
    let Ok(v) = gpu.download_f32(buf) else {
        return;
    };
    let l2: f64 = v
        .iter()
        .map(|&x| {
            let x = x as f64;
            x * x
        })
        .sum::<f64>()
        .sqrt();
    eprintln!(
        "PROD_STAGE_NORM pos={position} layer={layer_idx} stage={tag} l2={l2:.9e} nelems={}",
        v.len()
    );
}

/// Opt-in dump of indexer top-k + compressed KV L2 after `indexer_forward`.
/// `HIPFIRE_DEEPSEEK4_DUMP_INDEXER=1`; optional
/// `HIPFIRE_DEEPSEEK4_DUMP_INDEXER_LAYERS=2,4,26` (default those).
pub(crate) fn dump_indexer_state(
    gpu: &mut Gpu,
    state: &DeepseekV4State,
    layer_idx: usize,
    position: u32,
    n_scored: usize,
) {
    use std::sync::LazyLock;
    static ON: LazyLock<bool> = LazyLock::new(|| {
        hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_DUMP_INDEXER")
            .ok()
            .as_deref()
            == Some("1")
    });
    if !*ON {
        return;
    }
    static LAYERS: LazyLock<Vec<usize>> = LazyLock::new(|| {
        hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_DUMP_INDEXER_LAYERS")
            .ok()
            .map(|s| {
                s.split(',')
                    .filter_map(|t| t.trim().parse().ok())
                    .collect::<Vec<_>>()
            })
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| vec![2, 4, 26])
    });
    if !LAYERS.contains(&layer_idx) {
        return;
    }
    let _ = gpu.hip.device_synchronize();
    let idx = &state._indexer[layer_idx];
    // topk_idx_indices is DType::F32 storage holding i32 bit patterns.
    let topk_head = match idx.topk_idx_indices.as_ref() {
        Some(t) => match gpu.download_f32(t) {
            Ok(v) => {
                let idxs: Vec<i32> = v
                    .iter()
                    .map(|x| i32::from_le_bytes(x.to_bits().to_le_bytes()))
                    .collect();
                let n_pos = idxs.iter().filter(|&&i| i >= 0).count();
                let head = idxs
                    .iter()
                    .take(32)
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                format!("n_pos={n_pos} head=[{head}]")
            }
            Err(_) => String::from("?"),
        },
        None => String::from("none"),
    };
    let main_l2 = idx
        .main_kv_cache
        .as_ref()
        .and_then(|t| gpu.download_f32(t).ok())
        .map(|v| {
            let n = n_scored.saturating_mul(512).min(v.len());
            v[..n]
                .iter()
                .map(|&x| {
                    let x = x as f64;
                    x * x
                })
                .sum::<f64>()
                .sqrt()
        })
        .unwrap_or(f64::NAN);
    let idx_l2 = idx
        .indexer_kv_cache
        .as_ref()
        .and_then(|t| gpu.download_f32(t).ok())
        .map(|v| {
            let n = n_scored.saturating_mul(512).min(v.len());
            v[..n]
                .iter()
                .map(|&x| {
                    let x = x as f64;
                    x * x
                })
                .sum::<f64>()
                .sqrt()
        })
        .unwrap_or(f64::NAN);
    eprintln!(
        "PROD_INDEXER pos={position} layer={layer_idx} n_scored={n_scored} \
         main_kv_l2={main_l2:.9e} idx_kv_l2={idx_l2:.9e} {topk_head}"
    );
}

/// Override `HIPFIRE_DEEPSEEK4_E8_BATCHED_GEMV` for the rest of the process.
///
/// Exists so a bench can measure the WMMA and batched-GEMV arms against one
/// loaded copy of the trunk. `0` restores the WMMA path for every batch size.
pub fn set_e8_batched_gemv_max_batch(n: usize) {
    config_cache::E8_BATCHED_GEMV_MAX.store(n, std::sync::atomic::Ordering::Relaxed);
}

/// `[rows, m]` = W · `[rows, k]` prepared rows through the dispatch batched
/// projection, with this model's route knobs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemv_auto_batched_wmma(
    gpu: &mut Gpu,
    mq2r_backend: Mq2rBackend,
    weight: &GpuTensor,
    x_rotated_batch: &GpuTensor,
    x_plain_batch: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    batch_size: usize,
    x_f16_scratch: Option<&GpuTensor>,
) -> Result<(), String> {
    let route = crate::rows_program::rows_projection(gpu, mq2r_backend);
    hipfire_dispatch::pipeline::mla::project_prepared_rows(
        gpu,
        weight,
        x_rotated_batch,
        x_plain_batch,
        y,
        m,
        k,
        batch_size,
        x_f16_scratch,
        &route,
    )
    .map_err(dispatch_err)
}

/// DeepSeek V4 Compressor decode step (phase 3b scaffold — not yet wired).
///
/// Implements the upstream `Compressor.forward` decode case
/// (start_pos != 0):
///
///   kv = wkv @ x_rotated     [coff * head_dim]
///   score = wgate @ x_rotated [coff * head_dim]
///   score += ape[pos % ratio]
///   kv_state[ratio + pos%ratio]    = kv     (overlap=true)
///   score_state[ratio + pos%ratio] = score
///   if (pos+1) % ratio == 0:
///     overlap_concat → [2*ratio, head_dim]  for kv and score
///     softmax_pool   → [head_dim] compressed
///     rmsnorm (compressor.norm)
///     if is_indexer: tail RoPE (compress_rope_theta = 160000)
///     kv_cache[pos // ratio] = compressed
///     shift kv_state[:ratio] = kv_state[ratio:]  (and score_state)
///
/// Parameterized by `is_indexer`:
///   - false → main attn compressor; head_dim = cfg.head_dim = 512;
///     no RoPE on output; targets `state._indexer[l].main_*`
///   - true  → indexer's sub-compressor; head_dim = idx_head_dim = 128;
///     applies tail RoPE with cfg.compress_rope_theta;
///     targets `state._indexer[l].indexer_*`
///
pub(crate) fn compressor_cache_uses_vmm(
    gpu: &Gpu,
    backend: hipfire_runtime::kv_backend::KvBackend,
) -> bool {
    backend == hipfire_runtime::kv_backend::KvBackend::Vmm
        && (gpu.arch_caps.is_gfx1151() || gpu.arch_caps.is_gfx1201())
}

fn ensure_cache_tensor_rows(
    gpu: &mut Gpu,
    slot: &mut Option<GpuTensor>,
    logical_rows: usize,
    required_rows: usize,
    row_elems: usize,
    dtype: DType,
    access_devices: &[i32],
    backend: hipfire_runtime::kv_backend::KvBackend,
    label: &str,
) -> Result<bool, String> {
    let required_rows = required_rows.min(logical_rows).max(1);
    let use_vmm = compressor_cache_uses_vmm(gpu, backend);
    let mut changed = false;

    if slot.is_none() {
        let tensor = if use_vmm {
            // Reserve the model horizon but map nothing until the request
            // preflight below computes the needed prefix.
            unsafe { gpu.alloc_vmm_tensor(&[logical_rows, row_elems], dtype, 0, access_devices) }
                .map_err(|e| format!("reserve VMM {label}: {e:?}"))?
        } else {
            gpu.zeros(&[required_rows, row_elems], dtype)
                .map_err(|e| format!("alloc {label}: {e:?}"))?
        };
        *slot = Some(tensor);
        changed = true;
    }

    let tensor = slot.as_mut().expect("cache tensor just materialized");
    if let Some(mapped) = gpu.vmm_mapped_bytes(tensor) {
        let granularity = gpu
            .vmm_granularity(tensor)
            .ok_or_else(|| format!("{label}: VMM allocation has no granularity"))?;
        let row_bytes = row_elems
            .checked_mul(dtype.size())
            .ok_or_else(|| format!("{label}: row-byte overflow"))?;
        let plan = hipfire_runtime::kv_backend::KvChunkPlan::new(
            row_bytes,
            logical_rows,
            crate::deepseek4::INITIAL_COMPRESSED_ROWS.min(logical_rows),
            granularity,
            hipfire_runtime::kv_backend::DEFAULT_VMM_PHYSICAL_CHUNK_BYTES,
        )
        .map_err(|e| format!("{label}: VMM growth plan: {e}"))?;
        if let Some(growth) = plan
            .growth(mapped, required_rows)
            .map_err(|e| format!("{label}: VMM growth: {e}"))?
        {
            gpu.grow_vmm_tensor(tensor, growth.size_bytes, access_devices)
                .map_err(|e| format!("map VMM {label}: {e:?}"))?;
            changed = true;
        }
        return Ok(changed);
    }

    if tensor.shape.first().copied().unwrap_or(0) >= required_rows {
        return Ok(changed);
    }

    // Fallback for architectures without a certified DS4 VMM route: preserve
    // cache contents while growing the pointer-changing dense allocation.
    let replacement = gpu
        .zeros(&[required_rows, row_elems], dtype)
        .map_err(|e| format!("grow {label}: {e:?}"))?;
    let copy_bytes = tensor.byte_size();
    gpu.hip
        .memcpy_dtod(&replacement.buf, &tensor.buf, copy_bytes)
        .map_err(|e| format!("copy old {label}: {e:?}"))?;
    let old = std::mem::replace(tensor, replacement);
    gpu.free_tensor(old)
        .map_err(|e| format!("free old {label}: {e:?}"))?;
    Ok(true)
}

fn ensure_indexer_scratch_rows(
    gpu: &mut Gpu,
    layer: &mut crate::deepseek4::IndexerLayerState,
    required_rows: usize,
    layer_idx: usize,
) -> Result<bool, String> {
    fn replace_if_short(
        gpu: &mut Gpu,
        slot: &mut Option<GpuTensor>,
        required_rows: usize,
        label: &str,
    ) -> Result<bool, String> {
        if slot
            .as_ref()
            .is_some_and(|tensor| tensor.shape.first().copied().unwrap_or(0) >= required_rows)
        {
            return Ok(false);
        }
        let replacement = gpu
            .alloc_tensor(&[required_rows], DType::F32)
            .map_err(|e| format!("alloc {label}: {e:?}"))?;
        if let Some(old) = slot.replace(replacement) {
            gpu.free_tensor(old)
                .map_err(|e| format!("free old {label}: {e:?}"))?;
        }
        Ok(true)
    }

    let score_grew = replace_if_short(
        gpu,
        &mut layer.index_score,
        required_rows,
        &format!("index_score l{layer_idx}"),
    )?;
    // Allocate both merge-tree workspaces with the same stable stride. The
    // product route selects them automatically on gfx1151 MQ2R, while other
    // routes simply retain inexpensive unused scratch.
    let scores_ws_grew = replace_if_short(
        gpu,
        &mut layer.topk_ws_scores,
        required_rows,
        &format!("topk_ws_scores l{layer_idx}"),
    )?;
    let indices_ws_grew = replace_if_short(
        gpu,
        &mut layer.topk_ws_indices,
        required_rows,
        &format!("topk_ws_indices l{layer_idx}"),
    )?;
    Ok(score_grew || scores_ws_grew || indices_ws_grew)
}

const COMPRESSOR_GROWTH_HEADROOM_BYTES: usize = 512 * 1024 * 1024;

fn checked_add_growth(total: &mut usize, bytes: usize, label: &str) -> Result<(), String> {
    *total = total
        .checked_add(bytes)
        .ok_or_else(|| format!("DeepSeek V4 growth-byte overflow at {label}"))?;
    Ok(())
}

fn cache_growth_bytes(
    gpu: &Gpu,
    slot: &Option<GpuTensor>,
    logical_rows: usize,
    required_rows: usize,
    row_elems: usize,
    dtype: DType,
    default_granularity: usize,
    backend: hipfire_runtime::kv_backend::KvBackend,
    label: &str,
) -> Result<usize, String> {
    let row_bytes = row_elems
        .checked_mul(dtype.size())
        .ok_or_else(|| format!("{label}: row-byte overflow"))?;
    if !compressor_cache_uses_vmm(gpu, backend) {
        return match slot {
            Some(tensor) if tensor.shape.first().copied().unwrap_or(0) >= required_rows => Ok(0),
            _ => required_rows
                .checked_mul(row_bytes)
                .ok_or_else(|| format!("{label}: dense growth-byte overflow")),
        };
    }

    let (mapped, granularity) = slot
        .as_ref()
        .and_then(|tensor| Some((gpu.vmm_mapped_bytes(tensor)?, gpu.vmm_granularity(tensor)?)))
        .unwrap_or((0, default_granularity));
    let plan = hipfire_runtime::kv_backend::KvChunkPlan::new(
        row_bytes,
        logical_rows,
        crate::deepseek4::INITIAL_COMPRESSED_ROWS.min(logical_rows),
        granularity,
        hipfire_runtime::kv_backend::DEFAULT_VMM_PHYSICAL_CHUNK_BYTES,
    )
    .map_err(|e| format!("{label}: VMM admission plan: {e}"))?;
    Ok(plan
        .growth(mapped, required_rows.min(logical_rows).max(1))
        .map_err(|e| format!("{label}: VMM admission growth: {e}"))?
        .map_or(0, |growth| growth.size_bytes))
}

/// Refuse a growth request before mutating any cache allocation when its
/// complete physical footprint cannot fit. This keeps the session retryable
/// with a smaller request or lower-precision cache instead of leaving a
/// partially mapped cache after a late-layer allocation failure.
fn admit_compressor_growth(
    cfg: &DeepseekV4Config,
    state: &DeepseekV4State,
    gpu: &Gpu,
    pbs: Option<&PrefillBatchScratch>,
    required_tokens: usize,
) -> Result<(), String> {
    let target_rows = state.compressor_capacity.rows_for_tokens(required_tokens)?;
    let prepared_target = state
        .compressor_capacity
        .prepared_target_for_tokens(required_tokens)?;
    let default_granularity = if compressor_cache_uses_vmm(gpu, state.compressor_cache_backend) {
        gpu.vmm_recommended_granularity()
            .map_err(|e| format!("query {} VMM granularity: {e:?}", gpu.arch))?
    } else {
        1
    };
    let mut growth_bytes = 0usize;

    for (layer_idx, layer) in state._indexer.iter().enumerate() {
        let ratio = layer.compress_ratio as usize;
        if ratio == 0 {
            continue;
        }
        let logical_rows = state.compressor_capacity.max_tokens().div_ceil(ratio);
        let required_layer_rows = prepared_target.div_ceil(ratio).max(1);
        let local_logical_rows = state.compressor_cache_placement.local_rows(logical_rows);
        let local_required_rows = state
            .compressor_cache_placement
            .local_rows(required_layer_rows)
            .max(1);
        checked_add_growth(
            &mut growth_bytes,
            cache_growth_bytes(
                gpu,
                &layer.main_kv_cache,
                local_logical_rows,
                local_required_rows,
                cfg.head_dim,
                state.compressor_cache_dtype,
                default_granularity,
                state.compressor_cache_backend,
                &format!("main_kv_cache l{layer_idx}"),
            )?,
            "main_kv_cache",
        )?;
        if ratio == 4 {
            checked_add_growth(
                &mut growth_bytes,
                cache_growth_bytes(
                    gpu,
                    &layer.indexer_kv_cache,
                    local_logical_rows,
                    local_required_rows,
                    cfg.index_head_dim,
                    state.compressor_cache_dtype,
                    default_granularity,
                    state.compressor_cache_backend,
                    &format!("indexer_kv_cache l{layer_idx}"),
                )?,
                "indexer_kv_cache",
            )?;
            for (slot, label) in [
                (&layer.index_score, "index_score"),
                (&layer.topk_ws_scores, "topk_ws_scores"),
                (&layer.topk_ws_indices, "topk_ws_indices"),
            ] {
                if !slot
                    .as_ref()
                    .is_some_and(|tensor| tensor.shape.first().copied().unwrap_or(0) >= target_rows)
                {
                    checked_add_growth(
                        &mut growth_bytes,
                        target_rows
                            .checked_mul(std::mem::size_of::<f32>())
                            .ok_or_else(|| {
                                format!("{label} l{layer_idx}: scratch growth-byte overflow")
                            })?,
                        label,
                    )?;
                }
            }
        }
    }

    if let Some(pbs) = pbs.filter(|pbs| target_rows > pbs.idx_score_capacity) {
        let bytes = pbs
            .max_batch
            .checked_mul(target_rows)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "idx_scores_batch growth-byte overflow".to_string())?;
        checked_add_growth(&mut growth_bytes, bytes, "idx_scores_batch")?;
    }

    let (free_bytes, total_bytes) = gpu
        .device_mem_info()
        .map_err(|e| format!("query VRAM before DeepSeek V4 cache growth: {e:?}"))?;
    let required_with_headroom = growth_bytes
        .checked_add(COMPRESSOR_GROWTH_HEADROOM_BYTES)
        .ok_or_else(|| "DeepSeek V4 admission-byte overflow".to_string())?;
    if required_with_headroom > free_bytes {
        return Err(format!(
            "DeepSeek V4 {:?} compressed cache cannot admit {required_tokens} tokens atomically: growth {:.2} GiB + {:.2} GiB headroom exceeds {:.2} GiB free ({:.2} GiB addressable); use a lower-precision compressor cache or a shorter request",
            state.compressor_cache_dtype,
            growth_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            COMPRESSOR_GROWTH_HEADROOM_BYTES as f64 / (1024.0 * 1024.0 * 1024.0),
            free_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        ));
    }
    Ok(())
}

/// Ensure every compressed cache covers a complete request before prefill or
/// decode capture starts. Returns whether the launch/scratch bucket changed.
pub fn ensure_compressor_capacity(
    cfg: &DeepseekV4Config,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    required_tokens: usize,
) -> Result<bool, String> {
    if required_tokens <= state.compressor_capacity.prepared_tokens() {
        return Ok(false);
    }
    admit_compressor_growth(cfg, state, gpu, None, required_tokens)?;
    let mut next_plan = state.compressor_capacity;
    let bucket_grew = next_plan.activate_for_tokens(required_tokens)?;
    let active_rows = next_plan.active_rows();
    let prepared_target = next_plan.prepared_target_for_tokens(required_tokens)?;
    let mut layout_grew = bucket_grew;
    let access_devices =
        &state.compressor_cache_access_devices[..state.compressor_cache_access_count];
    let placement = state.compressor_cache_placement;
    let backend = state.compressor_cache_backend;

    for (layer_idx, layer) in state._indexer.iter_mut().enumerate() {
        let ratio = layer.compress_ratio as usize;
        if ratio == 0 {
            continue;
        }
        let logical_rows = next_plan.max_tokens().div_ceil(ratio);
        let required_layer_rows = prepared_target.div_ceil(ratio).max(1);
        let local_logical_rows = placement.local_rows(logical_rows);
        let local_required_rows = placement.local_rows(required_layer_rows).max(1);
        layout_grew |= ensure_cache_tensor_rows(
            gpu,
            &mut layer.main_kv_cache,
            local_logical_rows,
            local_required_rows,
            cfg.head_dim,
            state.compressor_cache_dtype,
            access_devices,
            backend,
            &format!("main_kv_cache l{layer_idx}"),
        )?;
        if ratio == 4 {
            layout_grew |= ensure_cache_tensor_rows(
                gpu,
                &mut layer.indexer_kv_cache,
                local_logical_rows,
                local_required_rows,
                cfg.index_head_dim,
                state.compressor_cache_dtype,
                access_devices,
                backend,
                &format!("indexer_kv_cache l{layer_idx}"),
            )?;
            layout_grew |= ensure_indexer_scratch_rows(gpu, layer, active_rows, layer_idx)?;
        }
    }

    next_plan.mark_prepared(prepared_target);
    state.compressor_capacity = next_plan;
    if layout_grew {
        state.ar_forward_warmed_up = false;
        gpu.invalidate_for_layout_growth();
        eprintln!(
            "[DeepSeek V4] compressed capacity prepared through {prepared_target} tokens ({active_rows} active ratio-4 rows)",
        );
    }
    Ok(layout_grew)
}

/// Request-boundary preflight for both persistent caches and batched score
/// scratch. This is the canonical automatic sizing entry point.
pub fn ensure_request_capacity(
    cfg: &DeepseekV4Config,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    pbs: &mut PrefillBatchScratch,
    required_tokens: usize,
) -> Result<bool, String> {
    let target_rows = state.compressor_capacity.rows_for_tokens(required_tokens)?;
    if required_tokens <= state.compressor_capacity.prepared_tokens()
        && target_rows <= pbs.idx_score_capacity
    {
        return Ok(false);
    }
    admit_compressor_growth(cfg, state, gpu, Some(pbs), required_tokens)?;
    let scratch_grew = pbs.ensure_idx_score_capacity(gpu, target_rows)?;
    let cache_grew = ensure_compressor_capacity(cfg, state, gpu, required_tokens)?;
    Ok(scratch_grew || cache_grew)
}

pub(crate) fn refresh_compressor_cache_shard_tables(
    states: &mut [DeepseekV4State],
) -> Result<(), String> {
    let world = states.len();
    if !matches!(world, 3 | 4) {
        return Err(format!(
            "DeepSeek V4 compressor shard table requires TP3/TP4 (got TP{world})"
        ));
    }
    if states.iter().all(|state| {
        matches!(
            state.compressor_cache_placement,
            crate::deepseek4::CompressorCachePlacement::Replicated
        )
    }) {
        for state in states.iter_mut() {
            for layer in &mut state._indexer {
                layer.main_kv_cache_shards = [0; 4];
                layer.indexer_kv_cache_shards = [0; 4];
                layer.cache_shard_count = 0;
            }
        }
        return Ok(());
    }
    for (rank, state) in states.iter().enumerate() {
        let crate::deepseek4::CompressorCachePlacement::BlockCyclic(shard) =
            state.compressor_cache_placement
        else {
            return Err(format!(
                "DeepSeek V4 compressor shard table rank {rank} is not block-cyclic"
            ));
        };
        if shard.rank() != rank || shard.world() != world {
            return Err(format!(
                "DeepSeek V4 compressor shard topology mismatch at rank {rank}: placement={shard:?}, world={world}"
            ));
        }
    }

    let n_layers = states[0]._indexer.len();
    if states.iter().any(|state| state._indexer.len() != n_layers) {
        return Err("DeepSeek V4 compressor shard layer-count mismatch".to_string());
    }
    for layer_idx in 0..n_layers {
        let ratio = states[0]._indexer[layer_idx].compress_ratio;
        if ratio == 0 {
            continue;
        }
        let mut main_ptrs = [0usize; 4];
        let mut indexer_ptrs = [0usize; 4];
        for rank in 0..world {
            let layer = &states[rank]._indexer[layer_idx];
            main_ptrs[rank] = layer
                .main_kv_cache
                .as_ref()
                .ok_or_else(|| format!("TP{world} rank {rank} main cache missing l{layer_idx}"))?
                .buf
                .as_ptr() as usize;
            if ratio == 4 {
                indexer_ptrs[rank] = layer
                    .indexer_kv_cache
                    .as_ref()
                    .ok_or_else(|| {
                        format!("TP{world} rank {rank} indexer cache missing l{layer_idx}")
                    })?
                    .buf
                    .as_ptr() as usize;
            }
        }
        for state in states.iter_mut() {
            let layer = &mut state._indexer[layer_idx];
            layer.main_kv_cache_shards = main_ptrs;
            layer.indexer_kv_cache_shards = indexer_ptrs;
            layer.cache_shard_count = world;
        }
    }
    Ok(())
}

#[allow(dead_code, clippy::too_many_arguments)]
fn compressor_forward(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
    x_rotated: &GpuTensor,
    position: u32,
    is_indexer: bool,
) -> Result<(), String> {
    compressor_forward_impl(
        cfg, weights, state, gpu, layer_idx, x_rotated, position, is_indexer,
        /*pre_batched=*/ None, /*state_buffer_driven=*/ true,
    )
}

#[allow(clippy::too_many_arguments)]
fn compressor_forward_impl(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
    x_rotated: &GpuTensor,
    position: u32,
    is_indexer: bool,
    pre_batched: Option<(&GpuTensor, &GpuTensor, usize)>,
    state_buffer_driven: bool,
) -> Result<(), String> {
    ensure_compressor(cfg, weights, state, gpu, layer_idx, is_indexer)?;
    if pre_batched.is_none() {
        let project =
            compressor_project_op(cfg, weights, state, gpu, layer_idx, x_rotated, is_indexer)?;
        if let Some(op) = project {
            execute_compressor_project(gpu, &op).map_err(dispatch_err)?;
        }
    }
    let op = compressor_op(
        cfg,
        weights,
        state,
        gpu,
        layer_idx,
        position,
        is_indexer,
        pre_batched,
        state_buffer_driven,
        None,
    )?;
    match op {
        Some(op) => execute_kv_compress(gpu, &op).map_err(dispatch_err),
        None => Ok(()),
    }
}

/// Compressor geometry of a layer: `(ratio, head_dim, proj_dim, state_rows)`,
/// or `None` when this flavor does not run on the layer (dense layers; the
/// indexer flavor only on ratio-4 layers).
fn compressor_geometry(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    layer_idx: usize,
    is_indexer: bool,
) -> Option<(usize, usize, usize, usize)> {
    let ratio = weights.resolve_layer(layer_idx).compress_ratio as usize;
    if ratio == 0 || (is_indexer && ratio != 4) {
        return None;
    }
    // The ratio-4 compressor overlaps two windows.
    let coff = if ratio == 4 { 2 } else { 1 };
    let head_dim = if is_indexer {
        cfg.index_head_dim
    } else {
        cfg.head_dim
    };
    Some((ratio, head_dim, coff * head_dim, coff * ratio))
}

/// The per-layer compressor rings, cache and scratch of one flavor.
pub(crate) fn ensure_compressor(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
    is_indexer: bool,
) -> Result<(), String> {
    let Some((ratio, head_dim, proj_dim, state_rows)) =
        compressor_geometry(cfg, weights, layer_idx, is_indexer)
    else {
        return Ok(());
    };
    let max_compressed = state.compressor_capacity.active_rows();
    let local_max_compressed = state
        .compressor_cache_placement
        .local_rows(max_compressed)
        .max(1);
    let cache_dtype = state.compressor_cache_dtype;
    let l_state = &mut state._indexer[layer_idx];
    if cache_dtype == DType::F16 && l_state.comp_cache_row_f32.is_none() {
        l_state.comp_cache_row_f32 = Some(
            gpu.zeros(&[cfg.head_dim], DType::F32)
                .map_err(|e| format!("alloc comp cache staging row l{layer_idx}: {e:?}"))?,
        );
    }
    let (kv_state, score_state, kv_cache, tag) = if is_indexer {
        (
            &mut l_state.indexer_kv_state,
            &mut l_state.indexer_score_state,
            &mut l_state.indexer_kv_cache,
            "idx",
        )
    } else {
        (
            &mut l_state.main_kv_state,
            &mut l_state.main_score_state,
            &mut l_state.main_kv_cache,
            "main",
        )
    };
    if kv_state.is_none() {
        *kv_state = Some(
            gpu.zeros(&[state_rows, proj_dim], DType::F32)
                .map_err(|e| format!("alloc {tag} kv_state l{layer_idx}: {e:?}"))?,
        );
    }
    if score_state.is_none() {
        // -inf init (reference `score_state = torch.full(-inf)`): unfilled
        // pool slots, e.g. block 0's missing overlap window, get zero weight.
        *score_state = Some(
            gpu.full_f32(&[state_rows, proj_dim], f32::NEG_INFINITY)
                .map_err(|e| format!("alloc {tag} score_state l{layer_idx}: {e:?}"))?,
        );
    }
    if kv_cache.is_none() {
        *kv_cache = Some(
            gpu.zeros(&[local_max_compressed, head_dim], cache_dtype)
                .map_err(|e| format!("alloc {tag} kv_cache l{layer_idx}: {e:?}"))?,
        );
    }
    let alloc = |gpu: &mut Gpu, slot: &mut Option<GpuTensor>, shape: &[usize], name: &str| {
        if slot.is_none() {
            *slot = Some(
                gpu.alloc_tensor(shape, DType::F32)
                    .map_err(|e| format!("alloc {name} l{layer_idx}: {e:?}"))?,
            );
        }
        Ok::<(), String>(())
    };
    // Per-step scratch, sized by the first flavor that runs on the layer
    // (the main compressor; the ratio-4 indexer's proj_dim is smaller).
    alloc(gpu, &mut l_state.comp_kv_buf, &[proj_dim], "comp_kv_buf")?;
    alloc(
        gpu,
        &mut l_state.comp_score_buf,
        &[proj_dim],
        "comp_score_buf",
    )?;
    if ratio == 4 {
        alloc(
            gpu,
            &mut l_state.comp_concat_kv,
            &[2 * ratio, head_dim],
            "comp_concat_kv",
        )?;
        alloc(
            gpu,
            &mut l_state.comp_concat_score,
            &[2 * ratio, head_dim],
            "comp_concat_score",
        )?;
    }
    Ok(())
}

/// The `wkv` / `wgate` projections of one compressor flavor of a decode row
/// over `x_rotated` (plain input `state.tmp_plain`) into the layer's `kv` /
/// `score` scratch, or `None` when this flavor does not run there.
pub(crate) fn compressor_project_op<'a>(
    cfg: &DeepseekV4Config,
    weights: &'a DeepseekV4Weights,
    state: &'a DeepseekV4State,
    gpu: &Gpu,
    layer_idx: usize,
    x_rotated: &'a GpuTensor,
    is_indexer: bool,
) -> Result<Option<CompressorProjectOp<'a>>, String> {
    let Some((_, _, proj_dim, _)) = compressor_geometry(cfg, weights, layer_idx, is_indexer) else {
        return Ok(None);
    };
    let layer = weights.resolve_layer(layer_idx);
    let req = |t: &'a Option<GpuTensor>, name: &str| {
        t.as_ref().ok_or_else(|| format!("{name} l{layer_idx}"))
    };
    let (wkv, wgate) = if is_indexer {
        (
            req(&layer.indexer_compressor_wkv, "idx_comp_wkv")?,
            req(&layer.indexer_compressor_wgate, "idx_comp_wgate")?,
        )
    } else {
        (
            req(&layer.compressor_wkv, "comp_wkv")?,
            req(&layer.compressor_wgate, "comp_wgate")?,
        )
    };
    let l_state = &state._indexer[layer_idx];
    Ok(Some(CompressorProjectOp {
        main: CompressorProjection {
            wkv,
            wgate,
            kv: l_state.comp_kv_buf.as_ref().unwrap(),
            score: l_state.comp_score_buf.as_ref().unwrap(),
            proj_dim,
        },
        indexer: None,
        tmp: x_rotated,
        tmp_plain: state.tmp_plain.as_ref().ok_or_else(|| {
            format!("comp l{layer_idx}: tmp_plain missing (q_lora must run first)")
        })?,
        hidden: cfg.hidden_size,
        rows: 1,
        layer: layer_idx,
        route: CompressorProjectRoute::Decode {
            e8_soa: e8_soa_kernel(&gpu.arch, weights.mq2r_backend),
        },
    }))
}

/// One compressor step of `layer_idx`, or `None` when this flavor does not
/// run there. `pre_batched` points at earlier `[rows, proj_dim]` projections
/// (row `b`); `state_buffer_driven` reads every slot from device state (decode
/// and capture), otherwise the host decides the window boundary. `row_slots`
/// replaces the decode position / attention-state arrays with one row's
/// stripes of a batched window.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compressor_op<'a>(
    cfg: &DeepseekV4Config,
    weights: &'a DeepseekV4Weights,
    state: &'a DeepseekV4State,
    gpu: &Gpu,
    layer_idx: usize,
    position: u32,
    is_indexer: bool,
    pre_batched: Option<(&'a GpuTensor, &'a GpuTensor, usize)>,
    state_buffer_driven: bool,
    row_slots: Option<(&'a GpuTensor, &'a GpuTensor)>,
) -> Result<Option<KvCompressOp<'a>>, String> {
    let Some((ratio, head_dim, _, _)) = compressor_geometry(cfg, weights, layer_idx, is_indexer)
    else {
        return Ok(None);
    };
    let layer = weights.resolve_layer(layer_idx);
    let req = |t: &'a Option<GpuTensor>, name: &str| {
        t.as_ref().ok_or_else(|| format!("{name} l{layer_idx}"))
    };
    let (norm, ape) = if is_indexer {
        (
            req(&layer.indexer_compressor_norm, "idx_comp_norm")?,
            req(&layer.indexer_compressor_ape, "idx_comp_ape")?,
        )
    } else {
        (
            req(&layer.compressor_norm, "comp_norm")?,
            req(&layer.compressor_ape, "comp_ape")?,
        )
    };
    let l_state = &state._indexer[layer_idx];
    // Without earlier batched projections, row 0 of the layer's `kv` /
    // `score` scratch, which a CompressorProject (or packed) step wrote.
    let (kv, score, row) = pre_batched.unwrap_or((
        l_state.comp_kv_buf.as_ref().unwrap(),
        l_state.comp_score_buf.as_ref().unwrap(),
        0,
    ));
    let input = CompressorInput { kv, score, row };
    let (kv_state, score_state, kv_cache) = if is_indexer {
        (
            &l_state.indexer_kv_state,
            &l_state.indexer_score_state,
            &l_state.indexer_kv_cache,
        )
    } else {
        (
            &l_state.main_kv_state,
            &l_state.main_score_state,
            &l_state.main_kv_cache,
        )
    };
    let pos = position as usize;
    let max_compressed = state.compressor_capacity.active_rows();
    let global_row = pos / ratio;
    let local_row = if global_row < max_compressed {
        state.compressor_cache_placement.global_to_local(global_row)
    } else {
        None
    };
    let commit = if state_buffer_driven {
        let (ring_slot, commit_slot, shift_slot) = if ratio == 4 { (6, 7, 10) } else { (8, 9, 11) };
        CompressorCommit::Device {
            ring_slot,
            commit_slot,
            shift_slot,
            // L1 (gfx942 MQ2R ordinary HIP): host-gate the commit stage off
            // the window boundary (where the device commit slot is -1).
            // Never during capture: the graph keeps the full node set.
            ring_only: config_cache::gfx942_compressor_gate_on(&gpu.arch, cfg.mq2r)
                && !gpu.graphs.capture_mode
                && !(pos + 1).is_multiple_of(ratio),
        }
    } else {
        let boundary = (pos + 1).is_multiple_of(ratio) && global_row < max_compressed;
        CompressorCommit::Host {
            ring_row: if ratio == 4 {
                ratio + pos % ratio
            } else {
                pos % ratio
            },
            commit: boundary.then_some(local_row),
        }
    };
    let rope = if is_indexer {
        CompressorRope {
            freq_base: cfg.compress_rope_theta,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            corr_low: 0.0,
            corr_high: 0.0,
            rope_dim: cfg.qk_rope_head_dim,
        }
    } else {
        let (freq_base, freq_scale, ext_factor, attn_factor, corr_low, corr_high) =
            layer_rope_params(cfg, layer.compress_ratio);
        CompressorRope {
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            corr_low,
            corr_high,
            rope_dim: cfg.qk_rope_head_dim,
        }
    };
    // Stage-bisect dump: HIPFIRE_COMP_DUMP="<pos>,<layer>" prints every
    // stage's fingerprint at that (position, layer). Diagnostic only.
    let debug = hipfire_config::developer_var("HIPFIRE_COMP_DUMP")
        .ok()
        .and_then(|s| {
            let mut it = s.split(',');
            let p: u32 = it.next()?.trim().parse().ok()?;
            let l: usize = it.next()?.trim().parse().ok()?;
            Some((p, l))
        })
        .filter(|&(p, l)| p == position && l == layer_idx)
        .map(|_| CompressorDebug {
            position,
            commit_row: local_row,
        });
    Ok(Some(KvCompressOp {
        rows: 1,
        ape,
        norm,
        kv_state: kv_state.as_ref().unwrap(),
        score_state: score_state.as_ref().unwrap(),
        kv_cache: kv_cache.as_ref().unwrap(),
        staging: l_state.comp_cache_row_f32.as_ref(),
        rope,
        ratio,
        head_dim,
        eps: cfg.rms_norm_eps,
        indexer: is_indexer,
        layer: layer_idx,
        route: KvCompressRoute::Decode {
            input,
            concat: match (&l_state.comp_concat_kv, &l_state.comp_concat_score) {
                (Some(kv), Some(score)) if ratio == 4 => Some((kv, score)),
                _ => None,
            },
            attn_state: match row_slots {
                Some((_, attn)) => Some(attn),
                None => state.attn_state_buf.as_ref(),
            },
            pos_array: match row_slots {
                Some((pos, _)) => pos,
                None => state
                    .pos_array_device
                    .as_ref()
                    .ok_or_else(|| "pos_array_device not initialised".to_string())?,
            },
            pos_index: layer_idx * POS_SLOTS_PER_LAYER + if is_indexer { 2 } else { 1 },
            commit,
            host_ape_row: (pre_batched.is_some() && !state_buffer_driven).then_some(pos % ratio),
            gfx1151_route: weights.mq2r_backend.is_gfx1151(),
            debug,
        },
    }))
}

/// DeepSeek V4 indexer scoring + top-K selection (phase 4b).
///
/// Run after `compressor_forward(is_indexer=true)` for layers with
/// `compress_ratio == 4`. Produces `state._indexer[l].topk_idx_indices`,
/// the indices into `indexer_kv_cache` that the modified main attention
/// (phase 5) will gather K/V from.
///
/// Pipeline:
///   q_idx     = indexer_wq_b @ q_lat_rot                  → [H, D]
///   tail-rope on q_idx (compress_rope_theta, current pos) → [H, D]
///   idx_w     = indexer_weights_proj @ state.tmp          → [H]
///   scores[n] = sum_h relu(q_idx[h] · K_cache[n]) * idx_w[h]
///   topk      = top-K(scores) — combined, not per-head
///
/// Returns the actual number of compressed slots scored (0 means no
/// scoring possible because the cache is still empty at this pos).
fn indexer_forward(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
    position: u32,
    idx_weights_preprojected: bool,
) -> Result<usize, String> {
    if weights.resolve_layer(layer_idx).compress_ratio != 4 {
        return Ok(0);
    }
    ensure_indexer(cfg, weights, state, gpu, layer_idx)?;
    let op = indexer_op(
        cfg,
        weights,
        state,
        gpu,
        layer_idx,
        idx_weights_preprojected,
    )?;
    execute_sparse_index(gpu, &op).map_err(dispatch_err)?;
    Ok(indexer_filled_rows(state, position))
}

/// Compressed indexer rows committed through `position` (diagnostics; the
/// kernels read the same count from `attn_state_buf`).
pub(crate) fn indexer_filled_rows(state: &DeepseekV4State, position: u32) -> usize {
    ((position as usize + 1) / 4).min(state.compressor_capacity.active_rows())
}

/// F3/G1 two-stage top-k: selected only when an arch lever is on AND the cap
/// is large enough for the merge tree's extra launches to pay off. Single
/// head only (the workspace has no head dimension).
fn indexer_two_stage_topk(weights: &DeepseekV4Weights, gpu: &Gpu, max_compressed: usize) -> bool {
    (config_cache::gfx942_indexer_topk_two_stage_on(&gpu.arch, weights.mq2r_backend.is_gfx942())
        || config_cache::gfx1151_indexer_topk_two_stage_on(&gpu.arch)
        || config_cache::gfx1100_indexer_topk_two_stage_on(&gpu.arch)
        || config_cache::gfx1201_indexer_topk_two_stage_on(&gpu.arch))
        && two_stage_topk_capacity_eligible(&gpu.arch, max_compressed)
}

pub(crate) fn ensure_indexer(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    let layer = weights.resolve_layer(layer_idx);
    layer
        .indexer_wq_b
        .as_ref()
        .ok_or_else(|| format!("idx_wq_b l{layer_idx}"))?;
    layer
        .indexer_weights_proj
        .as_ref()
        .ok_or_else(|| format!("idx_weights_proj l{layer_idx}"))?;
    let (h, d, k) = (cfg.index_n_heads, cfg.index_head_dim, cfg.index_topk);
    let max_compressed = state.compressor_capacity.active_rows();
    let two_stage = indexer_two_stage_topk(weights, gpu, max_compressed);
    let l_state = &mut state._indexer[layer_idx];
    let mut alloc = |slot: &mut Option<GpuTensor>, shape: &[usize], name: &str| {
        if slot.is_none() {
            *slot = Some(
                gpu.alloc_tensor(shape, DType::F32)
                    .map_err(|e| format!("alloc {name} l{layer_idx}: {e:?}"))?,
            );
        }
        Ok::<(), String>(())
    };
    alloc(&mut l_state.q_idx, &[h, d], "q_idx")?;
    alloc(&mut l_state.idx_weights, &[h], "idx_weights")?;
    alloc(&mut l_state.index_score, &[max_compressed], "index_score")?;
    alloc(&mut l_state.topk_idx_indices, &[k], "topk_idx")?;
    // The two-stage workspace (~2*max_compressed*4 B per layer) is spent only
    // when that route can dispatch.
    if two_stage {
        alloc(
            &mut l_state.topk_ws_scores,
            &[max_compressed],
            "topk_ws_scores",
        )?;
        alloc(
            &mut l_state.topk_ws_indices,
            &[max_compressed],
            "topk_ws_indices",
        )?;
    }
    Ok(())
}

/// One layer's compressed cache as the attention kernels address it.
fn compressed_cache<'a>(
    state: &'a DeepseekV4State,
    cache: &'a GpuTensor,
    shards: &'a [usize; 4],
    what: &str,
    layer_idx: usize,
) -> Result<CompressedCache<'a>, String> {
    Ok(match state.compressor_cache_placement {
        crate::deepseek4::CompressorCachePlacement::BlockCyclic(shard) => {
            if cache.dtype == DType::F16 {
                return Err(format!(
                    "F16 compressor cache does not support block-cyclic {what} l{layer_idx}"
                ));
            }
            CompressedCache::Sharded {
                shards,
                world: shard.world(),
                block_rows: shard.block_rows(),
            }
        }
        _ if cache.dtype == DType::F16 => CompressedCache::F16(cache),
        _ => CompressedCache::F32(cache),
    })
}

pub(crate) fn indexer_op<'a>(
    cfg: &DeepseekV4Config,
    weights: &'a DeepseekV4Weights,
    state: &'a DeepseekV4State,
    gpu: &Gpu,
    layer_idx: usize,
    weights_preprojected: bool,
) -> Result<SparseIndexOp<'a>, String> {
    let layer = weights.resolve_layer(layer_idx);
    let (h, d) = (cfg.index_n_heads, cfg.index_head_dim);
    let max_compressed = state.compressor_capacity.active_rows();
    let l_state = &state._indexer[layer_idx];
    let kv_cache = l_state
        .indexer_kv_cache
        .as_ref()
        .ok_or_else(|| "indexer: kv_cache missing".to_string())?;
    let cache = compressed_cache(
        state,
        kv_cache,
        &l_state.indexer_kv_cache_shards,
        "placement",
        layer_idx,
    )?;
    if let CompressedCache::Sharded { world, .. } = cache {
        if l_state.cache_shard_count != world {
            return Err(format!(
                "indexer shard table missing l{layer_idx}: have {}, want {world}",
                l_state.cache_shard_count
            ));
        }
    }
    let wide = gpu.arch.eq_ignore_ascii_case("gfx1151") || gpu.arch.eq_ignore_ascii_case("gfx1201");
    let top_k = if indexer_two_stage_topk(weights, gpu, max_compressed) {
        IndexTopK::TwoStage {
            ws_scores: l_state.topk_ws_scores.as_ref().unwrap(),
            ws_indices: l_state.topk_ws_indices.as_ref().unwrap(),
        }
    } else if config_cache::gfx942_indexer_topk_parallel_on(
        &gpu.arch,
        weights.mq2r_backend.is_gfx942(),
    ) {
        IndexTopK::BackendParallel {
            bounded: config_cache::gfx942_indexer_topk_bounded_on(
                &gpu.arch,
                weights.mq2r_backend.is_gfx942(),
            ),
            wide,
        }
    } else {
        IndexTopK::Portable { wide }
    };
    Ok(SparseIndexOp {
        rows: 1,
        wq_b: layer
            .indexer_wq_b
            .as_ref()
            .ok_or_else(|| format!("idx_wq_b l{layer_idx}"))?,
        weights_proj: layer
            .indexer_weights_proj
            .as_ref()
            .ok_or_else(|| format!("idx_weights_proj l{layer_idx}"))?,
        q_lat: state
            .q_lat
            .as_ref()
            .ok_or_else(|| "indexer: q_lat not allocated".to_string())?,
        q_lat_rot: state
            .q_lat_rot
            .as_ref()
            .ok_or_else(|| "indexer: q_lat_rot not allocated".to_string())?,
        tmp: state
            .tmp
            .as_ref()
            .ok_or_else(|| "indexer: state.tmp missing".to_string())?,
        tmp_plain: state
            .tmp_plain
            .as_ref()
            .ok_or_else(|| "indexer: tmp_plain missing".to_string())?,
        q_idx: l_state.q_idx.as_ref().unwrap(),
        idx_weights: l_state.idx_weights.as_ref().unwrap(),
        scores: l_state.index_score.as_ref().unwrap(),
        topk_indices: l_state.topk_idx_indices.as_ref().unwrap(),
        pos: state
            .pos_buf
            .as_ref()
            .ok_or_else(|| "indexer: pos_buf missing".to_string())?,
        weights_preprojected,
        heads: h,
        head_dim: d,
        top_k_max: cfg.index_topk,
        q_lora_rank: cfg.q_lora_rank,
        hidden: cfg.hidden_size,
        rope_dim: cfg.qk_rope_head_dim,
        rope_theta: cfg.compress_rope_theta,
        layer: layer_idx,
        route: SparseIndexRoute::Decode {
            cache,
            attn_state: state
                .attn_state_buf
                .as_ref()
                .ok_or_else(|| "indexer: attn_state_buf missing".to_string())?,
            top_k,
            backend: &weights.mq2r_backend,
            head_parallel_rope: config_cache::gfx1201_indexer_rope_heads_on(&gpu.arch, cfg.mq2r)
                && weights.mq2r_backend.is_gfx1201()
                && h == 64
                && d == 128
                && cfg.qk_rope_head_dim == 64,
            max_compressed,
            e8_soa: e8_soa_kernel(&gpu.arch, weights.mq2r_backend),
        },
    })
}

/// Keep gfx1151's initial 2,048-row bucket on its certified one-launch
/// parallel top-K route. Selecting the merge tree from bucket capacity alone
/// added three launches per ratio-4 layer even at tiny `n`, changing the
/// short-context tape from 2,320/32 to 2,383/34 and dropping retained PM4 to
/// linear AQL. The first automatic growth bucket (4,096 rows) is the point at
/// which the long-context two-stage route becomes eligible; capacity growth
/// already rearms graph/replay state before that route is captured.
fn two_stage_topk_capacity_eligible(arch: &str, max_compressed: usize) -> bool {
    max_compressed >= config_cache::indexer_topk_two_stage_min()
        && (!matches!(arch, "gfx1151" | "gfx1201")
            || max_compressed > crate::deepseek4::INITIAL_COMPRESSED_ROWS)
}

/// Single-token decode step. Takes the token id of the previous
/// position, returns the logits over `vocab_size`.
///
/// Caller is responsible for sampler integration and KV-state
/// advancement.
#[allow(unused_variables, dead_code)]
pub fn decode_step(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    ensure_compressor_capacity(cfg, state, gpu, (position as usize).saturating_add(1))?;
    // HIP-graphs prerequisite: lift the ~130 per-token pos_buf
    // `memcpy_htod` calls out of the per-layer code into a single
    // bulk write at decode-step entry. Per-layer kernels then read
    // their slot via `pos_slot(state, layer_idx, slot)`.
    precompute_positions(cfg, state, gpu, position)?;
    // Stage current token_id to device for the GPU hash-router
    // (consumed by `hash_router_normalize_f32_buf` on hash layers).
    precompute_token_id(state, gpu, token_id)?;

    // 1. Token embedding → initial residual streams.
    //    DeepSeek V4 uses `hc_mult = 4` parallel streams. Init pattern is
    //    [embed, 0, 0, 0] (paper-specified; verify against the DeepSeek V4
    //    reference code before optimising).
    init_residual_streams(cfg, weights, state, gpu, token_id)?;

    let _ = decode_step_body(cfg, weights, state, gpu, token_id, position)?;
    let logits = state.logits.as_ref().unwrap();
    gpu.download_f32(logits)
        .map_err(|e| format!("download logits: {e:?}"))
}

/// Direct-HIP heterogeneous token step for the frozen gfx1100+gfx1151 MQ2R
/// route. Dense state and all non-routed arithmetic remain canonical on
/// gfx1100. The layer loop transfers only the prepared FWHT activation and
/// the six normalized routes to gfx1151, then returns a routed-only F32
/// partial for the original ordered add and HC mix.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_step_heterogeneous(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4HeterogeneousWeights,
    state: &mut DeepseekV4State,
    dense_gpu: &mut Gpu,
    routed_gpu: &mut Gpu,
    execution: &mut DeepseekV4HeterogeneousExecution,
    token_id: u32,
    position: u32,
    abort_requested: &dyn Fn() -> bool,
) -> Result<Option<Vec<f32>>, String> {
    if dense_gpu.arch != "gfx1100" || routed_gpu.arch != "gfx1151" {
        return Err(format!(
            "deepseek4 heterogeneous decode requires gfx1100+gfx1151, got {}+{}",
            dense_gpu.arch, routed_gpu.arch
        ));
    }
    if dense_gpu.replay.is_enabled()
        || routed_gpu.replay.is_enabled()
        || dense_gpu.graphs.capture_mode
        || routed_gpu.graphs.capture_mode
    {
        return Err("deepseek4 heterogeneous G4 admits direct HIP only".into());
    }
    if !config_cache::moe_on() {
        return Err("deepseek4 heterogeneous route requires routed MoE enabled".into());
    }

    let dense_weights = &weights.dense.inner;
    ensure_compressor_capacity(cfg, state, dense_gpu, (position as usize).saturating_add(1))?;
    precompute_positions(cfg, state, dense_gpu, position)?;
    precompute_token_id(state, dense_gpu, token_id)?;
    init_residual_streams(cfg, dense_weights, state, dense_gpu, token_id)?;
    ensure_ffn_split_resource(cfg, state, dense_gpu)?;

    for layer_idx in 0..cfg.num_hidden_layers {
        if abort_requested() {
            return Ok(None);
        }
        ds4_attn_block_heterogeneous(
            cfg,
            dense_weights,
            state,
            dense_gpu,
            execution,
            layer_idx,
            position,
        )?;
        mhc_pre(
            cfg,
            dense_weights,
            state,
            dense_gpu,
            layer_idx,
            /*is_attn=*/ false,
        )?;
        ffn_prepare(cfg, dense_weights, state, dense_gpu, layer_idx)?;
        heterogeneous_select_routes(cfg, dense_weights, state, dense_gpu, layer_idx, token_id)?;

        let epoch = execution.next_epoch()?;
        heterogeneous_publish_routes(cfg, state, dense_gpu, routed_gpu, execution, epoch)?;

        // Same dense stream, immediately after publication. The routed stream
        // wakes when the signal is visible, so this shared projection and the
        // selected experts execute concurrently without a host wait.
        ffn_shared_project(cfg, dense_weights, state, dense_gpu, layer_idx)?;

        heterogeneous_run_selected(
            cfg,
            &weights.routed,
            state.ffn_routed_overlap.as_ref().unwrap(),
            dense_gpu.device_id,
            routed_gpu,
            execution,
            layer_idx,
            epoch,
        )?;
        heterogeneous_join(cfg, state, dense_gpu, execution, layer_idx, epoch)?;
        // The callback may have latched while either branch was in flight.
        // Observe it only after the join so no cross-device signal or packet
        // remains outstanding when the serving layer resets request state.
        if abort_requested() {
            return Ok(None);
        }
        hc_ffn_mix(cfg, dense_weights, state, dense_gpu, layer_idx)?;
        dump_residual_layer_norm(dense_gpu, state, layer_idx, position);
    }

    final_norm_and_head(cfg, dense_weights, state, dense_gpu)?;
    state.n_tokens += 1;
    dense_gpu
        .download_f32(state.logits.as_ref().unwrap())
        .map(Some)
        .map_err(|error| format!("download heterogeneous logits: {error:?}"))
}

/// HIP-graphs-aware decode_step. Opt-in via `HIPFIRE_DEEPSEEK4_GRAPH=1`.
///
/// Three-state machine driven by `state.ar_forward_warmed_up` and
/// `gpu.graphs.graph_exec`:
///   1. !warmed_up                   → direct dispatch (warmup so JIT
///                                       and lazy alloc happen out of
///                                       the captured region), set flag
///   2. warmed_up && no graph        → wrap layer loop + head in
///                                       `begin_graph_capture`/`end_graph_capture`,
///                                       instantiate, run it once
///   3. graph already instantiated   → update `pos_array_host[]` on
///                                       the host (stable Box source),
///                                       `graph_launch()` re-runs the
///                                       captured ops which re-read
///                                       pos_array_host; download logits
///
/// Returns logits same as `decode_step`. Falls back to plain
/// `decode_step` when `HIPFIRE_DEEPSEEK4_GRAPH` is unset / "0".
fn ensure_ffn_overlap_resources(
    cfg: &DeepseekV4Config,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
) -> Result<(), String> {
    let mut created = false;
    if state.ffn_routed_overlap.is_none() {
        state.ffn_routed_overlap = Some(
            gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
                .map_err(|e| format!("alloc ffn_routed_overlap: {e:?}"))?,
        );
        created = true;
    }
    if state.ffn_overlap_stream.is_none() {
        state.ffn_overlap_stream = Some(
            gpu.hip
                .stream_create()
                .map_err(|e| format!("create FFN overlap stream: {e:?}"))?,
        );
        created = true;
    }
    if state.ffn_overlap_fork_event.is_none() {
        state.ffn_overlap_fork_event = Some(
            gpu.hip
                .event_create()
                .map_err(|e| format!("create FFN overlap fork event: {e:?}"))?,
        );
        created = true;
    }
    if state.ffn_overlap_join_event.is_none() {
        state.ffn_overlap_join_event = Some(
            gpu.hip
                .event_create()
                .map_err(|e| format!("create FFN overlap join event: {e:?}"))?,
        );
        created = true;
    }
    if created {
        eprintln!("[DeepSeek V4] FFN overlap resources ready");
    }
    Ok(())
}

/// Materialize the primary and side streams before ordinary gfx942 decode.
/// The gfx1151 graph path creates its primary capture stream separately, so
/// this wrapper does not alter that route's launch ordering.
fn ensure_gfx942_ffn_overlap_resources(
    cfg: &DeepseekV4Config,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
) -> Result<(), String> {
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(
            gpu.hip
                .stream_create()
                .map_err(|e| format!("create gfx942 FFN primary stream: {e:?}"))?,
        );
    }
    ensure_ffn_overlap_resources(cfg, state, gpu)
}

fn ensure_ffn_split_resource(
    cfg: &DeepseekV4Config,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
) -> Result<(), String> {
    if state.ffn_routed_overlap.is_none() {
        state.ffn_routed_overlap = Some(
            gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
                .map_err(|e| format!("alloc Redline FFN routed partial: {e:?}"))?,
        );
    }
    Ok(())
}

pub fn decode_step_with_graph(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    use std::sync::OnceLock;
    // Mapping and any launch-geometry bucket transition must complete before
    // HipGraph/Redline decides whether to capture or replay this step.
    ensure_compressor_capacity(cfg, state, gpu, (position as usize).saturating_add(1))?;
    // State-dependent kernargs (SWA slot/n_valid, indexer n_compressed/k_active,
    // compressor ring/commit slots) all live in `state.attn_state_buf` and
    // `state.pos_array_device` device buffers now. The captured graph re-reads
    // those on every replay → byte-equivalent against direct dispatch out to
    // 200+ steps on gfx1151 (graph_drift_check). Default ON for RDNA3+
    // (gfx11xx/gfx12xx) where graph capture is mature; opt out with
    // `HIPFIRE_DEEPSEEK4_GRAPH=0`. Force on for older archs with
    // `HIPFIRE_DEEPSEEK4_GRAPH=1` (untested — beware kernarg-bake regressions).
    static GRAPH_OPT_ENV: OnceLock<Option<bool>> = OnceLock::new();
    let env_override = *GRAPH_OPT_ENV.get_or_init(|| {
        match hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_GRAPH")
            .ok()
            .as_deref()
        {
            Some("1") => Some(true),
            Some("0") => Some(false),
            _ => None,
        }
    });
    let graph_on = env_override.unwrap_or_else(|| {
        let a = gpu.arch.as_str();
        a.starts_with("gfx11") || a.starts_with("gfx12")
    });
    // The gfx942 concurrency route is ordinary HIP with two streams. Create
    // the resources before the direct decode return below; stream creation or
    // allocation inside graph capture would be illegal.
    if config_cache::gfx942_ffn_overlap_on(weights.mq2r_backend.is_gfx942()) {
        ensure_gfx942_ffn_overlap_resources(cfg, state, gpu)?;
    }
    // DeepSeek V4 retained replay is explicit-opt-in while the MQ2R fixture is
    // being certified. Its dynamic token/position state is staged outside the
    // tape, matching the mature HipGraph boundary. The HC ping-pong route is a
    // hard prerequisite: without it, 86 D2D copies live inside the body but
    // outside Redline's typed-kernel tape. The two-stream FFN overlap is also
    // held off for the conservative single-queue tape; it can be reintroduced
    // only after the serial route passes multi-position parity.
    if gpu.replay.is_enabled() {
        let retained_eligible = config_cache::hc_pingpong_on(&gpu.arch, cfg.mq2r)
            && !config_cache::ffn_overlap_on(&gpu.arch, cfg.mq2r);
        gpu.replay.set_forward_eligible(retained_eligible);
        if !retained_eligible {
            let reason = "DeepSeek V4 retained replay requires \
                          HIPFIRE_DEEPSEEK4_HC_PINGPONG=1 and \
                          HIPFIRE_DEEPSEEK4_FFN_OVERLAP=0";
            gpu.replay.poison(reason);
            eprintln!("[DeepSeek V4 redline] falling back to HIP: {reason}");
        } else {
            if config_cache::redline_ffn_split_on(&gpu.arch, cfg.mq2r) {
                ensure_ffn_split_resource(cfg, state, gpu)?;
            }
            // First eligible token remains ordinary HIP so all lazy state,
            // kernels, and the ping-pong allocation exist before recording.
            if !state.ar_forward_warmed_up {
                state.ar_forward_warmed_up = true;
                return decode_step(cfg, weights, state, gpu, token_id, position);
            }

            // External adapter boundary: dynamic scalar/position values are
            // refreshed before either capture or replay.
            precompute_positions(cfg, state, gpu, position)?;
            precompute_token_id(state, gpu, token_id)?;
            let retained_embedding = config_cache::retained_embedding_on(&gpu.arch, cfg.mq2r);
            if !retained_embedding {
                init_residual_streams(cfg, weights, state, gpu, token_id)?;

                // Ownership now crosses from HIP's queue to Redline's ROCr
                // queue. Keep the conservative full-device handoff for the
                // legacy external embedding route.
                gpu.hip
                    .device_synchronize()
                    .map_err(|e| format!("DeepSeek V4 redline adapter sync: {e:?}"))?;
            }

            gpu.replay
                .begin_auto_capture_if_armed()
                .map_err(|reason| format!("DeepSeek V4 redline begin capture: {reason}"))?;

            // Diagnostic oracle: once a retained route is ready, relaunch its
            // exact captured blobs through HIP instead of lowering them to an
            // HSA executable. This separates a bad capture/dynamic-state
            // contract from an AQL/PM4 loader or queue-state mismatch.
            if hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_REDLINE_HIP_BLOB")
                .ok()
                .as_deref()
                == Some("1")
                && (gpu.replay.should_route_aql() || gpu.replay.should_route_pm4())
            {
                let launches = gpu.replay.recorded_launches().len();
                gpu.replay_recorded_hip_prefix(launches)
                    .map_err(|e| format!("DeepSeek V4 retained HIP-blob replay: {e:?}"))?;
                if cfg.load_dspark {
                    let streams = state.residual_streams.as_ref().unwrap();
                    let mtp_hidden = state.mtp_last_hidden.as_ref().unwrap();
                    gpu.memcpy_dtod_auto(
                        &mtp_hidden.buf,
                        &streams.buf,
                        cfg.hc_mult * cfg.hidden_size * 4,
                    )
                    .map_err(|e| format!("DeepSeek V4 retained HIP-blob MTP snapshot: {e:?}"))?;
                }
                state.n_tokens += 1;
                return gpu
                    .download_f32(state.logits.as_ref().unwrap())
                    .map_err(|e| format!("download logits (retained HIP blob): {e:?}"));
            }

            let replay_result = if gpu.replay.should_route_aql() {
                Some(unsafe { gpu.replay.replay_linear_aql(position as usize) })
            } else {
                None
            };
            if let Some(result) = replay_result {
                if let Err(reason) = result {
                    gpu.replay
                        .poison(format!("DeepSeek V4 retained AQL replay failed: {reason}"));
                    return Err(reason);
                }
                // The MTP snapshot is the one intentional D2D adapter outside
                // the retained body. Plain AR never consumes it.
                if cfg.load_dspark {
                    let streams = state.residual_streams.as_ref().unwrap();
                    let mtp_hidden = state.mtp_last_hidden.as_ref().unwrap();
                    gpu.memcpy_dtod_auto(
                        &mtp_hidden.buf,
                        &streams.buf,
                        cfg.hc_mult * cfg.hidden_size * 4,
                    )
                    .map_err(|e| format!("DeepSeek V4 redline MTP snapshot: {e:?}"))?;
                }
                state.n_tokens += 1;
                return gpu
                    .download_f32(state.logits.as_ref().unwrap())
                    .map_err(|e| format!("download logits (retained AQL): {e:?}"));
            }

            let replay_result = if gpu.replay.should_route_pm4() {
                Some(unsafe { gpu.replay.replay_pm4(position as usize) })
            } else {
                None
            };
            if let Some(result) = replay_result {
                if let Err(reason) = result {
                    gpu.replay
                        .poison(format!("DeepSeek V4 retained PM4 replay failed: {reason}"));
                    return Err(reason);
                }
                if cfg.load_dspark {
                    let streams = state.residual_streams.as_ref().unwrap();
                    let mtp_hidden = state.mtp_last_hidden.as_ref().unwrap();
                    gpu.memcpy_dtod_auto(
                        &mtp_hidden.buf,
                        &streams.buf,
                        cfg.hc_mult * cfg.hidden_size * 4,
                    )
                    .map_err(|e| format!("DeepSeek V4 redline MTP snapshot: {e:?}"))?;
                }
                state.n_tokens += 1;
                return gpu
                    .download_f32(state.logits.as_ref().unwrap())
                    .map_err(|e| format!("download logits (retained PM4): {e:?}"));
            }

            // On the buffer-driven route, embedding is the first typed tape
            // dispatch. The token-id H2D above is synchronous, so no HIP queue
            // work remains to hand off to Redline.
            if retained_embedding {
                let token_embd = weights
                    .token_embd
                    .as_ref()
                    .ok_or_else(|| "retained embedding: token_embd missing".to_string())?;
                let streams = state
                    .residual_streams
                    .as_ref()
                    .ok_or_else(|| "retained embedding: residual streams missing".to_string())?;
                let token_id_buf = state
                    .token_id_buf
                    .as_ref()
                    .ok_or_else(|| "retained embedding: token-id buffer missing".to_string())?;
                gpu.embedding_lookup_q8_buf_broadcast(
                    token_embd,
                    streams,
                    token_id_buf,
                    cfg.hidden_size,
                    cfg.hc_mult,
                )
                .map_err(|e| format!("DeepSeek V4 retained embedding: {e:?}"))?;
            }

            // Recording warmup executes through ordinary HIP while
            // launch_maybe_blob records the exact typed kernel sequence.
            let _ = decode_step_body(cfg, weights, state, gpu, token_id, position)?;
            if gpu.replay.should_auto_finalize_capture() {
                gpu.hip
                    .device_synchronize()
                    .map_err(|e| format!("DeepSeek V4 redline capture sync: {e:?}"))?;
                let capture = gpu
                    .replay
                    .finish_capture()
                    .map_err(|reason| format!("DeepSeek V4 redline finish capture: {reason}"))?;
                if hipfire_config::developer_var("HIPFIRE_DS4_REPLAY_INVENTORY")
                    .ok()
                    .as_deref()
                    == Some("1")
                {
                    let mut inventory = std::collections::BTreeMap::<
                        (&str, usize, [u32; 3], [u32; 3]),
                        usize,
                    >::new();
                    for launch in gpu.replay.recorded_launches() {
                        *inventory
                            .entry((
                                launch.kernel.as_str(),
                                launch.kernarg.len(),
                                launch.grid,
                                launch.block,
                            ))
                            .or_default() += 1;
                    }
                    for ((kernel, kernarg, grid, block), count) in inventory {
                        eprintln!(
                            "DS4REPLAY kernel={kernel} count={count} kernarg={kernarg} \
                             grid={grid:?} block={block:?}"
                        );
                    }
                }
                let launches = gpu.replay.recorded_launches().len();
                let prepare = if gpu.replay.uses_pm4_transport() {
                    gpu.replay
                        .prepare_pm4_prefix(gpu.device_id as usize, launches)
                        .map(|_| ())
                } else {
                    gpu.replay
                        .prepare_linear_aql(gpu.device_id as usize)
                        .map(|_| ())
                };
                match prepare {
                    Ok(()) => eprintln!(
                        "[DeepSeek V4 redline] retained route ready: capture={capture:?} identity={:?}",
                        gpu.replay.prepared_route_identity()
                    ),
                    Err(reason) => {
                        gpu.replay.poison(format!(
                            "DeepSeek V4 Redline prepare after warmup failed: {reason}"
                        ));
                        eprintln!("[DeepSeek V4 redline] falling back to HIP: {reason}");
                    }
                }
            }
            return gpu
                .download_f32(state.logits.as_ref().unwrap())
                .map_err(|e| format!("download logits (retained capture): {e:?}"));
        }
    }
    // Note: prior to bc6353e the hash-routed MoE path did a d2h of
    // router scores inside the layer body — that broke HIP graph
    // capture. Replaced by `hash_router_normalize_f32_buf` which
    // reads token_id from a device buffer (staged by
    // `precompute_token_id` at decode entry), so MoE+hash layers
    // are now graph-safe and no guard is needed.
    if !graph_on {
        return decode_step(cfg, weights, state, gpu, token_id, position);
    }

    // Stream/event creation and the routed-partial allocation are illegal
    // during capture. Materialize them on the direct warmup call instead.
    if config_cache::ffn_overlap_on(&gpu.arch, cfg.mq2r) {
        ensure_ffn_overlap_resources(cfg, state, gpu)?;
    }

    // ── Warmup phase: direct dispatch, no capture ──────────────────
    if !state.ar_forward_warmed_up {
        state.ar_forward_warmed_up = true;
        return decode_step(cfg, weights, state, gpu, token_id, position);
    }

    // From here on we need an explicit stream for capture/replay.
    if gpu.active_stream.is_none() {
        let s = gpu
            .hip
            .stream_create()
            .map_err(|e| format!("decode_step_with_graph: stream_create: {e:?}"))?;
        gpu.active_stream = Some(s);
    }

    // Embedding lookup and pos-array host write run OUTSIDE the captured
    // region. token_id is baked into the embedding kernel arg, so capture
    // would lock the graph to a single token. Pos-array host source must
    // be a stable `Box<[i32]>` — the captured memcpy re-reads it on each
    // replay. We update those host bytes BEFORE launching the graph.
    init_residual_streams(cfg, weights, state, gpu, token_id)?;

    if gpu.graphs.graph_exec.is_none() {
        // ── Capture phase ──────────────────────────────────────────
        // precompute_positions + precompute_token_id are called INSIDE
        // the capture so the captured memcpy nodes re-read their stable
        // host sources on each replay.
        gpu.graphs
            .begin_graph_capture(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("begin_graph_capture: {e:?}"))?;
        precompute_positions(cfg, state, gpu, position)?;
        precompute_token_id(state, gpu, token_id)?;
        let _ = decode_step_body(cfg, weights, state, gpu, token_id, position)?;
        gpu.graphs
            .end_graph_capture(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("end_graph_capture: {e:?}"))?;
        // Captured kernels were RECORDED, not executed. Launch the
        // freshly-instantiated graph once so this position's forward
        // actually runs and `state.logits` gets fresh values.
        gpu.graphs
            .graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("graph_launch (capture-end): {e:?}"))?;
        eprintln!(
            "[DeepSeek V4 hipGraph] captured forward — {} kernarg blobs retained",
            gpu.graphs.capture_blobs.len()
        );
    } else {
        // ── Replay phase ───────────────────────────────────────────
        // Host-only update of the stable pos_array_host[], attn_state
        // _host[], and token_id_host[]. The captured memcpy nodes
        // re-read these bytes on graph_launch and propagate them to
        // the device-side pos_array_device / attn_state_buf /
        // token_id_buf which all per-layer kernels read.
        update_pos_array_host(cfg, state, position);
        // attn_state depends on state.n_tokens BEFORE increment (the
        // current position being processed). decode_step normally
        // increments state.n_tokens at the END of the body, so replay
        // sees the right pre-increment value.
        update_attn_state_host(cfg, state, state.n_tokens as u32);
        update_token_id_host(state, token_id);
        gpu.graphs
            .graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("graph_launch (replay): {e:?}"))?;
        state.n_tokens += 1;
    }

    // Logits download is outside the captured region (sync memcpy_dtoh
    // on the null stream — completes after the captured kernels finish
    // because the captured stream is observed by the device).
    let logits = state.logits.as_ref().unwrap();
    gpu.download_f32(logits)
        .map_err(|e| format!("download logits (graph path): {e:?}"))
}

/// Update `state.attn_state_host = [slot, n_valid]` and copy to the
/// device buffer. Called from `precompute_positions` (which is itself
/// inside the captured region during graph capture) so the captured
/// memcpy node re-reads the stable host source on every replay.
///
/// `slot = state.n_tokens % sliding_window`
/// `n_valid = min(state.n_tokens + 1, sliding_window)`
///
/// Layer-independent: all 43 layers read the same two values.
pub(crate) fn precompute_attn_state(
    cfg: &DeepseekV4Config,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
) -> Result<(), String> {
    if state.attn_state_buf.is_none() {
        state.attn_state_buf = Some(
            gpu.alloc_tensor(&[12], DType::F32)
                .map_err(|e| format!("alloc attn_state_buf: {e:?}"))?,
        );
    }
    if state.attn_state_host.is_none() {
        state.attn_state_host = Some(Box::new([0i32; 12]));
    }
    fill_attn_state_host(cfg, state, state.n_tokens as u32);
    let host = state.attn_state_host.as_ref().unwrap();
    let dev = state.attn_state_buf.as_ref().unwrap();
    let bytes = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, 12 * 4) };
    gpu.memcpy_htod_auto(&dev.buf, bytes)
        .map_err(|e| format!("htod attn_state: {e:?}"))
}

/// Internal helper: fill `state.attn_state_host[0..12]` from `position`
/// using DeepSeek V4's compress-ratio + index_topk constants. Used by both
/// `precompute_attn_state` (decode entry) and `update_attn_state_host`
/// (graph replay path).
#[inline]
fn capped_compressed_count(position: i32, ratio: i32, max_compressed: i32) -> i32 {
    debug_assert!(position >= 0);
    debug_assert!(ratio > 0);
    debug_assert!(max_compressed >= 0);
    ((position + 1) / ratio).min(max_compressed)
}

fn fill_attn_state_host(cfg: &DeepseekV4Config, state: &mut DeepseekV4State, position: u32) {
    let win = cfg.sliding_window as i32;
    let topk = cfg.index_topk as i32; // DeepSeek V4: 512
    let pos = position as i32;
    let max_compressed = state.compressor_capacity.active_rows() as i32;
    let swa_slot = pos % win;
    let n_valid_swa = (pos + 1).min(win);
    let n_compressed_4 = capped_compressed_count(pos, 4, max_compressed);
    let n_compressed_128 = capped_compressed_count(pos, 128, max_compressed);
    let k_active_4 = topk.min(n_compressed_4);
    let k_active_128 = topk.min(n_compressed_128);
    // Compressor ring/commit slots. For overlap=true (ratio=4 in DeepSeek V4),
    // the state ring is sized [2*ratio, proj_dim] and writes go to the
    // second half: `ring + ratio + (pos % ratio)`. Commit slot is
    // pos/ratio at commit positions, -1 otherwise (commit kernels
    // early-return on -1).
    let ring_slot_4 = 4 + (pos % 4);
    let global_commit_slot_4 = if (pos + 1) % 4 == 0 {
        let s = pos / 4;
        if s < max_compressed {
            s
        } else {
            -1
        }
    } else {
        -1
    };
    let ring_slot_128 = pos % 128; // overlap=false (ratio=128)
    let global_commit_slot_128 = if (pos + 1) % 128 == 0 {
        let s = pos / 128;
        if s < max_compressed {
            s
        } else {
            -1
        }
    } else {
        -1
    };
    let commit_slot_4 = if global_commit_slot_4 >= 0 {
        state
            .compressor_cache_placement
            .global_to_local(global_commit_slot_4 as usize)
            .map(|slot| slot as i32)
            .unwrap_or(-1)
    } else {
        -1
    };
    let commit_slot_128 = if global_commit_slot_128 >= 0 {
        state
            .compressor_cache_placement
            .global_to_local(global_commit_slot_128 as usize)
            .map(|slot| slot as i32)
            .unwrap_or(-1)
    } else {
        -1
    };
    let host = state
        .attn_state_host
        .as_mut()
        .expect("fill_attn_state_host: attn_state_host not initialised");
    host[0] = swa_slot;
    host[1] = n_valid_swa;
    host[2] = n_compressed_4;
    host[3] = n_compressed_128;
    host[4] = k_active_4;
    host[5] = k_active_128;
    host[6] = ring_slot_4;
    host[7] = commit_slot_4;
    host[8] = ring_slot_128;
    host[9] = commit_slot_128;
    // Cache writes use rank-local slots above. Ring state remains replicated,
    // so its overlap shift must fire on every rank at each global commit.
    host[10] = global_commit_slot_4;
    host[11] = global_commit_slot_128;
}

/// Update host-only `attn_state_host[]` (no device copy). Used by the
/// HIP-graphs replay path — the captured memcpy node re-reads this
/// buffer when graph_launch fires.
pub(crate) fn update_attn_state_host(
    cfg: &DeepseekV4Config,
    state: &mut DeepseekV4State,
    position: u32,
) {
    fill_attn_state_host(cfg, state, position);
}

/// Host-only update of `state.pos_array_host[]` for the given position.
/// Used by the HIP-graphs replay path; the captured memcpy node will
/// re-read these bytes when `graph_launch` runs.
pub(crate) fn update_pos_array_host(
    cfg: &DeepseekV4Config,
    state: &mut DeepseekV4State,
    position: u32,
) {
    let pos_array_host = state.pos_array_host.as_mut().expect(
        "update_pos_array_host: pos_array_host not initialised (call precompute_positions first)",
    );
    fill_pos_array_host(cfg, pos_array_host, position);
}

/// Shared host-side fill of the per-layer `[qk_pos, main_comp_rope_pos,
/// indexer_comp_rope_pos]` triples. Called by both `precompute_positions`
/// (initial alloc + htod path) and `update_pos_array_host` (graph-replay
/// host-only path).
///
/// Reference ds4 uses `comp_pos = pos + 1 - ratio` at compress events
/// (i.e. start of the just-closed window). Equivalent to
/// `pos / ratio * ratio` when `(pos+1) % ratio == 0`, which is exactly
/// when an event fires. "start" matches the reference; "mid" / "end"
/// remain available for diagnostic A/B via `HIPFIRE_DEEPSEEK4_COMP_ROPE_POS`.
///
/// Why one helper: prior to this refactor `precompute_positions` and
/// `update_pos_array_host` carried independently-edited copies of this
/// loop with DIFFERENT defaults (capture path: "mid", replay path:
/// "start"). The captured graph then read one rope_pos at capture time
/// and a different value at replay time, drifting compressor RoPE
/// across the capture/replay boundary.
fn fill_pos_array_host(cfg: &DeepseekV4Config, pos_array_host: &mut [i32], position: u32) {
    let comp_rope_mode = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_COMP_ROPE_POS").ok();
    let comp_rope_mode = comp_rope_mode.as_deref();
    for layer_idx in 0..=cfg.num_hidden_layers {
        let ratio = if layer_idx < cfg.num_hidden_layers {
            cfg.compress_ratios[layer_idx] as usize
        } else {
            0
        };
        let base = layer_idx * POS_SLOTS_PER_LAYER;
        pos_array_host[base] = position as i32;
        if ratio > 0 {
            let main_rope_pos: i32 = match comp_rope_mode {
                Some("end") => position as i32,
                Some("mid") => (((position as usize) / ratio * ratio) + ratio / 2) as i32,
                _ => ((position as usize) / ratio * ratio) as i32,
            };
            let indexer_rope_pos = ((position as usize) / ratio * ratio) as i32;
            pos_array_host[base + 1] = main_rope_pos;
            pos_array_host[base + 2] = indexer_rope_pos;
        } else {
            pos_array_host[base + 1] = 0;
            pos_array_host[base + 2] = 0;
        }
    }
}

/// Captured-region body of `decode_step`: the per-layer forward loop
/// + final norm + head. Token-id-dependent embedding lookup and
/// position-array setup must be done BEFORE calling this — they are
/// non-graph-safe (token_id is kernarg, position-array htod source must
/// stay alive across replays).
///
/// `position` is still passed through so position-derived sizing logic
/// (e.g. `n_filled = (pos + 1) / ratio` in `indexer_forward`) gets the
/// real value — these are HOST computations that select which slots of
/// the captured kernel to read, not kernarg-side position writes.
///
/// Public so the HIP-graphs capture/replay wrapper can call it directly.
pub fn decode_step_body(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    let skip_ffn = config_cache::skip_ffn();
    for layer_idx in 0..cfg.num_hidden_layers {
        program::ensure_layer(cfg, weights, state, gpu, layer_idx)?;
        let layer = program::layer_program(
            cfg, weights, state, gpu, layer_idx, position, token_id, skip_ffn,
        )?;
        program::run(
            &layer, cfg, weights, state, gpu, layer_idx, position, skip_ffn,
        )?;
        dump_residual_layer_norm(gpu, state, layer_idx, position);
    }
    // Head HC combine (also captures the full HC streams into
    // `mtp_last_hidden`, the h_n DeepSeek V4 MTP / DSpark consume), final
    // norm and LM head into `state.logits`. The download stays with the
    // caller so the captured-graph path can place it after `graph_launch`.
    final_norm_and_head(cfg, weights, state, gpu)?;
    state.n_tokens += 1;
    Ok(Vec::new())
}

// The EP executor (`hipfire_runtime::ep::run_layer_program_ep`) drives each
// layer as `[Attend, Moe]` through `Deepseek4Bindings`; both handlers run the
// same sublayer step builders as single-lane decode, with the HC mix deferred
// when a cross-rank reduction must assemble the sublayer output first.

/// Attention block core. `do_mix=false` leaves the rank-local `wo_b` partial
/// in `state.attn_out`; exact gfx1201 attention TP all-reduces that tensor
/// before invoking `hc_attn_mix` through the EP finish hook.
fn ds4_attn_block_core(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
    position: u32,
    do_mix: bool,
) -> Result<(), String> {
    program::ensure_attention_sublayer(cfg, weights, state, gpu, layer_idx)?;
    let mix = if do_mix { Mix::Fold } else { Mix::Deferred };
    {
        let mut steps = LayerProgram::new();
        program::attention_steps(
            &mut steps,
            cfg,
            weights,
            state,
            gpu,
            layer_idx,
            position,
            state.residual_streams.as_ref().unwrap(),
            mix,
            OloraSchedule::Default,
        )?;
        program::run(&steps, cfg, weights, state, gpu, layer_idx, position, false)?;
    }
    if do_mix {
        swap_streams_after_mix(cfg, state, gpu);
    }
    Ok(())
}

/// A sublayer that mixed into the ping-pong buffer hands the streams on in
/// `residual_streams` (the per-sublayer owners read them from there).
fn swap_streams_after_mix(cfg: &DeepseekV4Config, state: &mut DeepseekV4State, gpu: &Gpu) {
    if config_cache::hc_pingpong_on(&gpu.arch, cfg.mq2r) {
        std::mem::swap(
            &mut state.residual_streams,
            &mut state.residual_streams_next,
        );
    }
}

/// Replicated attention block (the established EP and single-GPU behavior).
fn ds4_attn_block(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
    position: u32,
) -> Result<(), String> {
    ds4_attn_block_core(cfg, weights, state, gpu, layer_idx, position, true)
}

/// Exact-gfx1100 attention schedule for the heterogeneous route.
///
/// Q-LoRA and the KV/compressor projections consume the same normalized input
/// but do not depend on each other. The ordinary single-queue path serializes
/// them. Here the primary queue retains Q-LoRA while a persistent side queue
/// evaluates KV plus both compressor branches; the queues rejoin before RoPE,
/// indexer scoring, attention, or any state consumer. Arithmetic within every
/// kernel is unchanged.
fn ds4_attn_block_heterogeneous(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    execution: &mut DeepseekV4HeterogeneousExecution,
    layer_idx: usize,
    position: u32,
) -> Result<(), String> {
    let layer = weights.resolve_layer(layer_idx);
    mhc_pre(cfg, weights, state, gpu, layer_idx, /*is_attn=*/ true)?;
    q_lora_prepare(cfg, weights, state, gpu, layer_idx)?;

    {
        let primary = gpu
            .active_stream
            .as_ref()
            .ok_or_else(|| "heterogeneous dense primary stream missing".to_string())?;
        let fork = execution
            .dense_attn_fork_event
            .as_ref()
            .ok_or_else(|| "heterogeneous dense attention fork missing".to_string())?;
        gpu.hip
            .event_record(fork, Some(primary))
            .map_err(|error| format!("heterogeneous attention fork l{layer_idx}: {error:?}"))?;
        let side = execution
            .dense_attn_stream
            .as_ref()
            .ok_or_else(|| "heterogeneous dense attention stream missing".to_string())?;
        gpu.hip.stream_wait_event(side, fork).map_err(|error| {
            format!("heterogeneous attention side wait l{layer_idx}: {error:?}")
        })?;
    }

    std::mem::swap(&mut gpu.active_stream, &mut execution.dense_attn_stream);
    let side_result = (|| {
        kv_joint(cfg, weights, state, gpu, layer_idx, false)?;
        if layer.compress_ratio > 0 {
            let tmp_view = {
                let tmp = state.tmp.as_ref().unwrap();
                tmp.sub_offset(0, tmp.numel())
            };
            compressor_forward(
                cfg, weights, state, gpu, layer_idx, &tmp_view, position,
                /*is_indexer=*/ false,
            )?;
            if layer.compress_ratio == 4 {
                compressor_forward(
                    cfg, weights, state, gpu, layer_idx, &tmp_view, position,
                    /*is_indexer=*/ true,
                )?;
            }
        }
        let side = gpu
            .active_stream
            .as_ref()
            .ok_or_else(|| "heterogeneous dense attention side stream missing".to_string())?;
        let join = execution
            .dense_attn_join_event
            .as_ref()
            .ok_or_else(|| "heterogeneous dense attention join missing".to_string())?;
        gpu.hip
            .event_record(join, Some(side))
            .map_err(|error| format!("heterogeneous attention join l{layer_idx}: {error:?}"))
    })();
    std::mem::swap(&mut gpu.active_stream, &mut execution.dense_attn_stream);
    side_result?;

    q_lora_project(cfg, weights, state, gpu, layer_idx, false)?;
    {
        let primary = gpu
            .active_stream
            .as_ref()
            .ok_or_else(|| "heterogeneous dense primary stream missing after Q-LoRA".to_string())?;
        let join = execution
            .dense_attn_join_event
            .as_ref()
            .ok_or_else(|| "heterogeneous dense attention join missing".to_string())?;
        gpu.hip.stream_wait_event(primary, join).map_err(|error| {
            format!("heterogeneous attention primary wait l{layer_idx}: {error:?}")
        })?;
    }

    apply_tail_rope(cfg, weights, state, gpu, position, layer_idx)?;
    if layer.compress_ratio == 4 {
        let n = indexer_forward(cfg, weights, state, gpu, layer_idx, position, false)?;
        dump_indexer_state(gpu, state, layer_idx, position, n);
    }
    attn_stub(
        cfg,
        weights,
        state,
        gpu,
        layer_idx,
        OloraSchedule::HeterogeneousGfx1100,
    )?;
    hc_attn_mix(cfg, weights, state, gpu, layer_idx)
}

/// FFN block (replays decode_step_body's FFN arm verbatim).
fn ds4_moe_block(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
    token_id: u32,
    skip_ffn: bool,
) -> Result<(), String> {
    // Non-EP: routed experts combine into `state.ffn_out` (alongside the
    // shared expert seeded by `ffn_stub`), and the HC mix folds ffn_out into
    // `residual_streams` in the same call.
    ds4_moe_block_core(
        cfg, weights, state, gpu, layer_idx, token_id, skip_ffn, None, /*do_mix=*/ true,
    )
}

/// MoE block core, parameterized for expert-parallel (EP).
///
/// - `routed_out = Some(partial)` redirects the routed-expert combine into a
///   zeroed per-rank partial (`partial = Σ_owned w_k · expert_k`), while the
///   SHARED expert (`ffn_stub`) still writes `state.ffn_out` replicated on
///   every rank. The cross-rank all-reduce of `partial` (in the EP executor)
///   then sums the routed contributions; `ds4_ep_add_into_residual` does
///   `ffn_out += all_reduced_partial` so each rank ends with `shared + routed`.
/// - `do_mix = false` defers `hc_ffn_mix` to AFTER the all-reduce (the mix
///   can't run until the full FFN output is assembled).
///
/// `routed_out = None, do_mix = true` is the byte-identical single-GPU path.
fn ds4_moe_block_core(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
    token_id: u32,
    skip_ffn: bool,
    routed_out: Option<&GpuTensor>,
    do_mix: bool,
) -> Result<(), String> {
    ensure_mhc_pre(cfg, weights, state, gpu)?;
    ensure_expert_ffn(cfg, weights, state, gpu, layer_idx)?;
    let mix = if do_mix { Mix::Fold } else { Mix::Deferred };
    {
        let mut steps = LayerProgram::new();
        program::ffn_steps(
            &mut steps,
            cfg,
            weights,
            state,
            gpu,
            layer_idx,
            token_id,
            skip_ffn,
            state.residual_streams.as_ref().unwrap(),
            routed_out,
            mix,
        )?;
        program::run(&steps, cfg, weights, state, gpu, layer_idx, 0, skip_ffn)?;
    }
    if do_mix {
        swap_streams_after_mix(cfg, state, gpu);
    }
    Ok(())
}

/// Scratch the FFN of one decode row writes: the shared-expert set and, on
/// layers whose routed experts run, the router and routed-expert scratch.
pub(crate) fn ensure_expert_ffn(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    ensure_ffn_prepare(cfg, state, gpu)?;
    if routed_experts_present(cfg, weights, layer_idx) {
        ensure_moe_route(cfg, state, gpu)?;
        ensure_routed_experts(cfg, state, gpu, layer_idx < cfg.num_hash_layers)?;
    }
    Ok(())
}

/// The FFN step of one decode row: the expert FFN, or a zero fill of
/// `ffn_out` when `skip_ffn` (diagnostic). `routed_out` redirects the routed
/// accumulation into an EP partial; `allow_split` admits the overlap and
/// retained-split schedules (single owner, HC mix right after).
#[allow(clippy::too_many_arguments)]
pub(crate) fn expert_ffn_step<'a>(
    cfg: &DeepseekV4Config,
    weights: &'a DeepseekV4Weights,
    state: &'a DeepseekV4State,
    gpu: &Gpu,
    layer_idx: usize,
    token_id: u32,
    skip_ffn: bool,
    routed_out: Option<&'a GpuTensor>,
    allow_split: bool,
) -> Result<Step<'a>, String> {
    let ffn_out = state.ffn_out.as_ref().unwrap();
    if skip_ffn {
        return Ok(Step::Clear(ClearOp {
            tensor: ffn_out,
            elements: cfg.hidden_size,
        }));
    }
    let schedule = if allow_split {
        expert_ffn_schedule(cfg, weights, state, gpu, layer_idx)
    } else {
        ExpertFfnSchedule::Serial
    };
    let routed_target = match &schedule {
        ExpertFfnSchedule::Serial => routed_out.unwrap_or(ffn_out),
        ExpertFfnSchedule::Overlap { partial, .. } | ExpertFfnSchedule::Split { partial } => {
            partial
        }
    };
    let router = routed_experts_present(cfg, weights, layer_idx);
    let routed = if router {
        Some(routed_experts_op(
            cfg,
            weights,
            state,
            gpu,
            layer_idx,
            token_id,
            routed_target,
        )?)
    } else {
        None
    };
    let op = expert_ffn_op(
        cfg, weights, state, gpu, layer_idx, router, routed, schedule,
    )?;
    Ok(Step::ExpertFfn(op))
}

/// Hash-routed and score-routed layers BOTH need the unbiased router scores
/// for the expert weights; score layers additionally select on `score +
/// gate.bias`. Upstream DeepSeek V4 gates on the post-ffn_norm input (the
/// `x` the experts see), so the norm must have run first.
#[allow(clippy::too_many_arguments)]
fn expert_ffn_op<'a>(
    cfg: &DeepseekV4Config,
    weights: &'a DeepseekV4Weights,
    state: &'a DeepseekV4State,
    gpu: &Gpu,
    layer_idx: usize,
    router: bool,
    routed: Option<RoutedExpertsOp<'a>>,
    schedule: ExpertFfnSchedule<'a>,
) -> Result<ExpertFfnOp<'a>, String> {
    let layer = weights.resolve_layer(layer_idx);
    let shared_w1 = layer.shared_w1.as_ref().unwrap();
    let shared_w3 = layer.shared_w3.as_ref().unwrap();
    let gate = if router {
        Some(
            layer
                .gate_weight
                .as_ref()
                .ok_or_else(|| format!("layer {layer_idx} gate.weight missing"))?,
        )
    } else {
        None
    };
    // On the gfx1151 MQ2R route the router GEMV lands on the E8-SoA U4 buffer
    // kernel; the separate sqrt(softplus(.)) costs ~1.43 ms/token on a 1x1x1
    // grid that is almost entirely launch and drain. The fused route applies
    // it in the GEMV's store, where the reduced row sum is in a register.
    let fused_router = gate.is_some_and(|gate| {
        config_cache::moe_route_fused_activation()
            && gpu.arch_caps.is_gfx1151()
            && gate.dtype == DType::MFP4G32E8SOA
            && config_cache::e8_u4_on(&gpu.arch, weights.mq2r_backend.is_gfx1151())
    });
    Ok(ExpertFfnOp {
        x_in: state.hc_x_in.as_ref().unwrap(),
        norm: layer.ffn_norm.as_ref().unwrap(),
        x_rot: state.ffn_x_rot.as_ref().ok_or_else(|| {
            "ffn_x_rot not allocated — ensure_ffn_prepare must run first".to_string()
        })?,
        x_plain: state.ffn_x_plain.as_ref().ok_or_else(|| {
            "ffn_x_plain not allocated — ensure_ffn_prepare must run first".to_string()
        })?,
        shared_w1,
        shared_w2: layer.shared_w2.as_ref().unwrap(),
        shared_w3,
        shared_gate: state.ffn_gate.as_ref().unwrap(),
        shared_up: state.ffn_up.as_ref().unwrap(),
        shared_rot: state.ffn_silu_rot.as_ref().unwrap(),
        out: state.ffn_out.as_ref().unwrap(),
        router: gate.map(|gate| ExpertFfnRouter {
            gate,
            scores: state.router_scores.as_ref().unwrap(),
            n_exp: cfg.n_routed_experts,
        }),
        hidden: cfg.hidden_size,
        shared_im: shared_expert_rows(cfg, weights, layer_idx),
        swiglu_limit: cfg.swiglu_limit,
        eps: cfg.rms_norm_eps,
        // CORRECTNESS: the routed MQ2-Lloyd experts read the FWHT-rotated input,
        // so the rotation stays live whenever MoE is on (default; opt out with
        // HIPFIRE_DEEPSEEK4_MOE=0), regardless of the shared weight dtype. With
        // MoE off and Q8/F16/F32 shared gate/up only the plain norm is written.
        rotate: config_cache::moe_on()
            || weight_needs_fwht(shared_w1)
            || weight_needs_fwht(shared_w3),
        rows: 1,
        layer: layer_idx,
        route: ExpertFfnRoute::Decode {
            rotate_nox: weights.mq2r_backend.is_gfx1151()
                || config_cache::gfx1201_rmsnorm_rotate_nox_on(&gpu.arch, cfg.mq2r),
            e8_soa: e8_soa_kernel(&gpu.arch, weights.mq2r_backend),
            fused_router,
            routed,
            schedule,
        },
    })
}

/// Shared/routed ordering of a single-owner decode FFN. The gfx1151 (and
/// gfx942) overlap forks the shared E8 expert onto a side stream while the
/// main stream routes and runs the six MQ2 experts; the streams join before
/// the routed-only partial is folded into `ffn_out`. Retained replay records
/// the same split topology serially so Redline's phase planner can place the
/// branches on separate queues.
fn expert_ffn_schedule<'a>(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &'a DeepseekV4State,
    gpu: &Gpu,
    layer_idx: usize,
) -> ExpertFfnSchedule<'a> {
    let layer = weights.resolve_layer(layer_idx);
    let blobs = layer.expert_gate_up_blob.is_some() && layer.expert_w2_blob.is_some();
    let moe = config_cache::moe_on();
    let gfx942_overlap = config_cache::gfx942_ffn_overlap_on(weights.mq2r_backend.is_gfx942())
        && !gpu.graphs.capture_mode;
    if moe
        && blobs
        && (config_cache::ffn_overlap_on(&gpu.arch, cfg.mq2r) || gfx942_overlap)
        && gpu.active_stream.is_some()
    {
        if let (Some(side), Some(fork), Some(join), Some(partial)) = (
            state.ffn_overlap_stream.as_ref(),
            state.ffn_overlap_fork_event.as_ref(),
            state.ffn_overlap_join_event.as_ref(),
            state.ffn_routed_overlap.as_ref(),
        ) {
            return ExpertFfnSchedule::Overlap {
                side,
                fork,
                join,
                partial,
            };
        }
    }
    if moe
        && blobs
        && gpu.replay.is_enabled()
        && config_cache::redline_ffn_split_on(&gpu.arch, cfg.mq2r)
    {
        if let Some(partial) = state.ffn_routed_overlap.as_ref() {
            return ExpertFfnSchedule::Split { partial };
        }
    }
    ExpertFfnSchedule::Serial
}

/// Calibration and route captures of one decode-row FFN. Every captured
/// tensor is final once the shared expert has run (the routed experts write
/// only their own scratch and `ffn_out`).
pub(crate) fn expert_ffn_dumps(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    ffn_shared_project_dumps(cfg, weights, state, gpu, layer_idx)?;
    if routed_experts_present(cfg, weights, layer_idx) {
        dump_moe_route_if_enabled(
            gpu,
            layer_idx,
            state.moe_topk_indices.as_ref().unwrap(),
            state.moe_topk_weights.as_ref().unwrap(),
        )?;
    }
    Ok(())
}

/// Per-layer execution context for the lowered decode path (rebuilt each layer).
pub(crate) struct Deepseek4Bindings<'a> {
    pub(crate) cfg: &'a DeepseekV4Config,
    pub(crate) weights: &'a DeepseekV4Weights,
    pub(crate) state: &'a mut DeepseekV4State,
    pub(crate) layer_idx: usize,
    pub(crate) position: u32,
    pub(crate) token_id: u32,
    pub(crate) skip_ffn: bool,
}

impl<'a> ForwardBindings for Deepseek4Bindings<'a> {
    fn run_attend(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        ds4_attn_block(
            self.cfg,
            self.weights,
            self.state,
            gpu,
            self.layer_idx,
            self.position,
        )
        .map_err(DispatchError::Hip)
    }
    fn attention_tp_enabled(&self) -> bool {
        self.weights.resolve_layer(self.layer_idx).attn_tp_size > 1
    }
    fn run_attend_ep(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let layer = self.weights.resolve_layer(self.layer_idx);
        let tp_size = layer.attn_tp_size;
        if tp_size <= 1
            || layer.attn_head_count == 0
            || layer.attn_group_count == 0
            || layer.attn_head_count * self.cfg.o_groups
                != layer.attn_group_count * self.cfg.num_attention_heads
        {
            return Err(DispatchError::Hip(format!(
                "deepseek4 attention TP invalid at layer {}: local_heads={} local_groups={} global_heads={} global_groups={} tp={tp_size}",
                self.layer_idx,
                layer.attn_head_count,
                layer.attn_group_count,
                self.cfg.num_attention_heads,
                self.cfg.o_groups
            )));
        }

        // Every rank owns a contiguous head range and the corresponding
        // contiguous O-LoRA groups. Head-local attention arithmetic is
        // unchanged; only the explicit shape view becomes rank-local.
        let mut local_cfg = self.cfg.clone();
        local_cfg.num_attention_heads = layer.attn_head_count;
        local_cfg.o_groups = layer.attn_group_count;
        ds4_attn_block_core(
            &local_cfg,
            self.weights,
            self.state,
            gpu,
            self.layer_idx,
            self.position,
            /*do_mix=*/ false,
        )
        .map_err(DispatchError::Hip)
    }
    fn ep_attention_partial(&self) -> Option<&GpuTensor> {
        self.state.attn_out.as_ref()
    }
    fn ep_finish_attend(&mut self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        hc_attn_mix(self.cfg, self.weights, self.state, gpu, self.layer_idx)
            .map_err(DispatchError::Hip)
    }
    fn supports_tp_peer_hc4(&self) -> bool {
        let layer = self.weights.resolve_layer(self.layer_idx);
        self.cfg.mq2r && layer.attn_tp_size == 4 && layer.shared_tp_size == 4
    }
    fn supports_tp_peer_hc3(&self) -> bool {
        let layer = self.weights.resolve_layer(self.layer_idx);
        self.cfg.mq2r && layer.attn_tp_size == 3 && layer.shared_tp_size == 3
    }
    fn ep_finish_attend_peer_hc3(
        &mut self,
        gpu: &mut Gpu,
        partials: [&GpuTensor; 3],
    ) -> Result<(), DispatchError> {
        hc_attn_mix_peer_hc3(self.cfg, self.state, gpu, partials).map_err(DispatchError::Hip)
    }
    fn ep_finish_attend_peer_hc4(
        &mut self,
        gpu: &mut Gpu,
        partials: [&GpuTensor; 4],
    ) -> Result<(), DispatchError> {
        hc_attn_mix_peer_hc4(self.cfg, self.state, gpu, partials).map_err(DispatchError::Hip)
    }
    fn run_moe(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        ds4_moe_block(
            self.cfg,
            self.weights,
            self.state,
            gpu,
            self.layer_idx,
            self.token_id,
            self.skip_ffn,
        )
        .map_err(DispatchError::Hip)
    }
    fn run_moe_ep(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
        routed_out: &GpuTensor,
        _skip_shared: bool,
    ) -> Result<(), DispatchError> {
        // EP: run mhc_pre + the SHARED expert (ffn_stub, replicated into
        // state.ffn_out on every rank) + the ROUTED experts redirected into the
        // zeroed `routed_out` partial. `hc_ffn_mix` is DEFERRED (do_mix=false)
        // to `ep_add_into_residual`, which runs after the cross-rank all-reduce
        // assembles the full routed output. `skip_shared` is intentionally
        // ignored: DeepSeek's shared expert lives in ffn_out (outside the
        // all-reduced partial), so replicating it per rank is correct — it is
        // never summed across ranks.
        ds4_moe_block_core(
            self.cfg,
            self.weights,
            self.state,
            gpu,
            self.layer_idx,
            self.token_id,
            self.skip_ffn,
            Some(routed_out),
            /*do_mix=*/ false,
        )
        .map_err(DispatchError::Hip)?;

        // Exact gfx1201 dense TP: the shared down projection is a local
        // input-column partial. Fold it into the already-zeroed routed partial
        // before the existing EP all-reduce, then clear ffn_out so the normal
        // post-reduce add installs the full shared+routed result exactly once.
        let layer = self.weights.resolve_layer(self.layer_idx);
        if layer.shared_tp_size > 1 {
            let ffn_out =
                self.state.ffn_out.as_ref().ok_or_else(|| {
                    DispatchError::Hip("run_moe_ep dense TP: ffn_out unset".into())
                })?;
            gpu.add_inplace_f32(routed_out, ffn_out)
                .map_err(|error| DispatchError::Hip(error.to_string()))?;
            gpu.zero_f32(ffn_out)
                .map_err(|error| DispatchError::Hip(error.to_string()))?;
        }
        Ok(())
    }
    fn ep_add_into_residual(
        &mut self,
        gpu: &mut Gpu,
        partial: &GpuTensor,
    ) -> Result<(), DispatchError> {
        // ffn_out (shared, replicated) += all-reduced routed partial → full FFN
        // output, then run the deferred HC mix to fold it into residual_streams.
        {
            let ffn_out =
                self.state.ffn_out.as_ref().ok_or_else(|| {
                    DispatchError::Hip("ep_add_into_residual: ffn_out unset".into())
                })?;
            gpu.add_inplace_f32(ffn_out, partial)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
        }
        hc_ffn_mix(self.cfg, self.weights, self.state, gpu, self.layer_idx)
            .map_err(DispatchError::Hip)
    }
    fn ep_finish_moe_peer_hc4(
        &mut self,
        gpu: &mut Gpu,
        partials: [&GpuTensor; 4],
    ) -> Result<(), DispatchError> {
        hc_ffn_mix_peer_hc4(self.cfg, self.state, gpu, partials).map_err(DispatchError::Hip)
    }
    fn ep_finish_moe_peer_hc3(
        &mut self,
        gpu: &mut Gpu,
        partials: [&GpuTensor; 3],
    ) -> Result<(), DispatchError> {
        hc_ffn_mix_peer_hc3(self.cfg, self.state, gpu, partials).map_err(DispatchError::Hip)
    }
    fn run_proj(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip("deepseek4 has no Proj super-op".into()))
    }
    fn run_residual_gemv(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(
            "deepseek4 has no ResidualGemv super-op".into(),
        ))
    }
    fn run_norm(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip("deepseek4 has no Norm super-op".into()))
    }
    fn run_conv(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip("deepseek4 has no Conv super-op".into()))
    }
    fn run_recurrent(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(
            "deepseek4 has no Recurrent super-op".into(),
        ))
    }
    fn run_escape(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
        kind: superop::EscapeKind,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(format!(
            "deepseek4 has no Escape super-op ({kind:?})"
        )))
    }
}

#[inline]
pub(crate) fn ds4_superop(kind: SuperOpKind) -> SuperOp {
    SuperOp {
        kind,
        binding: OpBinding {
            key: None,
            weights: Vec::new(),
            scratch: Vec::new(),
            flavor: OpFlavor::None,
        },
    }
}

/// deepseek4 has ONE layer shape ([Attend, Moe]); the per-layer conditionals are
/// inside the handlers. Pure → unit-testable.
pub(crate) fn ds4_lower_program() -> superop::LayerProgram {
    vec![
        ds4_superop(SuperOpKind::Attend),
        ds4_superop(SuperOpKind::Moe),
    ]
}

pub use crate::ep::forward_ep;
pub use crate::mtp::{mtp_forward, mtp_forward_batched, mtp_forward_ep};

/// FFN block (partial — shared expert only; routed experts pending).
///
/// DeepSeek V4 has one shared expert + 256 routed experts (top-6 selected
/// per token). The shared expert is a standard SwiGLU:
///   gate = x @ shared_w1   [moe_intermediate=2048]
///   up   = x @ shared_w3   [moe_intermediate]
///   silu_gated = silu(gate) * up
///   out  = silu_gated @ shared_w2   [hidden]
///
/// Then x_ffn = shared_out + routed_scaling_factor * routed_out.
/// Routed_out is currently 0 (router/expert dispatch pending), so
/// ffn_out = shared_out.
pub(crate) fn ffn_prepare(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    ensure_ffn_prepare(cfg, state, gpu)?;
    let op = expert_ffn_op(
        cfg,
        weights,
        state,
        gpu,
        layer_idx,
        false,
        None,
        ExpertFfnSchedule::Serial,
    )?;
    execute_ffn_norm(gpu, &op).map_err(dispatch_err)
}

fn ensure_ffn_prepare(
    cfg: &DeepseekV4Config,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
) -> Result<(), String> {
    let im = cfg.moe_intermediate_size;
    if state.ffn_out.is_none() {
        state.ffn_out = Some(
            gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
                .map_err(|e| format!("alloc ffn_out: {e:?}"))?,
        );
    }
    if state.ffn_x_rot.is_none() {
        state.ffn_x_rot = Some(
            gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
                .map_err(|e| format!("alloc ffn_x_rot: {e:?}"))?,
        );
    }
    if state.ffn_gate.is_none() {
        state.ffn_gate = Some(
            gpu.alloc_tensor(&[im], DType::F32)
                .map_err(|e| format!("alloc ffn_gate: {e:?}"))?,
        );
    }
    if state.ffn_up.is_none() {
        state.ffn_up = Some(
            gpu.alloc_tensor(&[im], DType::F32)
                .map_err(|e| format!("alloc ffn_up: {e:?}"))?,
        );
    }
    if state.ffn_silu_rot.is_none() {
        state.ffn_silu_rot = Some(
            gpu.alloc_tensor(&[im], DType::F32)
                .map_err(|e| format!("alloc ffn_silu_rot: {e:?}"))?,
        );
    }
    if state.ffn_x_plain.is_none() {
        state.ffn_x_plain = Some(
            gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
                .map_err(|e| format!("alloc ffn_x_plain: {e:?}"))?,
        );
    }
    Ok(())
}

/// Evaluate only the shared expert after `ffn_prepare` has populated the
/// normalized/rotated FFN input and all persistent scratch.
pub(crate) fn ffn_shared_project(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    let op = expert_ffn_op(
        cfg,
        weights,
        state,
        gpu,
        layer_idx,
        false,
        None,
        ExpertFfnSchedule::Serial,
    )?;
    execute_shared_expert(gpu, &op).map_err(dispatch_err)?;
    ffn_shared_project_dumps(cfg, weights, state, gpu, layer_idx)
}

/// Intermediate rows of this rank's shared expert (dense TP shards them).
fn shared_expert_rows(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    layer_idx: usize,
) -> usize {
    let layer = weights.resolve_layer(layer_idx);
    let im = cfg.moe_intermediate_size;
    let local_im = if layer.shared_tp_size > 1 {
        layer.shared_intermediate_count
    } else {
        im
    };
    debug_assert!(local_im > 0 && local_im <= im);
    local_im
}

/// P3 calibration captures of the shared expert's inputs. The FFN norm output
/// and the gate/up projections are unchanged after the shared expert runs.
fn ffn_shared_project_dumps(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    if !dense_activation_dump_enabled()? {
        return Ok(());
    }
    let local_im = shared_expert_rows(cfg, weights, layer_idx);
    // shared w1/w3 and the routed-expert selector all consume the exact
    // same post-ffn_norm, pre-FWHT activation.
    let names = [
        format!("layers.{layer_idx}.ffn.shared_experts.w1.weight"),
        format!("layers.{layer_idx}.ffn.shared_experts.w3.weight"),
        format!("layers.{layer_idx}.ffn.gate.weight"),
    ];
    let ffn_x_plain = state.ffn_x_plain.as_ref().unwrap();
    dump_dense_activations_if_enabled(gpu, &names, ffn_x_plain, cfg.hidden_size)?;
    let gate = state.ffn_gate.as_ref().unwrap().sub_offset(0, local_im);
    let shared_w2 = weights.resolve_layer(layer_idx).shared_w2.as_ref().unwrap();
    let w2_input = if weight_needs_fwht(shared_w2) {
        // The shipping route fuses the activation into the FWHT; materialize
        // the logical pre-FWHT input in diagnostic scratch only.
        let up = state.ffn_up.as_ref().unwrap().sub_offset(0, local_im);
        let scratch = state
            .embed_scratch
            .as_ref()
            .ok_or_else(|| "dense capture: embed_scratch missing".to_string())?
            .sub_offset(0, local_im);
        gpu.deepseek4_silu_mul_clamp_f32(&gate, &up, &scratch, cfg.swiglu_limit)
            .map_err(|e| format!("dense capture shared silu layer {layer_idx}: {e:?}"))?;
        scratch
    } else {
        gate
    };
    dump_dense_activation_if_enabled(
        gpu,
        &format!("layers.{layer_idx}.ffn.shared_experts.w2.weight"),
        &w2_input,
        local_im,
    )
}

/// Routed experts run on this layer: MoE on, expert blobs uploaded and, on
/// hash layers, a `tid2eid` table present (pre-FP4-fix quants skipped it and
/// fall back to the shared expert alone).
fn routed_experts_present(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    layer_idx: usize,
) -> bool {
    if !config_cache::moe_on() {
        return false;
    }
    let layer = weights.resolve_layer(layer_idx);
    if layer.expert_gate_up_blob.is_none() || layer.expert_w2_blob.is_none() {
        return false;
    }
    !(layer_idx < cfg.num_hash_layers && layer.tid2eid_host.is_empty())
}

fn ensure_routed_experts(
    cfg: &DeepseekV4Config,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    hash: bool,
) -> Result<(), String> {
    let k_top = cfg.num_experts_per_tok;
    let im = cfg.moe_intermediate_size;
    let tag = if hash { " hash" } else { "" };
    let alloc = |gpu: &mut Gpu, shape: &[usize], name: &str| {
        gpu.alloc_tensor(shape, DType::F32)
            .map_err(|e| format!("alloc {name}{tag}: {e:?}"))
    };
    if state.moe_topk_indices.is_none() {
        state.moe_topk_indices = Some(alloc(gpu, &[k_top], "moe_topk_indices")?);
    }
    if state.moe_topk_weights.is_none() {
        state.moe_topk_weights = Some(alloc(gpu, &[k_top], "moe_topk_weights")?);
    }
    if state.moe_gate_batch.is_none() {
        state.moe_gate_batch = Some(alloc(gpu, &[k_top, im], "moe_gate_batch")?);
    }
    if state.moe_up_batch.is_none() {
        state.moe_up_batch = Some(alloc(gpu, &[k_top, im], "moe_up_batch")?);
    }
    if state.moe_rot_batch.is_none() {
        state.moe_rot_batch = Some(alloc(gpu, &[k_top, im], "moe_rot_batch")?);
    }
    // [k_top × hidden] per-expert down outputs for the deterministic
    // (atomic-free) bias-aware combine.
    if !hash && state.moe_down_expert_outputs.is_none() {
        state.moe_down_expert_outputs = Some(alloc(
            gpu,
            &[k_top, cfg.hidden_size],
            "moe_down_expert_outputs",
        )?);
    }
    Ok(())
}

/// The routed experts of one row accumulating into `out` (`ffn_out`, an EP
/// partial or the overlap partial). Hash layers select `tid2eid[token]`;
/// score layers select bias-aware top-k on device.
fn routed_experts_op<'a>(
    cfg: &DeepseekV4Config,
    weights: &'a DeepseekV4Weights,
    state: &'a DeepseekV4State,
    gpu: &Gpu,
    layer_idx: usize,
    token_id: u32,
    out: &'a GpuTensor,
) -> Result<RoutedExpertsOp<'a>, String> {
    let layer = weights.resolve_layer(layer_idx);
    let k = cfg.num_experts_per_tok;
    let route_scale = config_cache::route_scale(cfg.routed_scaling_factor, cfg.mq2r);
    if layer_idx >= cfg.num_hash_layers {
        let gate_bias = layer
            .gate_bias
            .as_ref()
            .ok_or_else(|| format!("ffn_routed l{layer_idx}: gate_bias missing"))?;
        return Ok(RoutedExpertsOp::BiasAware {
            params: MoeBiasAwareParams {
                hidden: cfg.hidden_size,
                mi: cfg.moe_intermediate_size,
                k_top: k,
                n_exp: cfg.n_routed_experts,
                route_scale,
                swiglu_limit: cfg.swiglu_limit,
                uses_atomic_moe_down: weights.mq2r_backend.uses_atomic_moe_down(),
                native_mq2_backend: weights.mq2r_backend.bias_aware_native_backend(),
                nonowned_gate_up_dummy: layer.expert_gate_up_dummy.as_ref(),
                batch_size: 1,
                x_rot: state.ffn_x_rot.as_ref().unwrap(),
                ffn_out: out,
                scores: state.router_scores.as_ref().unwrap(),
                gate_bias,
                expert_gate_up_ptrs: layer.expert_gate_up_ptrs.as_ref().unwrap(),
                expert_down_ptrs: layer.expert_w2_ptrs.as_ref().unwrap(),
                topk_indices: state.moe_topk_indices.as_ref().unwrap(),
                topk_weights: state.moe_topk_weights.as_ref().unwrap(),
                gate_batch: state.moe_gate_batch.as_ref().unwrap(),
                up_batch: state.moe_up_batch.as_ref().unwrap(),
                rot_batch: state.moe_rot_batch.as_ref().unwrap(),
                down_expanded: state.moe_down_expert_outputs.as_ref().unwrap(),
            },
            layer: layer_idx,
        });
    }
    // Host-side bounds check (tid2eid_dev mirrors tid2eid_host).
    let row = (token_id as usize) * k;
    if row + k > layer.tid2eid_host.len() {
        return Err(format!(
            "hash l{layer_idx}: token_id {token_id} out of tid2eid range \
             ({} entries)",
            layer.tid2eid_host.len()
        ));
    }
    // The `_buf` variant reads the token id from device so a captured graph
    // re-reads it on every replay; the kernarg variant and the host gather
    // are fallbacks for missing prerequisites.
    let selection = match (layer.tid2eid_dev.as_ref(), state.token_id_buf.as_ref()) {
        (Some(tid2eid), Some(token)) => HashSelection::Device {
            tid2eid,
            token: HashToken::Device(token),
        },
        (Some(tid2eid), None) => HashSelection::Device {
            tid2eid,
            token: HashToken::Host(token_id),
        },
        (None, _) => HashSelection::Host {
            row: &layer.tid2eid_host[row..row + k],
        },
    };
    Ok(RoutedExpertsOp::Hash(HashRoutedExpertsOp {
        selection,
        scores: state.router_scores.as_ref().unwrap(),
        x_rot: state.ffn_x_rot.as_ref().unwrap(),
        topk_indices: state.moe_topk_indices.as_ref().unwrap(),
        topk_weights: state.moe_topk_weights.as_ref().unwrap(),
        gate_batch: state.moe_gate_batch.as_ref().unwrap(),
        up_batch: state.moe_up_batch.as_ref().unwrap(),
        rot_batch: state.moe_rot_batch.as_ref().unwrap(),
        gate_up_ptrs: layer.expert_gate_up_ptrs.as_ref().unwrap(),
        down_ptrs: layer.expert_w2_ptrs.as_ref().unwrap(),
        gate_up_dummy: layer.expert_gate_up_dummy.as_ref(),
        native: weights.mq2r_backend.bias_aware_native_backend(),
        out,
        hidden: cfg.hidden_size,
        im: cfg.moe_intermediate_size,
        k,
        n_exp: cfg.n_routed_experts,
        route_scale,
        swiglu_limit: cfg.swiglu_limit,
        down_nox: weights.mq2r_backend.is_gfx1151()
            || config_cache::gfx1201_rmsnorm_rotate_nox_on(&gpu.arch, cfg.mq2r),
        layer: layer_idx,
    }))
}

fn heterogeneous_select_routes(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
    token_id: u32,
) -> Result<(), String> {
    let k = cfg.num_experts_per_tok;
    if state.moe_topk_indices.is_none() {
        state.moe_topk_indices = Some(
            gpu.alloc_tensor(&[k], DType::F32)
                .map_err(|error| format!("heterogeneous alloc route indices: {error:?}"))?,
        );
    }
    if state.moe_topk_weights.is_none() {
        state.moe_topk_weights = Some(
            gpu.alloc_tensor(&[k], DType::F32)
                .map_err(|error| format!("heterogeneous alloc route weights: {error:?}"))?,
        );
    }

    moe_route(cfg, weights, state, gpu, layer_idx)?;
    let layer = weights.resolve_layer(layer_idx);
    let scores = state.router_scores.as_ref().unwrap();
    let topk_indices = state.moe_topk_indices.as_ref().unwrap();
    let topk_weights = state.moe_topk_weights.as_ref().unwrap();
    let route_scale = config_cache::route_scale(cfg.routed_scaling_factor, cfg.mq2r);

    if layer_idx < cfg.num_hash_layers {
        let tid2eid = layer.tid2eid_dev.as_ref().ok_or_else(|| {
            format!("heterogeneous hash route l{layer_idx}: tid2eid device table missing")
        })?;
        let token_id_buf = state.token_id_buf.as_ref().ok_or_else(|| {
            format!("heterogeneous hash route l{layer_idx}: token-id buffer missing")
        })?;
        gpu.hash_router_normalize_f32_buf(
            tid2eid,
            scores,
            token_id_buf,
            topk_indices,
            topk_weights,
            cfg.n_routed_experts as i32,
            k as i32,
            route_scale,
        )
        .map_err(|error| format!("heterogeneous hash route l{layer_idx}: {error:?}"))?;
    } else {
        let gate_bias = layer.gate_bias.as_ref().ok_or_else(|| {
            format!("heterogeneous bias-aware route l{layer_idx}: gate bias missing")
        })?;
        gpu.deepseek4_moe_topk_bias_aware_f32(
            scores,
            gate_bias,
            topk_indices,
            topk_weights,
            cfg.n_routed_experts as i32,
            k as i32,
            route_scale,
        )
        .map_err(|error| format!("heterogeneous bias-aware route l{layer_idx}: {error:?}"))?;
    }
    dump_moe_route_if_enabled(gpu, layer_idx, topk_indices, topk_weights)
}

const HETEROGENEOUS_WAIT_EQ: u32 = 0x1;
const HETEROGENEOUS_SIGNAL_FLAGS: u32 = 0;
const HETEROGENEOUS_SIGNAL_MASK: u32 = u32::MAX;

fn heterogeneous_publish_routes(
    cfg: &DeepseekV4Config,
    state: &DeepseekV4State,
    dense_gpu: &mut Gpu,
    routed_gpu: &Gpu,
    execution: &DeepseekV4HeterogeneousExecution,
    epoch: u32,
) -> Result<(), String> {
    dense_gpu
        .bind_thread()
        .map_err(|error| format!("heterogeneous publish bind dense: {error}"))?;
    let stream = dense_gpu
        .active_stream
        .as_ref()
        .ok_or_else(|| "heterogeneous dense stream missing".to_string())?;
    let x_rot = state
        .ffn_x_rot
        .as_ref()
        .ok_or_else(|| "heterogeneous dense x_rot missing".to_string())?;
    let topk_indices = state
        .moe_topk_indices
        .as_ref()
        .ok_or_else(|| "heterogeneous dense route indices missing".to_string())?;
    let topk_weights = state
        .moe_topk_weights
        .as_ref()
        .ok_or_else(|| "heterogeneous dense route weights missing".to_string())?;
    let routed_x_rot = execution
        .routed_x_rot
        .as_ref()
        .ok_or_else(|| "heterogeneous routed x_rot missing".to_string())?;
    let routed_topk_indices = execution
        .routed_topk_indices
        .as_ref()
        .ok_or_else(|| "heterogeneous routed route indices missing".to_string())?;
    let routed_topk_weights = execution
        .routed_topk_weights
        .as_ref()
        .ok_or_else(|| "heterogeneous routed route weights missing".to_string())?;

    dense_gpu
        .hip
        .memcpy_peer_async(
            &routed_x_rot.buf,
            routed_gpu.device_id,
            &x_rot.buf,
            dense_gpu.device_id,
            cfg.hidden_size * std::mem::size_of::<f32>(),
            stream,
        )
        .map_err(|error| format!("heterogeneous publish x_rot: {error}"))?;
    dense_gpu
        .hip
        .memcpy_peer_async(
            &routed_topk_indices.buf,
            routed_gpu.device_id,
            &topk_indices.buf,
            dense_gpu.device_id,
            cfg.num_experts_per_tok * std::mem::size_of::<f32>(),
            stream,
        )
        .map_err(|error| format!("heterogeneous publish route indices: {error}"))?;
    dense_gpu
        .hip
        .memcpy_peer_async(
            &routed_topk_weights.buf,
            routed_gpu.device_id,
            &topk_weights.buf,
            dense_gpu.device_id,
            cfg.num_experts_per_tok * std::mem::size_of::<f32>(),
            stream,
        )
        .map_err(|error| format!("heterogeneous publish route weights: {error}"))?;
    dense_gpu
        .hip
        .stream_write_value32(
            stream,
            execution
                .signal_to_routed
                .as_ref()
                .ok_or_else(|| "heterogeneous routed signal missing".to_string())?,
            epoch,
            HETEROGENEOUS_SIGNAL_FLAGS,
        )
        .map_err(|error| format!("heterogeneous publish signal: {error}"))
}

#[allow(clippy::too_many_arguments)]
fn heterogeneous_run_selected(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4RoutedWeights,
    dense_partial: &GpuTensor,
    dense_device_id: i32,
    routed_gpu: &mut Gpu,
    execution: &DeepseekV4HeterogeneousExecution,
    layer_idx: usize,
    epoch: u32,
) -> Result<(), String> {
    routed_gpu
        .bind_thread()
        .map_err(|error| format!("heterogeneous selected bind routed: {error}"))?;
    let stream = routed_gpu
        .active_stream
        .as_ref()
        .ok_or_else(|| "heterogeneous routed stream missing".to_string())?;
    routed_gpu
        .hip
        .stream_wait_value32(
            stream,
            execution
                .signal_to_routed
                .as_ref()
                .ok_or_else(|| "heterogeneous routed signal missing".to_string())?,
            epoch,
            HETEROGENEOUS_WAIT_EQ,
            HETEROGENEOUS_SIGNAL_MASK,
        )
        .map_err(|error| format!("heterogeneous wait for routes l{layer_idx}: {error}"))?;

    let layer = weights.resolve_layer(layer_idx);
    let expert_gate_up_ptrs = layer.expert_gate_up_ptrs.as_ref().ok_or_else(|| {
        format!("heterogeneous routed gate/up pointer table missing l{layer_idx}")
    })?;
    let expert_down_ptrs = layer
        .expert_w2_ptrs
        .as_ref()
        .ok_or_else(|| format!("heterogeneous routed down pointer table missing l{layer_idx}"))?;
    let x_rot = execution.routed_x_rot.as_ref().unwrap();
    let topk_indices = execution.routed_topk_indices.as_ref().unwrap();
    let topk_weights = execution.routed_topk_weights.as_ref().unwrap();
    let gate_batch = execution.routed_gate_batch.as_ref().unwrap();
    let up_batch = execution.routed_up_batch.as_ref().unwrap();
    let rot_batch = execution.routed_rot_batch.as_ref().unwrap();
    let down_expanded = execution.routed_down_expanded.as_ref().unwrap();
    let routed_partial = execution.routed_partial.as_ref().unwrap();
    routed_gpu
        .hip
        .memset_async(&routed_partial.buf, 0, routed_partial.byte_size(), stream)
        .map_err(|error| format!("heterogeneous zero routed partial l{layer_idx}: {error}"))?;

    let params = hipfire_dispatch::families::moe::MoeSelectedParams {
        hidden: cfg.hidden_size,
        mi: cfg.moe_intermediate_size,
        k_top: cfg.num_experts_per_tok,
        swiglu_limit: cfg.swiglu_limit,
        uses_atomic_moe_down: weights.mq2r_backend.uses_atomic_moe_down(),
        native_mq2_backend: weights.mq2r_backend.bias_aware_native_backend(),
        nonowned_gate_up_dummy: layer.expert_gate_up_dummy.as_ref(),
        batch_size: 1,
        x_rot,
        ffn_out: routed_partial,
        expert_gate_up_ptrs,
        expert_down_ptrs,
        topk_indices,
        topk_weights,
        gate_batch,
        up_batch,
        rot_batch,
        down_expanded,
    };
    hipfire_runtime::llama::moe_family()
        .run_selected(routed_gpu, &params)
        .map_err(|error| format!("heterogeneous selected experts l{layer_idx}: {error}"))?;

    let stream = routed_gpu.active_stream.as_ref().unwrap();
    routed_gpu
        .hip
        .memcpy_peer_async(
            &dense_partial.buf,
            dense_device_id,
            &routed_partial.buf,
            routed_gpu.device_id,
            cfg.hidden_size * std::mem::size_of::<f32>(),
            stream,
        )
        .map_err(|error| format!("heterogeneous return partial l{layer_idx}: {error}"))?;
    routed_gpu
        .hip
        .stream_write_value32(
            stream,
            execution
                .signal_to_dense
                .as_ref()
                .ok_or_else(|| "heterogeneous dense signal missing".to_string())?,
            epoch,
            HETEROGENEOUS_SIGNAL_FLAGS,
        )
        .map_err(|error| format!("heterogeneous return signal l{layer_idx}: {error}"))
}

fn heterogeneous_join(
    cfg: &DeepseekV4Config,
    state: &DeepseekV4State,
    dense_gpu: &mut Gpu,
    execution: &DeepseekV4HeterogeneousExecution,
    layer_idx: usize,
    epoch: u32,
) -> Result<(), String> {
    dense_gpu
        .bind_thread()
        .map_err(|error| format!("heterogeneous join bind dense: {error}"))?;
    let stream = dense_gpu
        .active_stream
        .as_ref()
        .ok_or_else(|| "heterogeneous dense stream missing".to_string())?;
    dense_gpu
        .hip
        .stream_wait_value32(
            stream,
            execution
                .signal_to_dense
                .as_ref()
                .ok_or_else(|| "heterogeneous dense signal missing".to_string())?,
            epoch,
            HETEROGENEOUS_WAIT_EQ,
            HETEROGENEOUS_SIGNAL_MASK,
        )
        .map_err(|error| format!("heterogeneous wait for partial l{layer_idx}: {error}"))?;
    let ffn_out = state
        .ffn_out
        .as_ref()
        .ok_or_else(|| format!("heterogeneous shared output missing l{layer_idx}"))?;
    let routed_partial = state
        .ffn_routed_overlap
        .as_ref()
        .ok_or_else(|| format!("heterogeneous dense routed partial missing l{layer_idx}"))?;
    dense_gpu
        .add_inplace_f32(ffn_out, &routed_partial.sub_offset(0, cfg.hidden_size))
        .map_err(|error| format!("heterogeneous ordered join l{layer_idx}: {error:?}"))
}

pub(crate) fn ffn_stub(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    ffn_prepare(cfg, weights, state, gpu, layer_idx)?;
    ffn_shared_project(cfg, weights, state, gpu, layer_idx)
}

/// Routed-expert dispatch (DeepSeek V4 top-6 MoE). Accumulates `routed_scaling
/// _factor · Σ_k w_k · expert_{idx_k}(ffn_x_rot)` into `ffn_out`
/// (which already holds the shared-expert output from `ffn_stub`).
///
/// Gated on HIPFIRE_DEEPSEEK4_MOE != "0" (default ON) AND expert blobs present
/// (uploaded by default; opt out with HIPFIRE_DEEPSEEK4_UPLOAD_EXPERTS=0) AND
/// layer is score-routed
/// (layer_idx >= num_hash_layers). Hash-routed layers 0..3 fall back
/// to shared-only (tid2eid lookup table is skipped at quant time).
///
/// Math (per upstream `inference/model.py:Gate.forward` and `Expert.
/// forward`):
///   scores = sqrt(softplus(gate.weight @ x))             [n_exp]
///   indices = topk(scores + bias, k=6)[1]                [k]   ← +bias for selection
///   weights = scores[indices]                            [k]   ← unbiased scores for weights
///   weights /= weights.sum(); weights *= route_scale     [k]
///   for each (idx, w) in (indices, weights):
///     gate_e = w1[idx] @ x                  ← clamp to swiglu_limit (skipped)
///     up_e   = w3[idx] @ x                  ← clamp to ±swiglu_limit (skipped)
///     e_out  = w2[idx] @ (silu(gate_e) * up_e * w)
///     ffn_out += e_out * routed_scaling_factor
pub(crate) fn ffn_routed(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
    routed_out: Option<&GpuTensor>,
) -> Result<(), String> {
    if layer_idx < cfg.num_hash_layers {
        // Hash layers route through `ffn_hash_routed`.
        return Ok(());
    }
    routed_experts(cfg, weights, state, gpu, layer_idx, 0, routed_out)
}

/// Router plus routed experts of one row (legacy entry of the MTP layer and
/// the hand-scheduled heterogeneous route), accumulating into `routed_out` or
/// `ffn_out`, which the shared expert must have seeded.
fn routed_experts(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
    token_id: u32,
    routed_out: Option<&GpuTensor>,
) -> Result<(), String> {
    if !routed_experts_present(cfg, weights, layer_idx) {
        return Ok(());
    }
    ensure_moe_route(cfg, state, gpu)?;
    ensure_routed_experts(cfg, state, gpu, layer_idx < cfg.num_hash_layers)?;
    let out = routed_out.unwrap_or(state.ffn_out.as_ref().unwrap());
    let routed = routed_experts_op(cfg, weights, state, gpu, layer_idx, token_id, out)?;
    let op = expert_ffn_op(
        cfg,
        weights,
        state,
        gpu,
        layer_idx,
        true,
        Some(routed),
        ExpertFfnSchedule::Serial,
    )?;
    execute_moe_route(gpu, &op).map_err(dispatch_err)?;
    execute_routed_experts(gpu, &op).map_err(dispatch_err)?;
    dump_moe_route_if_enabled(
        gpu,
        layer_idx,
        state.moe_topk_indices.as_ref().unwrap(),
        state.moe_topk_weights.as_ref().unwrap(),
    )
}

/// Diagnostic-only route capture used to compare a quantized router against
/// the untouched model on a byte-identical teacher-forced stream.
///
/// Format:
/// `DS4RTR01 | repeated { layer:u32, k:u32, ids:[i32;k], weights:[f32;k] }`.
/// Records are naturally ordered token-major, layer-minor by the decode loop.
fn dump_moe_route_if_enabled(
    gpu: &Gpu,
    layer_idx: usize,
    topk_indices: &GpuTensor,
    topk_weights: &GpuTensor,
) -> Result<(), String> {
    use std::io::Write;
    use std::sync::{Mutex, OnceLock};

    static DUMP: OnceLock<Option<Mutex<std::io::BufWriter<std::fs::File>>>> = OnceLock::new();
    let dump = DUMP.get_or_init(|| {
        let path = hipfire_config::developer_var("HIPFIRE_DS4_ROUTE_DUMP").ok()?;
        let file = std::fs::File::create(&path)
            .unwrap_or_else(|e| panic!("create HIPFIRE_DS4_ROUTE_DUMP {path}: {e}"));
        let mut writer = std::io::BufWriter::new(file);
        writer
            .write_all(b"DS4RTR01")
            .expect("write DS4 route dump header");
        Some(Mutex::new(writer))
    });
    let Some(dump) = dump else {
        return Ok(());
    };

    let ids_bits = gpu
        .download_f32(topk_indices)
        .map_err(|e| format!("route dump indices l{layer_idx}: {e:?}"))?;
    let weights = gpu
        .download_f32(topk_weights)
        .map_err(|e| format!("route dump weights l{layer_idx}: {e:?}"))?;
    if ids_bits.len() != weights.len() {
        return Err(format!(
            "route dump l{layer_idx}: index/weight length mismatch {} != {}",
            ids_bits.len(),
            weights.len()
        ));
    }

    let mut writer = dump
        .lock()
        .map_err(|_| "route dump mutex poisoned".to_string())?;
    writer
        .write_all(&(layer_idx as u32).to_le_bytes())
        .map_err(|e| format!("route dump layer: {e}"))?;
    writer
        .write_all(&(ids_bits.len() as u32).to_le_bytes())
        .map_err(|e| format!("route dump k: {e}"))?;
    for bits in ids_bits {
        writer
            .write_all(&(bits.to_bits() as i32).to_le_bytes())
            .map_err(|e| format!("route dump id: {e}"))?;
    }
    for weight in weights {
        writer
            .write_all(&weight.to_le_bytes())
            .map_err(|e| format!("route dump weight: {e}"))?;
    }
    writer.flush().map_err(|e| format!("route dump flush: {e}"))
}

/// Maps a dispatch error to the legacy `String` form: the bare HIP message, without the `HIP error:` prefix.
pub(crate) fn dispatch_err(err: DispatchError) -> String {
    match err {
        DispatchError::Hip(msg) => msg,
        other => other.to_string(),
    }
}

/// Binds one HC mix step reading `streams_in`; [`hc_mix_target`] picks its output.
pub(crate) fn hc_mix_op<'a>(
    cfg: &DeepseekV4Config,
    state: &'a DeepseekV4State,
    pingpong: bool,
    streams_in: &'a GpuTensor,
    branch: HcMixBranch<'a>,
    kernel_what: &'static str,
    copy_what: &'static str,
) -> HcPostOp<'a> {
    let (streams_out, writeback) = hc_mix_target(state, pingpong, streams_in);
    HcPostOp {
        rows: 1,
        branch,
        streams_in,
        streams_out,
        writeback,
        hidden: cfg.hidden_size,
        copy_bytes: cfg.hc_mult * cfg.hidden_size * 4,
        route: HcPostRoute::Decode {
            hc_c: state.hc_c.as_ref().unwrap(),
            kernel_what,
            copy_what,
        },
    }
}

/// Output of an HC mix reading `streams_in`. Ping-pong writes the other stream
/// buffer (`residual_streams` ↔ `residual_streams_next`); otherwise the mix goes
/// to the `q` scratch and is copied back over `streams_in`.
pub(crate) fn hc_mix_target<'a>(
    state: &'a DeepseekV4State,
    pingpong: bool,
    streams_in: &'a GpuTensor,
) -> (&'a GpuTensor, Option<&'a GpuTensor>) {
    if !pingpong {
        return (state.q.as_ref().unwrap(), Some(streams_in));
    }
    let current = state.residual_streams.as_ref().unwrap();
    let next = state.residual_streams_next.as_ref().unwrap();
    if std::ptr::eq(streams_in.buf.as_ptr(), current.buf.as_ptr()) {
        (next, None)
    } else {
        (current, None)
    }
}

/// HC FFN mix — same pattern as `hc_attn_mix` but with `hc_ffn_*`
/// tensors and `ffn_out` as transform_out.
pub(crate) fn hc_ffn_mix(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    let _ = (weights, layer_idx);
    let pingpong = config_cache::hc_pingpong_on(&gpu.arch, cfg.mq2r);
    // Same reasoning as hc_attn_mix: mhc_pre(is_attn=false) has already populated
    // state.hc_c with the FFN block's post and comb.
    let op = hc_mix_op(
        cfg,
        state,
        pingpong,
        state.residual_streams.as_ref().unwrap(),
        HcMixBranch::Transform(state.ffn_out.as_ref().unwrap()),
        "hc_mix_4stream ffn",
        "d2d hc_ffn_mix → streams",
    );
    execute_hc_post(gpu, &op).map_err(dispatch_err)?;
    if pingpong {
        std::mem::swap(
            &mut state.residual_streams,
            &mut state.residual_streams_next,
        );
    }
    Ok(())
}

/// gfx1201 TP3 twin of [`hc_ffn_mix`] that reduces three rank-local FFN
/// partials inside the HC consumer instead of materializing an RCCL result.
fn hc_ffn_mix_peer_hc3(
    cfg: &DeepseekV4Config,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    partials: [&GpuTensor; 3],
) -> Result<(), String> {
    let pingpong = config_cache::hc_pingpong_on(&gpu.arch, cfg.mq2r);
    let op = hc_mix_op(
        cfg,
        state,
        pingpong,
        state.residual_streams.as_ref().unwrap(),
        HcMixBranch::Peer3(partials),
        "hc_mix_4stream_peer3_gfx1201 ffn",
        "d2d peer HC3 FFN mix to streams",
    );
    execute_hc_post(gpu, &op).map_err(dispatch_err)?;
    if pingpong {
        std::mem::swap(
            &mut state.residual_streams,
            &mut state.residual_streams_next,
        );
    }
    Ok(())
}

/// gfx1201 TP4 twin of [`hc_ffn_mix`] that reduces four rank-local FFN
/// partials inside the HC consumer instead of materializing an RCCL result.
fn hc_ffn_mix_peer_hc4(
    cfg: &DeepseekV4Config,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    partials: [&GpuTensor; 4],
) -> Result<(), String> {
    let pingpong = config_cache::hc_pingpong_on(&gpu.arch, cfg.mq2r);
    let op = hc_mix_op(
        cfg,
        state,
        pingpong,
        state.residual_streams.as_ref().unwrap(),
        HcMixBranch::Peer4(partials),
        "hc_mix_4stream_peer4_gfx1201 ffn",
        "d2d peer HC FFN mix to streams",
    );
    execute_hc_post(gpu, &op).map_err(dispatch_err)?;
    if pingpong {
        std::mem::swap(
            &mut state.residual_streams,
            &mut state.residual_streams_next,
        );
    }
    Ok(())
}

/// Allocates the head prologue buffers after checking the head weights.
pub(crate) fn ensure_final_norm_compute(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
) -> Result<(), String> {
    head_weights(weights)?;
    if state.final_norm.is_none() {
        state.final_norm = Some(
            gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
                .map_err(|e| format!("alloc final_norm: {e:?}"))?,
        );
    }
    if state.final_norm_rot.is_none() {
        state.final_norm_rot = Some(
            gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
                .map_err(|e| format!("alloc final_norm_rot: {e:?}"))?,
        );
    }
    if state.head_hc_pre.is_none() {
        state.head_hc_pre = Some(
            gpu.alloc_tensor(&[cfg.hc_mult], DType::F32)
                .map_err(|e| format!("alloc head_hc_pre: {e:?}"))?,
        );
    }
    if state.head_hc_out.is_none() {
        state.head_hc_out = Some(
            gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
                .map_err(|e| format!("alloc head_hc_out: {e:?}"))?,
        );
    }
    if cfg.load_dspark {
        let mtp_hidden_len = cfg.hc_mult * cfg.hidden_size;
        let mtp_needs_realloc = state
            .mtp_last_hidden
            .as_ref()
            .map(|t| t.numel() != mtp_hidden_len)
            .unwrap_or(true);
        if mtp_needs_realloc {
            state.mtp_last_hidden = Some(
                gpu.alloc_tensor(&[cfg.hc_mult, cfg.hidden_size], DType::F32)
                    .map_err(|e| format!("alloc mtp_last_hidden in final_norm_and_head: {e:?}"))?,
            );
        }
    }
    Ok(())
}

/// Binds the head prologue of `streams`: head HC mix, input map, optional MTP
/// capture, RMSNorm, optional FWHT.
pub(crate) fn final_norm_compute_op<'a>(
    cfg: &DeepseekV4Config,
    weights: &'a DeepseekV4Weights,
    state: &'a DeepseekV4State,
    streams: &'a GpuTensor,
    rows: usize,
    route: HcHeadRoute<'a>,
) -> Result<HcHeadOp<'a>, String> {
    let [output_norm, head, hc_head_fn, hc_head_base] = head_weights(weights)?;
    Ok(HcHeadOp {
        rows,
        streams,
        hc_head_fn,
        hc_head_base,
        head_hc_pre: state.head_hc_pre.as_ref().unwrap(),
        head_hc_out: state.head_hc_out.as_ref().unwrap(),
        output_norm,
        final_norm: state.final_norm.as_ref().unwrap(),
        final_norm_rot: weight_needs_fwht(head).then(|| state.final_norm_rot.as_ref().unwrap()),
        mtp_last_hidden: if cfg.load_dspark {
            state.mtp_last_hidden.as_ref()
        } else {
            None
        },
        head_scale: weights.hc_head_scale,
        hc_mult: cfg.hc_mult,
        hidden: cfg.hidden_size,
        rms_norm_eps: cfg.rms_norm_eps,
        hc_eps: cfg.hc_eps,
        route,
    })
}

/// Head weights for the prologue; each error names the missing tensor.
fn head_weights(weights: &DeepseekV4Weights) -> Result<[&GpuTensor; 4], String> {
    let output_norm = weights
        .output_norm
        .as_ref()
        .ok_or_else(|| "output_norm not uploaded".to_string())?;
    let head = weights
        .head
        .as_ref()
        .ok_or_else(|| "head not uploaded".to_string())?;
    let hc_head_fn = weights
        .hc_head_fn
        .as_ref()
        .ok_or_else(|| "hc_head_fn not uploaded".to_string())?;
    let hc_head_base = weights
        .hc_head_base
        .as_ref()
        .ok_or_else(|| "hc_head_base not uploaded".to_string())?;
    Ok([output_norm, head, hc_head_fn, hc_head_base])
}

/// Full per-position head of `streams` (`state.residual_streams` when
/// `None`): head prologue, then the lm_head GEMV into `state.logits`.
pub(crate) fn final_norm_and_head_impl(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    capture_head_activation: bool,
    streams: Option<&GpuTensor>,
) -> Result<(), String> {
    ensure_head(cfg, weights, state, gpu)?;
    {
        let ctx = DispatchCtx::new(gpu);
        let streams = streams.unwrap_or_else(|| state.residual_streams.as_ref().unwrap());
        let steps = head_steps(cfg, weights, state, gpu, streams)?;
        execute_steps(gpu, &ctx, &steps).map_err(dispatch_err)?;
    }
    if capture_head_activation {
        dump_dense_activation_if_enabled(
            gpu,
            "head.weight",
            state.final_norm.as_ref().unwrap(),
            cfg.hidden_size,
        )?;
    }
    Ok(())
}

/// Head prologue buffers and the logits.
pub(crate) fn ensure_head(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
) -> Result<(), String> {
    ensure_final_norm_compute(cfg, weights, state, gpu)?;
    if state.logits.is_none() {
        state.logits = Some(
            gpu.alloc_tensor(&[cfg.vocab_size], DType::F32)
                .map_err(|e| format!("alloc logits: {e:?}"))?,
        );
    }
    Ok(())
}

/// Head HC combine + final norm of `streams`, then the lm_head projection
/// into `state.logits` (rotated input for rotated heads, plain otherwise).
pub(crate) fn head_steps<'a>(
    cfg: &DeepseekV4Config,
    weights: &'a DeepseekV4Weights,
    state: &'a DeepseekV4State,
    gpu: &Gpu,
    streams: &'a GpuTensor,
) -> Result<[Step<'a>; 2], String> {
    let head = weights
        .head
        .as_ref()
        .ok_or_else(|| "head not uploaded".to_string())?;
    Ok([
        Step::HcHead(final_norm_compute_op(
            cfg, weights, state, streams, 1, HcHeadRoute::Decode,
        )?),
        Step::PreparedProjection(PreparedProjectionOp {
            rows: 1,
            weight: head,
            x_rotated: state.final_norm_rot.as_ref().unwrap(),
            x_plain: state.final_norm.as_ref().unwrap(),
            y: state.logits.as_ref().unwrap(),
            m: cfg.vocab_size,
            k: cfg.hidden_size,
            route: PreparedProjectionRoute::Decode {
                e8_soa: e8_soa_kernel(&gpu.arch, weights.mq2r_backend),
            },
        }),
    ])
}

pub(crate) fn final_norm_and_head(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
) -> Result<(), String> {
    final_norm_and_head_impl(cfg, weights, state, gpu, true, None)
}

/// Step 6: Single-position attention (position-0 degenerate case).
///
/// DeepSeek V4's attention with `o_groups = 8` means the 64 query heads
/// are reduced over groups of 8 heads → 8 grouped outputs each of
/// `head_dim = 512`, yielding `[8 * 512 = 4096]` = hidden directly.
/// No separate O-projection needed (wo_a/wo_b's role TBD per paper).
///
/// For position-0 (no past KV history), each query head attends
/// only to the current token's K/V. softmax over 1 position = 1.0,
/// so attn_per_head = V. With o_groups grouping: each of 8 groups
/// sums 8 identical V vectors → attn_per_group = 8 * V.
///
/// Output `[hidden = o_groups * head_dim]`: 8 copies of V (each
/// scaled by 8 due to the in-group sum), giving [8*V, 8*V, ..., 8*V].
///
/// This handles position 0. For position > 0 we need SWA cache +
/// real Q·K·V over history — pending.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OloraSchedule {
    Default,
    HeterogeneousGfx1100,
}

pub(crate) fn attn_stub(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
    olora_schedule: OloraSchedule,
) -> Result<(), String> {
    ensure_attention(cfg, weights, state, gpu, layer_idx)?;
    let steps = attention_ops(cfg, weights, state, gpu, layer_idx, olora_schedule)?;
    let ctx = DispatchCtx::new(gpu);
    execute_steps(gpu, &ctx, &steps).map_err(dispatch_err)?;
    attention_dumps(cfg, state, gpu, layer_idx)
}

pub(crate) fn ensure_attention(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    let (heads, head_dim) = (cfg.num_attention_heads, cfg.head_dim);
    let groups_o_lora = cfg.o_groups * cfg.o_lora_rank;
    let alloc = |gpu: &mut Gpu, slot: &mut Option<GpuTensor>, shape: &[usize], name: &str| {
        if slot.is_none() {
            *slot = Some(
                gpu.alloc_tensor(shape, DType::F32)
                    .map_err(|e| format!("alloc {name}: {e:?}"))?,
            );
        }
        Ok::<(), String>(())
    };
    // Final attention contribution [hidden], consumed by the HC mix.
    alloc(gpu, &mut state.attn_out, &[cfg.hidden_size], "attn_out")?;
    alloc(
        gpu,
        &mut state.attn_out_raw,
        &[heads, head_dim],
        "attn_out_raw",
    )?;
    alloc(
        gpu,
        &mut state.attn_out_raw_rot,
        &[heads * head_dim],
        "attn_out_raw_rot",
    )?;
    alloc(gpu, &mut state.wo_a_out, &[groups_o_lora], "wo_a_out")?;
    alloc(
        gpu,
        &mut state.wo_a_out_rot,
        &[groups_o_lora],
        "wo_a_out_rot",
    )?;
    if config_cache::attn_pos0() {
        return Ok(());
    }
    let (n_kv, win) = (cfg.num_key_value_heads, cfg.sliding_window);
    let zeros = |gpu: &mut Gpu, slot: &mut Option<GpuTensor>, shape: &[usize], name: &str| {
        if slot.is_none() {
            *slot = Some(
                gpu.zeros(shape, DType::F32)
                    .map_err(|e| format!("alloc {name} l{layer_idx}: {e:?}"))?,
            );
        }
        Ok::<(), String>(())
    };
    let attn = &mut state._attention[layer_idx];
    zeros(gpu, &mut attn.swa_k, &[n_kv, head_dim, win], "swa_k")?;
    zeros(gpu, &mut attn.swa_v, &[n_kv, head_dim, win], "swa_v")?;
    let mixed = weights.resolve_layer(layer_idx).compress_ratio > 0
        && state._indexer[layer_idx].main_kv_cache.is_some();
    if mixed {
        zeros(
            gpu,
            &mut state._attention[layer_idx].gathered_k,
            &[n_kv, head_dim, cfg.index_topk],
            "gathered_k",
        )?;
    }
    Ok(())
}

/// The attention of one decode row after its Q/KV (and, ratio 4, indexer)
/// steps, in launch order: SWA ring write, compressed-row gather and the
/// joint softmax (or the window-only softmax), inverse tail RoPE, grouped
/// O-LoRA. The same operations run batched rows on their batched routes.
pub(crate) fn attention_ops<'a>(
    cfg: &DeepseekV4Config,
    weights: &'a DeepseekV4Weights,
    state: &'a DeepseekV4State,
    gpu: &Gpu,
    layer_idx: usize,
    olora_schedule: OloraSchedule,
) -> Result<Vec<Step<'a>>, String> {
    let layer = weights.resolve_layer(layer_idx);
    let (heads, head_dim, groups) = (cfg.num_attention_heads, cfg.head_dim, cfg.o_groups);
    let (o_lora_rank, win, topk) = (cfg.o_lora_rank, cfg.sliding_window, cfg.index_topk);
    let sink = layer
        .attn_sink
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} attn_sink not uploaded"))?;
    let (q, kv, raw) = (
        state.q.as_ref().unwrap(),
        state.kv.as_ref().unwrap(),
        state.attn_out_raw.as_ref().unwrap(),
    );
    let gfx1151_route = weights.mq2r_backend.is_gfx1151();
    let mut steps = Vec::with_capacity(5);
    if config_cache::attn_pos0() {
        steps.push(Step::SwaAttention(SwaAttentionOp {
            q,
            sink,
            out: raw,
            n_heads: heads,
            head_dim,
            n_groups: groups,
            win,
            rows: 1,
            layer: layer_idx,
            route: SwaAttentionRoute::Pos0 { kv },
        }));
    } else {
        let attn_state = state
            .attn_state_buf
            .as_ref()
            .ok_or("attn_state_buf missing (precompute_positions must run first)")?;
        let attn = &state._attention[layer_idx];
        let (swa_k, swa_v) = (attn.swa_k.as_ref().unwrap(), attn.swa_v.as_ref().unwrap());
        steps.push(Step::SwaRingWrite(SwaRingWriteOp {
            kv,
            swa_k,
            swa_v,
            n_kv: cfg.num_key_value_heads,
            head_dim,
            win,
            rows: 1,
            layer: layer_idx,
            route: SwaRingWriteRoute::Decode { attn_state },
        }));
        let l_state = &state._indexer[layer_idx];
        // Antirez-faithful mixed attention (ds4.c:7559-7566): dense layers
        // attend the SWA window; compressed layers run ONE softmax over the
        // window, the compressor cache rows (ratio 4: the indexer's top-k,
        // ratio 128: all) and the sink.
        match l_state.main_kv_cache.as_ref() {
            Some(main_kv_cache) if layer.compress_ratio > 0 => {
                let topk_idx = l_state
                    .topk_idx_indices
                    .as_ref()
                    .filter(|_| layer.compress_ratio == 4);
                let cache = compressed_cache(
                    state,
                    main_kv_cache,
                    &l_state.main_kv_cache_shards,
                    if topk_idx.is_some() {
                        "gather"
                    } else {
                        "identity gather"
                    },
                    layer_idx,
                )?;
                let (n_compressed_slot, k_active_slot) = if layer.compress_ratio == 4 {
                    (2, 4)
                } else {
                    (3, 5)
                };
                let gathered = attn.gathered_k.as_ref().unwrap();
                if topk_idx.is_none() {
                    steps.push(Step::IdentityGather(IdentityGatherOp {
                        cache,
                        staged: gathered,
                        head_dim,
                        top_k_max: topk,
                        rows: 1,
                        layer: layer_idx,
                        route: IdentityGatherRoute::Decode {
                            attn_state,
                            k_active_slot,
                        },
                    }));
                }
                steps.push(Step::SparseAttention(SparseAttentionOp {
                    q,
                    sink,
                    out: raw,
                    n_heads: heads,
                    head_dim,
                    win,
                    topk,
                    rows: 1,
                    layer: layer_idx,
                    route: SparseAttentionRoute::Decode {
                        swa_k,
                        swa_v,
                        gathered,
                        attn_state,
                        k_active_slot,
                        gather: topk_idx.map(|indices| TopkGather {
                            cache,
                            indices,
                            n_compressed_slot,
                        }),
                        gfx1151_route,
                    },
                }));
            }
            _ => steps.push(Step::SwaAttention(SwaAttentionOp {
                q,
                sink,
                out: raw,
                n_heads: heads,
                head_dim,
                n_groups: groups,
                win,
                rows: 1,
                layer: layer_idx,
                route: SwaAttentionRoute::Decode {
                    swa_k,
                    swa_v,
                    attn_state,
                },
            })),
        }
    }
    // Inverse tail RoPE with the forward parameters, so the rotation cancels
    // across attention (antirez rope_tail_layer_inplace(.., inverse)).
    let (freq_base, freq_scale, ext_factor, attn_factor, corr_low, corr_high) =
        layer_rope_params(cfg, layer.compress_ratio);
    steps.push(Step::ApplyTailRope(ApplyTailRopeOp {
        q: raw,
        kv: raw,
        pos: state
            .pos_buf
            .as_ref()
            .ok_or_else(|| "pos_buf not allocated".to_string())?,
        heads,
        kv_heads: 0,
        head_dim,
        rope: CompressorRope {
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            corr_low,
            corr_high,
            rope_dim: cfg.qk_rope_head_dim,
        },
        inverse: true,
        rows: 1,
        layer: layer_idx,
        route: ApplyTailRopeRoute::Decode { gfx1151_route },
    }));
    let wo_a = layer
        .wo_a
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} wo_a missing"))?;
    let wo_b = layer
        .wo_b
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} wo_b missing"))?;
    if !heads.is_multiple_of(groups) {
        return Err(format!(
            "layer {layer_idx}: {heads} heads do not split into {groups} O-LoRA groups"
        ));
    }
    let per_group_in = (heads / groups) * head_dim;
    let olora_a = if wo_a.dtype == DType::MQ4G256 {
        // MQ2RXT keeps the same one-launch grouped O-LoRA topology as MQ2R.
        OloraA::Mq4Grouped
    } else if wo_a.dtype == DType::MFP4G32E8SOA
        && config_cache::e8_wo_grouped_on(&gpu.arch, cfg.mq2r)
    {
        OloraA::E8Gfx1151
    } else if wo_a.dtype == DType::MFP4G32E8SOA
        && config_cache::gfx1201_e8_wo_grouped_on(&gpu.arch, cfg.mq2r)
        && per_group_in % 256 == 0
    {
        OloraA::E8Gfx1201
    } else if wo_a.dtype == DType::MFP4G32E8SOA
        && olora_schedule == OloraSchedule::HeterogeneousGfx1100
    {
        OloraA::E8Gfx1100
    } else if wo_a.dtype == DType::MFP4G32E8SOA
        && config_cache::gfx942_e8_wo_grouped_on(&gpu.arch, weights.mq2r_backend.is_gfx942())
        && per_group_in % 256 == 0
    {
        // The weight-owned DS4 backend reacquires an exact Gfx942Device proof
        // before launching; no broad arch match or env flag grants it.
        OloraA::E8Backend
    } else {
        // One group's wo_a slice: F32 by element, the quantized formats by
        // bytes (MQ4G256 136 B / 256, Q8_0 34 B / 32, E8 per-row layout).
        let per_group_elems = o_lora_rank * per_group_in;
        let group_bytes = match wo_a.dtype {
            DType::F32 => per_group_elems,
            DType::Q8_0 => (per_group_elems / 32) * 34,
            DType::MFP4G32E8 | DType::MFP4G32E8SOA | DType::MFP3G32E8 => {
                o_lora_rank * mfp_e8_row_bytes(wo_a.dtype, per_group_in)
            }
            _ => (per_group_elems / 256) * 136,
        };
        OloraA::PerGroup { group_bytes }
    };
    steps.push(Step::OLora(OLoraOp {
        wo_a,
        wo_b,
        raw,
        raw_rot: state.attn_out_raw_rot.as_ref().unwrap(),
        wo_a_out: state.wo_a_out.as_ref().unwrap(),
        wo_a_out_rot: state.wo_a_out_rot.as_ref().unwrap(),
        out: state.attn_out.as_ref().unwrap(),
        n_groups: groups,
        o_lora_rank,
        per_group_in,
        hidden: cfg.hidden_size,
        rotate_raw: weight_needs_fwht(wo_a),
        rotate_mid: weight_needs_fwht(wo_b),
        rows: 1,
        layer: layer_idx,
        route: OLoraRoute::Decode {
            olora_a,
            backend: &weights.mq2r_backend,
            e8_soa: e8_soa_kernel(&gpu.arch, weights.mq2r_backend),
        },
    }));
    Ok(steps)
}

/// P3 calibration captures of the O-LoRA inputs: the inverse-rotated head
/// outputs (wo_a) and the grouped A output (wo_b). Neither is written after
/// its consumer projection.
pub(crate) fn attention_dumps(
    cfg: &DeepseekV4Config,
    state: &DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    let per_group_in = (cfg.num_attention_heads / cfg.o_groups) * cfg.head_dim;
    dump_dense_activation_if_enabled(
        gpu,
        &format!("layers.{layer_idx}.attn.wo_a.weight"),
        state.attn_out_raw.as_ref().unwrap(),
        per_group_in,
    )?;
    dump_dense_activation_if_enabled(
        gpu,
        &format!("layers.{layer_idx}.attn.wo_b.weight"),
        state.wo_a_out.as_ref().unwrap(),
        cfg.o_groups * cfg.o_lora_rank,
    )
}

/// Router scores of one row: unbiased `sqrt(softplus(gate · x))` into
/// `state.router_scores` (the heterogeneous route selects on the dense
/// owner from these; the decode program runs the router inside its FFN step).
fn moe_route(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    ensure_moe_route(cfg, state, gpu)?;
    let op = expert_ffn_op(
        cfg,
        weights,
        state,
        gpu,
        layer_idx,
        true,
        None,
        ExpertFfnSchedule::Serial,
    )?;
    execute_moe_route(gpu, &op).map_err(dispatch_err)
}

fn ensure_moe_route(
    cfg: &DeepseekV4Config,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
) -> Result<(), String> {
    if state.router_scores.is_none() {
        state.router_scores = Some(
            gpu.alloc_tensor(&[cfg.n_routed_experts], DType::F32)
                .map_err(|e| format!("alloc router_scores: {e:?}"))?,
        );
    }
    Ok(())
}

/// mHC pre-step: compute c = X · W_fn + base [24], split into
/// Ã/B̃/C̃, apply sigmoid/exp+Sinkhorn/2σ, then compute
/// state.hc_x_in = A_l · streams (the input mapping).
///
/// After this runs, the layer's transform (attn or FFN) reads
/// hc_x_in as its [hidden]-shaped input.
pub(crate) fn mhc_pre(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
    is_attn: bool,
) -> Result<(), String> {
    ensure_mhc_pre(cfg, weights, state, gpu)?;
    let op = mhc_pre_op(
        cfg,
        weights,
        state,
        gpu,
        layer_idx,
        is_attn,
        state.residual_streams.as_ref().unwrap(),
    );
    execute_hc_pre(gpu, &op).map_err(dispatch_err)
}

/// Allocates `hc_x_in` and sizes `hc_c` for the active fused route.
pub(crate) fn ensure_mhc_pre(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
) -> Result<(), String> {
    if state.hc_x_in.is_none() {
        state.hc_x_in = Some(
            gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
                .map_err(|e| format!("alloc hc_x_in: {e:?}"))?,
        );
    }
    let flags = MhcPreFlags::new(cfg, weights, gpu);
    let hc_c_len = flags.hc_c_len();
    let hc_c_needs_realloc = state
        .hc_c
        .as_ref()
        .map(|tensor| tensor.numel() != hc_c_len)
        .unwrap_or(true);
    if hc_c_needs_realloc {
        let hc_c = gpu
            .alloc_tensor(&[hc_c_len], DType::F32)
            .map_err(|e| format!("alloc hc_c: {e:?}"))?;
        if flags.fused_control_finalize {
            gpu.hip
                .memset(&hc_c.buf, 0, hc_c.byte_size())
                .map_err(|e| format!("zero hc_c fusion ticket: {e:?}"))?;
        }
        if flags.fused_control_finalize && flags.control_rsqrt_once {
            // Rows 1.. wait (bounded) on row 0's rsqrt. Allocate the wait-error
            // block with the control state, never inside a graph capture.
            gpu.ensure_device_wait_block()
                .map_err(|e| format!("alloc hc_c rsqrt_once wait-error block: {e:?}"))?;
        }
        state.hc_c = Some(hc_c);
    }
    Ok(())
}

/// Fused-kernel routes that size `hc_c` and bind the mHC pre step.
struct MhcPreFlags {
    fused_control_finalize: bool,
    fused_finalize: bool,
    fused_input_map: bool,
    control_rsqrt_once: bool,
}

impl MhcPreFlags {
    fn new(cfg: &DeepseekV4Config, weights: &DeepseekV4Weights, gpu: &Gpu) -> Self {
        Self {
            fused_control_finalize: config_cache::hc_control_finalize_fused_on(&gpu.arch, cfg.mq2r),
            fused_finalize: config_cache::hc_finalize_fused_on(&gpu.arch, cfg.mq2r)
                || config_cache::gfx942_hc_finalize_fused_on(
                    &gpu.arch,
                    weights.mq2r_backend.is_gfx942(),
                ),
            fused_input_map: config_cache::hc_finalize_input_map_on(&gpu.arch, cfg.mq2r),
            control_rsqrt_once: config_cache::hc_control_rsqrt_once_on(&gpu.arch, cfg.mq2r),
        }
    }

    /// 24 controls, plus the extra slots the fused control finalize uses.
    fn hc_c_len(&self) -> usize {
        match (self.fused_control_finalize, self.control_rsqrt_once) {
            (true, true) => 27,
            (true, false) => 25,
            (false, _) => 24,
        }
    }
}

/// Binds the mHC pre step for one layer; `is_attn` picks the attention or FFN control weights.
pub(crate) fn mhc_pre_op<'a>(
    cfg: &DeepseekV4Config,
    weights: &'a DeepseekV4Weights,
    state: &'a DeepseekV4State,
    gpu: &Gpu,
    layer_idx: usize,
    is_attn: bool,
    streams: &'a GpuTensor,
) -> HcPreOp<'a> {
    let layer = weights.resolve_layer(layer_idx);
    let (hc_fn, hc_base, hc_scale) = if is_attn {
        (
            layer.hc_attn_fn.as_ref().unwrap(),
            layer.hc_attn_base.as_ref().unwrap(),
            layer.hc_attn_scale.as_ref().unwrap(),
        )
    } else {
        (
            layer.hc_ffn_fn.as_ref().unwrap(),
            layer.hc_ffn_base.as_ref().unwrap(),
            layer.hc_ffn_scale.as_ref().unwrap(),
        )
    };
    let flags = MhcPreFlags::new(cfg, weights, gpu);
    HcPreOp {
        rows: 1,
        streams,
        hc_fn,
        hc_base,
        hc_scale,
        hc_c: state.hc_c.as_ref().unwrap(),
        hc_x_in: state.hc_x_in.as_ref().unwrap(),
        layer: layer_idx,
        hidden: cfg.hidden_size,
        x_dim: cfg.hidden_size * cfg.hc_mult,
        hc_eps: cfg.hc_eps,
        post_scale: hc_post_scale(),
        sinkhorn_iters: cfg.hc_sinkhorn_iters as i32,
        route: HcPreRoute::Decode {
            gfx1151_route: weights.mq2r_backend.is_gfx1151(),
            fused_control_finalize: flags.fused_control_finalize,
            fused_finalize: flags.fused_finalize,
            fused_input_map: flags.fused_input_map,
            control_rsqrt_once: flags.control_rsqrt_once,
        },
    }
}

/// Pre/post sigmoid scale. Default 1.5 is the empirical optimum under mixed attention + YaRN
/// (antirez hardcodes 2.0). Env override: `HIPFIRE_DEEPSEEK4_POST_SCALE`.
fn hc_post_scale() -> f32 {
    use std::sync::OnceLock;
    static POST_SCALE: OnceLock<f32> = OnceLock::new();
    *POST_SCALE.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_POST_SCALE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.5)
    })
}

/// Step 8 (attention block): full manifold-constrained Hyper-Connection mix.
///
/// Per DeepSeek_V4.pdf §2.2:
///   c     = α · (X · W_fn) + base                  [24]
///   Ã,B̃,C̃ = c[0..4], c[4..20], c[20..24]
///   A_l   = σ(Ã_l)                                  [4]    (input mapping)
///   B_l   = Sinkhorn(exp(B̃_l))                     [4,4]  (residual matrix)
///   C_l   = 2σ(C̃_l)                                 [4]    (output mapping)
///   x_in  = A_l · X_l                               [hidden]   (NOT YET — uses stream0)
///   y     = F_l(x_in)
///   X_l+1 = B_l · X_l + C_l · y
///
/// Currently `α · X · W_fn` is computed without the α scaling, and
/// the input mapping `A·X` is stubbed (transform input = stream 0 not
/// the weighted-sum across streams). These approximations make HC
/// numerically non-canonical but kept bounded by the doubly-stochastic
/// B and bounded-magnitude C.
pub(crate) fn hc_attn_mix(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    let _ = weights; // post/comb already in state.hc_c from mhc_pre
    let _ = layer_idx;
    let pingpong = config_cache::hc_pingpong_on(&gpu.arch, cfg.mq2r);
    // X_{l+1} = comb · X_l + post · attn_out
    let op = hc_mix_op(
        cfg,
        state,
        pingpong,
        state.residual_streams.as_ref().unwrap(),
        HcMixBranch::Transform(state.attn_out.as_ref().unwrap()),
        "hc_mix_4stream layer",
        "d2d hc_mix → streams",
    );
    execute_hc_post(gpu, &op).map_err(dispatch_err)?;
    if pingpong {
        std::mem::swap(
            &mut state.residual_streams,
            &mut state.residual_streams_next,
        );
    }
    Ok(())
}

/// gfx1201 TP3 twin of [`hc_attn_mix`] that consumes three peer-visible
/// O-projection partials directly inside the HC residual update.
fn hc_attn_mix_peer_hc3(
    cfg: &DeepseekV4Config,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    partials: [&GpuTensor; 3],
) -> Result<(), String> {
    let pingpong = config_cache::hc_pingpong_on(&gpu.arch, cfg.mq2r);
    let op = hc_mix_op(
        cfg,
        state,
        pingpong,
        state.residual_streams.as_ref().unwrap(),
        HcMixBranch::Peer3(partials),
        "hc_mix_4stream_peer3_gfx1201 attention",
        "d2d peer HC3 attention mix to streams",
    );
    execute_hc_post(gpu, &op).map_err(dispatch_err)?;
    if pingpong {
        std::mem::swap(
            &mut state.residual_streams,
            &mut state.residual_streams_next,
        );
    }
    Ok(())
}

/// gfx1201 TP4 twin of [`hc_attn_mix`] that consumes the four peer-visible
/// O-projection partials directly inside the HC residual update.
fn hc_attn_mix_peer_hc4(
    cfg: &DeepseekV4Config,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    partials: [&GpuTensor; 4],
) -> Result<(), String> {
    let pingpong = config_cache::hc_pingpong_on(&gpu.arch, cfg.mq2r);
    let op = hc_mix_op(
        cfg,
        state,
        pingpong,
        state.residual_streams.as_ref().unwrap(),
        HcMixBranch::Peer4(partials),
        "hc_mix_4stream_peer4_gfx1201 attention",
        "d2d peer HC attention mix to streams",
    );
    execute_hc_post(gpu, &op).map_err(dispatch_err)?;
    if pingpong {
        std::mem::swap(
            &mut state.residual_streams,
            &mut state.residual_streams_next,
        );
    }
    Ok(())
}

/// Step 5 (attention block): Tail-only RoPE on Q and KV.
///
/// DeepSeek V4's `qk_rope_head_dim = 64` of `head_dim = 512`. Only the last
/// 64 dims of each head's 512-dim vector get rotated; the first 448
/// are pass-through. Same rotation applies to KV's 512-dim vector
/// (treated as 1 head).
///
/// Uses `rope_tail_halfsplit_f32` with DeepSeek V4's `rope_theta = 10000`.
/// YaRN correction dim: per-dim-pair index at which the high-vs-low
/// frequency split happens. Matches antirez ds4's `rope_yarn_corr_dim`.
fn rope_yarn_corr_dim(n_dims: u32, n_ctx_orig: u64, n_rot: f32, base: f32) -> f32 {
    n_dims as f32 * ((n_ctx_orig as f32 / (n_rot * 2.0 * std::f32::consts::PI)).ln())
        / (2.0 * base.ln())
}

/// Per-layer RoPE parameters: returns (freq_base, freq_scale, ext_factor,
/// attn_factor, corr_low, corr_high). Mirrors antirez's
/// `layer_rope_freq_base` / `layer_rope_freq_scale` + the attn_factor
/// cancellation in `rope_tail_layer_inplace`.
pub(crate) fn layer_rope_params(
    cfg: &DeepseekV4Config,
    compress_ratio: u32,
) -> (f32, f32, f32, f32, f32, f32) {
    let compressed = compress_ratio != 0;
    let freq_base = if compressed {
        cfg.compress_rope_theta
    } else {
        cfg.rope_theta
    };
    let scale_factor = cfg.rope_scaling_factor;
    let freq_scale = if compressed && scale_factor > 1.0 {
        1.0 / scale_factor
    } else {
        1.0
    };
    let ext_factor = if compressed && scale_factor > 1.0 {
        1.0
    } else {
        0.0
    };
    // attn_factor: antirez pre-divides by (1+0.1*log(1/fs)) here so the
    // kernel's inner `mscale *= (1+0.1*log(1/fs))` cancels it back to 1.0
    // (see ds4.c:4769-4778). For dense (ext_factor=0) the kernel skips the
    // log multiplication, so attn_factor stays 1.0.
    let attn_factor = if ext_factor != 0.0 && freq_scale > 0.0 {
        1.0 / (1.0 + 0.1 * (1.0_f32 / freq_scale).ln())
    } else {
        1.0
    };
    let n_rot = cfg.qk_rope_head_dim as u32;
    let n_ctx_orig = cfg.rope_scaling_original_max_position_embeddings as u64;
    let beta_fast = cfg.rope_scaling_beta_fast as f32;
    let beta_slow = cfg.rope_scaling_beta_slow as f32;
    let (corr_low, corr_high) = if ext_factor != 0.0 {
        let lo = rope_yarn_corr_dim(n_rot, n_ctx_orig, beta_fast, freq_base)
            .floor()
            .max(0.0);
        let hi = rope_yarn_corr_dim(n_rot, n_ctx_orig, beta_slow, freq_base)
            .ceil()
            .min((n_rot - 1) as f32);
        (lo, hi)
    } else {
        (0.0, 0.0)
    };
    (
        freq_base,
        freq_scale,
        ext_factor,
        attn_factor,
        corr_low,
        corr_high,
    )
}

pub(crate) fn apply_tail_rope(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    position: u32,
    layer_idx: usize,
) -> Result<(), String> {
    let _ = position; // silence unused; precompute_positions already used it
    ensure_apply_tail_rope(state, layer_idx)?;
    let op = apply_tail_rope_op(cfg, weights, state, layer_idx);
    execute_apply_tail_rope(gpu, &op).map_err(dispatch_err)
}

pub(crate) fn ensure_apply_tail_rope(
    state: &mut DeepseekV4State,
    layer_idx: usize,
) -> Result<(), String> {
    // Position is pre-loaded into `state.pos_array_device` at decode_step
    // entry (single htod for all layers). Slice the qk_pos slot for this
    // layer. Also seed the legacy `state.pos_buf` field so other code
    // paths that still read it (inverse RoPE on attn_out, indexer) work
    // unchanged — they get the SAME slice. The per-layer memcpy_htod is
    // gone, lifting it out of any future HIP-graph captured region.
    let pos_slice = pos_slot(state, layer_idx, 0)?;
    state.pos_buf = Some(pos_slice);
    Ok(())
}

/// Binds [`ApplyTailRopeOp`] from `state`; [`ensure_apply_tail_rope`] must run first.
pub(crate) fn apply_tail_rope_op<'a>(
    cfg: &DeepseekV4Config,
    weights: &'a DeepseekV4Weights,
    state: &'a DeepseekV4State,
    layer_idx: usize,
) -> ApplyTailRopeOp<'a> {
    // DeepSeek V4 upstream (per antirez ds4 reference):
    //   compress_ratio == 0 (layers 0, 1, MTP): rope_theta = 10000, no YaRN
    //   compress_ratio  > 0 (layers 2..42):      compress_rope_theta = 160000,
    //                                            YaRN with scale_factor = 16
    let layer = weights.resolve_layer(layer_idx);
    let (freq_base, freq_scale, ext_factor, attn_factor, corr_low, corr_high) =
        layer_rope_params(cfg, layer.compress_ratio);
    ApplyTailRopeOp {
        q: state.q.as_ref().unwrap(),
        kv: state.kv.as_ref().unwrap(),
        pos: state.pos_buf.as_ref().unwrap(),
        heads: cfg.num_attention_heads,
        kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        rope: CompressorRope {
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            corr_low,
            corr_high,
            rope_dim: cfg.qk_rope_head_dim,
        },
        inverse: false,
        rows: 1,
        layer: layer_idx,
        route: ApplyTailRopeRoute::Decode {
            gfx1151_route: weights.mq2r_backend.is_gfx1151(),
        },
    }
}

/// Step 4 (attention block): Joint KV projection.
///
/// DeepSeek V4 has `n_kv_heads = 1`, `head_dim = 512`, so the entire KV
/// stream per token is one 512-dim vector. `wkv` shape on disk is
/// `[512, 4096]` — a standard small GEMV producing 512 outputs from
/// 4096 hidden inputs.
///
/// Tail-only RoPE applies to the last `qk_rope_head_dim = 64` dims.
/// The leading 448 dims are pass-through.
///
/// Caller assumes `state.tmp` is still the FWHT-rotated post-RMSNorm
/// input from `q_lora` (gemv_mq4g256_prerotated doesn't modify x).
pub(crate) fn kv_joint(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
    preprojected: bool,
) -> Result<(), String> {
    ensure_kv_joint(cfg, state, gpu)?;
    let op = kv_joint_op(cfg, weights, state, gpu, layer_idx, preprojected)?;
    execute_kv_joint(gpu, &op).map_err(dispatch_err)
}

pub(crate) fn ensure_kv_joint(
    cfg: &DeepseekV4Config,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
) -> Result<(), String> {
    let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
    if state.kv.is_none() {
        state.kv = Some(
            gpu.alloc_tensor(&[kv_dim], DType::F32)
                .map_err(|e| format!("alloc kv: {e:?}"))?,
        );
    }
    Ok(())
}

/// Binds [`KvJointOp`] from `state`; [`ensure_kv_joint`] must run first.
pub(crate) fn kv_joint_op<'a>(
    cfg: &DeepseekV4Config,
    weights: &'a DeepseekV4Weights,
    state: &'a DeepseekV4State,
    gpu: &Gpu,
    layer_idx: usize,
    preprojected: bool,
) -> Result<KvJointOp<'a>, String> {
    let layer = weights.resolve_layer(layer_idx);
    let wkv = layer
        .wkv
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} wkv missing"))?;
    let kv_norm = layer
        .kv_norm
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} kv_norm missing"))?;
    Ok(KvJointOp {
        wkv,
        kv_norm,
        tmp: state.tmp.as_ref().unwrap(),
        tmp_plain: state
            .tmp_plain
            .as_ref()
            .ok_or_else(|| "kv_joint: tmp_plain missing (q_lora must run first)".to_string())?,
        kv: state.kv.as_ref().unwrap(),
        kv_dim: cfg.num_key_value_heads * cfg.head_dim,
        hidden: cfg.hidden_size,
        eps: cfg.rms_norm_eps,
        preprojected,
        rows: 1,
        layer: layer_idx,
        route: KvJointRoute::Decode {
            e8_soa: e8_soa_kernel(&gpu.arch, weights.mq2r_backend),
        },
    })
}

/// Step 3 (attention block): Q via Q-LoRA + tail-only RoPE.
///
///   x = state.tmp (post-RMSNorm)  -- but actually we should re-do
///       RMSNorm here with the fused-rotate variant so x is in the
///       FWHT-rotated domain that MQ4 expects.
///
///   Algorithm:
///     1. fused_rmsnorm_rotate_mq(stream0, attn_norm, x_rot, hidden, eps)
///        → x_rot [hidden] in MQ-rotated domain
///     2. gemv_mq4g256_prerotated(wq_a, x_rot, q_lat, q_lora_rank, hidden)
///        → q_lat [q_lora_rank=1024]
///     3. rotate_x_mq(q_lat, q_lat_rot, q_lora_rank)
///        → q_lat_rot [q_lora_rank]
///     4. gemv_mq4g256_prerotated(wq_b, q_lat_rot, q, n_heads*head_dim, q_lora_rank)
///        → q [n_heads*head_dim = 32768]
///     5. rope_tail_halfsplit on q (only last qk_rope_head_dim=64 of each
///        head's 512 dims)
///
/// Reuses `state.tmp` as the rotated post-RMSNorm input. Reuses
/// `state.q_lat`, `state.q_lat_rot`, `state.q`.
fn q_lora_prepare(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    ensure_q_lora_prepare(cfg, state, gpu)?;
    let op = q_lora_prepare_op(cfg, weights, state, layer_idx)?;
    execute_q_lora_prepare(gpu, &op).map_err(dispatch_err)?;
    q_lora_prepare_dumps(cfg, weights, state, gpu, layer_idx)
}

/// P3 calibration capture of the attention-normalized hidden state, fanned
/// out to every projection that consumes it.
pub(crate) fn q_lora_prepare_dumps(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    if !dense_activation_dump_enabled()? {
        return Ok(());
    }
    let layer = weights.resolve_layer(layer_idx);
    let mut names = vec![
        format!("layers.{layer_idx}.attn.wq_a.weight"),
        format!("layers.{layer_idx}.attn.wkv.weight"),
    ];
    if layer.compress_ratio != 0 {
        names.push(format!("layers.{layer_idx}.attn.compressor.wkv.weight"));
        names.push(format!("layers.{layer_idx}.attn.compressor.wgate.weight"));
    }
    if layer.compress_ratio == 4 {
        names.push(format!(
            "layers.{layer_idx}.attn.indexer.weights_proj.weight"
        ));
        names.push(format!(
            "layers.{layer_idx}.attn.indexer.compressor.wkv.weight"
        ));
        names.push(format!(
            "layers.{layer_idx}.attn.indexer.compressor.wgate.weight"
        ));
    }
    let tmp_plain = state.tmp_plain.as_ref().unwrap();
    dump_dense_activations_if_enabled(gpu, &names, tmp_plain, cfg.hidden_size)
}

/// Lazily allocates the Q-LoRA scratch slots used by [`q_lora_prepare_op`].
pub(crate) fn ensure_q_lora_prepare(
    cfg: &DeepseekV4Config,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
) -> Result<(), String> {
    // Allocate Q-LoRA state slots once.
    if state.q_lat.is_none() {
        state.q_lat = Some(
            gpu.alloc_tensor(&[cfg.q_lora_rank], DType::F32)
                .map_err(|e| format!("alloc q_lat: {e:?}"))?,
        );
    }
    if state.q_lat_rot.is_none() {
        state.q_lat_rot = Some(
            gpu.alloc_tensor(&[cfg.q_lora_rank], DType::F32)
                .map_err(|e| format!("alloc q_lat_rot: {e:?}"))?,
        );
    }
    if state.q.is_none() {
        // Keep enough storage for both the rank-local Q tensor and the legacy
        // non-pingpong HC mix scratch (`hc_mult * hidden`). Q operations below
        // take an explicitly shaped local view, so attention TP does not make
        // RMSNorm touch the unused tail.
        let q_total = cfg.num_attention_heads * cfg.head_dim;
        let q_storage = q_total.max(cfg.hc_mult * cfg.hidden_size);
        state.q = Some(
            gpu.alloc_tensor(&[q_storage], DType::F32)
                .map_err(|e| format!("alloc q: {e:?}"))?,
        );
    }
    if state.q_head_ones.is_none() {
        let ones = vec![1.0f32; cfg.head_dim];
        state.q_head_ones = Some(
            gpu.upload_f32(&ones, &[cfg.head_dim])
                .map_err(|e| format!("upload q_head_ones: {e:?}"))?,
        );
    }
    // Plain rmsnorm output for F16 non-expert GEMVs (antirez recipe).
    if state.tmp_plain.is_none() {
        state.tmp_plain = Some(
            gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
                .map_err(|e| format!("alloc tmp_plain: {e:?}"))?,
        );
    }
    Ok(())
}

/// Binds [`QLoraPrepareOp`] from `state`; [`ensure_q_lora_prepare`] must run first.
pub(crate) fn q_lora_prepare_op<'a>(
    cfg: &DeepseekV4Config,
    weights: &'a DeepseekV4Weights,
    state: &'a DeepseekV4State,
    layer_idx: usize,
) -> Result<QLoraPrepareOp<'a>, String> {
    let layer = weights.resolve_layer(layer_idx);
    let attn_norm = layer
        .attn_norm
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} attn_norm missing"))?;
    let wq_a = layer
        .wq_a
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} wq_a missing"))?;
    Ok(QLoraPrepareOp {
        hc_x_in: state.hc_x_in.as_ref().unwrap(),
        attn_norm,
        tmp: state.tmp.as_ref().unwrap(),
        tmp_plain: state.tmp_plain.as_ref().unwrap(),
        hidden: cfg.hidden_size,
        eps: cfg.rms_norm_eps,
        // Skip dead FWHT rotations when wq_a is Q8/F16. The projection helper
        // independently handles the second Q-LoRA projection's rotation contract.
        rotate: weight_needs_fwht(wq_a),
        rows: 1,
        layer: layer_idx,
        route: QLoraPrepareRoute::Decode {
            gfx1151_route: weights.mq2r_backend.is_gfx1151(),
        },
    })
}

/// Exact-gfx1201 MQ2R attention-input pack. Q-LoRA A, joint KV, the main
/// compressor pair, and (on ratio-4 layers) indexer weights all consume the
/// same rotated K=4096 activation. One mixed-row dispatch preserves every
/// incumbent per-row operation while replacing four or five graph nodes.
///
/// The weights of the attention-input pack when the exact-gfx1201 route admits
/// it for this layer: Q-LoRA A, joint KV, the main compressor pair and, on
/// ratio-4 layers, the indexer weights projection, all E8 SoA.
fn attention_input_pack_weights<'a>(
    cfg: &DeepseekV4Config,
    weights: &'a DeepseekV4Weights,
    gpu: &Gpu,
    layer_idx: usize,
) -> Result<Option<SmallVec<[&'a GpuTensor; 5]>>, String> {
    if gpu.arch != "gfx1201" || !cfg.mq2r || !weights.mq2r_backend.is_gfx1201() {
        return Ok(None);
    }
    let layer = weights.resolve_layer(layer_idx);
    let ratio = layer.compress_ratio as usize;
    if ratio != 4 && ratio != 128 {
        return Ok(None);
    }
    let wq_a = layer
        .wq_a
        .as_ref()
        .ok_or_else(|| format!("wq_a l{layer_idx}"))?;
    let wkv = layer
        .wkv
        .as_ref()
        .ok_or_else(|| format!("wkv l{layer_idx}"))?;
    let comp_wkv = layer
        .compressor_wkv
        .as_ref()
        .ok_or_else(|| format!("comp_wkv l{layer_idx}"))?;
    let comp_wgate = layer
        .compressor_wgate
        .as_ref()
        .ok_or_else(|| format!("comp_wgate l{layer_idx}"))?;
    let mut projection_weights: SmallVec<[&GpuTensor; 5]> =
        smallvec![wq_a, wkv, comp_wkv, comp_wgate];
    if ratio == 4 {
        projection_weights.push(
            layer
                .indexer_weights_proj
                .as_ref()
                .ok_or_else(|| format!("idx_weights_proj l{layer_idx}"))?,
        );
    }
    if projection_weights
        .iter()
        .any(|weight| weight.dtype != DType::MFP4G32E8SOA)
    {
        return Ok(None);
    }
    Ok(Some(projection_weights))
}

/// Main compressor projection rows of the attention-input pack.
fn attention_input_pack_main_proj_dim(cfg: &DeepseekV4Config, ratio: u32) -> usize {
    if ratio == 4 {
        2 * cfg.head_dim
    } else {
        cfg.head_dim
    }
}

pub(crate) fn ensure_attention_input_pack(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    if attention_input_pack_weights(cfg, weights, gpu, layer_idx)?.is_none() {
        return Ok(());
    }
    let ratio = weights.resolve_layer(layer_idx).compress_ratio;
    let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
    if state.kv.is_none() {
        state.kv = Some(
            gpu.alloc_tensor(&[kv_dim], DType::F32)
                .map_err(|error| format!("alloc packed kv l{layer_idx}: {error:?}"))?,
        );
    }
    let main_proj_dim = attention_input_pack_main_proj_dim(cfg, ratio);
    let indexer = &mut state._indexer[layer_idx];
    if indexer.comp_kv_buf.is_none() {
        indexer.comp_kv_buf = Some(
            gpu.alloc_tensor(&[main_proj_dim], DType::F32)
                .map_err(|error| format!("alloc packed comp kv l{layer_idx}: {error:?}"))?,
        );
    } else if indexer.comp_kv_buf.as_ref().unwrap().numel() < main_proj_dim {
        return Err(format!("packed comp kv is undersized l{layer_idx}"));
    }
    if indexer.comp_score_buf.is_none() {
        indexer.comp_score_buf = Some(
            gpu.alloc_tensor(&[main_proj_dim], DType::F32)
                .map_err(|error| format!("alloc packed comp score l{layer_idx}: {error:?}"))?,
        );
    } else if indexer.comp_score_buf.as_ref().unwrap().numel() < main_proj_dim {
        return Err(format!("packed comp score is undersized l{layer_idx}"));
    }
    if ratio == 4 && indexer.idx_weights.is_none() {
        indexer.idx_weights = Some(
            gpu.alloc_tensor(&[cfg.index_n_heads], DType::F32)
                .map_err(|error| format!("alloc packed idx weights l{layer_idx}: {error:?}"))?,
        );
    }
    Ok(())
}

/// The exact-gfx1201 attention-input pack of this layer, or `None` when the
/// route does not admit it (then every projection runs on its own).
pub(crate) fn attention_input_pack_op<'a>(
    cfg: &DeepseekV4Config,
    weights: &'a DeepseekV4Weights,
    state: &'a DeepseekV4State,
    gpu: &Gpu,
    layer_idx: usize,
) -> Result<Option<PackedProjectionsOp<'a>>, String> {
    let Some(projection_weights) = attention_input_pack_weights(cfg, weights, gpu, layer_idx)?
    else {
        return Ok(None);
    };
    let ratio = weights.resolve_layer(layer_idx).compress_ratio;
    let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
    let main_proj_dim = attention_input_pack_main_proj_dim(cfg, ratio);
    let x = state
        .tmp
        .as_ref()
        .ok_or_else(|| format!("packed tmp l{layer_idx}"))?;
    let indexer = &state._indexer[layer_idx];
    let mut outputs: SmallVec<[&GpuTensor; 5]> = smallvec![
        state
            .q_lat
            .as_ref()
            .ok_or_else(|| format!("packed q_lat l{layer_idx}"))?,
        state.kv.as_ref().unwrap(),
        indexer.comp_kv_buf.as_ref().unwrap(),
        indexer.comp_score_buf.as_ref().unwrap(),
    ];
    let mut rows: SmallVec<[usize; 5]> =
        smallvec![cfg.q_lora_rank, kv_dim, main_proj_dim, main_proj_dim];
    if ratio == 4 {
        outputs.push(indexer.idx_weights.as_ref().unwrap());
        rows.push(cfg.index_n_heads);
    }
    let mut jobs = [None; 5];
    for (slot, ((weight, output), rows)) in projection_weights
        .into_iter()
        .zip(outputs)
        .zip(rows)
        .enumerate()
    {
        jobs[slot] = Some(PackedProjection {
            weight,
            output,
            rows,
        });
    }
    Ok(Some(PackedProjectionsOp {
        x,
        k: cfg.hidden_size,
        rows: 1,
        layer: layer_idx,
        route: PackedProjectionsRoute::Decode {
            jobs,
            what: format!("packed attention input E8 l{layer_idx}"),
        },
    }))
}

/// The exact-gfx1201 ratio-4 indexer compressor pack, or `None` when the route
/// does not admit it.
pub(crate) fn indexer_compressor_pack_op<'a>(
    cfg: &DeepseekV4Config,
    weights: &'a DeepseekV4Weights,
    state: &'a DeepseekV4State,
    gpu: &Gpu,
    layer_idx: usize,
) -> Result<Option<PackedProjectionsOp<'a>>, String> {
    if gpu.arch != "gfx1201" || !cfg.mq2r || !weights.mq2r_backend.is_gfx1201() {
        return Ok(None);
    }
    let layer = weights.resolve_layer(layer_idx);
    if layer.compress_ratio != 4 {
        return Ok(None);
    }
    let wkv = layer
        .indexer_compressor_wkv
        .as_ref()
        .ok_or_else(|| format!("idx comp wkv l{layer_idx}"))?;
    let wgate = layer
        .indexer_compressor_wgate
        .as_ref()
        .ok_or_else(|| format!("idx comp wgate l{layer_idx}"))?;
    if wkv.dtype != DType::MFP4G32E8SOA || wgate.dtype != DType::MFP4G32E8SOA {
        return Ok(None);
    }
    let indexer = &state._indexer[layer_idx];
    let kv = indexer
        .comp_kv_buf
        .as_ref()
        .ok_or_else(|| format!("idx packed comp kv l{layer_idx}"))?;
    let score = indexer
        .comp_score_buf
        .as_ref()
        .ok_or_else(|| format!("idx packed comp score l{layer_idx}"))?;
    let x = state
        .tmp
        .as_ref()
        .ok_or_else(|| format!("idx packed tmp l{layer_idx}"))?;
    let proj_dim = 2 * cfg.index_head_dim;
    let job = |weight, output| {
        Some(PackedProjection {
            weight,
            output,
            rows: proj_dim,
        })
    };
    Ok(Some(PackedProjectionsOp {
        x,
        k: cfg.hidden_size,
        rows: 1,
        layer: layer_idx,
        route: PackedProjectionsRoute::Decode {
            jobs: [job(wkv, kv), job(wgate, score), None, None, None],
            what: format!("packed indexer compressor E8 l{layer_idx}"),
        },
    }))
}

/// Finish Q-LoRA after [`q_lora_prepare`] has materialized the shared
/// attention-normalized input. Keeping this boundary explicit lets the exact
/// heterogeneous gfx1100 route overlap the independent KV/compressor branch
/// without changing any projection arithmetic or reduction order.
fn q_lora_project(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
    wq_a_preprojected: bool,
) -> Result<(), String> {
    let op = q_lora_project_op(cfg, weights, state, gpu, layer_idx, wq_a_preprojected)?;
    execute_q_lora_project(gpu, &op).map_err(dispatch_err)?;
    q_lora_project_dumps(weights, state, gpu, &op)
}

/// Binds [`QLoraProjectOp`] from `state`; `q_lora_prepare` must have allocated its slots.
pub(crate) fn q_lora_project_op<'a>(
    cfg: &DeepseekV4Config,
    weights: &'a DeepseekV4Weights,
    state: &'a DeepseekV4State,
    gpu: &Gpu,
    layer_idx: usize,
    wq_a_preprojected: bool,
) -> Result<QLoraProjectOp<'a>, String> {
    let layer = weights.resolve_layer(layer_idx);
    let q_norm = layer
        .q_norm
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} q_norm missing"))?;
    let wq_a = layer
        .wq_a
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} wq_a missing"))?;
    let wq_b = layer
        .wq_b
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} wq_b missing"))?;
    let wq_b_needs_fwht = weight_needs_fwht(wq_b);
    Ok(QLoraProjectOp {
        wq_a,
        wq_b,
        q_norm,
        tmp: state.tmp.as_ref().unwrap(),
        tmp_plain: state.tmp_plain.as_ref().unwrap(),
        q_lat: state.q_lat.as_ref().unwrap(),
        q_lat_rot: state.q_lat_rot.as_ref().unwrap(),
        q: state.q.as_ref().unwrap(),
        q_head_ones: state.q_head_ones.as_ref().unwrap(),
        hidden: cfg.hidden_size,
        q_lora_rank: cfg.q_lora_rank,
        heads: cfg.num_attention_heads,
        head_dim: cfg.head_dim,
        eps: cfg.rms_norm_eps,
        wq_a_preprojected,
        wq_b_needs_fwht,
        rows: 1,
        layer: layer_idx,
        route: QLoraProjectRoute::Decode {
            e8_soa: e8_soa_kernel(&gpu.arch, weights.mq2r_backend),
            fused_qnorm_rotate: wq_b_needs_fwht
                && config_cache::qnorm_rotate_fused_on(&gpu.arch, cfg.mq2r),
        },
    })
}

/// Diagnostic dense-activation capture for [`q_lora_project`]. Runs after the
/// projection, which takes no `embed_scratch`, so the captured values are unchanged.
pub(crate) fn q_lora_project_dumps(
    weights: &DeepseekV4Weights,
    state: &DeepseekV4State,
    gpu: &mut Gpu,
    op: &QLoraProjectOp<'_>,
) -> Result<(), String> {
    if !dense_activation_dump_enabled()? {
        return Ok(());
    }
    let layer_idx = op.layer;
    let layer = weights.resolve_layer(layer_idx);
    let mut names = vec![format!("layers.{layer_idx}.attn.wq_b.weight")];
    if layer.compress_ratio == 4 {
        names.push(format!("layers.{layer_idx}.attn.indexer.wq_b.weight"));
    }
    let fused = matches!(op.route, QLoraProjectRoute::Decode { fused_qnorm_rotate: true, .. });
    if fused {
        // The shipping fused route intentionally leaves q_lat in its
        // pre-norm state. Materialize the logical, pre-FWHT wq_b input in
        // diagnostic scratch without perturbing q_lat_rot or the forward.
        let scratch = state
            .embed_scratch
            .as_ref()
            .ok_or_else(|| "dense capture: embed_scratch missing".to_string())?
            .sub_offset(0, op.q_lora_rank);
        gpu.rmsnorm_f32(op.q_lat, op.q_norm, &scratch, op.eps)
            .map_err(|e| format!("dense capture q_norm layer {layer_idx}: {e:?}"))?;
        dump_dense_activations_if_enabled(gpu, &names, &scratch, op.q_lora_rank)
    } else {
        dump_dense_activations_if_enabled(gpu, &names, op.q_lat, op.q_lora_rank)
    }
}

pub(crate) fn q_lora(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    q_lora_prepare(cfg, weights, state, gpu, layer_idx)?;
    q_lora_project(cfg, weights, state, gpu, layer_idx, false)
}

/// Step 1 of forward: embedding lookup + 4-stream residual init.
///
/// DeepSeek V4's HC pattern starts with `[embed, 0, 0, 0]` — stream 0 gets
/// the embedding, streams 1-3 zero-initialised. Subsequent layers'
/// HC mixes propagate signal across all four streams.
///
/// Allocates `state.residual_streams` and `state.embed_scratch`
/// lazily on first call.
/// Per-layer slot count in `pos_array_*`. Layout per layer:
///   [0] qk_pos              = position
///   [1] main_comp_rope_pos  = mid-of-window  (depends on ratio + COMP_ROPE_POS env)
///   [2] indexer_comp_rope_pos = start-of-window
/// Used by the HIP-graphs-friendly position-array path (default in
/// `decode_step` since 2026-05-21). Direct-dispatch path uses the same
/// array but doesn't strictly need the stable host source.
pub(crate) const POS_SLOTS_PER_LAYER: usize = 3;

/// Compute per-layer derived positions and update `state.pos_array_*`.
///
/// Single host-to-device copy of the entire `[(num_layers + 1) * 3]` i32
/// array, with `pos_array_host` as the stable source pointer (required so
/// captured graph nodes re-read valid values on replay).
///
/// Reads env vars HIPFIRE_DEEPSEEK4_COMP_ROPE_POS once into a cache (TODO:
/// migrate to OnceLock once we settle on a fixed default).
pub fn precompute_positions(
    cfg: &DeepseekV4Config,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    position: u32,
) -> Result<(), String> {
    let total_slots = (cfg.num_hidden_layers + 1) * POS_SLOTS_PER_LAYER;

    // Lazy-alloc device buffer + stable host source (Box<[i32]>).
    if state.pos_array_device.is_none() {
        state.pos_array_device = Some(
            gpu.alloc_tensor(&[total_slots], DType::F32)
                .map_err(|e| format!("alloc pos_array_device: {e:?}"))?,
        );
    }
    if state.pos_array_host.is_none() {
        state.pos_array_host = Some(vec![0i32; total_slots].into_boxed_slice());
    }

    let pos_array_host = state.pos_array_host.as_mut().unwrap();
    fill_pos_array_host(cfg, pos_array_host, position);

    // ONE htod for the whole array. Source is the stable Box<[i32]> on
    // the heap, so captured graph nodes can re-read it on replay.
    let pos_array_device = state.pos_array_device.as_ref().unwrap();
    let bytes = unsafe {
        std::slice::from_raw_parts(
            pos_array_host.as_ptr() as *const u8,
            pos_array_host.len() * 4,
        )
    };
    gpu.memcpy_htod_auto(&pos_array_device.buf, bytes)
        .map_err(|e| format!("htod pos_array: {e:?}"))?;

    // Also write SWA state (slot, n_valid) — same stable-host-source
    // pattern. The captured memcpy re-reads this on every graph_launch.
    precompute_attn_state(cfg, state, gpu)?;
    Ok(())
}

/// Lazy-alloc + populate `state.token_id_buf` (and stable host source
/// `state.token_id_host`) for the current step's token. The captured
/// htod node re-reads `token_id_host` on every graph replay, so the
/// HIP-graphs-safe `hash_router_normalize_f32_buf` kernel sees the
/// per-replay token_id.
pub(crate) fn precompute_token_id(
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    token_id: u32,
) -> Result<(), String> {
    if state.token_id_buf.is_none() {
        state.token_id_buf = Some(
            gpu.alloc_tensor(&[1], DType::F32)
                .map_err(|e| format!("alloc token_id_buf: {e:?}"))?,
        );
    }
    if state.token_id_host.is_none() {
        state.token_id_host = Some(Box::new([0i32; 1]));
    }
    let host = state.token_id_host.as_mut().unwrap();
    host[0] = token_id as i32;
    let dev = state.token_id_buf.as_ref().unwrap();
    let bytes = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, 4) };
    gpu.memcpy_htod_auto(&dev.buf, bytes)
        .map_err(|e| format!("htod token_id: {e:?}"))?;
    Ok(())
}

/// Stage the device-resident token and position inputs consumed by a retained
/// DeepSeek4 decode tape without launching the tape itself.
///
/// This is the committed Redline prefix-profiler adapter. It deliberately
/// reuses the production staging helpers so the decode, EP, and MTP paths keep
/// their existing behavior and call graph.
#[doc(hidden)]
pub fn prepare_retained_decode_inputs(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<(), String> {
    precompute_positions(cfg, state, gpu, position)?;
    precompute_token_id(state, gpu, token_id)?;
    if !config_cache::retained_embedding_on(&gpu.arch, cfg.mq2r) {
        init_residual_streams(cfg, weights, state, gpu, token_id)?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("DeepSeek V4 retained profiler adapter sync: {e:?}"))?;
    }
    Ok(())
}

/// Host-only update of `token_id_host[0]`. Used by the HIP-graphs
/// replay path — the captured memcpy node re-reads this byte on
/// graph_launch and propagates to `token_id_buf`.
pub(crate) fn update_token_id_host(state: &mut DeepseekV4State, token_id: u32) {
    let host = state.token_id_host.as_mut().expect(
        "update_token_id_host: token_id_host not initialised \
                 (call precompute_token_id first)",
    );
    host[0] = token_id as i32;
}

/// Per-batch twin of `precompute_positions`. Fills B contiguous stripes
/// of `(num_hidden_layers + 1) * POS_SLOTS_PER_LAYER` slots in
/// `pbs.pos_array_device_batch` — one stripe per batch row b at absolute
/// position `start_pos + b`. Single host-side build, single htod.
///
/// Stripe b layout matches the single-position `state.pos_array_device`:
/// `[layer_idx * 3 + slot]` where slot ∈ {0=qk_pos, 1=main_rope_pos,
/// 2=indexer_rope_pos}.
pub(crate) fn precompute_positions_batched(
    cfg: &DeepseekV4Config,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    start_pos: u32,
    batch_size: usize,
) -> Result<(), String> {
    let slots_per_pos = (cfg.num_hidden_layers + 1) * POS_SLOTS_PER_LAYER;
    let total_i32s = batch_size * slots_per_pos;

    let comp_rope_mode = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_COMP_ROPE_POS").ok();
    let comp_rope_mode = comp_rope_mode.as_deref();

    let mut host: Vec<i32> = vec![0i32; total_i32s];
    for b in 0..batch_size {
        let pos = (start_pos as usize) + b;
        let stripe = b * slots_per_pos;
        for layer_idx in 0..=cfg.num_hidden_layers {
            let ratio = if layer_idx < cfg.num_hidden_layers {
                cfg.compress_ratios[layer_idx] as usize
            } else {
                0
            };
            let base = stripe + layer_idx * POS_SLOTS_PER_LAYER;
            host[base] = pos as i32;
            if ratio > 0 {
                // Default MUST be "start" — `(pos/ratio)*ratio` — to match the
                // decode path (`fill_pos_array_host`) and the reference ds4
                // (comp_pos = start of the just-closed window). This previously
                // defaulted to "mid" (+ ratio/2) while decode defaults to
                // "start", so the compressed KV was BUILT here with a different
                // compressor-RoPE phase than it is READ with at decode → far-
                // context (compressed) recall lost the tail of the prompt.
                // Keep the named modes identical to `fill_pos_array_host`.
                let main_rope_pos: i32 = match comp_rope_mode {
                    Some("end") => pos as i32,
                    Some("mid") => (((pos / ratio) * ratio) + ratio / 2) as i32,
                    _ => ((pos / ratio) * ratio) as i32,
                };
                let indexer_rope_pos = ((pos / ratio) * ratio) as i32;
                host[base + 1] = main_rope_pos;
                host[base + 2] = indexer_rope_pos;
            } else {
                host[base + 1] = 0;
                host[base + 2] = 0;
            }
        }
    }

    let bytes = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, total_i32s * 4) };
    gpu.memcpy_htod_auto(&pbs.pos_array_device_batch.buf, bytes)
        .map_err(|e| format!("htod pos_array_device_batch: {e:?}"))
}

/// Per-batch twin of `precompute_attn_state`. Fills B contiguous stripes
/// of 10 slots in `pbs.attn_state_buf_batch` — one stripe per batch row
/// b at absolute position `start_pos + b`. Slot layout matches
/// `fill_attn_state_host` (see line ~1389).
pub(crate) fn precompute_attn_state_batched(
    cfg: &DeepseekV4Config,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    start_pos: u32,
    batch_size: usize,
) -> Result<(), String> {
    let slots_per_pos = 10;
    let total_i32s = batch_size * slots_per_pos;

    let win = cfg.sliding_window as i32;
    let topk = cfg.index_topk as i32;
    let max_compressed = pbs.idx_score_capacity as i32;

    let mut host: Vec<i32> = vec![0i32; total_i32s];
    for b in 0..batch_size {
        let pos = (start_pos as i32) + b as i32;
        let stripe = b * slots_per_pos;

        let swa_slot = pos % win;
        let n_valid_swa = (pos + 1).min(win);
        let n_compressed_4 = capped_compressed_count(pos, 4, max_compressed);
        let n_compressed_128 = capped_compressed_count(pos, 128, max_compressed);
        let k_active_4 = topk.min(n_compressed_4);
        let k_active_128 = topk.min(n_compressed_128);
        let ring_slot_4 = 4 + (pos % 4);
        let commit_slot_4 = if (pos + 1) % 4 == 0 {
            let s = pos / 4;
            if s < max_compressed {
                s
            } else {
                -1
            }
        } else {
            -1
        };
        let ring_slot_128 = pos % 128;
        let commit_slot_128 = if (pos + 1) % 128 == 0 {
            let s = pos / 128;
            if s < max_compressed {
                s
            } else {
                -1
            }
        } else {
            -1
        };

        host[stripe] = swa_slot;
        host[stripe + 1] = n_valid_swa;
        host[stripe + 2] = n_compressed_4;
        host[stripe + 3] = n_compressed_128;
        host[stripe + 4] = k_active_4;
        host[stripe + 5] = k_active_128;
        host[stripe + 6] = ring_slot_4;
        host[stripe + 7] = commit_slot_4;
        host[stripe + 8] = ring_slot_128;
        host[stripe + 9] = commit_slot_128;
    }

    let bytes = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, total_i32s * 4) };
    gpu.memcpy_htod_auto(&pbs.attn_state_buf_batch.buf, bytes)
        .map_err(|e| format!("htod attn_state_buf_batch: {e:?}"))
}

/// Slice the pos_array for a given layer's slot. Caller passes the slot
/// constant (0=qk_pos, 1=main_comp_rope, 2=indexer_comp_rope).
pub(crate) fn pos_slot(
    state: &DeepseekV4State,
    layer_idx: usize,
    slot: usize,
) -> Result<rdna_compute::GpuTensor, String> {
    let arr = state
        .pos_array_device
        .as_ref()
        .ok_or_else(|| "pos_array_device not initialised".to_string())?;
    let offset = layer_idx * POS_SLOTS_PER_LAYER + slot;
    Ok(arr.sub_offset(offset, 1))
}

pub(crate) fn init_residual_streams(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    token_id: u32,
) -> Result<(), String> {
    let token_embd = weights
        .token_embd
        .as_ref()
        .ok_or_else(|| "init_residual_streams: token_embd not uploaded".to_string())?;
    let hidden = cfg.hidden_size;
    let hc_mult = cfg.hc_mult;

    if state.embed_scratch.is_none() {
        state.embed_scratch = Some(
            gpu.alloc_tensor(&[hidden], DType::F32)
                .map_err(|e| format!("alloc embed_scratch: {e:?}"))?,
        );
    }
    if state.residual_streams.is_none() {
        // Zero-init: alloc_tensor leaves memory uninitialized, but the
        // [embed, 0, 0, 0] init pattern relies on streams 1..hc_mult
        // being zero. `gpu.zeros` is the right primitive.
        let t = gpu
            .zeros(&[hc_mult, hidden], DType::F32)
            .map_err(|e| format!("alloc residual_streams: {e:?}"))?;
        state.residual_streams = Some(t);
    }
    if config_cache::hc_pingpong_on(&gpu.arch, cfg.mq2r) && state.residual_streams_next.is_none() {
        state.residual_streams_next = Some(
            gpu.alloc_tensor(&[hc_mult, hidden], DType::F32)
                .map_err(|e| format!("alloc residual_streams_next: {e:?}"))?,
        );
    }
    if state.tmp.is_none() {
        state.tmp = Some(
            gpu.alloc_tensor(&[hidden], DType::F32)
                .map_err(|e| format!("alloc tmp: {e:?}"))?,
        );
    }

    // Dequant + lookup token row → embed_scratch [hidden].
    let embed_scratch = state.embed_scratch.as_ref().unwrap();
    gpu.embedding_lookup_q8(token_embd, embed_scratch, token_id, hidden)
        .map_err(|e| format!("embedding_lookup_q8: {e:?}"))?;

    // **HC init**: per antirez ds4 `hc_from_plain_embedding` (ds4.c:4358),
    // ALL `hc_mult` streams are initialised with a COPY of the embedding,
    // NOT `[embed, 0, 0, 0]` as our prior comment claimed. The "0 streams"
    // pattern would have forced HC pre/post/comb to propagate signal from
    // stream 0 across layers, producing wrong magnitudes throughout the
    // forward.
    let streams = state.residual_streams.as_ref().unwrap();
    let bytes_per_stream = hidden * 4; // F32 = 4 bytes
    for h in 0..hc_mult {
        let dst_view = streams.sub_offset(h * hidden, hidden);
        gpu.memcpy_dtod_auto(&dst_view.buf, &embed_scratch.buf, bytes_per_stream)
            .map_err(|e| format!("d2d copy stream {h}: {e:?}"))?;
    }

    Ok(())
}

/// Reusable per-call scratch for the batched-prefill driver.
///
/// **Phase B status (2026-05-18):** growing. Currently holds the
/// per-layer batched intermediates needed by `q_lora_batched`. Future
/// per-stage batched helpers (kv_joint_batched, attn_batched,
/// ffn_batched, hc_mix_batched) extend this struct as they land.
///
/// Sized to `max_batch` rows everywhere; tensors are reused across
/// per-chunk layer iterations.
pub struct PrefillBatchScratch {
    pub max_batch: usize,
    /// Allocated row stride of `idx_scores_batch`. It tracks the state's
    /// active compressed-capacity bucket and is resized only before a request,
    /// never inside a captured verify body.
    pub idx_score_capacity: usize,
    /// DSpark verify retained routes, keyed by the full target batch B. These
    /// controllers are separate from `Gpu::replay`, which owns the certified
    /// ordinary-AR tape.
    pub dspark_verify_pm4: std::collections::BTreeMap<usize, ReplayController>,
    /// Embedding-lookup output `[max_batch, hidden]`. Source for the
    /// HC stream-broadcast init at chunk start.
    pub embed_batch: GpuTensor,
    /// HC residual streams `[max_batch, hc_mult, hidden]`. Lives across
    /// the full per-layer loop within a chunk.
    pub streams_batch: GpuTensor,
    /// Token-ids buffer feeding `embedding_lookup_q8_batched`.
    /// `[max_batch]` stored as F32 (same i32-in-F32-slots dtype-cosmetic
    /// pattern as qwen35's `pbs.tokens`).
    pub tokens: GpuTensor,
    /// FWHT-rotated attn_norm output `[max_batch, hidden]` feeding MQ4
    /// non-expert GEMMs.
    pub tmp_batch: GpuTensor,
    /// Plain attn_norm output `[max_batch, hidden]` feeding F32/Q8
    /// non-expert GEMMs.
    pub tmp_plain_batch: GpuTensor,
    /// Q-LoRA bottleneck `[max_batch, q_lora_rank]`. Reused: wq_a output
    /// → q_norm in place → fed to wq_b (after rotate into q_lat_rot_batch).
    pub q_lat_batch: GpuTensor,
    /// FWHT-rotated q_lat for the MQ4 wq_b path `[max_batch, q_lora_rank]`.
    pub q_lat_rot_batch: GpuTensor,
    /// Q output `[max_batch, n_heads, head_dim]`. wq_b output, then
    /// per-(batch, head) RMSNormed by `q_head_ones`.
    pub q_batch: GpuTensor,
    /// Per-head ones vector `[head_dim]` reused as the rmsnorm weight
    /// for the per-(batch, head) Q normalisation. Shared across batch.
    pub q_head_ones: GpuTensor,
    /// Joint KV `[max_batch, kv_dim]` where `kv_dim = n_kv_heads * head_dim`.
    /// wkv output, then kv_norm RMSNormed in place.
    pub kv_batch: GpuTensor,
    /// Per-batch absolute KV positions `[max_batch]` stored as F32 (the
    /// rope_tail_*_batched kernels read it as i32). Uploaded once per
    /// chunk: positions[b] = start_pos + b.
    pub positions: GpuTensor,
    /// HC control vector `[max_batch, 24]` — output of hc_compute_control
    /// _batched, in-place rescaled by hc_apply_alpha_batched, then split
    /// into pre/post/comb by hc_split_finalize_batched.
    pub hc_c_batch: GpuTensor,
    /// HC `pre` weights `[max_batch, hc_mult=4]`. Used by
    /// hc_input_map_4stream_batched and hc_mix_4stream_batched.
    pub hc_pre_batch: GpuTensor,
    /// HC `post` weights `[max_batch, hc_mult=4]`. Scale-multiplied
    /// sigmoid output. Feeds hc_mix_4stream_batched as the per-stream
    /// scale.
    pub hc_post_batch: GpuTensor,
    /// HC `comb` matrix `[max_batch, 4, 4]` — Sinkhorn-normalised to be
    /// doubly stochastic per batch row.
    pub hc_comb_batch: GpuTensor,
    /// HC transform input `[max_batch, hidden]` — output of mhc_pre's
    /// hc_input_map_4stream_batched. Feeds q_lora_batched / kv_joint
    /// _batched on the attention side, and the FFN gate/up on the FFN
    /// side.
    pub hc_x_in_batch: GpuTensor,
    /// Attention contribution `[max_batch, hidden]` produced by the
    /// attention block (Q · K → softmax → V → wo). Consumed by
    /// hc_attn_mix_batched as the `transform_out` argument.
    pub attn_out_batch: GpuTensor,
    /// FFN contribution `[max_batch, hidden]` produced by the routed
    /// MoE FFN. Consumed by hc_ffn_mix_batched as `transform_out`.
    pub ffn_out_batch: GpuTensor,
    /// Temporary `[max_batch, hc_mult, hidden]` for the hc_mix output
    /// before it's memcpy'd back into streams_batch. Mirrors the
    /// sequential path's reuse of `state.q` as the mix-output buffer.
    pub streams_out_batch: GpuTensor,
    /// Per-row visible SWA window `[max_batch, head_dim, swa_window]`
    /// produced by swa_visibility_stage_batched. DeepSeek V4 has K=V tied so
    /// one buffer feeds both the K and V args of the attention kernel.
    pub swa_staged_batch: GpuTensor,
    /// Per-row top-K K/V gather buffer `[max_batch, head_dim, topk_max]`
    /// produced by deepseek4_topk_kv_gather_batched (or the identity variant
    /// for ratio=128). Same K=V tied semantics.
    pub topk_staged_batch: GpuTensor,
    /// Per-row n_valid_swa array `[max_batch]` (i32-in-F32 slots).
    /// Tells deepseek4_attn_swa_topk_batched_f32 how many SWA entries are
    /// valid for each batch row.
    pub n_valid_swa_arr: GpuTensor,
    /// Per-row n_active_topk array `[max_batch]` (i32-in-F32 slots).
    pub n_active_topk_arr: GpuTensor,
    /// Per-row compressed-count array for ratio-4 indexer layers. Kept
    /// separate from `n_active_topk_arr` so a captured verify graph never
    /// needs to overwrite a live kernel input with an in-capture H2D copy.
    pub n_compressed_4_arr: GpuTensor,
    /// Capture-safe per-row active-topk arrays. DeepSeek V4 only has ratio-4
    /// and ratio-128 compressed-attention layers, so both arrays are computed
    /// and uploaded once at the verify-window boundary.
    pub n_active_topk_4_arr: GpuTensor,
    pub n_active_topk_128_arr: GpuTensor,
    /// Raw attention output `[max_batch, n_heads, head_dim]`. Output of
    /// deepseek4_attn_swa_topk_batched_f32; consumed by inverse RoPE + the
    /// O-LoRA wo_a/wo_b projection chain.
    pub attn_out_raw_batch: GpuTensor,
    /// FWHT-rotated attn_out_raw `[max_batch, n_heads * head_dim]`.
    /// Input to per-group wo_a batched GEMV (MQ4 weight path).
    pub attn_out_raw_rot_batch: GpuTensor,
    /// wo_a output `[max_batch, n_groups, o_lora_rank]`.
    pub wo_a_out_batch: GpuTensor,
    /// FWHT-rotated wo_a output `[max_batch, n_groups * o_lora_rank]`.
    /// Input to wo_b batched GEMV (MQ4 weight path).
    pub wo_a_out_rot_batch: GpuTensor,
    // ── FFN-side scratch ──
    pub ffn_x_rot_batch: GpuTensor,        // [B, hidden]
    pub ffn_x_plain_batch: GpuTensor,      // [B, hidden]
    pub ffn_shared_gate_batch: GpuTensor,  // [B, IM]
    pub ffn_shared_up_batch: GpuTensor,    // [B, IM]
    pub ffn_shared_rot_batch: GpuTensor,   // [B, IM]
    pub moe_scores_batch: GpuTensor,       // [B, n_exp]
    pub moe_topk_indices_batch: GpuTensor, // [B, k_top]  i32-in-F32
    pub moe_topk_weights_batch: GpuTensor, // [B, k_top]
    pub moe_gate_batch: GpuTensor,         // [B, k_top, IM]
    pub moe_up_batch: GpuTensor,           // [B, k_top, IM]
    pub moe_rot_batch: GpuTensor,          // [B, k_top, IM]
    /// Per-(token, krank) expert outputs for the atomic-free down path.
    /// Gated by HIPFIRE_DEEPSEEK4_MOE_DETERMINISTIC=1 (or grouped path which
    /// uses its own scratch). Sized [B, k_top, hidden] f32 — at DeepSeek V4
    /// max_batch=512 hidden=4096 K_TOP=6 = 48 MB.
    pub moe_down_expert_outputs: GpuTensor,
    // ── Indexer chain scratch (Step-1 perf pass) ──
    pub idx_q_batch: GpuTensor,      // [B, idx_n_heads, idx_head_dim]
    pub idx_w_batch: GpuTensor,      // [B, idx_n_heads]
    pub idx_scores_batch: GpuTensor, // [B, max_compressed]
    pub idx_topk_indices_batch: GpuTensor, // [B, index_topk]  i32-in-F32
    // ── Compressor batched-GEMV scratch (Phase 2.5 perf pass) ──
    // Holds the wkv / wgate compressor outputs across all B positions
    // so the GEMVs can be batched out of the per-position loop. Main
    // and indexer compressors get separate buffers because the proj_dim
    // differs (main=2*head_dim=1024, idx=2*idx_head_dim=256 for DeepSeek V4).
    pub comp_main_kv_batch: GpuTensor,    // [B, 2*head_dim]
    pub comp_main_score_batch: GpuTensor, // [B, 2*head_dim]
    pub comp_idx_kv_batch: GpuTensor,     // [B, 2*idx_head_dim]
    pub comp_idx_score_batch: GpuTensor,  // [B, 2*idx_head_dim]
    /// F32 compressor output stage `[B, head_dim]` used only when the
    /// long-lived cache is F16. Main and indexer commits reuse it serially.
    pub comp_cache_batch_f32: GpuTensor,
    // ── Scatter-by-expert MoE sort outputs ──
    // Single counting-sort produces these per layer; the grouped MoE
    // GEMVs then read each expert weight slab once with cache reuse.
    pub moe_sorted_b: GpuTensor,      // [B * K_TOP] i32
    pub moe_sorted_krank: GpuTensor,  // [B * K_TOP] i32
    pub moe_sorted_expert: GpuTensor, // [B * K_TOP] i32
    pub moe_expert_starts: GpuTensor, // [n_exp + 1] i32

    // ── SGLang-style scatter-grouped MoE pipeline (chunk_size ≥ 256) ──
    // Outputs of moe_scatter_fused_k8 feed gemm_mq2g256_lloyd_moe_grouped
    // _wmma_k2 and the unscatter/down-combine kernels. Sized for the
    // worst-case `m_total_max = max_batch * K_TOP + n_exp * BLOCK_M(=16)`
    // padded scatter layout.
    pub moe_expert_token_counts: GpuTensor, // [n_exp]      i32 (Raw)
    pub moe_expert_offsets: GpuTensor,      // [n_exp + 1]  i32 (Raw)
    pub moe_sorted_slot_index: GpuTensor,   // [m_total_max] i32 (Raw)
    pub moe_expert_tile_ids: GpuTensor,     // [m_total_max / 16] i32 (Raw)
    pub moe_inverse_perm: GpuTensor,        // [B * K_TOP]  i32 (Raw)
    /// Output of grouped gate_up GEMM. [m_total_max × 2*mi] f32.
    pub moe_y_gate_up_grouped: GpuTensor,
    /// Permuted, post-silu-mul intermediate. [m_total_max × mi] f32.
    /// Built by `moe_gate_up_unscatter_k8` then re-scattered by silu-mul,
    /// or written directly by a fused silu-mul kernel into grouped order.
    /// Input to the grouped down GEMM.
    pub moe_x_grouped: GpuTensor,
    /// Output of grouped down GEMM. [m_total_max × hidden] f32.
    /// Combined into the residual stream by `moe_down_combine_grouped_k8`.
    pub moe_y_down_grouped: GpuTensor,
    // ── F16 staging for WMMA compressor GEMMs ──
    // F32 attention-norm output gets converted once per layer into
    // these buffers, then the four compressor GEMMs (wkv/wgate ×
    // main/idx) consume F16 inputs directly. Sized at [max_batch,
    // hidden] like tmp_batch; 1/2 the per-element bytes of F32.
    pub tmp_batch_f16: GpuTensor, // [B, hidden] F16 (stored as Raw)
    pub tmp_plain_batch_f16: GpuTensor, // [B, hidden] F16 (stored as Raw)
    /// Generic F16 staging buffer for WMMA HFQ4 GEMMs. Sized at
    /// `max_batch * max_dim * 2 bytes` so any batched GEMM input can
    /// be converted F32→F16 in place before dispatch. max_dim is the
    /// largest K dim across all DeepSeek V4 batched GEMM call sites — wo_b's
    /// K = groups × o_lora_rank for DeepSeek V4 (= 8 × 1024 = 8192).
    pub wmma_x_scratch_f16: GpuTensor,
    /// Per-compress-event RoPE positions buffer for the Phase A
    /// batched compressor pipeline. Sized [max_batch] F32 (i32-in-F32)
    /// since at B=64 ratio=4 we have at most 16 events per layer and
    /// ratio=128 has at most 1; total fits well under max_batch slots.
    /// Separate from `pbs.positions` (which holds the chunk's
    /// [batch_size] absolute positions and is read by the indexer).
    pub comp_positions: GpuTensor,
    /// Per-batch-position pos_array (Option B per-batch state).
    /// `[max_batch * (num_hidden_layers + 1) * POS_SLOTS_PER_LAYER]` i32
    /// stored as F32 — each batch row b occupies a stripe of
    /// `(L+1) * 3` slots matching the single-position `state.pos_array_device`
    /// layout. Populated once per chunk by `precompute_positions_batched`,
    /// then sub-viewed into `state.pos_array_device` during the per-position
    /// compressor fallback loop so existing per-position kernels read the
    /// right batch row.
    pub pos_array_device_batch: GpuTensor,
    /// Per-batch-position attn_state buffer (Option B per-batch state).
    /// `[max_batch * ATTN_STATE_SLOTS=10]` i32 stored as F32. Same
    /// swap-and-sub-view pattern as pos_array_device_batch.
    pub attn_state_buf_batch: GpuTensor,
    /// Batched MTP next-token ids `[max_batch]` stored as F32 (i32-in-F32
    /// slot pattern). Per-position next-token id, fed to the batched
    /// embedding lookup at the start of `mtp_forward_batched`.
    pub mtp_tokens_batch: GpuTensor,
    /// Batched MTP embed output `[max_batch, hidden]`. embedding_lookup_q8
    /// _batched writes one row per batch position from `mtp_tokens_batch`.
    pub mtp_embed_batch: GpuTensor,
    /// Batched MTP e_norm output `[max_batch, hidden]`. `mtp_enorm`
    /// applied to `mtp_embed_batch` per batch row.
    pub mtp_e_norm_batch: GpuTensor,
    /// Batched MTP h_norm output `[max_batch, hc_mult, hidden]`. `mtp_hnorm`
    /// applied to the main forward's `streams_batch` per (batch, HC row).
    /// Consumed by the per-HC `mtp_h_proj` GEMV that writes the new
    /// `streams_batch` contents at the start of the MTP layer block.
    pub mtp_h_norm_batch: GpuTensor,
    /// Batched MTP x_e output `[max_batch, hidden]`. `mtp_e_proj @ e_norm`
    /// per batch row; broadcast-added to every HC row of the rebuilt
    /// streams_batch.
    pub mtp_x_e_batch: GpuTensor,
}

/// Constructor-local owner for one PBS allocation. `GpuTensor` deliberately
/// has no `Drop`, so a struct literal with dozens of fallible allocations leaks
/// every earlier tensor if a later allocation fails. Each staged tensor returns
/// itself to the exact owning GPU unless `into_tensor` publishes it into the
/// completed `PrefillBatchScratch`.
struct StagedPrefillTensor {
    gpu: *mut Gpu,
    tensor: Option<GpuTensor>,
}

impl StagedPrefillTensor {
    fn new(gpu: &mut Gpu, tensor: GpuTensor) -> Self {
        Self {
            gpu,
            tensor: Some(tensor),
        }
    }

    fn into_tensor(mut self) -> GpuTensor {
        self.tensor
            .take()
            .expect("PBS staged tensor disarmed twice")
    }
}

impl Drop for StagedPrefillTensor {
    fn drop(&mut self) {
        let Some(tensor) = self.tensor.take() else {
            return;
        };
        // SAFETY: every staging value is created and destroyed inside
        // `PrefillBatchScratch::new`, before its caller's `&mut Gpu` can leave
        // scope. Staged values are not returned or stored anywhere.
        unsafe {
            let _ = (&mut *self.gpu).free_tensor(tensor);
        }
    }
}

impl PrefillBatchScratch {
    /// Exact requested allocation bytes for [`Self::new`]. This mirrors the
    /// constructor's complete tensor inventory and exists so a transactional
    /// multi-device loader can fail before allocating either owner. Values are
    /// HIP allocation requests (all are >= the pool's 256-byte minimum for the
    /// DS4 product shape); driver-internal page rounding is recorded separately
    /// from post-load `hipMemGetInfo` deltas.
    pub fn projected_allocation_bytes(
        cfg: &DeepseekV4Config,
        max_batch: usize,
    ) -> Result<usize, String> {
        let b = max_batch as u128;
        let hidden = cfg.hidden_size as u128;
        let q_rank = cfg.q_lora_rank as u128;
        let n_heads = cfg.num_attention_heads as u128;
        let head_dim = cfg.head_dim as u128;
        let hc = cfg.hc_mult as u128;
        let n_exp = cfg.n_routed_experts as u128;
        let topk = cfg.num_experts_per_tok as u128;
        let moe = cfg.moe_intermediate_size as u128;
        let idx_heads = cfg.index_n_heads as u128;
        let idx_dim = cfg.index_head_dim as u128;
        let idx_topk = cfg.index_topk as u128;
        let kv_dim = (cfg.num_key_value_heads * cfg.head_dim) as u128;
        let idx_score_capacity =
            crate::deepseek4::CompressorCapacityPlan::new(cfg.max_position_embeddings)?
                .active_rows() as u128;
        let block_m = 16u128;
        let m_total = b * topk + n_exp * block_m;
        let mut f32_elements = 0u128;
        let mut raw_bytes = 0u128;
        let mut add = |elements: u128| -> Result<(), String> {
            f32_elements = f32_elements
                .checked_add(elements)
                .ok_or_else(|| "PrefillBatchScratch projection overflow".to_string())?;
            Ok(())
        };

        add(head_dim)?; // q_head_ones
        add(b * hidden)?; // embed
        add(b * hc * hidden)?; // streams
        add(b)?; // tokens
        add(2 * b * hidden)?; // tmp + tmp_plain
        add(2 * b * q_rank)?; // q_lat + q_lat_rot
        add(b * n_heads * head_dim)?; // q
        add(b * kv_dim)?;
        add(b)?; // positions
        add(b * 24)?;
        add(2 * b * hc)?; // hc pre/post
        add(b * hc * hc)?;
        add(2 * b * hidden)?; // hc_x + attn_out
        add(b * hidden)?; // ffn_out
        add(b * hc * hidden)?; // streams_out
        add(b * head_dim * cfg.sliding_window as u128)?;
        add(b * head_dim * idx_topk)?;
        add(5 * b)?; // per-row attention/indexer counts
        add(2 * b * n_heads * head_dim)?; // raw attn + rotated
        add(2 * b * cfg.o_groups as u128 * cfg.o_lora_rank as u128)?;
        add(2 * b * hidden)?; // FFN x rotated/plain
        add(3 * b * moe)?; // shared gate/up/rotated
        add(b * n_exp)?;
        add(2 * b * topk)?; // top-k indices + weights
        add(3 * b * topk * moe)?; // routed gate/up/rotated
        add(b * topk * hidden)?; // routed down outputs
        add(b * idx_heads * idx_dim)?;
        add(b * idx_heads)?;
        add(b * idx_score_capacity)?;
        add(b * idx_topk)?;
        add(4 * b * head_dim)?; // main compressor kv + score
        add(4 * b * idx_dim)?; // indexer compressor kv + score
        add(b * head_dim)?; // F16 compressor-cache commit staging
        add(3 * b * topk)?; // sorted b/krank/expert
        add(n_exp + 1)?; // expert starts
        raw_bytes += n_exp * 4; // expert token counts
        raw_bytes += (n_exp + 1) * 4; // expert offsets
        raw_bytes += m_total * 4; // sorted slot index
        raw_bytes += (m_total / block_m) * 4; // expert tile ids
        raw_bytes += b * topk * 4; // inverse permutation
        add(m_total * 2 * moe)?; // grouped gate/up
        add(m_total * moe)?; // grouped activation
        add(m_total * hidden)?; // grouped down
        raw_bytes += 2 * b * hidden * 2; // two F16 activation staging slabs
        add(b)?; // compressor positions
        add(b * (cfg.num_hidden_layers as u128 + 1) * POS_SLOTS_PER_LAYER as u128)?;
        add(b * 10)?;
        add(b)?; // MTP tokens
        add(b * hidden)?; // MTP embed
        add(b * hidden)?; // MTP e norm
        add(b * hc * hidden)?; // MTP h norm
        add(b * hidden)?; // MTP projected embed
        let per_group_in = (cfg.num_attention_heads / cfg.o_groups) * cfg.head_dim;
        let max_dim = (cfg.o_groups * cfg.o_lora_rank)
            .max(cfg.hidden_size)
            .max(cfg.q_lora_rank)
            .max(cfg.o_groups * per_group_in) as u128;
        raw_bytes += b * max_dim * 2; // WMMA F16 staging

        let bytes = f32_elements
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_add(raw_bytes))
            .ok_or_else(|| "PrefillBatchScratch byte projection overflow".to_string())?;
        usize::try_from(bytes)
            .map_err(|_| "PrefillBatchScratch projection exceeds usize".to_string())
    }

    /// Allocate scratch for prefill chunks of up to `max_batch` tokens.
    /// Sizes track the DeepSeek V4 config's hidden_size / q_lora_rank /
    /// num_attention_heads × head_dim.
    pub fn new(gpu: &mut Gpu, cfg: &DeepseekV4Config, max_batch: usize) -> Result<Self, String> {
        let hidden = cfg.hidden_size;
        let q_rank = cfg.q_lora_rank;
        let n_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim;
        let hc_mult = cfg.hc_mult;
        let idx_score_capacity =
            crate::deepseek4::CompressorCapacityPlan::new(cfg.max_position_embeddings)?
                .active_rows();

        let alloc =
            |gpu: &mut Gpu, shape: &[usize], label: &str| -> Result<StagedPrefillTensor, String> {
                let tensor = gpu
                    .alloc_tensor(shape, DType::F32)
                    .map_err(|e| format!("PrefillBatchScratch alloc {label}: {e:?}"))?;
                Ok(StagedPrefillTensor::new(gpu, tensor))
            };
        let zeros = |gpu: &mut Gpu,
                     shape: &[usize],
                     dtype: DType,
                     label: &str|
         -> Result<StagedPrefillTensor, String> {
            let tensor = gpu
                .zeros(shape, dtype)
                .map_err(|e| format!("PrefillBatchScratch zeros {label}: {e:?}"))?;
            Ok(StagedPrefillTensor::new(gpu, tensor))
        };

        let ones_host = vec![1.0f32; head_dim];
        let q_head_ones = {
            let tensor = gpu
                .upload_f32(&ones_host, &[head_dim])
                .map_err(|e| format!("PrefillBatchScratch upload q_head_ones: {e:?}"))?;
            StagedPrefillTensor::new(gpu, tensor)
        };
        let kv_dim = cfg.num_key_value_heads * head_dim;
        let block_m = 16;
        let m_total_max = max_batch * cfg.num_experts_per_tok + cfg.n_routed_experts * block_m;
        let per_group_in = (n_heads / cfg.o_groups) * head_dim;
        let wmma_max_dim = (cfg.o_groups * cfg.o_lora_rank)
            .max(hidden)
            .max(cfg.q_lora_rank)
            .max(cfg.o_groups * per_group_in);

        macro_rules! alloc_f32 {
            ($shape:expr, $label:literal) => {
                alloc(gpu, $shape, $label)?
            };
        }
        macro_rules! zero_f32 {
            ($shape:expr, $label:literal) => {
                zeros(gpu, $shape, DType::F32, $label)?
            };
        }
        macro_rules! zero_raw {
            ($shape:expr, $label:literal) => {
                zeros(gpu, $shape, DType::Raw, $label)?
            };
        }

        // Allocate every tensor while each previous allocation remains armed.
        // Only the infallible publication block below disarms them.
        let embed_batch = alloc_f32!(&[max_batch, hidden], "embed_batch");
        let streams_batch = zero_f32!(&[max_batch, hc_mult, hidden], "streams_batch");
        let tokens = alloc_f32!(&[max_batch], "tokens");
        let tmp_batch = alloc_f32!(&[max_batch, hidden], "tmp_batch");
        let tmp_plain_batch = alloc_f32!(&[max_batch, hidden], "tmp_plain_batch");
        let q_lat_batch = alloc_f32!(&[max_batch, q_rank], "q_lat_batch");
        let q_lat_rot_batch = alloc_f32!(&[max_batch, q_rank], "q_lat_rot_batch");
        let q_batch = alloc_f32!(&[max_batch, n_heads, head_dim], "q_batch");
        let kv_batch = alloc_f32!(&[max_batch, kv_dim], "kv_batch");
        let positions = alloc_f32!(&[max_batch], "positions");
        let hc_c_batch = alloc_f32!(&[max_batch, 24], "hc_c_batch");
        let hc_pre_batch = alloc_f32!(&[max_batch, hc_mult], "hc_pre_batch");
        let hc_post_batch = alloc_f32!(&[max_batch, hc_mult], "hc_post_batch");
        let hc_comb_batch = alloc_f32!(&[max_batch, hc_mult, hc_mult], "hc_comb_batch");
        let hc_x_in_batch = alloc_f32!(&[max_batch, hidden], "hc_x_in_batch");
        let attn_out_batch = alloc_f32!(&[max_batch, hidden], "attn_out_batch");
        let ffn_out_batch = alloc_f32!(&[max_batch, hidden], "ffn_out_batch");
        let streams_out_batch = alloc_f32!(&[max_batch, hc_mult, hidden], "streams_out_batch");
        let swa_staged_batch = alloc_f32!(
            &[max_batch, head_dim, cfg.sliding_window],
            "swa_staged_batch"
        );
        let topk_staged_batch =
            alloc_f32!(&[max_batch, head_dim, cfg.index_topk], "topk_staged_batch");
        let n_valid_swa_arr = alloc_f32!(&[max_batch], "n_valid_swa_arr");
        let n_active_topk_arr = alloc_f32!(&[max_batch], "n_active_topk_arr");
        let n_compressed_4_arr = alloc_f32!(&[max_batch], "n_compressed_4_arr");
        let n_active_topk_4_arr = alloc_f32!(&[max_batch], "n_active_topk_4_arr");
        let n_active_topk_128_arr = alloc_f32!(&[max_batch], "n_active_topk_128_arr");
        let attn_out_raw_batch = alloc_f32!(&[max_batch, n_heads, head_dim], "attn_out_raw_batch");
        let attn_out_raw_rot_batch =
            alloc_f32!(&[max_batch, n_heads * head_dim], "attn_out_raw_rot_batch");
        let wo_a_out_batch = alloc_f32!(
            &[max_batch, cfg.o_groups, cfg.o_lora_rank],
            "wo_a_out_batch"
        );
        let wo_a_out_rot_batch = alloc_f32!(
            &[max_batch, cfg.o_groups * cfg.o_lora_rank],
            "wo_a_out_rot_batch"
        );
        let ffn_x_rot_batch = alloc_f32!(&[max_batch, hidden], "ffn_x_rot_batch");
        let ffn_x_plain_batch = alloc_f32!(&[max_batch, hidden], "ffn_x_plain_batch");
        let ffn_shared_gate_batch = alloc_f32!(
            &[max_batch, cfg.moe_intermediate_size],
            "ffn_shared_gate_batch"
        );
        let ffn_shared_up_batch = alloc_f32!(
            &[max_batch, cfg.moe_intermediate_size],
            "ffn_shared_up_batch"
        );
        let ffn_shared_rot_batch = alloc_f32!(
            &[max_batch, cfg.moe_intermediate_size],
            "ffn_shared_rot_batch"
        );
        let moe_scores_batch = alloc_f32!(&[max_batch, cfg.n_routed_experts], "moe_scores_batch");
        let moe_topk_indices_batch = alloc_f32!(
            &[max_batch, cfg.num_experts_per_tok],
            "moe_topk_indices_batch"
        );
        let moe_topk_weights_batch = alloc_f32!(
            &[max_batch, cfg.num_experts_per_tok],
            "moe_topk_weights_batch"
        );
        let moe_gate_batch = alloc_f32!(
            &[
                max_batch,
                cfg.num_experts_per_tok,
                cfg.moe_intermediate_size
            ],
            "moe_gate_batch"
        );
        let moe_up_batch = alloc_f32!(
            &[
                max_batch,
                cfg.num_experts_per_tok,
                cfg.moe_intermediate_size
            ],
            "moe_up_batch"
        );
        let moe_rot_batch = alloc_f32!(
            &[
                max_batch,
                cfg.num_experts_per_tok,
                cfg.moe_intermediate_size
            ],
            "moe_rot_batch"
        );
        let moe_down_expert_outputs = alloc_f32!(
            &[max_batch, cfg.num_experts_per_tok, hidden],
            "moe_down_expert_outputs"
        );
        let idx_q_batch = alloc_f32!(
            &[max_batch, cfg.index_n_heads, cfg.index_head_dim],
            "idx_q_batch"
        );
        let idx_w_batch = alloc_f32!(&[max_batch, cfg.index_n_heads], "idx_w_batch");
        let idx_scores_batch = alloc_f32!(&[max_batch, idx_score_capacity], "idx_scores_batch");
        let idx_topk_indices_batch =
            alloc_f32!(&[max_batch, cfg.index_topk], "idx_topk_indices_batch");
        let comp_main_kv_batch = alloc_f32!(&[max_batch, 2 * head_dim], "comp_main_kv_batch");
        let comp_main_score_batch = alloc_f32!(&[max_batch, 2 * head_dim], "comp_main_score_batch");
        let comp_idx_kv_batch =
            alloc_f32!(&[max_batch, 2 * cfg.index_head_dim], "comp_idx_kv_batch");
        let comp_idx_score_batch =
            alloc_f32!(&[max_batch, 2 * cfg.index_head_dim], "comp_idx_score_batch");
        let comp_cache_batch_f32 = alloc_f32!(&[max_batch, head_dim], "comp_cache_batch_f32");
        let moe_sorted_b = alloc_f32!(&[max_batch * cfg.num_experts_per_tok], "moe_sorted_b");
        let moe_sorted_krank =
            alloc_f32!(&[max_batch * cfg.num_experts_per_tok], "moe_sorted_krank");
        let moe_sorted_expert =
            alloc_f32!(&[max_batch * cfg.num_experts_per_tok], "moe_sorted_expert");
        let moe_expert_starts = alloc_f32!(&[cfg.n_routed_experts + 1], "moe_expert_starts");
        let moe_expert_token_counts =
            zero_raw!(&[cfg.n_routed_experts * 4], "moe_expert_token_counts");
        let moe_expert_offsets = zero_raw!(&[(cfg.n_routed_experts + 1) * 4], "moe_expert_offsets");
        let moe_sorted_slot_index = zero_raw!(&[m_total_max * 4], "moe_sorted_slot_index");
        let moe_expert_tile_ids = zero_raw!(&[(m_total_max / block_m) * 4], "moe_expert_tile_ids");
        let moe_inverse_perm = zero_raw!(
            &[max_batch * cfg.num_experts_per_tok * 4],
            "moe_inverse_perm"
        );
        let moe_y_gate_up_grouped = alloc_f32!(
            &[m_total_max, 2 * cfg.moe_intermediate_size],
            "moe_y_gate_up_grouped"
        );
        let moe_x_grouped = alloc_f32!(&[m_total_max, cfg.moe_intermediate_size], "moe_x_grouped");
        let moe_y_down_grouped = alloc_f32!(&[m_total_max, hidden], "moe_y_down_grouped");
        let mut tmp_batch_f16 = zero_raw!(&[max_batch * hidden * 2], "tmp_batch_f16");
        if let Some(tensor) = tmp_batch_f16.tensor.as_mut() {
            tensor.dtype = DType::F16;
            tensor.shape = vec![max_batch, hidden];
        }
        let mut tmp_plain_batch_f16 = zero_raw!(&[max_batch * hidden * 2], "tmp_plain_batch_f16");
        if let Some(tensor) = tmp_plain_batch_f16.tensor.as_mut() {
            tensor.dtype = DType::F16;
            tensor.shape = vec![max_batch, hidden];
        }
        let comp_positions = alloc_f32!(&[max_batch], "comp_positions");
        let pos_array_device_batch = alloc_f32!(
            &[max_batch * (cfg.num_hidden_layers + 1) * POS_SLOTS_PER_LAYER],
            "pos_array_device_batch"
        );
        let attn_state_buf_batch = alloc_f32!(&[max_batch * 10], "attn_state_buf_batch");
        let mtp_tokens_batch = alloc_f32!(&[max_batch], "mtp_tokens_batch");
        let mtp_embed_batch = alloc_f32!(&[max_batch, hidden], "mtp_embed_batch");
        let mtp_e_norm_batch = alloc_f32!(&[max_batch, hidden], "mtp_e_norm_batch");
        let mtp_h_norm_batch = alloc_f32!(&[max_batch, hc_mult, hidden], "mtp_h_norm_batch");
        let mtp_x_e_batch = alloc_f32!(&[max_batch, hidden], "mtp_x_e_batch");
        let mut wmma_x_scratch_f16 =
            zero_raw!(&[max_batch * wmma_max_dim * 2], "wmma_x_scratch_f16");
        if let Some(tensor) = wmma_x_scratch_f16.tensor.as_mut() {
            tensor.dtype = DType::F16;
            tensor.shape = vec![max_batch, wmma_max_dim];
        }

        Ok(Self {
            max_batch,
            idx_score_capacity,
            dspark_verify_pm4: std::collections::BTreeMap::new(),
            embed_batch: embed_batch.into_tensor(),
            streams_batch: streams_batch.into_tensor(),
            tokens: tokens.into_tensor(),
            tmp_batch: tmp_batch.into_tensor(),
            tmp_plain_batch: tmp_plain_batch.into_tensor(),
            q_lat_batch: q_lat_batch.into_tensor(),
            q_lat_rot_batch: q_lat_rot_batch.into_tensor(),
            q_batch: q_batch.into_tensor(),
            q_head_ones: q_head_ones.into_tensor(),
            kv_batch: kv_batch.into_tensor(),
            positions: positions.into_tensor(),
            hc_c_batch: hc_c_batch.into_tensor(),
            hc_pre_batch: hc_pre_batch.into_tensor(),
            hc_post_batch: hc_post_batch.into_tensor(),
            hc_comb_batch: hc_comb_batch.into_tensor(),
            hc_x_in_batch: hc_x_in_batch.into_tensor(),
            attn_out_batch: attn_out_batch.into_tensor(),
            ffn_out_batch: ffn_out_batch.into_tensor(),
            streams_out_batch: streams_out_batch.into_tensor(),
            swa_staged_batch: swa_staged_batch.into_tensor(),
            topk_staged_batch: topk_staged_batch.into_tensor(),
            n_valid_swa_arr: n_valid_swa_arr.into_tensor(),
            n_active_topk_arr: n_active_topk_arr.into_tensor(),
            n_compressed_4_arr: n_compressed_4_arr.into_tensor(),
            n_active_topk_4_arr: n_active_topk_4_arr.into_tensor(),
            n_active_topk_128_arr: n_active_topk_128_arr.into_tensor(),
            attn_out_raw_batch: attn_out_raw_batch.into_tensor(),
            attn_out_raw_rot_batch: attn_out_raw_rot_batch.into_tensor(),
            wo_a_out_batch: wo_a_out_batch.into_tensor(),
            wo_a_out_rot_batch: wo_a_out_rot_batch.into_tensor(),
            ffn_x_rot_batch: ffn_x_rot_batch.into_tensor(),
            ffn_x_plain_batch: ffn_x_plain_batch.into_tensor(),
            ffn_shared_gate_batch: ffn_shared_gate_batch.into_tensor(),
            ffn_shared_up_batch: ffn_shared_up_batch.into_tensor(),
            ffn_shared_rot_batch: ffn_shared_rot_batch.into_tensor(),
            moe_scores_batch: moe_scores_batch.into_tensor(),
            moe_topk_indices_batch: moe_topk_indices_batch.into_tensor(),
            moe_topk_weights_batch: moe_topk_weights_batch.into_tensor(),
            moe_gate_batch: moe_gate_batch.into_tensor(),
            moe_up_batch: moe_up_batch.into_tensor(),
            moe_rot_batch: moe_rot_batch.into_tensor(),
            moe_down_expert_outputs: moe_down_expert_outputs.into_tensor(),
            idx_q_batch: idx_q_batch.into_tensor(),
            idx_w_batch: idx_w_batch.into_tensor(),
            idx_scores_batch: idx_scores_batch.into_tensor(),
            idx_topk_indices_batch: idx_topk_indices_batch.into_tensor(),
            comp_main_kv_batch: comp_main_kv_batch.into_tensor(),
            comp_main_score_batch: comp_main_score_batch.into_tensor(),
            comp_idx_kv_batch: comp_idx_kv_batch.into_tensor(),
            comp_idx_score_batch: comp_idx_score_batch.into_tensor(),
            comp_cache_batch_f32: comp_cache_batch_f32.into_tensor(),
            moe_sorted_b: moe_sorted_b.into_tensor(),
            moe_sorted_krank: moe_sorted_krank.into_tensor(),
            moe_sorted_expert: moe_sorted_expert.into_tensor(),
            moe_expert_starts: moe_expert_starts.into_tensor(),
            moe_expert_token_counts: moe_expert_token_counts.into_tensor(),
            moe_expert_offsets: moe_expert_offsets.into_tensor(),
            moe_sorted_slot_index: moe_sorted_slot_index.into_tensor(),
            moe_expert_tile_ids: moe_expert_tile_ids.into_tensor(),
            moe_inverse_perm: moe_inverse_perm.into_tensor(),
            moe_y_gate_up_grouped: moe_y_gate_up_grouped.into_tensor(),
            moe_x_grouped: moe_x_grouped.into_tensor(),
            moe_y_down_grouped: moe_y_down_grouped.into_tensor(),
            tmp_batch_f16: tmp_batch_f16.into_tensor(),
            tmp_plain_batch_f16: tmp_plain_batch_f16.into_tensor(),
            comp_positions: comp_positions.into_tensor(),
            pos_array_device_batch: pos_array_device_batch.into_tensor(),
            attn_state_buf_batch: attn_state_buf_batch.into_tensor(),
            mtp_tokens_batch: mtp_tokens_batch.into_tensor(),
            mtp_embed_batch: mtp_embed_batch.into_tensor(),
            mtp_e_norm_batch: mtp_e_norm_batch.into_tensor(),
            mtp_h_norm_batch: mtp_h_norm_batch.into_tensor(),
            mtp_x_e_batch: mtp_x_e_batch.into_tensor(),
            wmma_x_scratch_f16: wmma_x_scratch_f16.into_tensor(),
        })
    }

    /// Resize the batched indexer score slab to a new stable row stride.
    /// Any retained verify route owns the old pointer and must be dropped
    /// before that allocation is returned to the pool.
    pub fn ensure_idx_score_capacity(
        &mut self,
        gpu: &mut Gpu,
        required_rows: usize,
    ) -> Result<bool, String> {
        if required_rows <= self.idx_score_capacity {
            return Ok(false);
        }
        let replacement = gpu
            .alloc_tensor(&[self.max_batch, required_rows], DType::F32)
            .map_err(|e| {
                format!(
                    "PrefillBatchScratch grow idx_scores_batch {} -> {required_rows}: {e:?}",
                    self.idx_score_capacity
                )
            })?;
        self.dspark_verify_pm4.clear();
        let old = std::mem::replace(&mut self.idx_scores_batch, replacement);
        gpu.free_tensor(old)
            .map_err(|e| format!("PrefillBatchScratch free old idx_scores_batch: {e:?}"))?;
        // This is a request-boundary geometry change, not a hot-loop free.
        // Keeping the old, potentially hundreds-of-MiB score slab in the pool
        // can make the cache's second admission check fail after the first one
        // passed. Return pooled blocks to HIP before committing the new stride.
        gpu.drain_pool();
        self.idx_score_capacity = required_rows;
        Ok(true)
    }

    /// Release every GPU buffer this prefill-batch scratch owns back to
    /// the pool. Consumes self. Called from `unload_model` on idle
    /// eviction / explicit unload so the ~50 sizeable per-chunk buffers
    /// (embed_batch, streams_batch, swa_staged_batch, MoE grouped
    /// scratches, …) actually return their VRAM rather than leaking.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        // Destroy retained verify queues/kernargs before returning any captured
        // PBS allocation to the pool.
        drop(self.dspark_verify_pm4);
        for t in [
            self.embed_batch,
            self.streams_batch,
            self.tokens,
            self.tmp_batch,
            self.tmp_plain_batch,
            self.q_lat_batch,
            self.q_lat_rot_batch,
            self.q_batch,
            self.q_head_ones,
            self.kv_batch,
            self.positions,
            self.hc_c_batch,
            self.hc_pre_batch,
            self.hc_post_batch,
            self.hc_comb_batch,
            self.hc_x_in_batch,
            self.attn_out_batch,
            self.ffn_out_batch,
            self.streams_out_batch,
            self.swa_staged_batch,
            self.topk_staged_batch,
            self.n_valid_swa_arr,
            self.n_active_topk_arr,
            self.n_compressed_4_arr,
            self.n_active_topk_4_arr,
            self.n_active_topk_128_arr,
            self.attn_out_raw_batch,
            self.attn_out_raw_rot_batch,
            self.wo_a_out_batch,
            self.wo_a_out_rot_batch,
            self.ffn_x_rot_batch,
            self.ffn_x_plain_batch,
            self.ffn_shared_gate_batch,
            self.ffn_shared_up_batch,
            self.ffn_shared_rot_batch,
            self.moe_scores_batch,
            self.moe_topk_indices_batch,
            self.moe_topk_weights_batch,
            self.moe_gate_batch,
            self.moe_up_batch,
            self.moe_rot_batch,
            self.moe_down_expert_outputs,
            self.idx_q_batch,
            self.idx_w_batch,
            self.idx_scores_batch,
            self.idx_topk_indices_batch,
            self.comp_main_kv_batch,
            self.comp_main_score_batch,
            self.comp_idx_kv_batch,
            self.comp_idx_score_batch,
            self.comp_cache_batch_f32,
            self.moe_sorted_b,
            self.moe_sorted_krank,
            self.moe_sorted_expert,
            self.moe_expert_starts,
            self.moe_expert_token_counts,
            self.moe_expert_offsets,
            self.moe_sorted_slot_index,
            self.moe_expert_tile_ids,
            self.moe_inverse_perm,
            self.moe_y_gate_up_grouped,
            self.moe_x_grouped,
            self.moe_y_down_grouped,
            self.tmp_batch_f16,
            self.tmp_plain_batch_f16,
            self.wmma_x_scratch_f16,
            self.comp_positions,
            self.pos_array_device_batch,
            self.attn_state_buf_batch,
            self.mtp_tokens_batch,
            self.mtp_embed_batch,
            self.mtp_e_norm_batch,
            self.mtp_h_norm_batch,
            self.mtp_x_e_batch,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}

/// A graph capture or replay recording is active: device copies must be
/// typed kernels, not raw D2D nodes.
#[inline]
pub(crate) fn dspark_requires_typed_device_copy(gpu: &Gpu) -> bool {
    gpu.replay.is_recording() || gpu.graphs.capture_mode || gpu.flags.force_blob_path
}

/// Batched-aware twin of `final_norm_and_head` — extracts the LAST
/// position's residual streams from pbs.streams_batch and runs the
/// existing per-position head pipeline against it.
///
/// Phase B2 chunk forward only needs logits at the last position
/// (matches qwen35::forward_prefill_batch's contract). All upstream
/// state.* scratch fields used by `final_norm_and_head` are sized for
/// one position and get reused unchanged.
///
/// Returns the logits at the last position. Caller is responsible for
/// any sampler integration.
#[allow(dead_code)]
pub fn final_norm_and_head_last_batched(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    batch_size: usize,
) -> Result<Vec<f32>, String> {
    if batch_size == 0 {
        return Err("final_norm_and_head_last_batched: empty batch".to_string());
    }
    let stream_len = cfg.hc_mult * cfg.hidden_size;
    let last_streams = pbs
        .streams_batch
        .sub_offset((batch_size - 1) * stream_len, stream_len);
    // `forward_prefill_batch_chunked` already captured every head input as
    // one active matrix. Do not append the last row a second time here.
    final_norm_and_head_impl(cfg, weights, state, gpu, false, Some(&last_streams))?;
    let logits_tensor = state
        .logits
        .as_ref()
        .ok_or_else(|| "logits not allocated".to_string())?;
    gpu.download_f32(logits_tensor)
        .map_err(|e| format!("download logits: {e:?}"))
}

/// Run final_norm + head on EVERY position of the batched chunk.
///
/// `final_norm_and_head_last_batched` only produces logits for the last
/// position (the only position whose token is sampled in normal prefill).
/// Speculative-decode verification needs per-position logits so each
/// draft can be compared against the verifier's preferred token —
/// that's what this helper provides.
///
/// Cost: K invocations of the per-position final_norm_and_head pipeline
/// (head HC + RMSNorm + rotate + lm_head GEMV + d2h). The lm_head is
/// [vocab=129280, hidden=4096]; per-position it's well under 5 ms on
/// gfx1151, so K=8 takes <40 ms. Acceptable for spec-decode windows.
///
/// Returns Vec<Vec<f32>> of length `batch_size`, each inner Vec sized
/// `vocab_size`.
pub fn final_norm_and_head_all_batched(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    batch_size: usize,
) -> Result<Vec<Vec<f32>>, String> {
    if batch_size == 0 {
        return Err("final_norm_and_head_all_batched: empty batch".to_string());
    }
    let stream_len = cfg.hc_mult * cfg.hidden_size;
    let vocab = cfg.vocab_size;

    // An MQ4 head needs a per-position FWHT rotation of the normed input that
    // we don't stage into the batch buffer — keep the scalar per-position loop
    // for that case. Q8/F16/F32 heads (this build's head is Q8) take the
    // batched path below.
    let head_needs_fwht = {
        let head = weights
            .head
            .as_ref()
            .ok_or_else(|| "head not uploaded".to_string())?;
        weight_needs_fwht(head)
    };
    // Opt-out: HIPFIRE_DEEPSEEK4_BATCH_HEAD=0 forces the legacy per-position
    // scalar loop — used for A/B measurement and as a safety fallback.
    let batch_head = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_BATCH_HEAD")
        .map(|s| s != "0")
        .unwrap_or(true);
    if head_needs_fwht || !batch_head {
        let mut all_logits: Vec<Vec<f32>> = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let streams_i = pbs.streams_batch.sub_offset(i * stream_len, stream_len);
            final_norm_and_head_impl(cfg, weights, state, gpu, true, Some(&streams_i))?;
            let logits_tensor = state
                .logits
                .as_ref()
                .ok_or_else(|| "logits not allocated".to_string())?;
            let logits_host = gpu
                .download_f32(logits_tensor)
                .map_err(|e| format!("download logits @pos {i}: {e:?}"))?;
            all_logits.push(logits_host);
        }
        return Ok(all_logits);
    }

    // ── Batched lm_head path ──────────────────────────────────────────────
    // Fill `state.head_logits_batch` ([K, vocab]) with one weight-read GEMV,
    // then download + split per position.
    compute_batched_head_logits(cfg, weights, state, pbs, gpu, batch_size)?;
    let logits_batch = state.head_logits_batch.as_ref().unwrap();
    let flat = gpu
        .download_f32(logits_batch)
        .map_err(|e| format!("download batched logits: {e:?}"))?;
    let mut all_logits: Vec<Vec<f32>> = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        all_logits.push(flat[i * vocab..(i + 1) * vocab].to_vec());
    }
    Ok(all_logits)
}

/// Fill `state.head_logits_batch` (`[batch_size, vocab]`, F32, cached on state)
/// with the batched lm_head over `batch_size` verify positions: the cheap
/// per-position head-HC + RMSNorm prologue staged into a `[K, hidden]` buffer,
/// then ONE weight-bandwidth-bound GEMV reading the ~565 MB Q8 lm_head a single
/// time for all K. Callers choose how to reduce the logits: download them
/// (`final_norm_and_head_all_batched`) or argmax on-GPU
/// (`final_norm_and_argmax_all_batched`). Assumes the Q8/F16/F32 batched head
/// (the FWHT fallback stays in the callers' per-position loop).
fn compute_batched_head_logits(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    batch_size: usize,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    if state
        .head_norm_batch
        .as_ref()
        .map(|t| t.numel() != batch_size * hidden)
        .unwrap_or(true)
    {
        state.head_norm_batch = Some(
            gpu.alloc_tensor(&[batch_size, hidden], DType::F32)
                .map_err(|e| format!("alloc head_norm_batch: {e:?}"))?,
        );
    }
    if state
        .head_x_f16
        .as_ref()
        .map(|t| t.numel() != batch_size * hidden)
        .unwrap_or(true)
    {
        state.head_x_f16 = Some(
            gpu.alloc_tensor(&[batch_size * hidden], DType::F16)
                .map_err(|e| format!("alloc head_x_f16: {e:?}"))?,
        );
    }
    if state
        .head_logits_batch
        .as_ref()
        .map(|t| t.numel() != batch_size * vocab)
        .unwrap_or(true)
    {
        state.head_logits_batch = Some(
            gpu.alloc_tensor(&[batch_size, vocab], DType::F32)
                .map_err(|e| format!("alloc head_logits_batch: {e:?}"))?,
        );
    }
    crate::rows_program::run_head_rows(cfg, weights, state, gpu, pbs, batch_size, true)
}

/// Calibration-only head prologue for batched prefill. Normal serving needs
/// logits only for the last prompt position, but the Hessian needs the
/// pre-lm_head activation for every row. Stage those rows without issuing the
/// enormous batched vocabulary projection, then download the completed matrix
/// once through the standard P3 activation dumper.
fn capture_head_norm_batched(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    batch_size: usize,
) -> Result<(), String> {
    if !dense_activation_dump_enabled()? {
        return Ok(());
    }
    let hidden = cfg.hidden_size;
    if state
        .head_norm_batch
        .as_ref()
        .map(|tensor| tensor.numel() < batch_size * hidden)
        .unwrap_or(true)
    {
        state.head_norm_batch = Some(
            gpu.alloc_tensor(&[batch_size, hidden], DType::F32)
                .map_err(|error| format!("alloc calibration head_norm_batch: {error:?}"))?,
        );
    }
    crate::rows_program::run_head_rows(cfg, weights, state, gpu, pbs, batch_size, false)?;
    let active = state
        .head_norm_batch
        .as_ref()
        .unwrap()
        .sub_offset(0, batch_size * hidden);
    dump_dense_activation_if_enabled(gpu, "head.weight", &active, hidden)
}

/// On-GPU greedy twin of [`final_norm_and_head_all_batched`]: runs the same
/// batched lm_head, then argmaxes each row ON GPU and downloads only the
/// `batch_size` token ids — avoiding the `batch_size × vocab` (~5 MB) logits
/// d2h the host-argmax path pays every spec window. Used by the DSpark verify.
pub fn final_norm_and_argmax_all_batched(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    batch_size: usize,
) -> Result<Vec<u32>, String> {
    if batch_size == 0 {
        return Err("final_norm_and_argmax_all_batched: empty batch".to_string());
    }
    let vocab = cfg.vocab_size;
    let stream_len = cfg.hc_mult * cfg.hidden_size;

    // FWHT-head fallback: no batched GEMV path — argmax each position on GPU.
    let head_needs_fwht = {
        let head = weights
            .head
            .as_ref()
            .ok_or_else(|| "head not uploaded".to_string())?;
        weight_needs_fwht(head)
    };
    let batch_head = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_BATCH_HEAD")
        .map(|s| s != "0")
        .unwrap_or(true);
    if head_needs_fwht || !batch_head {
        let mut ids: Vec<u32> = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let streams_i = pbs.streams_batch.sub_offset(i * stream_len, stream_len);
            final_norm_and_head_impl(cfg, weights, state, gpu, true, Some(&streams_i))?;
            let logits_tensor = state
                .logits
                .as_ref()
                .ok_or_else(|| "logits not allocated".to_string())?;
            let idx = gpu
                .argmax_f32(logits_tensor, vocab)
                .map_err(|e| format!("argmax @pos {i}: {e:?}"))?;
            ids.push(idx);
        }
        return Ok(ids);
    }

    // Batched path: fill logits on GPU, argmax on GPU, download only the ids.
    compute_batched_head_logits(cfg, weights, state, pbs, gpu, batch_size)?;
    let logits_batch = state.head_logits_batch.as_ref().unwrap();
    let argmax_buf = gpu
        .alloc_tensor(&[batch_size], DType::F32)
        .map_err(|e| format!("alloc argmax buf: {e:?}"))?;
    gpu.argmax_f32_batched(logits_batch, &argmax_buf, vocab, batch_size)
        .map_err(|e| format!("argmax_f32_batched: {e:?}"))?;
    let mut host_idx = vec![0i32; batch_size];
    {
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(host_idx.as_mut_ptr() as *mut u8, batch_size * 4)
        };
        gpu.hip
            .memcpy_dtoh(bytes, &argmax_buf.buf)
            .map_err(|e| format!("download argmax ids: {e:?}"))?;
    }
    let _ = gpu.free_tensor(argmax_buf);
    Ok(host_idx.into_iter().map(|i| i as u32).collect())
}

/// LAZY greedy twin of [`final_norm_and_argmax_all_batched`]: per position,
/// final-norm + head → logits → GPU argmax, with a prefix stop against the
/// drafted `block` (once a position's argmax != its draft `block[i+1]`, all later
/// positions reject — skip their heads). Byte-identical committed output vs the
/// eager argmax (the caller's `accept_greedy_prefix` reads only up to the
/// mismatch), just fewer (152k-vocab) head GEMVs. Rejected picks are padded.
pub fn final_norm_and_argmax_all_batched_lazy(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    block: &[u32],
) -> Result<Vec<u32>, String> {
    let n = block.len();
    if n == 0 {
        return Err("final_norm_and_argmax_all_batched_lazy: empty batch".to_string());
    }
    let vocab = cfg.vocab_size;
    let stream_len = cfg.hc_mult * cfg.hidden_size;
    let mut ids: Vec<u32> = Vec::with_capacity(n);
    for i in 0..n {
        let streams_i = pbs.streams_batch.sub_offset(i * stream_len, stream_len);
        final_norm_and_head_impl(cfg, weights, state, gpu, true, Some(&streams_i))?;
        let logits_tensor = state
            .logits
            .as_ref()
            .ok_or_else(|| "logits not allocated".to_string())?;
        let tok = gpu
            .argmax_f32(logits_tensor, vocab)
            .map_err(|e| format!("argmax @pos {i}: {e:?}"))?;
        ids.push(tok);
        if i + 1 < n && block[i + 1] != tok {
            while ids.len() < n {
                ids.push(u32::MAX);
            }
            break;
        }
    }
    Ok(ids)
}

/// Sampled + LAZY twin of [`final_norm_and_argmax_all_batched`] for temp>0 DSpark
/// verify. Mirrors the greedy sibling's architecture: **one** batched lm-head
/// read (`compute_batched_head_logits` → resident `[n × vocab]` logits) followed
/// by a single fused on-GPU kernel (`sample_accept_lazy_f32`) that samples every
/// position on the device, threads the xorshift32 RNG, and LAZILY early-exits on
/// the first token that differs from its drafted successor (`block[i+1]`) —
/// padding the rejected tail with `u32::MAX` (the caller's `accept_greedy_prefix`
/// reads only up to the mismatch). This replaces the previous per-position host
/// loop (which re-read the ~565 MB lm-head τ times and paid an 8-byte D2H +
/// stream sync per position) with one head read + one `(n+1)×4`-byte D2H; the
/// sampled token sequence stays distribution-identical to the AR sampler.
///
/// FWHT heads (no batched GEMV path) and `HIPFIRE_DEEPSEEK4_BATCH_HEAD=0` fall
/// back to the original per-position `sample_top_p_pf` loop. `rng` advances per
/// draw in both paths.
#[allow(clippy::too_many_arguments)]
pub fn final_norm_and_sample_all_batched_lazy(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    block: &[u32],
    temp: f32,
    top_p: f32,
    top_k: usize,
    // CACTUS acceptance-boost δ (0 = lossless/byte-identical). Plumbed from
    // `Speculator::set_sampling` via the drafter + `verify_block_sampled_capture_gpu`.
    // Deliberately lossy at δ>0 (trades distribution fidelity for higher τ).
    cactus_delta: f32,
    rng: &mut u32,
    result_buf: &GpuTensor,
    repeat_buf: &GpuTensor,
) -> Result<Vec<u32>, String> {
    let n = block.len();
    if n == 0 {
        return Err("final_norm_and_sample_all_batched_lazy: empty batch".to_string());
    }
    let vocab = cfg.vocab_size;
    let stream_len = cfg.hc_mult * cfg.hidden_size;
    let top_p_eff = if top_p > 0.0 { top_p.min(1.0) } else { 1.0 };
    let top_k_opt = if top_k > 0 { Some(top_k as u32) } else { None };

    // FWHT-head fallback (no batched GEMV path) or batched head disabled: the
    // original per-position head + host `sample_top_p_pf` loop. Mirrors the
    // `weight_needs_fwht || !batch_head` branch in `final_norm_and_argmax_all_batched`.
    let head_needs_fwht = {
        let head = weights
            .head
            .as_ref()
            .ok_or_else(|| "head not uploaded".to_string())?;
        weight_needs_fwht(head)
    };
    let batch_head = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_BATCH_HEAD")
        .map(|s| s != "0")
        .unwrap_or(true);
    if head_needs_fwht || !batch_head {
        let mut ids: Vec<u32> = Vec::with_capacity(n);
        for i in 0..n {
            let streams_i = pbs.streams_batch.sub_offset(i * stream_len, stream_len);
            final_norm_and_head_impl(cfg, weights, state, gpu, true, Some(&streams_i))?;
            let logits_tensor = state
                .logits
                .as_ref()
                .ok_or_else(|| "logits not allocated".to_string())?;
            let (tok, new_rng) = gpu
                .sample_top_p_pf(
                    logits_tensor,
                    result_buf,
                    repeat_buf,
                    vocab,
                    temp,
                    top_p_eff,
                    *rng,
                    0,
                    1.0,
                    0.0,
                    0.0,
                    top_k_opt,
                    None,
                )
                .map_err(|e| format!("sample_top_p_pf @pos {i}: {e:?}"))?;
            *rng = new_rng;
            ids.push(tok);
            // LAZY prefix stop: draft for position i is block[i+1] (i<n-1).
            if i + 1 < n && block[i + 1] != tok {
                while ids.len() < n {
                    ids.push(u32::MAX);
                }
                break;
            }
        }
        return Ok(ids);
    }

    // Batched path: one lm-head read fills resident `[n × vocab]` logits, then
    // the fused on-GPU sample+accept kernel does the lazy per-position draw and
    // early-exit entirely on the device (one `(n+1)×4`-byte D2H). `result_buf` /
    // `repeat_buf` are unused here (they feed only the FWHT fallback above).
    compute_batched_head_logits(cfg, weights, state, pbs, gpu, n)?;
    let logits_batch = state.head_logits_batch.as_ref().unwrap();

    // Upload the drafted block (n u32) and allocate the (n+1)-u32 result buffer.
    // Tiny per-window allocs, matching `final_norm_and_argmax_all_batched`'s
    // per-call `argmax_buf`. Stored in F32 tensors (4 bytes/elem = one u32).
    let draft_buf = gpu
        .alloc_tensor(&[n], DType::F32)
        .map_err(|e| format!("alloc draft buf: {e:?}"))?;
    let draft_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(block.as_ptr() as *const u8, n * 4) };
    gpu.memcpy_htod_auto(&draft_buf.buf, draft_bytes)
        .map_err(|e| format!("upload draft block: {e:?}"))?;
    let out_buf = gpu
        .alloc_tensor(&[n + 1], DType::F32)
        .map_err(|e| format!("alloc sample-accept out: {e:?}"))?;

    let (ids, new_rng) = gpu
        .sample_accept_lazy_f32(
            logits_batch,
            &draft_buf,
            &out_buf,
            n,
            vocab,
            temp,
            top_p_eff,
            top_k_opt,
            *rng,
            cactus_delta,
        )
        .map_err(|e| format!("sample_accept_lazy_f32: {e:?}"))?;
    *rng = new_rng;
    let _ = gpu.free_tensor(draft_buf);
    let _ = gpu.free_tensor(out_buf);
    Ok(ids)
}

/// Batched-prefill entry point for DeepSeek V4.
///
/// Processes the `tokens` slice starting at absolute KV position
/// `start_pos`. Returns the logits at the LAST position only (matches
/// the qwen35 forward_prefill_batch contract).
///
/// **Phase B status (2026-05-18):** scaffold. The body falls back to a
/// per-token `decode_step` loop — byte-identical to the existing
/// sequential prefill semantics. Phase B2 will replace the loop body
/// with a `forward_prefill_batch_chunk` call that processes `max_batch`
/// positions at once using the Phase A batched kernels (A1: SWA-topK,
/// A2: SWA, A3: indexer top-K, A5: HC mix).
///
/// The entry-point shape is finalised now so callers (eval harnesses,
/// daemon, eventual prefill API) can wire against the stable signature
/// while the inner batched body grows behind it. Production callers
/// should use `forward_prefill_batch_chunked` for actual batched prefill.
pub fn forward_prefill_batch(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: u32,
    _scratch: &mut PrefillBatchScratch,
) -> Result<Vec<f32>, String> {
    if tokens.is_empty() {
        return Err("forward_prefill_batch: empty tokens slice".to_string());
    }
    // Per-token fallback until forward_prefill_batch_chunk is end-to-end.
    let mut last_logits = Vec::new();
    for (i, &tok) in tokens.iter().enumerate() {
        last_logits = decode_step(cfg, weights, state, gpu, tok, start_pos + i as u32)?;
    }
    Ok(last_logits)
}

/// Single-chunk batched forward pass — Phase B2 work in progress.
///
/// Processes a chunk of `tokens.len()` ≤ `pbs.max_batch` positions
/// starting at `start_pos` through one batched forward. Mirrors
/// `decode_step` but with each per-layer stage swapped for its batched
/// twin. Returns the logits at the LAST position only.
///
/// Currently a partial wiring — runs through the stages that have
/// shipped batched bodies (embedding, HC stream init, q_lora,
/// kv_joint, tail RoPE) then errors out at the first unbatched stage
/// (the indexer + mixed attention dispatch). Each subsequent commit
/// replaces one error path with a real batched body until the chunk
/// runs end-to-end.
///
/// **Stages and their status (2026-05-18):**
///   ✓ token-ids upload → pbs.tokens
///   ✓ positions upload → pbs.positions
///   ✓ batched embedding lookup → pbs.embed_batch
///   ✓ HC streams broadcast init → pbs.streams_batch
///   ✓ per-layer q_lora_batched (Phase B2)
///   ✓ per-layer kv_joint_batched (Phase B2)
///   ✓ per-layer apply_tail_rope_batched (Phase B2)
///   ☐ per-layer mhc_pre_batched
///   ☐ per-layer compressor (loop sequential per A4 deferral)
///   ☐ per-layer indexer_forward_batched
///   ☐ per-layer mixed attention (wire deepseek4_attn_swa_topk_batched)
///   ☐ per-layer wo projection (gemv_auto_batched, two-stage O-LoRA)
///   ☐ per-layer hc_attn_mix_batched
///   ☐ per-layer ffn_routed_batched + hc_ffn_mix_batched
///   ☐ final_norm + lm_head (last position only)
///
/// Until all stages are wired this function returns an error from the
/// first unimplemented stage; callers should keep dispatching through
/// `forward_prefill_batch`'s per-token fallback for now.
/// Upload every host-varying input consumed by a batched DS4 forward.
///
/// This is deliberately a DS4-owned boundary: verify graph / retained replay
/// callers invoke it before capture or replay, while ordinary prefill keeps the
/// same public `forward_prefill_batch_chunk` contract below. No H2D copy is
/// permitted in the capture-safe body.
pub(crate) fn upload_prefill_batch_inputs(
    cfg: &DeepseekV4Config,
    gpu: &mut Gpu,
    pbs: &PrefillBatchScratch,
    tokens: &[u32],
    start_pos: u32,
) -> Result<(), String> {
    let n = tokens.len();
    if n == 0 || n > pbs.max_batch {
        return Err(format!(
            "upload_prefill_batch_inputs: invalid batch {n} (max {})",
            pbs.max_batch
        ));
    }
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(
            gpu.hip
                .stream_create()
                .map_err(|e| format!("stream_create for async htod: {e:?}"))?,
        );
    }

    let token_ids_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let token_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(token_ids_host.as_ptr() as *const u8, n * 4) };
    gpu.memcpy_htod_auto(&pbs.tokens.buf, token_bytes)
        .map_err(|e| format!("htod tokens: {e:?}"))?;

    let positions_host: Vec<i32> = (0..n).map(|i| start_pos as i32 + i as i32).collect();
    let positions_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(positions_host.as_ptr() as *const u8, n * 4) };
    gpu.memcpy_htod_auto(&pbs.positions.buf, positions_bytes)
        .map_err(|e| format!("htod positions: {e:?}"))?;

    let win = cfg.sliding_window;
    let n_valid_host: Vec<i32> = (0..n)
        .map(|b| ((start_pos as usize + b + 1).min(win)) as i32)
        .collect();
    let max_compressed = pbs.idx_score_capacity;
    let n_compressed_4_host: Vec<i32> = (0..n)
        .map(|b| ((start_pos as usize + b + 1) / 4).min(max_compressed) as i32)
        .collect();
    let n_active_4_host: Vec<i32> = n_compressed_4_host
        .iter()
        .map(|&v| cfg.index_topk.min(v as usize) as i32)
        .collect();
    let n_active_128_host: Vec<i32> = (0..n)
        .map(|b| {
            cfg.index_topk
                .min(((start_pos as usize + b + 1) / 128).min(max_compressed)) as i32
        })
        .collect();
    let as_bytes = |values: &[i32]| unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4)
    };
    gpu.memcpy_htod_auto(&pbs.n_valid_swa_arr.buf, as_bytes(&n_valid_host))
        .map_err(|e| format!("htod n_valid_swa_arr (chunk-level): {e:?}"))?;
    gpu.memcpy_htod_auto(&pbs.n_compressed_4_arr.buf, as_bytes(&n_compressed_4_host))
        .map_err(|e| format!("htod n_compressed_4_arr: {e:?}"))?;
    gpu.memcpy_htod_auto(&pbs.n_active_topk_4_arr.buf, as_bytes(&n_active_4_host))
        .map_err(|e| format!("htod n_active_topk_4_arr: {e:?}"))?;
    gpu.memcpy_htod_auto(&pbs.n_active_topk_128_arr.buf, as_bytes(&n_active_128_host))
        .map_err(|e| format!("htod n_active_topk_128_arr: {e:?}"))?;

    // These stripes were formerly rebuilt inside every mixed layer's
    // compressor fallback. They depend only on (start_pos, B), so one upload
    // is both faster for direct prefill and mandatory for capture/replay.
    precompute_positions_batched(cfg, pbs, gpu, start_pos, n)?;
    precompute_attn_state_batched(cfg, pbs, gpu, start_pos, n)?;
    Ok(())
}

#[allow(dead_code)]
pub fn forward_prefill_batch_chunk(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    pbs: &mut PrefillBatchScratch,
    tokens: &[u32],
    start_pos: u32,
) -> Result<(), String> {
    ensure_request_capacity(
        cfg,
        state,
        gpu,
        pbs,
        (start_pos as usize).saturating_add(tokens.len()),
    )?;
    upload_prefill_batch_inputs(cfg, gpu, pbs, tokens, start_pos)?;
    forward_prefill_batch_chunk_impl(cfg, weights, state, gpu, pbs, tokens, start_pos, false)
}

/// Capture-safe DS4 verify body. The caller must first invoke
/// [`upload_prefill_batch_inputs`]. `capture_safe=true` selects only
/// device-buffer-driven position/count paths and a fixed compressor node set.
pub(crate) fn forward_prefill_batch_chunk_preuploaded(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    pbs: &PrefillBatchScratch,
    tokens: &[u32],
    start_pos: u32,
) -> Result<(), String> {
    let required_tokens = (start_pos as usize).saturating_add(tokens.len());
    let required_rows = state.compressor_capacity.rows_for_tokens(required_tokens)?;
    if required_rows > state.compressor_capacity.active_rows()
        || required_tokens > state.compressor_capacity.prepared_tokens()
        || required_rows > pbs.idx_score_capacity
    {
        return Err(format!(
            "capture-safe DS4 verify requires preflight: rows={required_rows}, state={}, pbs={}",
            state.compressor_capacity.active_rows(),
            pbs.idx_score_capacity
        ));
    }
    forward_prefill_batch_chunk_impl(cfg, weights, state, gpu, pbs, tokens, start_pos, true)
}

#[allow(clippy::too_many_arguments)]
fn forward_prefill_batch_chunk_impl(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    pbs: &PrefillBatchScratch,
    tokens: &[u32],
    start_pos: u32,
    capture_safe: bool,
) -> Result<(), String> {
    let n = tokens.len();
    if n == 0 {
        return Err("forward_prefill_batch_chunk: empty tokens".to_string());
    }
    if n > pbs.max_batch {
        return Err(format!(
            "forward_prefill_batch_chunk: chunk size {n} > max_batch {}",
            pbs.max_batch
        ));
    }

    // Phase C: ensure we have an active stream so all the small h2d
    // uploads in this chunk forward go async-on-stream via
    // `memcpy_htod_auto`. Subsequent kernels submitted to the same
    // stream order naturally — no host blocking on each tiny upload.
    if gpu.active_stream.is_none() {
        let new_stream = gpu
            .hip
            .stream_create()
            .map_err(|e| format!("stream_create for async htod: {e:?}"))?;
        gpu.active_stream = Some(new_stream);
    }

    let window = RowsWindow::new(pbs, start_pos, n, tokens, capture_safe);
    let embed = rows_program::embed_rows_program(cfg, weights, window)?;
    rows_program::run_rows(&embed, cfg, gpu, window, Observed::Embed)?;
    drop(embed);
    let slots = RowSlots::new(cfg, pbs, n);
    for layer_idx in 0..cfg.num_hidden_layers {
        rows_program::ensure_rows_layer(cfg, weights, state, gpu, layer_idx)?;
        // DSpark priming / verify captures the stream mean of its target
        // layers (`state.dspark_capture_active`).
        let capture_slot = state
            .dspark_capture_active
            .then(|| {
                state
                    .dspark_target_layers
                    .iter()
                    .position(|&l| l == layer_idx)
            })
            .flatten();
        if capture_slot.is_some() {
            rows_program::ensure_stream_mean_capture(cfg, state, gpu, pbs.max_batch)?;
        }
        let program = rows_program::rows_layer_program(
            cfg,
            weights,
            state,
            gpu,
            layer_idx,
            window,
            &slots,
            capture_slot,
        )?;
        let observed = Observed::Layer(layer_idx, weights.resolve_layer(layer_idx));
        rows_program::run_rows(&program, cfg, gpu, window, observed)?;
    }
    Ok(())
}

/// Assemble the captured per-target-layer hidden states for batch position
/// `batch_pos` into a contiguous `[n_target_layers * hidden]` tensor — the
/// `main_hidden` input to [`dspark_forward`]. Target layers appear in
/// ascending capture-slot order (slot 0, 1, … = `dspark_target_layers[0..]`).
///
/// In the `[max_batch, n_target_layers, hidden]` layout the slice
/// `dspark_caps[batch_pos, 0..n_targets, :]` is already contiguous, so this
/// is a single d2d copy into the reused `state.dspark_main_hidden` buffer.
pub fn dspark_assemble_main_hidden<'a>(
    state: &'a mut DeepseekV4State,
    gpu: &mut Gpu,
    cfg: &DeepseekV4Config,
    batch_pos: usize,
) -> Result<&'a GpuTensor, String> {
    let hidden = cfg.hidden_size;
    let n_targets = state.dspark_target_layers.len();
    if n_targets == 0 {
        return Err("dspark_assemble_main_hidden: no target layers set".to_string());
    }
    let caps = state
        .dspark_caps
        .as_ref()
        .ok_or("dspark_assemble_main_hidden: no captures (capture inactive?)")?;
    let total = n_targets * hidden;

    if state.dspark_main_hidden.is_none() {
        state.dspark_main_hidden = Some(
            gpu.alloc_tensor(&[total], DType::F32)
                .map_err(|e| format!("dspark_assemble alloc main_hidden: {e:?}"))?,
        );
    }
    let dst = state.dspark_main_hidden.as_ref().unwrap();
    let src_off = batch_pos * n_targets * hidden * 4;
    gpu.memcpy_dtod_at_auto(&dst.buf, 0, &caps.buf, src_off, total * 4)
        .map_err(|e| format!("dspark_assemble d2d: {e:?}"))?;
    Ok(state.dspark_main_hidden.as_ref().unwrap())
}

/// Top-level batched-prefill driver — chunks the prompt by max_batch
/// and dispatches each chunk through `forward_prefill_batch_chunk`.
///
/// Returns logits at the LAST position only (matches the qwen35
/// contract). Falls back to per-token decode_step if any chunk fails
/// (typically because a layer's compress_ratio path isn't yet wired —
/// pure-SWA-only for now, mixed-attention layers error out).
///
/// **Phase B2 status (2026-05-18):** the chunk-forward path handles
/// pure-SWA layers (compress_ratio == 0) end-to-end including the
/// MoE FFN; mixed-attention layers (compress_ratio > 0) still bail
/// at the indexer chain. Until mixed is wired, this function falls
/// back to per-token decode_step for any chunk that contains a
/// mixed-attention layer (i.e. all DeepSeek V4 prompts except the trivial
/// case where all 43 layers are dense, which doesn't exist).
#[allow(dead_code)]
pub fn forward_prefill_batch_chunked(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: u32,
    pbs: &mut PrefillBatchScratch,
) -> Result<Vec<f32>, String> {
    if tokens.is_empty() {
        return Err("forward_prefill_batch_chunked: empty tokens".to_string());
    }
    ensure_request_capacity(
        cfg,
        state,
        gpu,
        pbs,
        (start_pos as usize).saturating_add(tokens.len()),
    )?;

    // Strict batched-only path. Any chunk failure surfaces immediately —
    // we do NOT silently fall back to per-token decode_step. The original
    // fallback masked a real correctness bug in chunk 2+ (per-batch state
    // not initialised; see Option B fix, memory entry
    // `feedback_deepseek4_chunked_silent_fallback_bug`). Keeping the fallback
    // hides any future regression in the same place.
    let mut pos_cursor = start_pos as usize;
    let mut remaining = tokens;
    while !remaining.is_empty() {
        let take = remaining.len().min(pbs.max_batch);
        let chunk = &remaining[..take];
        forward_prefill_batch_chunk(cfg, weights, state, gpu, pbs, chunk, pos_cursor as u32)?;
        capture_head_norm_batched(cfg, weights, state, pbs, gpu, take)?;
        if take == remaining.len() {
            return final_norm_and_head_last_batched(cfg, weights, state, pbs, gpu, take);
        }
        pos_cursor += take;
        remaining = &remaining[take..];
    }
    Err("forward_prefill_batch_chunked: chunk loop completed without producing logits".to_string())
}

/// Exact gfx1201 TP3/TP4 batched prefill for the sharded DeepSeek V4 route.
///
/// Attention heads/O-LoRA groups and the shared expert are evaluated at each
/// rank's local width. Routed experts remain expert-parallel. The two hidden
/// contributions are reduced with a deterministic rank-ordered rooted peer
/// sum after attention and FFN, then broadcast before the HC mix so every
/// rank enters the next stage with bit-identical residual streams.
#[allow(clippy::too_many_arguments)]
pub fn forward_ep_prefill_batch_chunked(
    gpus: &mut hipfire_runtime::multi_gpu::Gpus,
    weights_per_rank: &[DeepseekV4Weights],
    cfg: &DeepseekV4Config,
    state_per_rank: &mut [DeepseekV4State],
    pbs_per_rank: &mut [PrefillBatchScratch],
    tokens: &[u32],
    start_pos: u32,
) -> Result<(), String> {
    let n_ranks = gpus.devices.len();
    if !matches!(n_ranks, 3 | 4)
        || weights_per_rank.len() != n_ranks
        || state_per_rank.len() != n_ranks
        || pbs_per_rank.len() != n_ranks
        || !cfg.mq2r
        || cfg.mq2rxt
        || !gpus.peer_access_enabled
        || !gpus
            .devices
            .iter()
            .all(|device| device.arch_caps.is_gfx1201())
    {
        return Err(format!(
            "forward_ep_prefill_batch_chunked requires exact gfx1201 MQ2R TP3/TP4 (ranks={n_ranks}, weights={}, state={}, pbs={}, mq2r={}, mq2rxt={}, peer={})",
            weights_per_rank.len(),
            state_per_rank.len(),
            pbs_per_rank.len(),
            cfg.mq2r,
            cfg.mq2rxt,
            gpus.peer_access_enabled,
        ));
    }
    if tokens.is_empty() {
        return Err("forward_ep_prefill_batch_chunked: empty tokens".to_string());
    }
    let max_batch = pbs_per_rank[0].max_batch;
    if max_batch == 0 || pbs_per_rank.iter().any(|pbs| pbs.max_batch != max_batch) {
        return Err("forward_ep_prefill_batch_chunked: inconsistent rank PBS capacity".to_string());
    }

    let required_tokens = (start_pos as usize).saturating_add(tokens.len());
    for rank in 0..n_ranks {
        gpus.devices[rank]
            .bind_thread()
            .map_err(|error| format!("TP{n_ranks} prefill bind rank {rank}: {error:?}"))?;
        ensure_request_capacity(
            cfg,
            &mut state_per_rank[rank],
            &mut gpus.devices[rank],
            &mut pbs_per_rank[rank],
            required_tokens,
        )?;
    }
    refresh_compressor_cache_shard_tables(state_per_rank)?;

    let mut consumed = 0usize;
    let mut last_batch = 0usize;
    while consumed < tokens.len() {
        let chunk_start = start_pos + consumed as u32;
        let mut batch_size = (tokens.len() - consumed).min(max_batch);
        if let crate::deepseek4::CompressorCachePlacement::BlockCyclic(shard) =
            state_per_rank[0].compressor_cache_placement
        {
            // Ratio-4 is the finest compressor. Do not let one batched commit
            // cross a physical ownership block; this keeps the incumbent
            // contiguous compressor kernel unchanged on the owning rank.
            let ownership_token_span = shard.block_rows() * 4;
            let until_boundary =
                ownership_token_span - (chunk_start as usize % ownership_token_span);
            batch_size = batch_size.min(until_boundary);
        }
        let chunk = &tokens[consumed..consumed + batch_size];
        last_batch = batch_size;

        // Dynamic request inputs and replicated embedding/HC initialization.
        let mut slots = Vec::with_capacity(n_ranks);
        for rank in 0..n_ranks {
            let gpu = &mut gpus.devices[rank];
            let pbs = &pbs_per_rank[rank];
            upload_prefill_batch_inputs(cfg, gpu, pbs, chunk, chunk_start)?;
            let window = RowsWindow::new(pbs, chunk_start, batch_size, chunk, false);
            let embed = rows_program::embed_rows_program(cfg, &weights_per_rank[rank], window)
                .map_err(|error| format!("TP{n_ranks} rank {rank}: {error}"))?;
            rows_program::run_rows(&embed, cfg, gpu, window, Observed::Embed)?;
            slots.push(RowSlots::new(cfg, pbs, batch_size));
        }
        let window = |rank: usize| {
            RowsWindow::new(&pbs_per_rank[rank], chunk_start, batch_size, chunk, false)
        };

        for layer_idx in 0..cfg.num_hidden_layers {
            let cache_owner = match state_per_rank[0].compressor_cache_placement {
                crate::deepseek4::CompressorCachePlacement::Replicated => None,
                crate::deepseek4::CompressorCachePlacement::BlockCyclic(shard) => {
                    let ratio =
                        weights_per_rank[0].resolve_layer(layer_idx).compress_ratio as usize;
                    (ratio > 0).then(|| shard.owner(chunk_start as usize / ratio))
                }
            };
            let rank_at = |ordinal: usize| match cache_owner {
                None => ordinal,
                Some(owner) if ordinal == 0 => owner,
                Some(owner) => {
                    let other = ordinal - 1;
                    if other >= owner {
                        other + 1
                    } else {
                        other
                    }
                }
            };
            // Rank-local attention projection and attention body.
            // A sharded compressor owner runs first. Its system-scope event
            // then releases the newly committed rows before the remaining
            // ranks enqueue sparse-attention reads of that peer allocation.
            for rank_order in 0..n_ranks {
                let rank = rank_at(rank_order);
                let weights = &weights_per_rank[rank];
                let layer = weights.resolve_layer(layer_idx);
                if layer.attn_tp_size != n_ranks
                    || layer.attn_tp_rank != rank
                    || layer.attn_head_count == 0
                    || layer.attn_group_count == 0
                    || layer.attn_head_count * cfg.o_groups
                        != layer.attn_group_count * cfg.num_attention_heads
                    || layer.shared_tp_size != n_ranks
                    || layer.shared_tp_rank != rank
                    || layer.shared_intermediate_count == 0
                {
                    return Err(format!(
                        "TP{n_ranks} prefill invalid shard l{layer_idx} r{rank}: heads={} groups={} shared={} attn_tp={}/{} shared_tp={}/{}",
                        layer.attn_head_count,
                        layer.attn_group_count,
                        layer.shared_intermediate_count,
                        layer.attn_tp_rank,
                        layer.attn_tp_size,
                        layer.shared_tp_rank,
                        layer.shared_tp_size,
                    ));
                }
                // The attention runs this rank's heads and O-LoRA groups.
                let mut local_cfg = cfg.clone();
                local_cfg.num_attention_heads = layer.attn_head_count;
                local_cfg.o_groups = layer.attn_group_count;
                let gpu = &mut gpus.devices[rank];
                rows_program::ensure_rows_layer(
                    &local_cfg,
                    weights,
                    &mut state_per_rank[rank],
                    gpu,
                    layer_idx,
                )?;
                let state = &state_per_rank[rank];
                let mut program = rows_program::RowsProgram::new();
                rows_program::push_attention(
                    &mut program,
                    &local_cfg,
                    weights,
                    state,
                    gpu,
                    layer_idx,
                    window(rank),
                    &slots[rank],
                    rows_program::rows_projection(gpu, weights.mq2r_backend),
                )?;
                let observed = Observed::Layer(layer_idx, layer);
                rows_program::run_rows(&program, &local_cfg, gpu, window(rank), observed)?;
                if cache_owner == Some(rank) {
                    gpus.handoff_rank_stream_reuse(rank).map_err(|error| {
                        format!(
                            "TP{n_ranks} prefill compressor handoff l{layer_idx} owner={rank}: {error:?}"
                        )
                    })?;
                }
            }

            // Rank-ordered rooted sum of the attention (then FFN) partials,
            // broadcast before every rank's HC mix.
            for ffn in [false, true] {
                let buffers: Vec<_> = pbs_per_rank
                    .iter()
                    .map(|pbs| {
                        if ffn {
                            &pbs.ffn_out_batch.buf
                        } else {
                            &pbs.attn_out_batch.buf
                        }
                    })
                    .collect();
                let what = if ffn { "FFN" } else { "attention" };
                gpus.all_reduce_sum_f32_peer_rooted(&buffers, batch_size * cfg.hidden_size)
                    .map_err(|error| {
                        format!("TP{n_ranks} prefill {what} reduce l{layer_idx}: {error:?}")
                    })?;
                for rank in 0..n_ranks {
                    let gpu = &mut gpus.devices[rank];
                    let weights = &weights_per_rank[rank];
                    let layer = weights.resolve_layer(layer_idx);
                    let mut program = rows_program::RowsProgram::new();
                    rows_program::push_mix(
                        &mut program,
                        cfg,
                        &pbs_per_rank[rank],
                        gpu,
                        ffn,
                        batch_size,
                    );
                    // Shared-intermediate TP plus routed-expert EP: the layer's
                    // local shared width and the global routed-expert shape.
                    if !ffn {
                        rows_program::push_ffn(
                            &mut program,
                            cfg,
                            layer,
                            &pbs_per_rank[rank],
                            gpu,
                            layer_idx,
                            window(rank),
                            layer_idx < cfg.num_hash_layers,
                            rows_program::rows_projection(gpu, weights.mq2r_backend),
                            weights.mq2r_backend,
                        )?;
                    }
                    let observed = Observed::Layer(layer_idx, layer);
                    rows_program::run_rows(&program, cfg, gpu, window(rank), observed)?;
                }
            }
        }

        consumed += batch_size;
        let resident = (start_pos as usize).saturating_add(consumed) as u64;
        for state in state_per_rank.iter_mut() {
            state.n_tokens = resident;
        }
    }

    gpus.devices[0]
        .bind_thread()
        .map_err(|error| format!("TP{n_ranks} prefill final bind: {error:?}"))?;
    final_norm_and_head_last_batched(
        cfg,
        &weights_per_rank[0],
        &mut state_per_rank[0],
        &pbs_per_rank[0],
        &mut gpus.devices[0],
        last_batch,
    )?;
    Ok(())
}

/// Manual-chunk prefill with per-position MTP fill interleaved.
///
/// Mirrors the deepseek4_mtp_smoke "batched main + per-position MTP" path.
/// Used by the spec-decode entry points (deepseek4_chat / daemon) so the MTP
/// layer's SWA cache is populated during prefill — without this the
/// first spec-decode draft step sees an empty MTP attention history
/// and accept rate collapses.
///
/// Returns logits at the LAST position (the prediction for the first
/// generated token). Side-effect: leaves `state.mtp_last_hidden`
/// populated and `state.n_tokens` advanced to `start_pos + prompt.len()`.
///
/// Temporarily sets `HIPFIRE_DEEPSEEK4_MTP_SKIP_HEAD=1` around the MTP pass
/// so `mtp_forward_batched` short-circuits the lm_head + logits
/// download — that's per-MTP-position waste during prefill fill (we
/// only need the MTP attention SWA state to be primed, not the
/// per-position MTP logits). Restored after each chunk.
pub fn prefill_with_mtp_fill(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    pbs: &mut PrefillBatchScratch,
    prompt_tokens: &[u32],
    start_pos: u32,
) -> Result<Vec<f32>, String> {
    let stream_len = cfg.hc_mult * cfg.hidden_size;
    if state.mtp_last_hidden.is_none() {
        state.mtp_last_hidden = Some(
            gpu.alloc_tensor(&[cfg.hc_mult, cfg.hidden_size], rdna_compute::DType::F32)
                .map_err(|e| format!("alloc mtp_last_hidden (spec prefill): {e:?}"))?,
        );
    }
    // compressor_forward_prebatched reads pos_array_device via pos_slot()
    // for any compressed layer. Init to start_pos.
    precompute_positions(cfg, state, gpu, start_pos)?;

    let mut last_logits: Vec<f32> = vec![];
    let mut pos_cursor: usize = 0;
    while pos_cursor < prompt_tokens.len() {
        let chunk_size = (prompt_tokens.len() - pos_cursor).min(pbs.max_batch);
        let chunk = &prompt_tokens[pos_cursor..pos_cursor + chunk_size];
        let abs_chunk_start = start_pos as usize + pos_cursor;
        let is_last_chunk = pos_cursor + chunk_size == prompt_tokens.len();

        // 1. Batched main forward over this chunk's positions. After this,
        //    pbs.streams_batch holds [chunk_size, hc_mult, hidden] residuals
        //    — these are the per-position h_n inputs the MTP layer needs.
        forward_prefill_batch_chunk(cfg, weights, state, gpu, pbs, chunk, abs_chunk_start as u32)?;

        // 2. Capture the last position's stream for the head on the last
        //    chunk, BEFORE mtp_forward_batched overwrites streams_batch.
        let last_stream_pre_mtp: Option<rdna_compute::GpuTensor> = if is_last_chunk {
            let off = (chunk_size - 1) * stream_len;
            let src = pbs.streams_batch.sub_offset(off, stream_len);
            let mut snap = gpu
                .alloc_tensor(&[cfg.hc_mult, cfg.hidden_size], rdna_compute::DType::F32)
                .map_err(|e| format!("alloc head_input_snap: {e:?}"))?;
            gpu.memcpy_dtod_auto(&snap.buf, &src.buf, stream_len * 4)
                .map_err(|e| format!("d2d streams[last] → head_input_snap: {e:?}"))?;
            snap.shape = vec![cfg.hc_mult, cfg.hidden_size];
            Some(snap)
        } else {
            None
        };

        // 3. Batched MTP fill — single pass through the MTP layer for all
        //    mtp_end_b positions in this chunk. Skip the global last
        //    position (next-token unknown, that's what we're about to
        //    generate).
        std::env::set_var("HIPFIRE_DEEPSEEK4_MTP_SKIP_HEAD", "1");
        let mtp_end_b = if is_last_chunk {
            chunk_size.saturating_sub(1)
        } else {
            chunk_size
        };
        if mtp_end_b > 0 {
            let h_n_streams = pbs.streams_batch.sub_offset(0, mtp_end_b * stream_len);
            let next_tokens: Vec<u32> = (0..mtp_end_b)
                .map(|b| prompt_tokens[pos_cursor + b + 1])
                .collect();
            mtp_forward_batched(
                cfg,
                weights,
                state,
                gpu,
                pbs,
                &h_n_streams,
                &next_tokens,
                abs_chunk_start as u32,
                mtp_end_b,
            )?;
        }
        std::env::remove_var("HIPFIRE_DEEPSEEK4_MTP_SKIP_HEAD");

        pos_cursor += chunk_size;
        state.n_tokens = (abs_chunk_start + chunk_size) as u64;

        // 4. Last chunk: run final_norm_and_head from the snapshot we
        //    captured pre-MTP (streams_batch is now MTP outputs). Write
        //    the snapshot back into the last slot so the existing
        //    final_norm_and_head_last_batched can read it.
        if is_last_chunk {
            if let Some(snap) = last_stream_pre_mtp {
                let off = (chunk_size - 1) * stream_len;
                let dst = pbs.streams_batch.sub_offset(off, stream_len);
                gpu.memcpy_dtod_auto(&dst.buf, &snap.buf, stream_len * 4)
                    .map_err(|e| format!("d2d restore streams[last] for head: {e:?}"))?;
            }
            last_logits =
                final_norm_and_head_last_batched(cfg, weights, state, pbs, gpu, chunk_size)?;
        }
    }
    Ok(last_logits)
}

// ════════════════════════════════════════════════════════════════════════
//  DSpark draft-module forward
//  (additive — see docs/design/2026-06-28-dspark-deepseek4-forward-spec.md)
// ════════════════════════════════════════════════════════════════════════

/// One block of DSpark draft output. Filled by [`dspark_forward`].
///
/// - `tokens`   — `block_size` drafted token ids (the markov sequential
///                sampling output, slots `1..=block_size`).
/// - `logits`   — flattened `[block_size, vocab]` per-slot logits (post
///                markov bias). Useful for a downstream verifier; may be
///                left empty by callers that only need `tokens`.
/// - `confidence` — `[block_size]` per-slot confidence scalars from the
///                  confidence head.
pub struct DraftResult {
    pub tokens: Vec<u32>,
    /// Per-slot draft logits `[block * vocab]`. Currently always EMPTY: the
    /// markov bias-add + argmax run on-GPU and no caller consumes the draft
    /// logits (the verify forward recomputes the trunk head). Populate this
    /// only if a future consumer needs them — it costs a `[block, vocab]` d2h.
    pub logits: Vec<f32>,
    pub confidence: Vec<f32>,
}

/// Single-token embedding lookup into row 0 of `out` (`[dim]` F32),
/// dispatching on the embedding-table dtype. Mirrors the dtype handling
/// the head / MTP paths use. Supports the formats DSpark sidecars ship
/// the trunk `token_embd` in (Q8_0 / HFQ4G256 / F16 / F32).
fn dspark_embed_one(
    gpu: &mut Gpu,
    table: &GpuTensor,
    out: &GpuTensor,
    token_id: u32,
    dim: usize,
) -> Result<(), String> {
    match table.dtype {
        DType::Q8_0 => gpu
            .embedding_lookup_q8(table, out, token_id, dim)
            .map_err(|e| format!("dspark embed q8: {e:?}")),
        DType::F32 => gpu
            .embedding_lookup(table, out, token_id, dim)
            .map_err(|e| format!("dspark embed f32: {e:?}")),
        DType::Raw => gpu
            .embedding_lookup_hfq4g256(table, out, token_id, dim)
            .map_err(|e| format!("dspark embed hfq4g256: {e:?}")),
        DType::F16 => {
            // No single-row F16 lookup kernel — extract the row to host,
            // convert F16→F32, upload into `out`. Cheap (dim ≤ 4096).
            let mut row_bytes = vec![0u8; dim * 2];
            let off = (token_id as usize) * dim * 2;
            gpu.hip
                .memcpy_dtoh_at(&mut row_bytes, &table.buf, off)
                .map_err(|e| format!("dspark embed f16 dtoh: {e:?}"))?;
            let row_f32: Vec<f32> = (0..dim)
                .map(|i| {
                    let h = u16::from_le_bytes([row_bytes[i * 2], row_bytes[i * 2 + 1]]);
                    hipfire_runtime::llama::f16_to_f32(h)
                })
                .collect();
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(row_f32.as_ptr() as *const u8, dim * 4) };
            gpu.hip
                .memcpy_htod(&out.buf, bytes)
                .map_err(|e| format!("dspark embed f16 htod: {e:?}"))
        }
        other => Err(format!("dspark embed: unsupported table dtype {other:?}")),
    }
}

/// DSpark draft forward: a 3-stage MTP chain over a `block_size`-slot noise
/// block, producing `block_size` draft tokens.
///
/// The crux is the bidirectional draft attention: all `block_size` block
/// slots attend over the SAME key window — the committed main_kv ring plus
/// the `block_size` block KVs. We reuse the existing
/// [`Gpu::deepseek4_attn_swa_batched`] kernel with a CUSTOM bidirectional
/// stager: for every query row `b` we stage `[committed main_kv (n_committed)
/// ++ block_size block kv]` and set `n_valid_arr[b] = n_committed + block_size`.
/// The kernel then does dense attention of each query over those keys.
///
/// Inputs:
///   - `main_hidden` `[3 * hidden]` — the 3 target layers' HC-mean-pooled
///     hidden states concatenated (captured in the trunk forward).
///   - `token_embd`  — the TRUNK token-embedding table (DSpark sidecars
///     don't carry their own embedding).
///   - `head` / `output_norm` — the TRUNK lm_head + its output norm.
///   - `prev_token` / `position` — the committed token and its absolute
///     KV position (the block predicts positions `position+1 ..= +block_size`).
///
/// Additive: touches no existing `mtp_forward` / `forward_prefill_*` path.
/// Per-stage main_kv rings live in `state.dspark_swa_k`; the block scratch
/// in `state.dspark_pbs`; `main_x` in `state.dspark_main_x`.
#[allow(clippy::too_many_arguments)]
pub fn dspark_forward(
    cfg: &DeepseekV4Config,
    dspark: &crate::deepseek4::DsparkWeights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    main_hidden: &GpuTensor, // [3 * hidden]
    token_embd: &GpuTensor,
    head: &GpuTensor,
    output_norm: &GpuTensor,
    prev_token: u32,
    position: u32,
) -> Result<DraftResult, String> {
    let hidden = cfg.hidden_size;
    let hc_mult = cfg.hc_mult;
    let head_dim = cfg.head_dim;
    let n_kv = cfg.num_key_value_heads;
    let win = cfg.sliding_window;
    let vocab = cfg.vocab_size;
    let block = dspark.cfg.block_size;
    let n_stages = dspark.stages.len();
    if n_stages == 0 {
        return Err("dspark_forward: no stages loaded".to_string());
    }

    // committed main_kv history visible to this step (ring fill level).
    //
    // NOTE: this counts as if the rings were warmed from position 0, but DSpark
    // writes only ONE seed main_kv per window (at `position % win`), so the
    // staged committed window holds real keys only at written columns and ZEROS
    // elsewhere — the bidirectional attention dilutes over those zeros. We TRIED
    // compacting the writes (slot = fill % win) + `n_committed = fill+1` so the
    // attention sees only real keys (no zero dilution). It is strictly more
    // faithful to the reference's full-history scheme, but A/B-measured a τ LOSS
    // on MQ2-Lloyd — code τ 3.64→3.14, prose τ 2.96→2.62 (both still coherent),
    // ~6-8% tok/s. Same lesson as priming the rings over the prompt (a618510a):
    // under 2-bit quant the drafter's attention is mis-calibrated such that the
    // zero-key denominator inflation is a NET BENEFIT to acceptance, not noise.
    // So we keep the zeros. Revisit on a higher-precision DSpark sidecar.
    let n_committed = (position as usize + 1).min(win);
    // custom-stager width: committed window + the block KVs.
    let stage_w = win + block;
    if n_committed + block > 1024 {
        // deepseek4_attn_swa_batched LDS scores buffer caps at MAX_WINDOW=1024.
        return Err(format!(
            "dspark_forward: n_valid {} exceeds kernel MAX_WINDOW 1024",
            n_committed + block
        ));
    }

    // ── Ensure async stream + per-call scratch ──────────────────────────
    if gpu.active_stream.is_none() {
        let s = gpu
            .hip
            .stream_create()
            .map_err(|e| format!("dspark stream_create: {e:?}"))?;
        gpu.active_stream = Some(s);
    }

    // Block scratch — allocate once, reuse across decode steps.
    if state.dspark_pbs.is_none() {
        state.dspark_pbs = Some(PrefillBatchScratch::new(gpu, cfg, block.max(8))?);
    }
    // Per-stage main_kv rings — lazy.
    if state.dspark_swa_k.len() != n_stages {
        state.dspark_swa_k = (0..n_stages).map(|_| None).collect();
    }
    for s in 0..n_stages {
        if state.dspark_swa_k[s].is_none() {
            state.dspark_swa_k[s] = Some(
                gpu.zeros(&[n_kv, head_dim, win], DType::F32)
                    .map_err(|e| format!("dspark alloc ring[{s}]: {e:?}"))?,
            );
        }
    }

    // ── A. forward_embed (once) ─────────────────────────────────────────
    // main_x = main_norm(main_proj(main_hidden)). main_proj: [hidden, 3*hidden].
    let main_proj = dspark
        .main_proj
        .as_ref()
        .ok_or("dspark_forward: main_proj missing")?;
    let main_norm = dspark
        .main_norm
        .as_ref()
        .ok_or("dspark_forward: main_norm missing")?;
    if state.dspark_main_x.is_none() {
        state.dspark_main_x = Some(
            gpu.alloc_tensor(&[hidden], DType::F32)
                .map_err(|e| format!("dspark alloc main_x: {e:?}"))?,
        );
    }
    {
        // main_proj GEMV: main_hidden[3*hidden] → tmp[hidden].
        // main_proj/main_hidden may need FWHT for MQ4 weights — for F32/Q8/F16
        // gemv_auto reads the plain input. We pre-rotate when needed.
        let main_x = state.dspark_main_x.as_ref().unwrap().shallow_clone();
        let three_h = 3 * hidden;
        if weight_needs_fwht(main_proj) {
            // Rotate main_hidden into a scratch, then GEMV. Reuse a per-call buffer.
            let rot = gpu
                .alloc_tensor(&[three_h], DType::F32)
                .map_err(|e| format!("dspark alloc main_proj rot: {e:?}"))?;
            gpu.rotate_x_mq(main_hidden, &rot, three_h)
                .map_err(|e| format!("dspark rotate main_hidden: {e:?}"))?;
            gemv_auto(
                gpu,
                Mq2rBackend::Portable,
                main_proj,
                &rot,
                main_hidden,
                &main_x,
                hidden,
                three_h,
            )?;
            let _ = gpu.free_tensor(rot);
        } else {
            gemv_auto(
                gpu,
                Mq2rBackend::Portable,
                main_proj,
                main_hidden,
                main_hidden,
                &main_x,
                hidden,
                three_h,
            )?;
        }
        // main_norm RMSNorm in place.
        gpu.rmsnorm_f32(&main_x, main_norm, &main_x, cfg.rms_norm_eps)
            .map_err(|e| format!("dspark main_norm: {e:?}"))?;
    }
    // noise block ids: [prev_token, noise, noise, ...] (block_size slots).
    let mut block_ids = vec![dspark.cfg.noise_token_id; block];
    block_ids[0] = prev_token;

    // Embed each block id into pbs.embed_batch rows, then broadcast → streams.
    {
        let pbs = state.dspark_pbs.as_ref().unwrap();
        // Embed the block ids via the SAME proven batched kernel the AR/prefill
        // path uses (the per-token embedding_lookup_q8 produced garbage). Upload
        // ids → pbs.tokens (i32-in-F32 slots), then batched lookup → embed_batch.
        // Called unconditionally — matches forward_prefill_batch_chunk, which
        // runs this on weights.token_embd regardless of its dtype enum.
        let tok_host: Vec<i32> = block_ids.iter().map(|&t| t as i32).collect();
        let tok_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(tok_host.as_ptr() as *const u8, block * 4) };
        gpu.memcpy_htod_auto(&pbs.tokens.buf, tok_bytes)
            .map_err(|e| format!("dspark htod block tokens: {e:?}"))?;
        gpu.embedding_lookup_q8_batched(token_embd, &pbs.embed_batch, &pbs.tokens, block, hidden)
            .map_err(|e| format!("dspark embedding_lookup_q8_batched: {e:?}"))?;
        gpu.hc_streams_init_from_embed_batched(
            &pbs.embed_batch,
            &pbs.streams_batch,
            hidden as i32,
            hc_mult as i32,
            block as i32,
        )
        .map_err(|e| format!("dspark hc_streams_init: {e:?}"))?;

        // Block-slot positions: position+1 ..= position+block (shared by q +
        // block-kv RoPE). Uploaded once; reused by every stage.
        let pos_host: Vec<i32> = (0..block).map(|i| position as i32 + 1 + i as i32).collect();
        let pos_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(pos_host.as_ptr() as *const u8, block * 4) };
        gpu.memcpy_htod_auto(&pbs.positions.buf, pos_bytes)
            .map_err(|e| format!("dspark htod block positions: {e:?}"))?;

        // n_valid_arr[b] = n_committed + block for all b (dense / bidirectional).
        let nv_host: Vec<i32> = vec![(n_committed + block) as i32; block];
        let nv_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(nv_host.as_ptr() as *const u8, block * 4) };
        gpu.memcpy_htod_auto(&pbs.n_valid_swa_arr.buf, nv_bytes)
            .map_err(|e| format!("dspark htod n_valid: {e:?}"))?;
    }

    // Custom staging buffer [block, head_dim, stage_w] — allocated once and
    // reused (see DeepseekV4State::dspark_staged). `dspark_stage_kv` rewrites
    // every column each stage (including the zeroed tail), so reuse is exact;
    // keeping it off the per-call path means an early `?` can't leak it.
    if state.dspark_staged.is_none() {
        state.dspark_staged = Some(
            gpu.alloc_tensor(&[block, head_dim, stage_w], DType::F32)
                .map_err(|e| format!("dspark alloc staged: {e:?}"))?,
        );
    }
    let staged = state.dspark_staged.as_ref().unwrap().shallow_clone();

    // ── B. stage chain: one rows program per stage ─────────────────────
    {
        let pbs = state.dspark_pbs.as_ref().unwrap();
        let views = rows_program::DrafterViews::new(cfg, pbs, block);
        let main_x = state.dspark_main_x.as_ref().unwrap();
        let window = RowsWindow::new(pbs, 0, block, &[], false);
        for (s, layer) in dspark.stages.iter().enumerate() {
            let program = rows_program::drafter_stage_program(
                cfg,
                layer,
                pbs,
                &views,
                gpu,
                s,
                block,
                main_x,
                vec![position],
                state.dspark_swa_k[s].as_ref().unwrap(),
                &staged,
                n_committed,
            )?;
            rows_program::run_rows(&program, cfg, gpu, window, Observed::DrafterStage(s, layer))?;
        }
    }

    // ── C. forward_head (last stage) ────────────────────────────────────
    let last = &dspark.stages[n_stages - 1];
    dspark_forward_head(
        cfg,
        dspark,
        last,
        state,
        gpu,
        head,
        output_norm,
        prev_token,
        block,
        vocab,
        hidden,
        hc_mult,
    )
}

/// DSpark body-only forward: runs body B (3-stage MoE/SWA chain) + body C.begin
/// (HC-gate reduction) from [`dspark_forward`], writing `x_head[block, hidden]`
/// into `x_head_out` WITHOUT running the markov/confidence/lm-head (part C.rest).
///
/// Called by `Deepseek4DsparkBody::draft_block` in `dspark_speculator.rs`: the
/// arch-agnostic `dspark_core::DsparkDrafter` calls `draft_block` to get
/// `x_head` and then passes it to `dspark_core::run_heads` for the rest.
///
/// **Stage 3 multi-slot main_kv population:**
/// `main_x_batch` is `[ctx_len * hidden]` F32 — the output of
/// `main_proj_ingest_batched` applied to the accepted-prefix multi-slot context.
/// For each of the `ctx_len` context slots at `ctx_positions[j]`, step 5 now
/// calls `dspark_stage_main_kv_to_ring` at the slot's absolute position, filling
/// the ring with all `ctx_len` context entries instead of just the single
/// bootstrap seed. With `ctx_len=1` this is identical to the prior single-slot
/// contract (backward-compatible).
///
/// `state.dspark_pbs` and `state.dspark_swa_k` are lazily allocated here.
#[allow(clippy::too_many_arguments)]
pub fn dspark_run_body_and_hc_gate(
    cfg: &DeepseekV4Config,
    dspark: &crate::deepseek4::DsparkWeights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    token_embd: &GpuTensor,
    main_x_batch: &GpuTensor, // [ctx_len * hidden] F32 — multi-slot main_proj output
    ctx_positions: &[usize],  // absolute positions for each context slot; len = ctx_len
    prev_token: u32,
    position: u32,
    block: usize,
    x_head_out: &GpuTensor, // [block, hidden] F32 output
) -> Result<(), String> {
    let hidden = cfg.hidden_size;
    let hc_mult = cfg.hc_mult;
    let head_dim = cfg.head_dim;
    let n_kv = cfg.num_key_value_heads;
    let win = cfg.sliding_window;
    let n_stages = dspark.stages.len();
    if n_stages == 0 {
        return Err("dspark_run_body_and_hc_gate: no stages loaded".to_string());
    }

    let n_committed = (position as usize + 1).min(win);
    let stage_w = win + block;
    if n_committed + block > 1024 {
        return Err(format!(
            "dspark_run_body_and_hc_gate: n_valid {} exceeds kernel MAX_WINDOW 1024",
            n_committed + block
        ));
    }

    // Ensure async stream.
    if gpu.active_stream.is_none() {
        let s = gpu
            .hip
            .stream_create()
            .map_err(|e| format!("dspark body stream_create: {e:?}"))?;
        gpu.active_stream = Some(s);
    }

    // Block scratch — allocate once, reuse across decode steps.
    if state.dspark_pbs.is_none() {
        state.dspark_pbs = Some(PrefillBatchScratch::new(gpu, cfg, block.max(8))?);
    }
    // Per-stage main_kv rings — lazy.
    if state.dspark_swa_k.len() != n_stages {
        state.dspark_swa_k = (0..n_stages).map(|_| None).collect();
    }
    for s in 0..n_stages {
        if state.dspark_swa_k[s].is_none() {
            state.dspark_swa_k[s] = Some(
                gpu.zeros(&[n_kv, head_dim, win], DType::F32)
                    .map_err(|e| format!("dspark body alloc ring[{s}]: {e:?}"))?,
            );
        }
    }

    // Build block token ids and embed.
    let mut block_ids = vec![dspark.cfg.noise_token_id; block];
    block_ids[0] = prev_token;
    {
        let pbs = state.dspark_pbs.as_ref().unwrap();
        let tok_host: Vec<i32> = block_ids.iter().map(|&t| t as i32).collect();
        let tok_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(tok_host.as_ptr() as *const u8, block * 4) };
        gpu.memcpy_htod_auto(&pbs.tokens.buf, tok_bytes)
            .map_err(|e| format!("dspark body htod block tokens: {e:?}"))?;
        gpu.embedding_lookup_q8_batched(token_embd, &pbs.embed_batch, &pbs.tokens, block, hidden)
            .map_err(|e| format!("dspark body embedding_lookup_q8_batched: {e:?}"))?;
        gpu.hc_streams_init_from_embed_batched(
            &pbs.embed_batch,
            &pbs.streams_batch,
            hidden as i32,
            hc_mult as i32,
            block as i32,
        )
        .map_err(|e| format!("dspark body hc_streams_init: {e:?}"))?;
        let pos_host: Vec<i32> = (0..block).map(|i| position as i32 + 1 + i as i32).collect();
        let pos_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(pos_host.as_ptr() as *const u8, block * 4) };
        gpu.memcpy_htod_auto(&pbs.positions.buf, pos_bytes)
            .map_err(|e| format!("dspark body htod block positions: {e:?}"))?;
        let nv_host: Vec<i32> = vec![(n_committed + block) as i32; block];
        let nv_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(nv_host.as_ptr() as *const u8, block * 4) };
        gpu.memcpy_htod_auto(&pbs.n_valid_swa_arr.buf, nv_bytes)
            .map_err(|e| format!("dspark body htod n_valid: {e:?}"))?;
    }

    // Per-call staging buffer.
    let staged = gpu
        .alloc_tensor(&[block, head_dim, stage_w], DType::F32)
        .map_err(|e| format!("dspark body alloc staged: {e:?}"))?;

    // ── B. stage chain (as dspark_forward), one context row per slot ───────
    {
        let pbs = state.dspark_pbs.as_ref().unwrap();
        let views = rows_program::DrafterViews::new(cfg, pbs, block);
        let window = RowsWindow::new(pbs, 0, block, &[], false);
        for (s, layer) in dspark.stages.iter().enumerate() {
            let program = rows_program::drafter_stage_program(
                cfg,
                layer,
                pbs,
                &views,
                gpu,
                s,
                block,
                main_x_batch,
                ctx_positions.iter().map(|&p| p as u32).collect(),
                state.dspark_swa_k[s].as_ref().unwrap(),
                &staged,
                n_committed,
            )?;
            rows_program::run_rows(&program, cfg, gpu, window, Observed::DrafterStage(s, layer))?;
        }
    }
    let _ = gpu.free_tensor(staged);

    // ── C.begin: HC-gate reduction → x_head_out[block, hidden] ───────────
    let last = &dspark.stages[n_stages - 1];
    let hc_head_fn = last
        .mtp_hc_head_fn
        .as_ref()
        .ok_or("dspark body: mtp_hc_head_fn missing on last stage")?;
    let hc_head_base = last
        .mtp_hc_head_base
        .as_ref()
        .ok_or("dspark body: mtp_hc_head_base missing on last stage")?;
    let hc_pre = gpu
        .alloc_tensor(&[hc_mult], DType::F32)
        .map_err(|e| format!("dspark body alloc head hc_pre: {e:?}"))?;
    {
        let pbs = state.dspark_pbs.as_ref().unwrap();
        let stream_len = hc_mult * hidden;
        let x_dim = hidden * hc_mult;
        for b in 0..block {
            let streams_b = pbs.streams_batch.sub_offset(b * stream_len, stream_len);
            gpu.hc_head_compute_pre(
                &streams_b,
                hc_head_fn,
                hc_head_base,
                &hc_pre,
                hc_mult as i32,
                x_dim as i32,
                last.mtp_hc_head_scale,
                cfg.rms_norm_eps,
                cfg.hc_eps,
            )
            .map_err(|e| format!("dspark body hc_head_compute_pre b{b}: {e:?}"))?;
            let x_head_b = x_head_out.sub_offset(b * hidden, hidden);
            gpu.hc_input_map_4stream(&hc_pre, &streams_b, &x_head_b, hidden as i32)
                .map_err(|e| format!("dspark body hc_input_map head b{b}: {e:?}"))?;
        }
    }
    let _ = gpu.free_tensor(hc_pre);

    Ok(())
}

/// forward_head: head-HC (SIGMOID gate) → x_head[block, hidden]; lm_head
/// over rmsnorm(x_head, mtp_final_norm) → logits[block, vocab]; sequential
/// markov in-block sampling → block_size draft tokens; confidence head.
#[allow(clippy::too_many_arguments)]
fn dspark_forward_head(
    cfg: &DeepseekV4Config,
    dspark: &crate::deepseek4::DsparkWeights,
    last: &crate::deepseek4::DeepseekV4LayerWeights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    head: &GpuTensor,
    output_norm: &GpuTensor,
    prev_token: u32,
    block: usize,
    vocab: usize,
    hidden: usize,
    hc_mult: usize,
) -> Result<DraftResult, String> {
    let markov_rank = dspark.cfg.markov_rank;

    let hc_head_fn = last
        .mtp_hc_head_fn
        .as_ref()
        .ok_or("dspark_forward_head: mtp_hc_head_fn missing")?;
    let hc_head_base = last
        .mtp_hc_head_base
        .as_ref()
        .ok_or("dspark_forward_head: mtp_hc_head_base missing")?;
    let mtp_final_norm = last
        .mtp_final_norm
        .as_ref()
        .ok_or("dspark_forward_head: mtp_final_norm missing")?;
    let markov_w1 = dspark
        .markov_w1
        .as_ref()
        .ok_or("dspark_forward_head: markov_w1 missing")?;
    let markov_w2 = dspark
        .markov_w2
        .as_ref()
        .ok_or("dspark_forward_head: markov_w2 missing")?;
    let confidence_proj = dspark
        .confidence_proj
        .as_ref()
        .ok_or("dspark_forward_head: confidence_proj missing")?;

    // x_head[block, hidden] — head-HC sigmoid-gate reduction per slot.
    let x_head = gpu
        .alloc_tensor(&[block, hidden], DType::F32)
        .map_err(|e| format!("dspark alloc x_head: {e:?}"))?;
    // Per-slot head-HC pre [hc_mult] scratch.
    let hc_pre = gpu
        .alloc_tensor(&[hc_mult], DType::F32)
        .map_err(|e| format!("dspark alloc head hc_pre: {e:?}"))?;
    let x_dim = hidden * hc_mult;
    {
        let pbs = state.dspark_pbs.as_ref().unwrap();
        let stream_len = hc_mult * hidden;
        for b in 0..block {
            let streams_b = pbs.streams_batch.sub_offset(b * stream_len, stream_len);
            // pre = sigmoid(mixes * scale + base) + eps  (NOT sinkhorn).
            gpu.hc_head_compute_pre(
                &streams_b,
                hc_head_fn,
                hc_head_base,
                &hc_pre,
                hc_mult as i32,
                x_dim as i32,
                last.mtp_hc_head_scale,
                cfg.rms_norm_eps,
                cfg.hc_eps,
            )
            .map_err(|e| format!("dspark hc_head_compute_pre b{b}: {e:?}"))?;
            // x_head[b] = sum_h pre[h] * streams[b, h, :].
            let x_head_b = x_head.sub_offset(b * hidden, hidden);
            gpu.hc_input_map_4stream(&hc_pre, &streams_b, &x_head_b, hidden as i32)
                .map_err(|e| format!("dspark hc_input_map head b{b}: {e:?}"))?;
        }
    }
    let _ = gpu.free_tensor(hc_pre);

    // x_head stays resident: the confidence head reads it ON GPU (per-slot
    // `proj · [x_head[i] ++ markov_embed[i]]` 1-row gemv in the markov loop),
    // so we never pay the [block, hidden] d2h. Neutral on UMA (gfx1151
    // VRAM==RAM) but removes a per-window PCIe d2h on discrete cards
    // (gfx1100/gfx1201) where it stalls the decode pipeline. Freed after loop.

    // logits[block, vocab] = lm_head(rmsnorm(x_head, mtp_final_norm)).
    let normed = gpu
        .alloc_tensor(&[block, hidden], DType::F32)
        .map_err(|e| format!("dspark alloc head normed: {e:?}"))?;
    let norm = [Step::RmsnormAutomatic {
        x: &x_head,
        norm_weight: mtp_final_norm,
        x_plain: &normed,
        out: &normed,
        awq_scale: None,
        k: hidden,
        eps: cfg.rms_norm_eps,
        rotation: hipfire_dispatch::types::RotationPlan::None,
        rows: block,
    }];
    let ctx = DispatchCtx::new(gpu);
    execute_steps(gpu, &ctx, &norm).map_err(|e| format!("dspark final rmsnorm: {e:?}"))?;
    // output_norm is the trunk lm_head norm — applied after the per-stage
    // mtp_final_norm? Spec uses head(norm(x)) with the stage's mtp_final_norm.
    // We feed the stage-normed activation directly to the trunk lm_head.
    let _ = output_norm;
    let normed_rot = if weight_needs_fwht(head) {
        let r = gpu
            .alloc_tensor(&[block, hidden], DType::F32)
            .map_err(|e| format!("dspark alloc normed_rot: {e:?}"))?;
        gpu.rotate_x_mq_batched(&normed, &r, hidden, block)
            .map_err(|e| format!("dspark rotate head input: {e:?}"))?;
        Some(r)
    } else {
        None
    };
    let logits_dev = gpu
        .alloc_tensor(&[block, vocab], DType::F32)
        .map_err(|e| format!("dspark alloc logits: {e:?}"))?;
    let x_f16 = gpu
        .alloc_tensor(&[block * hidden], DType::F16)
        .map_err(|e| format!("dspark alloc head x_f16: {e:?}"))?;
    gemv_auto_batched_wmma(
        gpu,
        Mq2rBackend::Portable,
        head,
        normed_rot.as_ref().unwrap_or(&normed),
        &normed,
        &logits_dev,
        vocab,
        hidden,
        block,
        Some(&x_f16),
    )?;
    if let Some(r) = normed_rot {
        let _ = gpu.free_tensor(r);
    }
    let _ = gpu.free_tensor(x_f16);
    let _ = gpu.free_tensor(normed);
    // x_head freed after the confidence head (below) — the markov loop's
    // per-slot confidence gemv reads x_head[i] on GPU.
    // `logits_dev` stays resident: the markov loop adds each slot's bias and
    // argmaxes ON-GPU (no per-slot 517 KB / upfront 2.5 MB full-vocab d2h).
    // Freed after the loop. The draft logits themselves are never consumed
    // downstream (verify re-runs the trunk head), so we never materialize them
    // on the host.

    // ── Sequential markov in-block sampling (greedy) ────────────────────
    // out_ids[0] = prev_token; out_ids[i+1] = argmax(logits[i] + markov_bias).
    // markov_bias = markov_w2 @ markov_w1_lookup(out_ids[i]). markov_embed[i]
    // (the [markov_rank] lookup) is also collected for the confidence head.
    let mut out_ids = vec![prev_token; block + 1];
    // Confidence head buffers (computed ON GPU per slot inside the loop):
    // `conf_batch[block]` holds the per-slot confidence logit; `concat_dev`
    // stages `[x_head[i] ++ markov_embed[i]]` for the 1-row `confidence_proj`
    // gemv. Downloaded once after the loop (block floats), so neither x_head
    // nor the per-slot markov embed crosses to the host.
    let proj_in = hidden + markov_rank;
    let conf_batch = gpu
        .alloc_tensor(&[block], DType::F32)
        .map_err(|e| format!("dspark alloc conf_batch: {e:?}"))?;
    let concat_dev = gpu
        .alloc_tensor(&[proj_in], DType::F32)
        .map_err(|e| format!("dspark alloc conf concat: {e:?}"))?;
    // Reusable device scratch for the markov head.
    let emb_dev = gpu
        .alloc_tensor(&[markov_rank], DType::F32)
        .map_err(|e| format!("dspark alloc markov emb: {e:?}"))?;
    let bias_dev = gpu
        .alloc_tensor(&[vocab], DType::F32)
        .map_err(|e| format!("dspark alloc markov bias: {e:?}"))?;
    let emb_rot = if weight_needs_fwht(markov_w2) {
        Some(
            gpu.alloc_tensor(&[markov_rank], DType::F32)
                .map_err(|e| format!("dspark alloc markov emb rot: {e:?}"))?,
        )
    } else {
        None
    };
    for i in 0..block {
        // markov_w1 lookup of out_ids[i] → emb_dev [markov_rank] (unrotated).
        dspark_embed_one(gpu, markov_w1, &emb_dev, out_ids[i], markov_rank)?;
        // Confidence slot i ON GPU: stage [x_head[i] ++ markov_embed[i]] then a
        // 1-row `confidence_proj` gemv → conf_batch[i]. Uses the UNROTATED
        // emb_dev (matches the reference, which dotted the raw markov embed).
        {
            let xh_i = x_head.sub_offset(i * hidden, hidden);
            let c_hidden = concat_dev.sub_offset(0, hidden);
            let c_markov = concat_dev.sub_offset(hidden, markov_rank);
            gpu.memcpy_dtod_auto(&c_hidden.buf, &xh_i.buf, hidden * 4)
                .map_err(|e| format!("dspark conf stage x_head {i}: {e:?}"))?;
            gpu.memcpy_dtod_auto(&c_markov.buf, &emb_dev.buf, markov_rank * 4)
                .map_err(|e| format!("dspark conf stage emb {i}: {e:?}"))?;
            let conf_i = conf_batch.sub_offset(i, 1);
            gemv_auto(
                gpu,
                Mq2rBackend::Portable,
                confidence_proj,
                &concat_dev,
                &concat_dev,
                &conf_i,
                1,
                proj_in,
            )?;
        }
        // bias = markov_w2 @ emb  ([vocab, markov_rank] · [markov_rank]).
        let x_for_w2 = if let Some(r) = emb_rot.as_ref() {
            gpu.rotate_x_mq(&emb_dev, r, markov_rank)
                .map_err(|e| format!("dspark rotate markov emb {i}: {e:?}"))?;
            r
        } else {
            &emb_dev
        };
        gemv_auto(
            gpu,
            Mq2rBackend::Portable,
            markov_w2,
            x_for_w2,
            &emb_dev,
            &bias_dev,
            vocab,
            markov_rank,
        )?;
        // logits[i] += bias, then argmax — both ON-GPU. The sequential
        // dependency (out_ids[i] → emb → bias → argmax → out_ids[i+1]) forces
        // per-slot argmax, but only the 4-byte token id crosses to the host
        // (argmax_f32 reduces on-GPU), not a 517 KB bias vector.
        let row = logits_dev.sub_offset(i * vocab, vocab);
        gpu.add_inplace_f32(&row, &bias_dev)
            .map_err(|e| format!("dspark markov bias add {i}: {e:?}"))?;
        out_ids[i + 1] = gpu
            .argmax_f32(&row, vocab)
            .map_err(|e| format!("dspark markov argmax {i}: {e:?}"))?;
    }
    let _ = gpu.free_tensor(emb_dev);
    let _ = gpu.free_tensor(bias_dev);
    let _ = gpu.free_tensor(logits_dev);
    if let Some(r) = emb_rot {
        let _ = gpu.free_tensor(r);
    }

    // ── Confidence head: computed ON GPU per slot in the markov loop above
    //    (`confidence_proj · [x_head[i] ++ markov_embed[i]]`). Download only the
    //    `block` confidence logits — no x_head / per-slot-embed / proj-weight
    //    d2h. ───────────────────────────────────────────────────────────────
    let mut confidence = vec![0.0f32; block];
    {
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(confidence.as_mut_ptr() as *mut u8, block * 4)
        };
        gpu.hip
            .memcpy_dtoh(bytes, &conf_batch.buf)
            .map_err(|e| format!("dspark d2h confidence: {e:?}"))?;
    }
    let _ = gpu.free_tensor(conf_batch);
    let _ = gpu.free_tensor(concat_dev);
    let _ = gpu.free_tensor(x_head);

    Ok(DraftResult {
        // Draft logits are not materialized on the host: argmax happens on-GPU
        // and nothing downstream consumes them (verify re-runs the trunk head).
        tokens: out_ids[1..=block].to_vec(),
        logits: Vec::new(),
        confidence,
    })
}

/// Download a `[rows, cols]` weight as F32, dispatching on dtype. Used for
/// the tiny confidence-head projection where a CPU dot-product is simplest.
/// Supports F32 / F16 (Q8_0 / MQ4 confidence projections are not expected —
/// the head ships F16/F32 — and error out clearly if encountered).
fn dspark_download_weight_f32(
    gpu: &Gpu,
    w: &GpuTensor,
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    match w.dtype {
        DType::F32 => gpu
            .download_f32(w)
            .map_err(|e| format!("dspark d2h weight f32: {e:?}")),
        DType::F16 => {
            let n = rows * cols;
            let mut bytes = vec![0u8; n * 2];
            gpu.hip
                .memcpy_dtoh(&mut bytes, &w.buf)
                .map_err(|e| format!("dspark d2h weight f16: {e:?}"))?;
            Ok((0..n)
                .map(|i| {
                    hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([
                        bytes[i * 2],
                        bytes[i * 2 + 1],
                    ]))
                })
                .collect())
        }
        DType::Q8_0 => {
            // Q8_0: 34-byte blocks (f16 scale + 32 int8) over rows*cols values.
            // The deepseek4-q8-mtp quant routes confidence_proj/markov to Q8F16
            // (= DType::Q8_0 on device), so handle it here rather than forcing a
            // re-quant. proj is tiny ([1, 4352]) → CPU dequant is cheap.
            let n = rows * cols;
            let nblocks = n.div_ceil(32);
            let mut bytes = vec![0u8; nblocks * 34];
            gpu.hip
                .memcpy_dtoh(&mut bytes, &w.buf)
                .map_err(|e| format!("dspark d2h weight q8_0: {e:?}"))?;
            Ok(hipfire_runtime::llama::dequantize_q8_0(&bytes, n))
        }
        other => Err(format!(
            "dspark confidence_proj: unsupported dtype {other:?} (expected F16/F32/Q8_0)"
        )),
    }
}

/// One GPU-vs-CPU numeric parity check for a DSpark novel-head kernel.
#[derive(Debug, Clone)]
pub struct DsparkParityCheck {
    pub name: &'static str,
    pub n: usize,
    pub max_abs_diff: f32,
    /// `max_abs_diff` relative to the largest CPU-reference magnitude.
    pub rel_max_abs: f32,
    pub cosine: f32,
    pub pass: bool,
}

/// Result of [`dspark_head_parity`] — one entry per novel-head kernel checked.
#[derive(Debug, Clone)]
pub struct DsparkParityReport {
    pub checks: Vec<DsparkParityCheck>,
}

impl DsparkParityReport {
    pub fn all_pass(&self) -> bool {
        self.checks.iter().all(|c| c.pass)
    }
}

/// Cosine + relative-max-abs between a GPU output and its CPU reference. A
/// transpose/layout bug collapses cosine toward 0; Q8 dequant + FMA-order
/// differences leave cosine ≈ 1 with a small relative error — so the pass gate
/// (cosine ≥ 0.9999 AND rel_max_abs ≤ 0.02) separates the two cleanly.
fn dspark_parity_stats(name: &'static str, gpu_v: &[f32], cpu_v: &[f32]) -> DsparkParityCheck {
    let n = gpu_v.len().min(cpu_v.len());
    let mut max_abs = 0.0f32;
    let mut max_cpu = 0.0f32;
    let (mut dot, mut ng, mut nc) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let (g, c) = (gpu_v[i], cpu_v[i]);
        max_abs = max_abs.max((g - c).abs());
        max_cpu = max_cpu.max(c.abs());
        dot += g as f64 * c as f64;
        ng += g as f64 * g as f64;
        nc += c as f64 * c as f64;
    }
    let cosine = if ng > 0.0 && nc > 0.0 {
        (dot / (ng.sqrt() * nc.sqrt())) as f32
    } else {
        0.0
    };
    let rel_max_abs = if max_cpu > 0.0 {
        max_abs / max_cpu
    } else {
        max_abs
    };
    let pass = cosine >= 0.9999 && rel_max_abs <= 0.02;
    DsparkParityCheck {
        name,
        n,
        max_abs_diff: max_abs,
        rel_max_abs,
        cosine,
        pass,
    }
}

/// GPU-vs-CPU numeric parity for the DSpark **novel** head kernels — the code
/// with NO trunk reuse: `main_proj`+`main_norm` (the target-hidden ingestion)
/// and the Markov head (`markov_w1` embed + `markov_w2` bias). Each check runs
/// the EXACT production GPU primitive `dspark_forward` uses on a fixed synthetic
/// input, against an independent CPU reference derived from `inference/model.py`
/// using the real quantized weights, and compares cosine + relative max-abs.
///
/// This is the runnable slice of the plan's mandated "numeric-parity spike".
/// The full fp8 `model.py` reference cannot run on an RDNA box (fp8 kernels +
/// the 167 GB trunk), so we validate the novel LINEAR heads — where a layout /
/// transpose / dequant bug would silently sink draft acceptance and masquerade
/// as "DSpark is just slow" rather than "the port is wrong". Coverage boundary:
///   - main_proj gemv, main_norm RMS, markov embed, markov bias gemv → HERE.
///   - MLA / MoE / HC kernels → reused from the trunk, covered by its gates.
///   - bidirectional KV staging.
///   - hc_head sigmoid gate, confidence dot → host/coherence-covered (the
///     confidence dot already IS a CPU computation; no GPU kernel to diff).
pub fn dspark_head_parity(
    cfg: &DeepseekV4Config,
    dspark: &crate::deepseek4::DsparkWeights,
    gpu: &mut Gpu,
) -> Result<DsparkParityReport, String> {
    let hidden = cfg.hidden_size;
    let three_h = 3 * hidden;
    let rank = dspark.cfg.markov_rank;
    let vocab = cfg.vocab_size;
    let mut checks = Vec::new();

    let main_proj = dspark
        .main_proj
        .as_ref()
        .ok_or("parity: main_proj missing")?;
    let main_norm = dspark
        .main_norm
        .as_ref()
        .ok_or("parity: main_norm missing")?;
    let markov_w1 = dspark
        .markov_w1
        .as_ref()
        .ok_or("parity: markov_w1 missing")?;
    let markov_w2 = dspark
        .markov_w2
        .as_ref()
        .ok_or("parity: markov_w2 missing")?;

    let upload = |gpu: &mut Gpu, v: &[f32]| -> Result<GpuTensor, String> {
        let t = gpu
            .alloc_tensor(&[v.len()], DType::F32)
            .map_err(|e| format!("parity alloc: {e:?}"))?;
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v))
        };
        gpu.memcpy_htod_auto(&t.buf, bytes)
            .map_err(|e| format!("parity htod: {e:?}"))?;
        Ok(t)
    };

    // ── 1+2. main_proj gemv (pre-norm) + main_norm → main_x ─────────────────
    // Deterministic synthetic main_hidden[3*hidden]; no RNG (reproducible).
    let mh: Vec<f32> = (0..three_h)
        .map(|i| ((i as f32) * 0.013).sin() * 0.5)
        .collect();
    let mh_dev = upload(gpu, &mh)?;
    let pre_dev = gpu
        .alloc_tensor(&[hidden], DType::F32)
        .map_err(|e| format!("parity alloc pre: {e:?}"))?;
    if weight_needs_fwht(main_proj) {
        let rot = gpu
            .alloc_tensor(&[three_h], DType::F32)
            .map_err(|e| format!("parity alloc rot: {e:?}"))?;
        gpu.rotate_x_mq(&mh_dev, &rot, three_h)
            .map_err(|e| format!("parity rotate mh: {e:?}"))?;
        gemv_auto(
            gpu,
            Mq2rBackend::Portable,
            main_proj,
            &rot,
            &mh_dev,
            &pre_dev,
            hidden,
            three_h,
        )?;
        let _ = gpu.free_tensor(rot);
    } else {
        gemv_auto(
            gpu,
            Mq2rBackend::Portable,
            main_proj,
            &mh_dev,
            &mh_dev,
            &pre_dev,
            hidden,
            three_h,
        )?;
    }
    let pre_gpu = gpu
        .download_f32(&pre_dev)
        .map_err(|e| format!("parity d2h pre: {e:?}"))?;
    // CPU reference: dequant main_proj [hidden, 3h] row-major, matmul.
    let wproj = dspark_download_weight_f32(gpu, main_proj, hidden, three_h)?;
    let pre_cpu: Vec<f32> = (0..hidden)
        .map(|o| {
            let base = o * three_h;
            (0..three_h).map(|i| wproj[base + i] * mh[i]).sum()
        })
        .collect();
    checks.push(dspark_parity_stats(
        "main_proj gemv (pre-norm)",
        &pre_gpu,
        &pre_cpu,
    ));

    // main_norm: RMS in place on the GPU pre-norm vector (out = x*w*rsqrt(mean(x²)+eps)).
    let x_dev = upload(gpu, &pre_gpu)?;
    gpu.rmsnorm_f32(&x_dev, main_norm, &x_dev, cfg.rms_norm_eps)
        .map_err(|e| format!("parity rmsnorm: {e:?}"))?;
    let x_gpu = gpu
        .download_f32(&x_dev)
        .map_err(|e| format!("parity d2h main_x: {e:?}"))?;
    let g = dspark_download_weight_f32(gpu, main_norm, 1, hidden)?;
    let ms = pre_cpu.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
    let inv = 1.0 / (ms + cfg.rms_norm_eps).sqrt();
    let x_cpu: Vec<f32> = (0..hidden).map(|o| pre_cpu[o] * inv * g[o]).collect();
    checks.push(dspark_parity_stats(
        "main_norm(main_proj) = main_x",
        &x_gpu,
        &x_cpu,
    ));

    // ── 3. markov_w1 embedding lookup (row tok) ─────────────────────────────
    let tok = (vocab / 3) as u32;
    let emb_dev = gpu
        .alloc_tensor(&[rank], DType::F32)
        .map_err(|e| format!("parity alloc emb: {e:?}"))?;
    dspark_embed_one(gpu, markov_w1, &emb_dev, tok, rank)?;
    let emb_gpu = gpu
        .download_f32(&emb_dev)
        .map_err(|e| format!("parity d2h emb: {e:?}"))?;
    let w1 = dspark_download_weight_f32(gpu, markov_w1, vocab, rank)?;
    let emb_cpu = w1[(tok as usize) * rank..(tok as usize + 1) * rank].to_vec();
    checks.push(dspark_parity_stats(
        "markov_w1 embed lookup",
        &emb_gpu,
        &emb_cpu,
    ));

    // ── 4. markov_w2 bias = markov_w2 @ emb ─────────────────────────────────
    let bias_dev = gpu
        .alloc_tensor(&[vocab], DType::F32)
        .map_err(|e| format!("parity alloc bias: {e:?}"))?;
    if weight_needs_fwht(markov_w2) {
        let rot = gpu
            .alloc_tensor(&[rank], DType::F32)
            .map_err(|e| format!("parity alloc emb rot: {e:?}"))?;
        gpu.rotate_x_mq(&emb_dev, &rot, rank)
            .map_err(|e| format!("parity rotate emb: {e:?}"))?;
        gemv_auto(
            gpu,
            Mq2rBackend::Portable,
            markov_w2,
            &rot,
            &emb_dev,
            &bias_dev,
            vocab,
            rank,
        )?;
        let _ = gpu.free_tensor(rot);
    } else {
        gemv_auto(
            gpu,
            Mq2rBackend::Portable,
            markov_w2,
            &emb_dev,
            &emb_dev,
            &bias_dev,
            vocab,
            rank,
        )?;
    }
    let bias_gpu = gpu
        .download_f32(&bias_dev)
        .map_err(|e| format!("parity d2h bias: {e:?}"))?;
    let w2 = dspark_download_weight_f32(gpu, markov_w2, vocab, rank)?;
    // Feed the GPU emb to the CPU reference so this isolates the w2 layout.
    let bias_cpu: Vec<f32> = (0..vocab)
        .map(|v| {
            let base = v * rank;
            (0..rank).map(|r| w2[base + r] * emb_gpu[r]).sum()
        })
        .collect();
    checks.push(dspark_parity_stats(
        "markov_w2 bias gemv",
        &bias_gpu,
        &bias_cpu,
    ));

    let _ = gpu.free_tensor(mh_dev);
    let _ = gpu.free_tensor(pre_dev);
    let _ = gpu.free_tensor(x_dev);
    let _ = gpu.free_tensor(emb_dev);
    let _ = gpu.free_tensor(bias_dev);

    Ok(DsparkParityReport { checks })
}

/// CPU reference implementation of bias-aware top-k: picks the `k` highest
/// `scores[i] + bias[i]` entries, then weights them by their UNBIASED
/// scores (per DeepSeek V4 router semantics — bias only steers selection).
/// Production routing goes through the GPU kernel
/// `deepseek4_moe_topk_bias_aware_f32`; this is kept as a tested reference.
#[cfg(test)]
fn bias_aware_topk_weights(scores: &[f32], bias: &[f32], k: usize) -> Option<(Vec<u32>, Vec<f32>)> {
    let n = scores.len();
    if k == 0 || n == 0 {
        return None;
    }
    let mut biased: Vec<f32> = (0..n)
        .map(|i| scores[i] + bias.get(i).copied().unwrap_or(0.0))
        .collect();
    let k = k.min(n);
    let mut indices: Vec<u32> = Vec::with_capacity(k);
    for _ in 0..k {
        let (best_i, _) = biased
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        indices.push(best_i as u32);
        biased[best_i] = f32::NEG_INFINITY;
    }
    let mut wts: Vec<f32> = indices.iter().map(|&i| scores[i as usize]).collect();
    let w_sum: f32 = wts.iter().sum();
    if w_sum <= 0.0 {
        return None;
    }
    for w in wts.iter_mut() {
        *w /= w_sum;
    }
    Some((indices, wts))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_activation_dump_writes_collector_contract_and_fails_closed() {
        let unique = format!(
            "hipfire-ds4-dense-acts-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let out_dir = std::env::temp_dir().join(unique);
        let tensor_name = "layers.0.attn.wq_b.weight";
        let path = out_dir.join(format!("{tensor_name}.acts"));
        let mut dump = DenseActivationDump::new(out_dir.clone()).unwrap();

        dump.record(tensor_name, 4, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        dump.record(tensor_name, 4, &[5.0, 6.0, 7.0, 8.0]).unwrap();
        assert!(dump
            .record("layers.0.attn.wo_b.weight", 3, &[1.0, 2.0])
            .unwrap_err()
            .contains("not whole"));
        assert_eq!(dump.finish().unwrap(), (1, 2));
        assert!(dump.record(tensor_name, 4, &[0.0; 4]).is_err());

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 8 + 2 * 4 * std::mem::size_of::<f32>());
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 4);
        let values: Vec<f32> = bytes[8..]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(out_dir).unwrap();
    }

    #[test]
    fn mq2r_pins_the_accepted_gfx1151_switches() {
        let arch = "gfx1151";
        assert!(config_cache::e8_wo_grouped_on(arch, true));
        assert!(config_cache::e8_u4_on(arch, true));
        assert!(config_cache::e8_prefill_b2_on(arch, true));
        assert!(config_cache::e8_prefill_b4_on(arch, true));
        assert!(!config_cache::ffn_overlap_on(arch, true));
        assert!(config_cache::hc_pingpong_on(arch, true));
        assert!(config_cache::hc_finalize_fused_on(arch, true));
        assert!(config_cache::hc_control_finalize_fused_on(arch, true));
        assert!(!config_cache::retained_embedding_on(arch, true));
        assert!(!config_cache::hc_control_rsqrt_once_on(arch, true));
        assert!(!config_cache::hc_finalize_input_map_on(arch, true));
        assert!(!config_cache::qnorm_rotate_fused_on(arch, true));
        assert!(!config_cache::redline_ffn_split_on(arch, true));
        // G1 is gfx1151's own certified two-stage route and defaults ON for
        // every DeepSeek4 weight tier; indexer state is format-independent.
        assert!(config_cache::gfx1151_indexer_topk_two_stage_on(arch));
        // gfx942 A2 levers must not bleed into gfx1151.
        assert!(!config_cache::gfx942_compressor_gate_on(arch, true));
        assert!(!config_cache::gfx942_indexer_topk_bounded_on(arch, true));
        assert!(!config_cache::gfx942_indexer_topk_two_stage_on(arch, true));
        assert!(!config_cache::gfx942_e8_wo_grouped_on(arch, true));
        assert!(!config_cache::gfx942_hc_finalize_fused_on(arch, true));
        assert!(!config_cache::gfx942_indexer_topk_parallel_on(arch, true));
        assert!(!config_cache::gfx942_ffn_overlap_on(false));
    }

    #[test]
    fn mq2r_pins_only_the_admitted_gfx1201_hc_fusions() {
        assert!(config_cache::hc_finalize_fused_on("gfx1201", true));
        assert!(!config_cache::hc_finalize_fused_on("gfx1201", false));
        assert!(config_cache::hc_control_finalize_fused_on("gfx1201", true));
        assert!(!config_cache::hc_control_finalize_fused_on(
            "gfx1201", false
        ));
        assert!(!config_cache::hc_finalize_fused_on("gfx1100", true));
        assert!(!config_cache::hc_finalize_fused_on("gfx942", true));
        assert!(config_cache::gfx1201_e8_wo_grouped_on("gfx1201", true));
        assert!(!config_cache::gfx1201_e8_wo_grouped_on("gfx1201", false));
        assert!(!config_cache::gfx1201_e8_wo_grouped_on("gfx1151", true));
        assert!(!config_cache::gfx1201_e8_wo_grouped_on("gfx942", true));
        assert!(config_cache::gfx1201_rmsnorm_rotate_nox_on("gfx1201", true));
        assert!(!config_cache::gfx1201_rmsnorm_rotate_nox_on(
            "gfx1201", false
        ));
        assert!(!config_cache::gfx1201_rmsnorm_rotate_nox_on(
            "gfx1151", true
        ));
        assert!(!config_cache::gfx1201_rmsnorm_rotate_nox_on(
            "gfx1100", true
        ));
    }

    #[test]
    fn wave32_two_stage_starts_after_the_short_bucket() {
        assert!(!two_stage_topk_capacity_eligible(
            "gfx1151",
            crate::deepseek4::INITIAL_COMPRESSED_ROWS,
        ));
        assert!(two_stage_topk_capacity_eligible(
            "gfx1151",
            crate::deepseek4::INITIAL_COMPRESSED_ROWS * 2,
        ));
        // gfx942 keeps its independently promoted capacity threshold.
        assert!(two_stage_topk_capacity_eligible("gfx942", 2_048));
        assert!(!two_stage_topk_capacity_eligible("gfx942", 512));
        assert!(!two_stage_topk_capacity_eligible(
            "gfx1201",
            crate::deepseek4::INITIAL_COMPRESSED_ROWS
        ));
        assert!(two_stage_topk_capacity_eligible(
            "gfx1201",
            crate::deepseek4::INITIAL_COMPRESSED_ROWS * 2
        ));
    }

    #[test]
    fn route_scale_never_uses_the_checkpoint_value_on_this_path() {
        // The checkpoint declares 1.5 and every DS4 artifact does, but 1.5 is
        // measurably bad on this crate's MoE path: 16.31 PPL at ctx2048 against
        // 10.81 at 2.0, a 51% penalty. 672373ce1 moved the default to the
        // checkpoint on provenance grounds and silently regressed every
        // non-mq2r artifact from 2.2 to 1.5. This is the guard against that
        // happening a third time.
        for cfg_scale in [1.5f32, 1.8, 3.0] {
            assert_ne!(
                config_cache::resolve_route_scale(cfg_scale, None, false),
                cfg_scale,
                "the checkpoint value must not reach the routed branch unmodified"
            );
        }
        // An explicit override always wins, so the compensation stays escapable.
        assert_eq!(
            config_cache::resolve_route_scale(1.5, Some(2.2), false),
            2.2,
            "explicit HIPFIRE_DEEPSEEK4_ROUTE_SCALE override must win"
        );
        assert_eq!(
            config_cache::resolve_route_scale(1.5, Some(1.0), true),
            1.0,
            "override must win over the mq2r default too"
        );
    }

    #[test]
    fn route_scale_per_build_defaults_mq2r_1_8_others_2_2() {
        // The 0731 `.mq2r` sweep at ctx2048 (wikitext2 md5 83b0205a…, fresh
        // process per run, effective_route_scale logged) puts the minimum at
        // 1.8: 16.306 / 10.688 / 10.810 / 11.174 / 11.408 across
        // 1.5 / 1.8 / 2.0 / 2.2 / 2.4. An older ctx256 sweep said 2.0, but a
        // second ctx256 table from the same period contradicts it, so the clean
        // long-context measurement wins.
        assert_eq!(
            config_cache::resolve_route_scale(1.5, None, true),
            1.8,
            "the .mq2r build must ship at its measured ctx2048 optimum"
        );
        // MQ2-Lloyd and every other DS4 artifact get the value the arch was
        // calibrated at in b263fb3cc and served on for two months. 672373ce1
        // dropped them to the checkpoint's 1.5, which is a ~51% PPL regression;
        // this asserts they are back.
        assert_eq!(
            config_cache::resolve_route_scale(1.5, None, false),
            2.2,
            "non-mq2r DS4 must use the calibrated 2.2, not the checkpoint 1.5"
        );
        // Both defaults sit well above the header value. That is the point:
        // the reference applies 1.5 and scores PPL 4.693, so 1.5 is right for
        // the model and wrong for this crate's MoE routed branch. Both builds
        // needing >1.5 makes it systematic, not per-artifact.
        for mq2r in [true, false] {
            assert!(
                config_cache::resolve_route_scale(1.5, None, mq2r) > 1.5,
                "routed branch is systematically weak here; compensation must not vanish"
            );
        }
    }

    #[test]
    fn gfx942_a2_lever_defaults_are_arch_gated() {
        // L3 and F1 are certified defaults for exact gfx942 route v1; L1 is
        // still an opt-in experiment.
        assert!(!config_cache::gfx942_compressor_gate_on("gfx942", true));
        assert!(config_cache::gfx942_indexer_topk_bounded_on("gfx942", true));
        assert!(config_cache::gfx942_indexer_topk_two_stage_on(
            "gfx942", true
        ));
        assert!(config_cache::gfx942_e8_wo_grouped_on("gfx942", true));
        assert!(!config_cache::gfx942_hc_finalize_fused_on("gfx942", true));
        assert!(config_cache::gfx942_indexer_topk_parallel_on(
            "gfx942", true
        ));
        // Fail closed on non-mq2r and non-gfx942.
        assert!(!config_cache::gfx942_compressor_gate_on("gfx942", false));
        assert!(!config_cache::gfx942_indexer_topk_bounded_on(
            "gfx942", false
        ));
        assert!(!config_cache::gfx942_indexer_topk_two_stage_on(
            "gfx942", false
        ));
        assert!(!config_cache::gfx942_e8_wo_grouped_on("gfx942", false));
        assert!(!config_cache::gfx942_hc_finalize_fused_on("gfx942", false));
        assert!(!config_cache::gfx942_indexer_topk_parallel_on(
            "gfx942", false
        ));
        assert!(!config_cache::gfx942_ffn_overlap_on(false));
        assert!(!config_cache::gfx942_compressor_gate_on("gfx1100", true));
        assert!(!config_cache::gfx942_indexer_topk_bounded_on(
            "gfx1100", true
        ));
        assert!(!config_cache::gfx942_indexer_topk_two_stage_on(
            "gfx1100", true
        ));
        assert!(!config_cache::gfx942_e8_wo_grouped_on("gfx1201", true));
        assert!(!config_cache::gfx942_hc_finalize_fused_on("gfx1201", true));
        assert!(!config_cache::gfx942_indexer_topk_parallel_on(
            "gfx1201", true
        ));
        assert!(!config_cache::gfx942_indexer_topk_bounded_on(
            "gfx1201", true
        ));
        assert!(!config_cache::gfx942_indexer_topk_two_stage_on(
            "gfx1201", true
        ));
        assert!(!config_cache::gfx942_compressor_gate_on("gfx1151", true));
        assert!(!config_cache::gfx942_indexer_topk_bounded_on(
            "gfx1151", true
        ));
        assert!(!config_cache::gfx942_indexer_topk_two_stage_on(
            "gfx1151", true
        ));
        assert!(!config_cache::gfx942_indexer_topk_parallel_on(
            "gfx1151", true
        ));
        // gfx1151 grouped flag stays gfx1151-only.
        assert!(!config_cache::e8_wo_grouped_on("gfx942", true));
        // G1 is the gfx1151 twin of F3 and must not bleed into gfx942 or any
        // other arch. Unlike the gfx942 native backend, its inputs are
        // format-independent DS4 F32 indexer state, so MQ2-Lloyd and MQ2R use
        // the same exact-device route.
        assert!(!config_cache::gfx1151_indexer_topk_two_stage_on("gfx942"));
        assert!(!config_cache::gfx1151_indexer_topk_two_stage_on("gfx1100"));
        assert!(!config_cache::gfx1151_indexer_topk_two_stage_on("gfx1201"));
        assert!(config_cache::gfx1151_indexer_topk_two_stage_on("gfx1151"));
        // The heterogeneous dense device gets an independently admitted
        // exact-gfx1100 code object.  It must not widen either G1 or F3.
        assert!(config_cache::gfx1100_indexer_topk_two_stage_on("gfx1100"));
        assert!(!config_cache::gfx1100_indexer_topk_two_stage_on("gfx942"));
        assert!(!config_cache::gfx1100_indexer_topk_two_stage_on("gfx1151"));
        assert!(!config_cache::gfx1100_indexer_topk_two_stage_on("gfx1201"));
        assert!(config_cache::gfx1201_indexer_topk_two_stage_on("gfx1201"));
        assert!(!config_cache::gfx1201_indexer_topk_two_stage_on("gfx942"));
        assert!(!config_cache::gfx1201_indexer_topk_two_stage_on("gfx1151"));
        assert!(!config_cache::gfx1201_indexer_topk_two_stage_on("gfx1100"));
    }

    #[test]
    fn bias_aware_topk_picks_biased_indices() {
        // Bias steers selection. scores=[1,1,1,1,1,1], bias=[0,0,0,3,2,0]
        // → biased=[1,1,1,4,3,1] → top-2 = [3, 4].
        let scores = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let bias = vec![0.0, 0.0, 0.0, 3.0, 2.0, 0.0];
        let (idx, wts) = bias_aware_topk_weights(&scores, &bias, 2).unwrap();
        assert_eq!(idx, vec![3, 4]);
        // Weights come from UNBIASED scores (both 1.0), normalized.
        assert!((wts[0] - 0.5).abs() < 1e-6);
        assert!((wts[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn bias_aware_topk_weights_use_unbiased_scores() {
        // scores=[5, 1, 1], bias=[0, 10, 10] → biased=[5, 11, 11].
        // Top-2 by biased = [1, 2]. Weights from unbiased = [1, 1] → [0.5, 0.5].
        let scores = vec![5.0, 1.0, 1.0];
        let bias = vec![0.0, 10.0, 10.0];
        let (idx, wts) = bias_aware_topk_weights(&scores, &bias, 2).unwrap();
        assert!(idx == vec![1, 2] || idx == vec![2, 1]);
        assert!((wts[0] - 0.5).abs() < 1e-6);
        assert!((wts[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn bias_aware_topk_falls_back_zero_bias() {
        // No bias → pure top-K from scores.
        let scores = vec![0.1, 0.9, 0.5, 0.7];
        let bias: Vec<f32> = vec![];
        let (idx, wts) = bias_aware_topk_weights(&scores, &bias, 2).unwrap();
        assert_eq!(idx, vec![1, 3]);
        let s = 0.9 + 0.7;
        assert!((wts[0] - 0.9 / s).abs() < 1e-6);
        assert!((wts[1] - 0.7 / s).abs() < 1e-6);
    }

    #[test]
    fn bias_aware_topk_returns_none_on_zero_sum() {
        // All scores zero → no positive weight sum.
        let scores = vec![0.0, 0.0, 0.0];
        let bias = vec![5.0, 0.0, 0.0]; // bias picks idx 0 but score is 0
        assert!(bias_aware_topk_weights(&scores, &bias, 1).is_none());
    }

    #[test]
    fn bias_aware_topk_handles_k_geq_n() {
        // k=4 but only n=2 scores — caller's job to set k correctly,
        // but we silently clamp rather than panic.
        let scores = vec![1.0, 2.0];
        let bias = vec![0.0, 0.0];
        let (idx, wts) = bias_aware_topk_weights(&scores, &bias, 4).unwrap();
        assert_eq!(idx.len(), 2);
        assert!(wts.iter().sum::<f32>() > 0.99 && wts.iter().sum::<f32>() < 1.01);
    }

    #[test]
    fn mfp_e8_row_bytes_match_wire_layouts() {
        assert_eq!(mfp_e8_row_bytes(DType::MFP4G32E8, 4096), 2192);
        assert_eq!(mfp_e8_row_bytes(DType::MFP4G32E8SOA, 4096), 2192);
        assert_eq!(mfp_e8_row_bytes(DType::MFP3G32E8, 4096), 1680);
        // At K=256 the SoA scale plane pads eight block scales to 16 B.
        assert_eq!(mfp_e8_row_bytes(DType::MFP4G32E8, 256), 152);
        assert_eq!(mfp_e8_row_bytes(DType::MFP4G32E8SOA, 256), 160);
        assert_eq!(mfp_e8_row_bytes(DType::MFP3G32E8, 256), 120);
    }

    #[test]
    fn compressed_count_never_exceeds_backing_cache() {
        let cap = 2048;

        assert_eq!(capped_compressed_count(0, 4, cap), 0);
        assert_eq!(capped_compressed_count(3, 4, cap), 1);
        assert_eq!(capped_compressed_count(8191, 4, cap), cap);
        assert_eq!(capped_compressed_count(8192, 4, cap), cap);
        assert_eq!(capped_compressed_count(8195, 4, cap), cap);

        assert_eq!(capped_compressed_count(262143, 128, cap), cap);
        assert_eq!(capped_compressed_count(262271, 128, cap), cap);
    }
}

#[cfg(test)]
mod ship6_lower_tests {
    use super::*;
    use superop::SuperOpKind::{Attend, Moe};

    // #397 Ship 6 — deepseek4 is one variant (every layer Attn+MoE; per-layer
    // conditionals live inside the handlers).
    #[test]
    fn ds4_program_is_attend_then_moe() {
        let kinds: Vec<_> = ds4_lower_program().iter().map(|o| o.kind).collect();
        assert_eq!(kinds, vec![Attend, Moe]);
    }
}
