// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#![allow(
    dead_code,
    unused_imports,
    unused_variables,
    non_snake_case,
    clippy::all
)]

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use crate::calibration::*;
use crate::cli::{guard_qwen3_arch_override, QuantizeArgs};
use crate::dequant::*;
use crate::e8;
use crate::e8_gptq;
use crate::gguf_input;
use crate::hfq::*;
use crate::model_filter::*;
use crate::quant_e8::*;
use crate::quant_fwht::*;
use crate::quant_hfp4::*;
use crate::quant_mq::*;
use crate::quant_q4::*;
use crate::reap_overlay;
use clap::Parser;
use hipfire_quantize::float16::{bf16_to_f32, f16_to_f32, f32_to_f16};
use hipfire_quantize::hessian_io;
use hipfire_quantize::safetensors_file::{SafetensorsFile, TensorMeta};

/// 2D-weight quantization target chosen at the per-tensor level. The choice
/// per format flag:
///
/// | --format | 2D weights      | embedding | comment                          |
/// |----------|-----------------|-----------|----------------------------------|
/// | hfq4     | HFQ4G256        | Q8F16     | dense default — no FWHT, plain   |
/// | hfq6     | HFQ6G256        | Q8F16     | dense + higher quality           |
/// | mq4      | MQ4G256         | Q8F16     | Qwen3.5+ (DeltaNet) — FWHT-rot   |
/// | mq5      | MQ5G256         | Q8F16     | 5-bit FWHT (5.25 bpw, 168 B/grp) |
/// | mq6      | MQ6G256         | Q8F16     | Qwen3.5+ (DeltaNet) + higher q   |
/// | mq3      | MQ3G256         | Q8F16     | Sub-4-bit FWHT (3.25 bpw)        |
/// | mq2      | MQ2G256         | Q8F16     | Sub-4-bit FWHT (2.25 bpw)        |
///
/// **MQ4/MQ6 for non-Qwen3.5 dense produces correct output on the Llama path
/// (the rotation cancels via `gemv_mq4g256_with_rotate`) but adds per-layer
/// `rotate_x_mq` overhead with no quality benefit — those rotations were
/// calibrated for Qwen3.5+ training.** Default is HFQ4 for dense GGUFs;
/// pass `--format mq4` only when the source is a Qwen3.5+ family model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GgufFormat {
    Hfq4,
    Hfq6,
    Mq4,
    Mq4V2,
    Mq4C,
    Mq5,
    Mq6,
    Mq3,
    Mq2,
    Mq2Lloyd,
    Mq2LloydAnchored,
    Mq3Lloyd,
    Mq4Lloyd,
    Mq6V2,
    Mq5V2,
    Mq3V2,
    Mq2V2,
    Hfp4,      // HFP4G32 — RDNA-optimal FP4 (E2M1 + UE8M0 g32 + FP16 row scale)
    Mfp4,      // MFP4G32 — HFP4G32 + offline FWHT rotation (drop-in MQ4 replacement)
    Mfp4Lloyd, // mfp4 + per-tensor 16-entry Lloyd codebook
    Mfp4P,     // mfp4+P — mfp4 with E4M3 (non-power-of-2) per-block scale
    Mfp4E8, // mfp4-E8 — mfp4+P container with E8-lattice vector quantization (4 codewords/32 weights)
    Mfp4E8Soa, // mfp4-E8 SoA — same E8 data in structure-of-arrays layout for coalesced GEMV
    Mfp3E8, // mfp3-E8 — mfp4-E8 frame with 3-bit lattice (13 B/blk, 3.25 bpw; drop-in for MQ3-Lloyd cold)
    Mfp2E8, // mfp2-E8 — mfp4-E8 frame with 2-bit lattice (9 B/blk, 2.25 bpw; drop-in for MQ2-Lloyd cold)
    /// Ternary — PrismML Bonsai family. Source GGUF is already mixed-precision
    /// (Q2_0 ternary matmuls + Q8_0/F16 embeddings + F32/F16 norms); the
    /// per-tensor precision PrismML chose is authoritative, so 2D matmul
    /// tensors already in Q2_0 pass through byte-verbatim to TQ2G128
    /// (see the dedicated arm in `run_gguf_pipeline`) rather than going
    /// through the kmap/Q8 rules. `quantize_tq2g128` is only the re-quant
    /// fallback for the rare non-Q2_0 matmul tensor under this format.
    Ternary,
    /// Binary — PrismML Bonsai family, 1-bit variant. Source GGUF is already
    /// mixed-precision (Q1_0 binary matmuls + Q8_0/F16 embeddings + F32/F16
    /// norms); the per-tensor precision PrismML chose is authoritative, so 2D
    /// matmul tensors already in Q1_0 pass through byte-verbatim to BQ1G128
    /// (see the dedicated arm in `run_gguf_pipeline`) rather than going
    /// through the kmap/Q8 rules. There is no re-quant fallback for this
    /// format — binary weights always take the byte-verbatim passthrough.
    Binary,
}
impl GgufFormat {
    pub(crate) fn from_flag(flag: &str) -> Option<Self> {
        match flag {
            "hfq4" | "hfq4g256" | "hf4" => Some(Self::Hfq4),
            "hfq6" | "hfq6g256" | "hf6" => Some(Self::Hfq6),
            "mq4v1" | "mq4g256" | "magnum" => Some(Self::Mq4),
            "mq4v2" | "mq4" | "mq4g256v2" => Some(Self::Mq4V2),
            "mq4c" | "mq4cg256" | "mq4g256c" => Some(Self::Mq4C),
            "mq5" | "mq5g256" => Some(Self::Mq5),
            "mq6" | "mq6g256" => Some(Self::Mq6),
            "mq3" | "mq3g256" => Some(Self::Mq3),
            "mq2" | "mq2g256" => Some(Self::Mq2),
            "mq6v2" | "mq6g256v2" => Some(Self::Mq6V2),
            "mq5v2" | "mq5g256v2" => Some(Self::Mq5V2),
            "mq3v2" | "mq3g256v2" => Some(Self::Mq3V2),
            "mq2v2" | "mq2g256v2" => Some(Self::Mq2V2),
            "mq2-lloyd" | "mq2g256-lloyd" | "mq2lloyd" => Some(Self::Mq2Lloyd),
            "mq2lloyd-anchored"
            | "mq2lloyd_anchored"
            | "mq2-lloyd-anchored"
            | "mq2-lloyd_anchored"
            | "mq2g256-lloyd-anchored"
            | "mq2g256-lloyd_anchored" => Some(Self::Mq2LloydAnchored),
            "mq3-lloyd" | "mq3g256-lloyd" | "mq3lloyd" => Some(Self::Mq3Lloyd),
            "mq4-lloyd" | "mq4g256-lloyd" | "mq4lloyd" => Some(Self::Mq4Lloyd),
            "hfp4" | "hfp4g32" | "hf4p" | "fp4" => Some(Self::Hfp4),
            "mfp4" | "mfp4g32" | "mf4p" => Some(Self::Mfp4),
            "mfp4l" | "mfp4-lloyd" | "mfp4g32-lloyd" | "mfp4lloyd" => Some(Self::Mfp4Lloyd),
            "mfp4p" | "mfp4+p" | "mfp4-p" => Some(Self::Mfp4P),
            "mfp4e8" | "mfp4-e8" | "mfp4l8" => Some(Self::Mfp4E8),
            "mfp4e8soa" | "mfp4-e8-soa" | "mfp4e8-soa" => Some(Self::Mfp4E8Soa),
            "mfp3e8" | "mfp3-e8" => Some(Self::Mfp3E8),
            "mfp2e8" | "mfp2-e8" => Some(Self::Mfp2E8),
            "ternary" | "tq2" | "tq2g128" => Some(Self::Ternary),
            "binary" | "bq1" | "bq1g128" => Some(Self::Binary),
            _ => None,
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Hfq4 => "HFQ4G256",
            Self::Hfq6 => "HFQ6G256",
            Self::Mq4 => "MQ4G256",
            Self::Mq4V2 => "MQ4G256V2",
            Self::Mq4C => "MQ4CG256",
            Self::Mq5 => "MQ5G256",
            Self::Mq6 => "MQ6G256",
            Self::Mq3 => "MQ3G256",
            Self::Mq2 => "MQ2G256",
            Self::Mq6V2 => "MQ6G256V2",
            Self::Mq5V2 => "MQ5G256V2",
            Self::Mq3V2 => "MQ3G256V2",
            Self::Mq2V2 => "MQ2G256V2",
            Self::Mq2Lloyd => "MQ2G256Lloyd",
            Self::Mq2LloydAnchored => "MQ2G256Lloyd",
            Self::Mq3Lloyd => "MQ3G256Lloyd",
            Self::Mq4Lloyd => "MQ4G256Lloyd",
            Self::Hfp4 => "HFP4G32",
            Self::Mfp4 => "MFP4G32",
            Self::Mfp4Lloyd => "MFP4G32Lloyd",
            Self::Mfp4P => "MFP4G32P",
            Self::Mfp4E8 => "MFP4G32E8",
            Self::Mfp4E8Soa => "MFP4G32E8SOA",
            Self::Mfp3E8 => "MFP3G32E8",
            Self::Mfp2E8 => "MFP2G32E8",
            Self::Ternary => "TQ2G128",
            Self::Binary => "BQ1G128",
        }
    }
}
/// True for MQ{2,3,5,6}V2 — dense-only formats (gfx1201 Qwen3.8).
/// MoE GGUFs must not select these: expert packs are often non-2D and would
/// silently fall through the GGUF `!is_2d` arm to F16.
pub(crate) fn gguf_format_is_dense_only_mq_v2(format: GgufFormat) -> bool {
    matches!(
        format,
        GgufFormat::Mq6V2 | GgufFormat::Mq5V2 | GgufFormat::Mq3V2 | GgufFormat::Mq2V2
    )
}

/// MoE / MoE-like arch ids that reject dense-only MQ V2 formats on the GGUF path.
/// Mirrors `is_moe_like` in `pipeline.rs` (qwen MoE, deepseek4, minimax, lfm2,
/// cohere2moe, gemma4).
pub(crate) fn gguf_arch_is_moe_like(arch_id: u32) -> bool {
    matches!(arch_id, 6 | 9 | 10 | 11 | 12 | 13)
}

/// Q1_0 on-disk layout == hipfire BQ1G128 layout: copy verbatim.
pub(crate) fn convert_binary_tensor(
    src: &[u8],
    dtype: gguf_input::GgmlType,
) -> (Vec<u8>, crate::hfq::QuantType, u32) {
    debug_assert_eq!(dtype, gguf_input::GgmlType::Q1_0);
    (src.to_vec(), crate::hfq::QuantType::BQ1G128, 128)
}

/// True for GGUF tensors that belong to the MTP (multi-token-prediction) head
/// rather than the trunk.
///
/// llama.cpp appends the head's block(s) AFTER the trunk and counts them in
/// `block_count`. hipfire loads MTP from a separate sidecar, so those tensors
/// must stay out of the trunk file. Two independent signals are used because
/// neither is sufficient alone:
///
/// * the `nextn.` slot prefix, which llama.cpp emits for the head's own tensors
///   (`nextn.eh_proj`, `nextn.enorm`, `nextn.hnorm`, `nextn.shared_head_norm`);
/// * the block index, because the head's attention and FFN tensors carry
///   ordinary slot names (`attn_q`, `ffn_gate`, ...) and are distinguishable
///   from trunk layers only by sitting at an index >= the trunk length.
///
/// `trunk_len` is `None` when the checkpoint does not advertise
/// `nextn_predict_layers`, in which case only the prefix test applies —
/// a model with no MTP head converts byte-identically to before.
fn gguf_is_mtp_tensor(name: &str, trunk_len: Option<usize>, has_nextn: bool) -> bool {
    if name.contains(".nextn.") || name.ends_with(".nextn") {
        return true;
    }
    if !has_nextn {
        return false;
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return false;
    };
    let Some(dot) = rest.find('.') else {
        return false;
    };
    match (rest[..dot].parse::<usize>(), trunk_len) {
        (Ok(idx), Some(t)) => idx >= t,
        _ => false,
    }
}

/// V-head 重排的作用轴。
#[derive(Clone, Copy)]
enum VReorderAxis {
    /// 行方向（输出维）：in_proj_qkv 的 V 段 / in_proj_z / in_proj_a / in_proj_b /
    /// conv1d 的 V 通道段，以及 1-D 的 A_log / dt_bias。
    Rows,
    /// 列方向（收缩维）：out_proj。
    Cols,
}

/// V-head 逆重排（行方向）。`[start_row, start_row + v_heads*blk)` 这段按 `blk` 切块，
/// 块序 tiled → grouped：`dst_blk = b*v_per_k + a` 取 `src_blk = a*k_heads + b`。
fn vreorder_rows_inplace(
    d: &mut [f32],
    cols: usize,
    start_row: usize,
    k_heads: usize,
    v_per_k: usize,
    blk: usize,
) {
    let src = d.to_vec();
    for b in 0..k_heads {
        for a in 0..v_per_k {
            let dst_row = start_row + (b * v_per_k + a) * blk;
            let src_row = start_row + (a * k_heads + b) * blk;
            for dd in 0..blk {
                let d_off = (dst_row + dd) * cols;
                let s_off = (src_row + dd) * cols;
                d[d_off..d_off + cols].copy_from_slice(&src[s_off..s_off + cols]);
            }
        }
    }
}

/// V-head 逆重排（列方向，out_proj 的收缩维）。
fn vreorder_cols_inplace(
    d: &mut [f32],
    rows: usize,
    cols: usize,
    k_heads: usize,
    v_per_k: usize,
    blk: usize,
) {
    let src = d.to_vec();
    for r in 0..rows {
        let base = r * cols;
        for b in 0..k_heads {
            for a in 0..v_per_k {
                let d_off = base + (b * v_per_k + a) * blk;
                let s_off = base + (a * k_heads + b) * blk;
                d[d_off..d_off + blk].copy_from_slice(&src[s_off..s_off + blk]);
            }
        }
    }
}

/// GGUF 写入的「已折算」约定 → hipfire 运行时期望的 HF 约定。
///
/// llama.cpp 系的 GGUF 转换器会预先把若干张量折算，因为 llama.cpp 直接消费折算后
/// 的值；hipfire 运行时则按 HuggingFace 的原始语义加载。以下四类必须折回，否则权重
/// 「看起来对」（形状 / 类型 / 字节数全对）而推理输出完全乱码。
///
/// 全部结论由与参考 HFQ `qwen3.8-27b.mq4-xt` 的逐元素比对实测得出（2026-09-28）。
///
/// 1. **norm**（`attn_norm` / `ffn_norm` / `attn_q_norm` / `attn_k_norm` /
///    `output_norm`，共 161 个）：HF 里存原始 `w`，加载时 `out = w + 1.0`
///    （运行时 `load_norm_weight` 的 `+= 1.0`）。GGUF 存的是已折算的 `w + 1`。
///    实测：GGUF `output_norm.weight` = 1.960938，参考 HFQ = 0.960938。
///    ⚠️ **必须排除 `ssm_norm`**：GGUF 名 `blk.N.ssm_norm.weight` 含 `_norm`，
///    会被 `gguf_is_norm_tensor()` 归成 norm，但实测它与参考**完全相同**
///    （GGUF 里存的就是 HF 原始值），减 1 会把 48 个线性注意力 RMSNorm 打歪。
/// 2. **ssm_a → linear_attn.A_log**：运行时自己算 `-exp(a_log)`
///    （`rdna-compute/src/norm.rs`: `alpha[i] = softplus(alpha[i] + dt_bias[i])
///    * (-exp(a_log[i]))`），即它要 **log 域**的 `A_log`；GGUF 存的是折算后的
///    衰减系数 `-exp(A_log)`（实测 `-0.040635 = -exp(-3.203125)`）。
/// 3. **V-head 逆重排**：llama.cpp 的 `_LinearAttentionVReorderBase`
///    (`conversion/qwen.py:453`) 在写 GGUF 时把 linear_attn 的 V 维从 HF 的
///    grouped 布局转成 ggml 广播用的 tiled 布局；本转换器直接吃 GGUF，必须转回来。
///    形式：沿某维 view 成 `[k_heads, v_per_k, blk]` → 转置前两轴 → reshape 回去
///    （转置自逆，正反向同式）；`blk` = head_v_dim（a/b 门控为 1）。
///    覆盖 1-D（`ssm_a` / `ssm_dt.bias`，实测置换 `[0,3,6,...,45, 1,4,...,46, ...]`）
///    与 2-D / conv1d（`attn_qkv` 的 V 段、`attn_gate`、`ssm_alpha`、`ssm_beta`、
///    `ssm_out`、`ssm_conv1d` 的 V 通道段）。2-D 部分的验收证据（2026-09-28）：
///    conv1d（Q8_0 无损）置换后与参考 **R² = 1.0000**（逐值一致）；5 个 2-D 族在
///    权重域用参考自带 `awq_scale` 侧车对账，R² 从 −0.87~0.51 升到 0.946~0.984。
/// 4. 其余张量原样。
///
/// 本函数只覆盖 `arch_id == 5`（`qwen35` / `qwen3_5` / `qwen3_5_text`）。
/// `qwen3_5_moe` 是 arch 6，**未覆盖** —— 它的 MoE 变体十有八九有同样的约定，
/// 但没有实测样本，按「只验证过才改」的原则留白。
/// `vreorder` 为 `None`（元数据缺失 / 非 arch 5）时跳过第 3 条，调用处会打印告警。
fn gguf_convention_to_hf(
    gguf_name: &str,
    arch_id: u32,
    is_norm: bool,
    shape: &[u32],
    vreorder: Option<(usize, usize, usize)>,
    d: &mut [f32],
) {
    // 只对 qwen3.5 / 3.8 的 hybrid 栈启用；其余 arch 的运行时约定未逐一核对。
    if arch_id != 5 || d.is_empty() {
        return;
    }
    let is_ssm_a = gguf_name.ends_with("ssm_a");
    let is_ssm_dt = gguf_name.ends_with("ssm_dt.bias");
    // `blk.N.ssm_norm.weight` 会被 `gguf_is_norm_tensor()` 判成 norm（含 `_norm`），
    // 但它存的就是 HF 原始值，实测与参考完全相同 —— 必须排除。
    let is_ssm_norm = gguf_name.ends_with("ssm_norm.weight");

    // (3) V-head 逆重排（1-D 与 2-D 统一处理）。
    if let Some((k_heads, v_heads, hdim)) = vreorder {
        if k_heads > 0 && v_heads % k_heads == 0 && hdim > 0 {
            let v_per_k = v_heads / k_heads;
            let rows = shape.first().copied().unwrap_or(0) as usize;
            // HFQ 的 shape 是行主序、最后一维最快；`cols` 就是每行的元素数。
            let cols: usize = shape
                .iter()
                .skip(1)
                .map(|&s| s as usize)
                .product::<usize>()
                .max(1);
            let plan: Option<(VReorderAxis, usize, usize)> =
                if is_ssm_a || is_ssm_dt {
                    // 1-D，每个 v-head 一个标量。
                    Some((VReorderAxis::Rows, 0, 1))
                } else if gguf_name.ends_with("attn_qkv.weight")
                    || gguf_name.ends_with("ssm_conv1d.weight")
                {
                    // q/k 各占 k_heads*hdim 行（conv1d 是「通道」），V 段在后。
                    Some((VReorderAxis::Rows, 2 * k_heads * hdim, hdim))
                } else if gguf_name.ends_with("attn_gate.weight") {
                    // DeltaNet 的 z 门控 = in_proj_z。
                    Some((VReorderAxis::Rows, 0, hdim))
                } else if gguf_name.ends_with("ssm_alpha.weight")
                    || gguf_name.ends_with("ssm_beta.weight")
                {
                    // a / b 门控：每个 v-head 一个标量，head_dim = 1。
                    Some((VReorderAxis::Rows, 0, 1))
                } else if gguf_name.ends_with("ssm_out.weight") {
                    Some((VReorderAxis::Cols, 0, hdim))
                } else {
                    None
                };
            if let Some((axis, start, blk)) = plan {
                let span = start as u64 + (v_heads * blk) as u64;
                let fits = rows > 0
                    && rows * cols == d.len()
                    && match axis {
                        VReorderAxis::Rows => span <= rows as u64,
                        VReorderAxis::Cols => span <= cols as u64,
                    };
                if !fits {
                    eprintln!(
                        "warning: V-head reorder skipped for {} (shape {:?}, rows {rows}, cols {cols}) \
                         — linear_attn output will be wrong",
                        gguf_name, shape
                    );
                } else {
                    match axis {
                        VReorderAxis::Rows => {
                            vreorder_rows_inplace(d, cols, start, k_heads, v_per_k, blk)
                        }
                        VReorderAxis::Cols => {
                            vreorder_cols_inplace(d, rows, cols, k_heads, v_per_k, blk)
                        }
                    }
                }
            }
        }
    }

    // (2) ssm_a：-exp(A_log) → A_log。
    if is_ssm_a {
        for v in d.iter_mut() {
            let a = -*v; // GGUF 侧是负的衰减系数
            if a > 0.0 {
                *v = a.ln();
            }
        }
    }

    // (1) norm：w + 1 → w。ssm_norm 除外（见上）。
    if is_norm && !is_ssm_norm {
        for v in d.iter_mut() {
            *v -= 1.0;
        }
    }
}

/// GGUF 张量 → f32，并翻成 hipfire 运行时的 HF 约定。
///
/// **所有** 取 f32 的调用点都必须走这里：漏一个，那个张量就带着 GGUF 的折算约定进了
/// `.hfq`，运行时会安静地算错（历史故障：linear_attn 整族错位 → 模型立刻吐 EOS）。
fn gguf_tensor_to_f32_hf(
    info: &gguf_input::TensorInfo,
    raw: &[u8],
    shape: &[u32],
    is_norm: bool,
    arch_id: u32,
    vreorder: Option<(usize, usize, usize)>,
) -> Vec<f32> {
    let mut d = gguf_input::tensor_to_f32(info, raw);
    gguf_convention_to_hf(&info.name, arch_id, is_norm, shape, vreorder, &mut d);
    d
}

/// MQ4G256V2 + 可选 AWQ，镜像 safetensors 管线（`pipeline.rs:5183-5196`）
/// 的写法：`compute_awq_scales` → 记下侧车 → `awq_pre_scale_weights` → 量化。
///
/// 两个名字都要传，因为两套查找用的键不同：
/// * `awq_eligible` 匹配 **HF** 后缀（`.in_proj_` / `q_proj.weight` / ...）；
/// * `imatrix_weights_for` 匹配 **imatrix 文件自己的**张量名 —— 对本管线就是
///   `info.name`（llama.cpp 的 `blk.N.*`，也正是 `llama-imatrix` 的输出命名）。
///   传 GGUF 名先直接命中；再回落 HF 名（内部会做 safetensors→ggml 换算），
///   这样两种 imatrix 命名约定都能用。
///
/// 预乘**必须在 FWHT 之前**（见 `awq_pre_scale_weights` 的文档）：这里就是把
/// `W·s` 交给 `quantize_mq4g256v2`，由它在内部旋转前使用。
///
/// 应用成功时 `slot` 拿到 per-input-channel 尺度（长度 K）供侧车写出；
/// 未应用时显式置 `None`，避免上一轮张量的尺度泄漏到下一个张量。
fn quantize_mq4g256v2_awq(
    slot: &mut Option<Vec<f32>>,
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
    gguf_name: &str,
    hf_name: &str,
) -> Vec<u8> {
    *slot = None;
    let alpha = match AWQ_ALPHA.get() {
        Some(a) => *a,
        // 未开 --awq / --awq-alpha：原样量化。
        None => return quantize_mq4g256v2(f32_data, m, k, signs1, signs2),
    };
    if !awq_eligible(hf_name) {
        return quantize_mq4g256v2(f32_data, m, k, signs1, signs2);
    }
    let im = match imatrix_weights_for(gguf_name).or_else(|| imatrix_weights_for(hf_name)) {
        Some(v) => v,
        None => return quantize_mq4g256v2(f32_data, m, k, signs1, signs2),
    };
    if im.len() != k {
        eprintln!(
            "    AWQ:      {hf_name} SKIPPED — imatrix len {} != K {k}",
            im.len()
        );
        return quantize_mq4g256v2(f32_data, m, k, signs1, signs2);
    }
    let scales = compute_awq_scales(im, alpha);
    let mut scaled = f32_data.to_vec();
    awq_pre_scale_weights(&mut scaled, m, k, &scales);
    *slot = Some(scales);
    quantize_mq4g256v2(&scaled, m, k, signs1, signs2)
}


/// MTP（nextn）槽 → `.mtp` 规范名 + 是否 norm。
///
/// 右列必须与 `bin/mtp_extract.rs::MTP_NAMING_MAP` 一致 —— 运行时
/// `mtp_head.rs::load_weight_raw` 按**裸名**取张量（"eh_proj" / "wq" / ...），
/// 不做任何前缀补全。
///
/// 匹配用「`blk.{idx}.` 之后的完整剩余」精确相等，不用后缀：
/// `attn_norm.weight` 是 `nextn.shared_head_norm.weight` 的真后缀，
/// 用 `ends_with` 会把两者混起来。
const MTP_SLOT_MAP: &[(&str, &str, bool)] = &[
    // Norms（F32, 1D）
    ("nextn.shared_head_norm.weight", "shared_head_norm", true),
    ("nextn.enorm.weight", "enorm", true),
    ("nextn.hnorm.weight", "hnorm", true),
    ("attn_norm.weight", "attn_norm", true),
    ("post_attention_norm.weight", "attn_post_norm", true),
    ("attn_q_norm.weight", "attn_q_norm", true),
    ("attn_k_norm.weight", "attn_k_norm", true),
    // 2-D 权重（MQ4G256）
    ("nextn.eh_proj.weight", "eh_proj", false),
    ("attn_q.weight", "wq", false),
    ("attn_k.weight", "wk", false),
    ("attn_v.weight", "wv", false),
    ("attn_output.weight", "wo", false),
    ("ffn_gate.weight", "ffn_gate", false),
    ("ffn_up.weight", "ffn_up", false),
    ("ffn_down.weight", "ffn_down", false),
];

fn mtp_canonical_name(gguf_name: &str) -> Option<(&'static str, bool)> {
    let rest = gguf_name.strip_prefix("blk.")?;
    let dot = rest.find('.')?;
    let slot = &rest[dot + 1..];
    MTP_SLOT_MAP
        .iter()
        .find(|(s, _, _)| *s == slot)
        .map(|(_, canon, is_norm)| (*canon, *is_norm))
}

/// 把一个 GGUF MTP 张量打成 `.mtp` 容器里的 `HfqTensor`。
///
/// 约定与主干**完全一致**（走同一个 `gguf_tensor_to_f32_hf`）：
/// norm 是 GGUF 的「HF+1」折算值，而 `.mtp` 存 HF 原始值
/// （`mtp_head.rs::load_norm_raw` 加载时 +1.0），所以必须做同一个 -1.0。
/// MTP 层是**全注意力**层（没用 linear_attn 槽），所以 V-head 重排天然不适用。
fn mtp_pack_one(
    info: &gguf_input::TensorInfo,
    raw: &[u8],
    arch_id: u32,
    vreorder: Option<(usize, usize, usize)>,
    signs1: &[f32],
    signs2: &[f32],
    out: &mut Vec<HfqTensor>,
) -> Result<(), String> {
    let (canon_name, _) = match mtp_canonical_name(&info.name) {
        Some(v) => v,
        None => return Err(format!("{} (no canonical slot)", info.name)),
    };
    // 与主干同一套轴序规则（GGUF dim[0] 是收缩轴 → HF `[out, in]`）。
    let shape: Vec<u32> = if info.shape.len() == 2 {
        vec![info.shape[1] as u32, info.shape[0] as u32]
    } else {
        info.shape.iter().map(|&s| s as u32).collect()
    };
    let is_norm = gguf_is_norm_tensor(&info.name);
    let f32_data = gguf_tensor_to_f32_hf(info, raw, &shape, is_norm, arch_id, vreorder);

    let (quant_type, group_size, bytes, label) = if is_norm {
        // F32 原样（qt=2）—— `load_norm_raw` 断言 quant_type==2。
        let b: Vec<u8> = f32_data.iter().flat_map(|&v| v.to_le_bytes()).collect();
        (QuantType::F32, 0u32, b, "F32")
    } else {
        let k = shape[1] as usize;
        if k % 256 != 0 {
            return Err(format!("{} (K={k} not %256)", info.name));
        }
        let b = quantize_mq4g256(&f32_data, signs1, signs2);
        (QuantType::MQ4G256, 256u32, b, "MQ4G256")
    };
    eprintln!(
        "  MTP [{label:>7}] {canon_name:>16}  <- {:<28} shape={:?} out={}B",
        info.name,
        shape,
        bytes.len()
    );
    out.push(HfqTensor {
        name: canon_name.to_string(),
        quant_type,
        shape,
        group_size,
        data: bytes,
        spilled_len: 0,
    });
    Ok(())
}


/// Convert a GGUF file to a hipfire `.hfq`. Per-format quantization target
/// applies to 2D weight matrices; the embedding table is always Q8F16
/// (Q4-grade is too lossy for embeddings) and 1D norms stay F16. Tensor
/// names are translated GGUF → safetensors style so the engine's existing
/// `load_weights_hfq` can consume the output.
pub(crate) fn run_gguf_pipeline(
    input: &Path,
    output: &Path,
    format: GgufFormat,
    no_kmap: bool,
    kmap_dense: bool,
    kmap_mode: u8,
    arch_id_override: Option<u32>,
    force_arch_id: bool,
    // conv1d (DeltaNet) defaults to Q8 — mirrors `pipeline.rs`'s
    // `flags.q8_conv1d_default`. Caller passes `!args.no_q8_conv1d`.
    q8_conv1d: bool,
    // Optional `.mtp` sidecar output. `None` = MTP tensors are skipped as
    // before (trunk-only output). `Some(path)` = additionally pack the head.
    mtp_out: Option<&Path>,
) -> std::io::Result<()> {
    eprintln!("=== GGUF → {} conversion ===", format.label());
    eprintln!("Input:  {}", input.display());
    eprintln!("Output: {}", output.display());

    let gguf = gguf_input::GgufFile::open(input)?;
    eprintln!("GGUF version: {}", gguf.version);
    eprintln!("Tensors: {}", gguf.tensors.len());

    let arch_str = gguf
        .meta_str("general.architecture")
        .unwrap_or("llama")
        .to_string();
    // Single source of truth lives in hipfire-runtime::arch_mapping. Fail-closed
    // on unknown architecture: error listing supported types, unless an explicit
    // --arch-id override is supplied. The old qwen3* pillar guard is subsumed
    // by this uniform check — any unknown qwen3* now fails with the same clear
    // error instead of a bespoke branch.
    let auto_arch_id: u32 = match hipfire_runtime::arch_mapping::lookup_model_type(
        arch_str.as_str(),
    ) {
        Some(id) => id,
        None => {
            if let Some(ov) = arch_id_override {
                eprintln!(
                    "warning: unknown GGUF architecture '{}' but --arch-id {} override supplied; proceeding with override",
                    arch_str, ov
                );
                ov
            } else {
                let supported = hipfire_runtime::arch_mapping::supported_model_types_display();
                eprintln!(
                    "error: unknown GGUF architecture '{}'; supported architectures are: [{}]. Hint: pass --arch-id <id> to override for this model",
                    arch_str, supported
                );
                std::process::exit(1);
            }
        }
    };
    // --arch-id <u32> overrides the auto-detected id. Use when the
    // model's family maps to a different crate than the default
    // (e.g. plain Qwen2 → arch_id=7 for the hipfire-arch-qwen2 crate
    // instead of the LLaMA-family default 1, which silently drops
    // Q/K/V bias on the LLaMA loader path). See docs/plans/
    // dots-ocr-devlog.md §7 (R1) for the bring-up context.
    let arch_id: u32 = arch_id_override.unwrap_or(auto_arch_id);
    guard_qwen3_arch_override(auto_arch_id, arch_id, force_arch_id);
    if arch_id != auto_arch_id {
        eprintln!(
            "Architecture: {arch_str} (auto id={auto_arch_id}, overridden via --arch-id to {arch_id})"
        );
    } else {
        eprintln!("Architecture: {arch_str} (id={arch_id})");
    }

    // Metadata JSON: must populate `config.*` so engine's `config_from_hfq`
    // can reconstruct LlamaConfig at load time. Also keep the raw GGUF
    // metadata tree under `gguf_meta` for any consumer that wants original
    // values (chat template, vocab, scores, merges, etc.).
    // Strict validation before worker threads — same parser as CLI.
    crate::model_filter::validate_env_fixed_tier_or_exit();
    let config_json = config_json_from_gguf(&gguf, &arch_str, arch_id);
    let metadata = serde_json::json!({
        "architecture": arch_str,
        "source": "gguf",
        "config": config_json,
        "gguf_meta": gguf_meta_to_json(&gguf.metadata),
    });
    let metadata_json = serde_json::to_string(&metadata)?;
    // safetensors path so the engine's runtime FWHT inverse stays identical.
    let needs_signs = matches!(
        format,
        GgufFormat::Mq4
            | GgufFormat::Mq4V2
            | GgufFormat::Mq4C
            | GgufFormat::Mq6
            | GgufFormat::Mq6V2
            | GgufFormat::Mq5
            | GgufFormat::Mq5V2
            | GgufFormat::Mq3
            | GgufFormat::Mq3V2
            | GgufFormat::Mq2
            | GgufFormat::Mq2V2
            | GgufFormat::Mq2Lloyd
            | GgufFormat::Mq2LloydAnchored
            | GgufFormat::Mq3Lloyd
            | GgufFormat::Mq4Lloyd
            | GgufFormat::Mfp4
            | GgufFormat::Mfp4P
            | GgufFormat::Mfp4E8
            | GgufFormat::Mfp3E8
            | GgufFormat::Mfp2E8
    );
    let signs1 = if needs_signs {
        gen_fwht_signs(42, 256)
    } else {
        Vec::new()
    };
    let signs2 = if needs_signs {
        gen_fwht_signs(1042, 256)
    } else {
        Vec::new()
    };

    // K-map setup for GGUF path
    let is_moe = arch_id == 6;
    // MQ{2,3,5,6}V2 are dense-only. Reject every MoE GGUF before any tensor
    // work or output write so non-2D expert packs cannot silently become F16.
    if gguf_format_is_dense_only_mq_v2(format) && gguf_arch_is_moe_like(arch_id) {
        eprintln!(
            "error: --format mq{{2,3,5,6}}v2 is dense-only (gfx1201 Qwen3.8); MoE model (arch_id={arch_id}) is not supported with this format. Use legacy mq{{2,3,5,6}} or mq4/mq4v2/mq4c for MoE, or run on a dense checkpoint."
        );
        std::process::exit(2);
    }
    if format == GgufFormat::Mq2LloydAnchored && gguf_arch_is_moe_like(arch_id) {
        eprintln!(
            "error: --format mq2lloyd-anchored is dense-only (Qwen 3.8 Lloyd rescue); MoE model (arch_id={arch_id}) is not supported with this format. Use --format mq2lloyd for MoE routed experts or a dense checkpoint."
        );
        std::process::exit(2);
    }

    let is_gemma4 = arch_id == 13;
    let n_layers: usize = config_json
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    // Build K-map using translated (safetensors-style) names where available,
    // falling back to raw GGUF names for untranslated tensors.
    //
    // K-map is gated to MoE models only. On dense models the author's own
    // bench shows a mixed picture (PPL +1.5% to +2.5% at 2K context on 4B
    // and 27B; PPL -4.8% on 27B at 8K context — crossover at ~3K). The
    // ship-default is the conservative shape per maintainer directive
    // (2026-05-08): never silently change dense quantization. Users who
    // want K-map on dense pass `--kmap-dense` (see flag parsing below).
    // K-map is enabled for: MoE models (default), gemma4 (arch_id 13,
    // default mode=2), or any dense model with --kmap-dense.
    // Suppress with --no-kmap / --uniform.
    let kmap: HashMap<String, QuantLevel> = if no_kmap || (!is_moe && !is_gemma4 && !kmap_dense) {
        HashMap::new()
    } else {
        let mut map = HashMap::new();
        let mut counts = [0u32; 4];
        for info in &gguf.tensors {
            let out_name =
                gguf_to_safetensors_name(&info.name, arch_id).unwrap_or_else(|| info.name.clone());
            let level = kmap_resolve_mode(&out_name, n_layers, is_moe, kmap_mode);
            match level {
                QuantLevel::F16 => counts[0] += 1,
                QuantLevel::Q8 => counts[1] += 1,
                QuantLevel::Promote6 => counts[2] += 1,
                QuantLevel::Override(_) => counts[3] += 1,
                QuantLevel::Base => counts[3] += 1,
            }
            map.insert(out_name, level);
        }
        if !map.is_empty() {
            let mode_label = match kmap_mode {
                0 => "full",
                1 => "alternating",
                2 => "typed",
                _ => "?",
            };
            eprintln!(
                "K-map plan ({} base, {n_layers} layers{}, mode={mode_label}):",
                format.label(),
                if is_moe { ", MoE" } else { "" }
            );
            eprintln!("  F16:       {:>4} tensors", counts[0]);
            eprintln!("  Q8:        {:>4} tensors", counts[1]);
            eprintln!("  Promote6:  {:>4} tensors", counts[2]);
            eprintln!("  Base:      {:>4} tensors", counts[3]);
        }
        map
    };

    let mut hfq_tensors: Vec<HfqTensor> = Vec::with_capacity(gguf.tensors.len());
    let mut total_params: u64 = 0;
    let mut quant_params: u64 = 0;
    let mut total_bytes_in: u64 = 0;
    let mut total_bytes_out: u64 = 0;

    // ── MTP head exclusion ─────────────────────────────────────────────────
    // `block_count` covers the trunk PLUS the MTP blocks appended after it
    // (Qwen3.8-27B: 65 = 64 trunk + 1 MTP). `n_layers` above already reflects
    // the trunk-only count, because the arch-specific config hook
    // (`apply_qwen35_fields`) subtracts `nextn_predict_layers`.
    let n_nextn: usize = gguf
        .metadata
        .get(&format!("{arch_str}.nextn_predict_layers"))
        .and_then(|v| v.as_u32())
        .unwrap_or(0) as usize;
    let mtp_trunk_len: Option<usize> = if n_nextn > 0 { Some(n_layers) } else { None };
    if n_nextn > 0 {
        eprintln!(
            "MTP head: nextn_predict_layers={n_nextn}, trunk={n_layers} layers — MTP tensors are excluded from this file (serve them from a sidecar)"
        );
    }

    // ── V-head 重排参数 ────────────────────────────────────────────────────
    // llama.cpp 的 `_LinearAttentionVReorderBase`（conversion/qwen.py:453）在写 GGUF
    // 时把 linear_attn 的 V 维从 HF 的 grouped 布局转成 ggml 广播用的 tiled 布局。
    // 本转换器直接吃 GGUF，所以必须做逆变换。参数取自 GGUF 自己的元数据，不硬编码。
    let vreorder: Option<(usize, usize, usize)> = if arch_id == 5 {
        let g = gguf
            .metadata
            .get(&format!("{arch_str}.ssm.group_count"))
            .and_then(|v| v.as_u32());
        let t = gguf
            .metadata
            .get(&format!("{arch_str}.ssm.time_step_rank"))
            .and_then(|v| v.as_u32());
        let s = gguf
            .metadata
            .get(&format!("{arch_str}.ssm.state_size"))
            .and_then(|v| v.as_u32());
        match (g, t, s) {
            (Some(g), Some(t), Some(s)) if g > 0 && s > 0 && t > 0 && t % g == 0 => {
                eprintln!(
                    "V-head reorder: {t} v-heads / {g} k-heads, head_dim {s} ({} per k-head)",
                    t / g
                );
                Some((g as usize, t as usize, s as usize))
            }
            _ => {
                eprintln!(
                    "warning: {arch_str}.ssm.{{group_count,time_step_rank,state_size}} missing \
                     — V-head reorder DISABLED; linear_attn tensors will be mis-ordered"
                );
                None
            }
        }
    } else {
        None
    };

    let mut skipped_mtp = 0usize;
    let mut mtp_tensors: Vec<HfqTensor> = Vec::new();
    let mut mtp_unmapped: Vec<String> = Vec::new();
    for info in &gguf.tensors {
        // MTP-head tensors never enter the trunk file.
        if gguf_is_mtp_tensor(&info.name, mtp_trunk_len, n_nextn > 0) {
            skipped_mtp += 1;
            // `--mtp-out` 时顺手打包成侧车（默认仍然只是丢弃）。
            if mtp_out.is_some() {
                let raw_mtp = gguf.tensor_data(info);
                if let Err(nm) =
                    mtp_pack_one(info, raw_mtp, arch_id, vreorder, &signs1, &signs2, &mut mtp_tensors)
                {
                    mtp_unmapped.push(nm);
                }
            }
            continue;
        }
        let raw = gguf.tensor_data(info);
        let n_elements = info.numel();
        total_params += n_elements as u64;
        total_bytes_in += raw.len() as u64;

        // GGUF declares dim[0] as the CONTIGUOUS (inner) axis and dim[1] as the
        // output axis; HF and this loader declare `[out, in]`. The payload
        // already lies row-major with the inner axis fastest, which is exactly
        // HF's `[out, in]` order — so the swap happens in METADATA ONLY and is
        // never a physical transpose.
        //
        // `m_dim` is the matching row count for the quantizers below. Those call
        // sites previously read `m` from shape[0], i.e. the INNER axis, which
        // only happens to be correct when out == in and silently mis-shapes
        // every rectangular projection. `k_dim` just below was already correct
        // (GGUF dim[0] IS the contraction axis), which is why this went
        // unnoticed: the two reads disagreed and nothing cross-checked them.
        let shape: Vec<u32> = match info.shape.len() {
            // DeltaNet conv filter. GGUF stores it 2-D as `[kernel, channels]`,
            // but hipfire reads it as a flat `raw_f32` at shape
            // `[channels, 1, kernel]`. Element order is kernel-fastest in both
            // layouts, so this is a pure reshape, not a transpose. Must be
            // matched BEFORE the generic 2-D arm, which would emit
            // `[channels, kernel]` and leave the loader one axis short.
            2 if crate::model_filter::is_conv1d_tensor(&info.name) => {
                vec![info.shape[1] as u32, 1, info.shape[0] as u32]
            }
            2 => vec![info.shape[1] as u32, info.shape[0] as u32],
            // A 3-D GGUF tensor is already (kernel, channels)-shaped elsewhere;
            // keep the same reshape rule for symmetry.
            3 => vec![info.shape[1] as u32, 1, info.shape[0] as u32],
            _ => info.shape.iter().map(|&s| s as u32).collect(),
        };

        // Tensor classification (uses the original GGUF name).
        let is_norm = gguf_is_norm_tensor(&info.name);
        let is_embed = gguf_is_embed_tensor(&info.name);
        let is_2d = info.shape.len() == 2;
        let k_dim = if is_2d { info.shape[0] } else { n_elements };
        // HF-facing matmul dims. See the axis-swap note above: `m` is the
        // OUTPUT axis (GGUF dim[1]) and `k` the contraction axis (GGUF dim[0]).
        let m_dim = if is_2d { info.shape[1] as usize } else { 0 };
        let k_dim_usize = k_dim as usize;

        // Translate to the safetensors-style name `hipfire_runtime::hfq::load_weights_hfq`
        // expects. If we don't have a translation, keep the original name —
        // the future loader can ignore unknown tensors.
        // AWQ 侧车槽：每个张量进来先清空，应用成功时由
        // `quantize_mq4g256v2_awq` 填上 per-input-channel 尺度，push 后立刻写出。
        let mut awq_sidecar: Option<Vec<f32>> = None;

        let out_name =
            gguf_to_safetensors_name(&info.name, arch_id).unwrap_or_else(|| info.name.clone());

        let kmap_level = kmap.get(&out_name).copied().unwrap_or(QuantLevel::Base);

        let (data, quant_type, group_size, label) = if is_norm || !is_2d {
            // Norms and 1D tensors always F16 (primary gate)
            let f32_data = gguf_tensor_to_f32_hf(info, raw, &shape, is_norm, arch_id, vreorder);
            let f16_bytes: Vec<u8> = f32_data
                .iter()
                .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                .collect();
            (f16_bytes, QuantType::F16, 0u32, "F16")
        } else if q8_conv1d && crate::model_filter::is_conv1d_tensor(&out_name) {
            // DeltaNet conv1d defaults to Q8 — same rule, and same position in the
            // chain, as the safetensors pipeline (`pipeline.rs`, gated by
            // `flags.q8_conv1d_default`). The tensor is small (~32K elem) but runs
            // every token, and 4-bit FWHT formats measurably hurt the gated-delta
            // path. Disable with --no-q8-conv1d.
            //
            // This arm matters here specifically: conv1d's K is the kernel width
            // (4), so `K % 256 != 0` and the tensor would otherwise fall through to
            // the HFQ4-G128 fallback arm below, which pads K up to 128 — inflating
            // the tensor from 43520 B (Q8F16, gs=32) to 737280 B (HFQ4G128, gs=128).
            let f32_data = gguf_tensor_to_f32_hf(info, raw, &shape, is_norm, arch_id, vreorder);
            let q = quantize_q8f16(&f32_data);
            quant_params += n_elements as u64;
            (q, QuantType::Q8F16, 32u32, "Q8_F16")
        } else if kmap_level == QuantLevel::Q8 || is_embed {
            // K-map Q8 or embedding
            let f32_data = gguf_tensor_to_f32_hf(info, raw, &shape, is_norm, arch_id, vreorder);
            let q = quantize_q8f16(&f32_data);
            quant_params += n_elements as u64;
            (q, QuantType::Q8F16, 32u32, "Q8_F16")
        } else if crate::model_filter::product_tier_cli().is_some_and(|t| {
            crate::model_filter::q8_class_of(&out_name).is_some_and(|cls| t.lifts(cls))
        }) || crate::model_filter::fixed_tier_override_applies(&out_name)
        {
            // Product tier lift and/or explicit --fixed-tier / HIPFIRE_FIXED_TIER
            // entry. Codec overrides (e.g. attn_full:mq6v2) apply even when the
            // ProductTier does not lift that class; missing override => Q8.
            let f32_data = gguf_tensor_to_f32_hf(info, raw, &shape, is_norm, arch_id, vreorder);
            quant_params += n_elements as u64;
            if let Some(dt) = crate::model_filter::fixed_tier_dtype_for(&out_name) {
                let m = m_dim;
                let k = k_dim_usize;
                if k % 256 != 0 && matches!(dt, "mq2v2" | "mq3v2" | "mq4v2" | "mq5v2" | "mq6v2") {
                    eprintln!(
                        "error: fixed-tier dtype {dt} requires K%256==0 for {out_name} (K={k})"
                    );
                    std::process::exit(2);
                }
                match dt {
                    "mq6v2" => {
                        let q = quantize_mq6g256v2(&f32_data, m, k, &signs1, &signs2);
                        (q, QuantType::MQ6G256V2, 256u32, "MQ6G256V2")
                    }
                    "mq5v2" => {
                        let q = quantize_mq5g256v2(&f32_data, m, k, &signs1, &signs2);
                        (q, QuantType::MQ5G256V2, 256u32, "MQ5G256V2")
                    }
                    "mq4v2" => {
                        let q = quantize_mq4g256v2_awq(&mut awq_sidecar, &f32_data, m, k, &signs1, &signs2, &info.name, &out_name);
                        (q, QuantType::MQ4G256V2, 256u32, "MQ4G256V2")
                    }
                    "mq3v2" => {
                        let q = quantize_mq3g256v2(&f32_data, m, k, &signs1, &signs2);
                        (q, QuantType::MQ3G256V2, 256u32, "MQ3G256V2")
                    }
                    "mq2v2" => {
                        let q = quantize_mq2g256v2(&f32_data, m, k, &signs1, &signs2);
                        (q, QuantType::MQ2G256V2, 256u32, "MQ2G256V2")
                    }
                    "mq4" => {
                        let q = quantize_mq4g256(&f32_data, &signs1, &signs2);
                        (q, QuantType::MQ4G256, 256u32, "MQ4G256")
                    }
                    "mq3l" => {
                        let q = quantize_mq3g256_lloyd(&f32_data, &signs1, &signs2);
                        (q, QuantType::MQ3G256Lloyd, 256u32, "MQ3G256Lloyd")
                    }
                    "mfp4e8" => {
                        let q = quantize_mfp4g32_e8_2d(&f32_data, m, k, &signs1, &signs2);
                        (q, QuantType::MFP4G32E8, 32u32, "MFP4G32E8")
                    }
                    "mfp4e8soa" => {
                        let q = quantize_mfp4g32_e8_soa_2d(&f32_data, m, k, &signs1, &signs2);
                        (q, QuantType::MFP4G32E8SOA, 32u32, "MFP4G32E8SOA")
                    }
                    _ => {
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    }
                }
            } else {
                let q = quantize_q8f16(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q8_F16")
            }
        } else if matches!(format, GgufFormat::Ternary) && info.dtype == gguf_input::GgmlType::Q2_0
        {
            // TQ2G128 byte-verbatim passthrough wins over the kmap/Q8 rules
            // when the source is already Q2_0 — PrismML's per-tensor format
            // layout is byte-identical between the GGUF Q2_0 block and hipfire TQ2G128
            let expected = (n_elements.div_ceil(128)) * 34;
            // Use exact check: n_elements already includes padding? For Q2_0 n_elements is logical count (raw.len()/34*128)
            // but info.numel() is logical; raw.len() should be n_blocks*34.
            if raw.len() != (n_elements.div_ceil(128) * 34) && raw.len() != (n_elements / 128 * 34)
            {
                // Fallback permissive check - just record
            }
            if raw.len() != expected && raw.len() != (n_elements / 128 * 34) {
                eprintln!(
                    "Q2_0 size mismatch for {}: got {} bytes, expected {}",
                    info.name,
                    raw.len(),
                    expected
                );
                std::process::exit(1);
            }
            (
                raw.to_vec(),
                QuantType::TQ2G128,
                128u32,
                "TQ2G128 (passthrough)",
            )
        } else if matches!(format, GgufFormat::Binary) && info.dtype == gguf_input::GgmlType::Q1_0 {
            // BQ1G128 byte-verbatim passthrough — mirrors the TQ2G128/Q2_0 arm
            // Q1_0 on-disk layout == hipfire BQ1G128 layout: copy verbatim.
            let expected = (n_elements.div_ceil(128)) * 18;
            if raw.len() != expected && raw.len() != (n_elements / 128 * 18) {
                eprintln!(
                    "Q1_0 size mismatch for {}: got {} bytes, expected {}",
                    info.name,
                    raw.len(),
                    expected
                );
                std::process::exit(1);
            }
            let (bytes, quant_type, group_size) =
                convert_binary_tensor(raw, gguf_input::GgmlType::Q1_0);
            (bytes, quant_type, group_size, "BQ1G128 (passthrough)")
        } else if kmap_level == QuantLevel::Promote6 && k_dim % 256 == 0 {
            // K-map promote to 6-bit
            let f32_data = gguf_tensor_to_f32_hf(info, raw, &shape, is_norm, arch_id, vreorder);
            quant_params += n_elements as u64;
            match format {
                GgufFormat::Mq4
                | GgufFormat::Mq4V2
                | GgufFormat::Mq4C
                | GgufFormat::Mq3
                | GgufFormat::Mq2
                | GgufFormat::Mq2Lloyd
                | GgufFormat::Mq2LloydAnchored
                | GgufFormat::Mq3Lloyd
                | GgufFormat::Mq4Lloyd
                | GgufFormat::Mq5
                | GgufFormat::Mq6 => {
                    let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                }
                GgufFormat::Mq6V2 | GgufFormat::Mq5V2 | GgufFormat::Mq3V2 | GgufFormat::Mq2V2 => {
                    let m = m_dim;
                    let k = k_dim_usize;
                    let q = quantize_mq6g256v2(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MQ6G256V2, 256u32, "MQ6G256V2")
                }
                GgufFormat::Hfq4 | GgufFormat::Hfq6 => {
                    let q = quantize_hfq6g256(&f32_data);
                    (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                }
                GgufFormat::Hfp4 => {
                    // No HFP6 variant in v1. Promote6 for HFP4 stays at HFP4G32 (4.25 bpw).
                    let m = m_dim;
                    let k = k_dim_usize;
                    let q = quantize_hfp4g32_2d(&f32_data, m, k);
                    (q, QuantType::HFP4G32, 32u32, "HFP4G32")
                }
                GgufFormat::Mfp4 => {
                    // No MFP6 variant. Promote6 for MFP4 stays at MFP4G32 (4.25 bpw).
                    let m = m_dim;
                    let k = k_dim_usize;
                    let q = quantize_mfp4g32_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32, 32u32, "MFP4G32")
                }
                GgufFormat::Mfp4Lloyd => {
                    let m = m_dim;
                    let k = k_dim_usize;
                    let q = quantize_mfp4g32_lloyd_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32Lloyd, 32u32, "MFP4G32Lloyd")
                }
                GgufFormat::Mfp4P => {
                    // No MFP6 variant. Promote6 for mfp4+P stays at MFP4G32P (4.25 bpw).
                    let m = m_dim;
                    let k = k_dim_usize;
                    let q = quantize_mfp4g32_p_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32P, 32u32, "MFP4G32P")
                }
                GgufFormat::Mfp4E8 => {
                    let m = m_dim;
                    let k = k_dim_usize;
                    let q = quantize_mfp4g32_e8_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32E8, 32u32, "MFP4G32E8")
                }
                GgufFormat::Mfp4E8Soa => {
                    let m = m_dim;
                    let k = k_dim_usize;
                    let q = quantize_mfp4g32_e8_soa_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32E8SOA, 32u32, "MFP4G32E8SOA")
                }
                GgufFormat::Mfp3E8 => {
                    let m = m_dim;
                    let k = k_dim_usize;
                    let q = quantize_mfp3g32_e8_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP3G32E8, 32u32, "MFP3G32E8")
                }
                GgufFormat::Mfp2E8 => {
                    let m = m_dim;
                    let k = k_dim_usize;
                    let q = quantize_mfp2g32_e8_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP2G32E8, 32u32, "MFP2G32E8")
                }
                GgufFormat::Ternary => {
                    // Terminal low-bit format — promotion is a no-op.
                    // Direct TQ2G128 semantics: scale-only ternary g128, 34 B/blk (2.125 bpw).
                    let q = quantize_tq2g128(&f32_data);
                    (q, QuantType::TQ2G128, 128u32, "TQ2G128")
                }
                GgufFormat::Binary => {
                    // Terminal low-bit format — promotion is a no-op.
                    // Direct BQ1G128 semantics: scale-only binary g128, 18 B/blk (1.14 bpw).
                    let q = quantize_bq1g128(&f32_data);
                    (q, QuantType::BQ1G128, 128u32, "BQ1G128")
                }
            }
        } else if let (QuantLevel::Override(override_fmt), true) = (kmap_level, k_dim % 256 == 0) {
            // K-map says override (lm_head when --lm-head-format set).
            // GGUF pipeline has no AWQ wiring (AWQ is safetensors-only today),
            // so this is a plain quantize on the carried target format.
            let f32_data = gguf_tensor_to_f32_hf(info, raw, &shape, is_norm, arch_id, vreorder);
            quant_params += n_elements as u64;
            match override_fmt {
                GgufFormat::Mq6 => {
                    let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                }
                GgufFormat::Hfq6 => {
                    let q = quantize_hfq6g256(&f32_data);
                    (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                }
                GgufFormat::Mq4 => {
                    let q = quantize_mq4g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ4G256, 256u32, "MQ4G256")
                }
                GgufFormat::Mq4V2 => {
                    let m = m_dim;
                    let k = k_dim_usize;
                    let q = quantize_mq4g256v2_awq(&mut awq_sidecar, &f32_data, m, k, &signs1, &signs2, &info.name, &out_name);
                    (q, QuantType::MQ4G256V2, 256u32, "MQ4G256V2")
                }
                GgufFormat::Mq4C => {
                    let m = m_dim;
                    let k = k_dim_usize;
                    let q = quantize_mq4cg256(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MQ4CG256, 256u32, "MQ4CG256")
                }
                GgufFormat::Mq5 => {
                    let q = quantize_mq5g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ5G256, 256u32, "MQ5G256")
                }
                GgufFormat::Mq6V2 => {
                    let m = m_dim;
                    let k = k_dim_usize;
                    let q = quantize_mq6g256v2(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MQ6G256V2, 256u32, "MQ6G256V2")
                }
                GgufFormat::Mq5V2 => {
                    let m = m_dim;
                    let k = k_dim_usize;
                    let q = quantize_mq5g256v2(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MQ5G256V2, 256u32, "MQ5G256V2")
                }
                GgufFormat::Mq3V2 => {
                    let m = m_dim;
                    let k = k_dim_usize;
                    let q = quantize_mq3g256v2(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MQ3G256V2, 256u32, "MQ3G256V2")
                }
                GgufFormat::Mq2V2 => {
                    let m = m_dim;
                    let k = k_dim_usize;
                    let q = quantize_mq2g256v2(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MQ2G256V2, 256u32, "MQ2G256V2")
                }
                GgufFormat::Hfq4 => {
                    let q = quantize_hfq4g256(&f32_data);
                    (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                }
                GgufFormat::Mq3 => {
                    let q = quantize_mq3g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ3G256, 256u32, "MQ3G256")
                }
                GgufFormat::Mq2 => {
                    let q = quantize_mq2g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ2G256, 256u32, "MQ2G256")
                }
                GgufFormat::Mq2Lloyd => {
                    let q = quantize_mq2g256_lloyd(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ2G256Lloyd, 256u32, "MQ2G256Lloyd")
                }
                GgufFormat::Mq2LloydAnchored => {
                    let q = quantize_mq2g256_lloyd_anchored(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ2G256Lloyd, 256u32, "MQ2G256Lloyd")
                }
                GgufFormat::Mq3Lloyd => {
                    let q = quantize_mq3g256_lloyd(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ3G256Lloyd, 256u32, "MQ3G256Lloyd")
                }
                GgufFormat::Mq4Lloyd => {
                    let q = quantize_mq4g256_lloyd(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ4G256Lloyd, 256u32, "MQ4G256Lloyd")
                }
                GgufFormat::Hfp4 => {
                    let m = m_dim;
                    let q = quantize_hfp4g32_2d(&f32_data, m, k_dim);
                    (q, QuantType::HFP4G32, 32u32, "HFP4G32")
                }
                GgufFormat::Mfp4 => {
                    let m = m_dim;
                    let q = quantize_mfp4g32_2d(&f32_data, m, k_dim, &signs1, &signs2);
                    (q, QuantType::MFP4G32, 32u32, "MFP4G32")
                }
                GgufFormat::Mfp4Lloyd => {
                    let m = m_dim;
                    let q = quantize_mfp4g32_lloyd_2d(&f32_data, m, k_dim, &signs1, &signs2);
                    (q, QuantType::MFP4G32Lloyd, 32u32, "MFP4G32Lloyd")
                }
                GgufFormat::Mfp4P => {
                    let m = m_dim;
                    let q = quantize_mfp4g32_p_2d(&f32_data, m, k_dim, &signs1, &signs2);
                    (q, QuantType::MFP4G32P, 32u32, "MFP4G32P")
                }
                GgufFormat::Mfp4E8 => {
                    let m = m_dim;
                    let q = quantize_mfp4g32_e8_2d(&f32_data, m, k_dim, &signs1, &signs2);
                    (q, QuantType::MFP4G32E8, 32u32, "MFP4G32E8")
                }
                GgufFormat::Mfp4E8Soa => {
                    let m = m_dim;
                    let q = quantize_mfp4g32_e8_soa_2d(&f32_data, m, k_dim, &signs1, &signs2);
                    (q, QuantType::MFP4G32E8SOA, 32u32, "MFP4G32E8SOA")
                }
                GgufFormat::Mfp3E8 => {
                    let m = m_dim;
                    let q = quantize_mfp3g32_e8_2d(&f32_data, m, k_dim, &signs1, &signs2);
                    (q, QuantType::MFP3G32E8, 32u32, "MFP3G32E8")
                }
                GgufFormat::Mfp2E8 => {
                    let m = m_dim;
                    let q = quantize_mfp2g32_e8_2d(&f32_data, m, k_dim, &signs1, &signs2);
                    (q, QuantType::MFP2G32E8, 32u32, "MFP2G32E8")
                }
                GgufFormat::Ternary => {
                    // Terminal low-bit: direct TQ2G128 semantics (plain, no rotation).
                    let q = quantize_tq2g128(&f32_data);
                    (q, QuantType::TQ2G128, 128u32, "TQ2G128")
                }
                GgufFormat::Binary => {
                    // Terminal low-bit: direct BQ1G128 semantics (plain, no rotation).
                    let q = quantize_bq1g128(&f32_data);
                    (q, QuantType::BQ1G128, 128u32, "BQ1G128")
                }
            }
        } else if k_dim % 256 == 0 {
            // 256-aligned 2D weight — quantize per the chosen format (Base level).
            let f32_data = gguf_tensor_to_f32_hf(info, raw, &shape, is_norm, arch_id, vreorder);
            quant_params += n_elements as u64;
            match format {
                GgufFormat::Hfq4 => {
                    let q = quantize_hfq4g256(&f32_data);
                    (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                }
                GgufFormat::Hfq6 => {
                    let q = quantize_hfq6g256(&f32_data);
                    (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                }
                GgufFormat::Mq4 => {
                    let q = quantize_mq4g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ4G256, 256u32, "MQ4G256")
                }
                GgufFormat::Mq4V2 => {
                    let m = m_dim;
                    let k = k_dim_usize;
                    let q = quantize_mq4g256v2_awq(&mut awq_sidecar, &f32_data, m, k, &signs1, &signs2, &info.name, &out_name);
                    (q, QuantType::MQ4G256V2, 256u32, "MQ4G256V2")
                }
                GgufFormat::Mq4C => {
                    let m = m_dim;
                    let k = k_dim_usize;
                    let q = quantize_mq4cg256(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MQ4CG256, 256u32, "MQ4CG256")
                }
                GgufFormat::Mq5 => {
                    let q = quantize_mq5g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ5G256, 256u32, "MQ5G256")
                }
                GgufFormat::Mq6 => {
                    let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                }
                GgufFormat::Mq6V2 => {
                    let m = m_dim;
                    let k = k_dim_usize;
                    let q = quantize_mq6g256v2(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MQ6G256V2, 256u32, "MQ6G256V2")
                }
                GgufFormat::Mq5V2 => {
                    let m = m_dim;
                    let k = k_dim_usize;
                    let q = quantize_mq5g256v2(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MQ5G256V2, 256u32, "MQ5G256V2")
                }
                GgufFormat::Mq3V2 => {
                    let m = m_dim;
                    let k = k_dim_usize;
                    let q = quantize_mq3g256v2(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MQ3G256V2, 256u32, "MQ3G256V2")
                }
                GgufFormat::Mq2V2 => {
                    let m = m_dim;
                    let k = k_dim_usize;
                    let q = quantize_mq2g256v2(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MQ2G256V2, 256u32, "MQ2G256V2")
                }
                GgufFormat::Mq3 => {
                    let q = quantize_mq3g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ3G256, 256u32, "MQ3G256")
                }
                GgufFormat::Mq2 => {
                    let q = quantize_mq2g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ2G256, 256u32, "MQ2G256")
                }
                GgufFormat::Mq2Lloyd => {
                    let q = quantize_mq2g256_lloyd(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ2G256Lloyd, 256u32, "MQ2G256Lloyd")
                }
                GgufFormat::Mq2LloydAnchored => {
                    let q = quantize_mq2g256_lloyd_anchored(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ2G256Lloyd, 256u32, "MQ2G256Lloyd")
                }
                GgufFormat::Mq3Lloyd => {
                    let q = quantize_mq3g256_lloyd(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ3G256Lloyd, 256u32, "MQ3G256Lloyd")
                }
                GgufFormat::Mq4Lloyd => {
                    let q = quantize_mq4g256_lloyd(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ4G256Lloyd, 256u32, "MQ4G256Lloyd")
                }
                GgufFormat::Hfp4 => {
                    let m = m_dim;
                    let k = k_dim_usize;
                    let q = quantize_hfp4g32_2d(&f32_data, m, k);
                    (q, QuantType::HFP4G32, 32u32, "HFP4G32")
                }
                GgufFormat::Mfp4 => {
                    let m = m_dim;
                    let k = k_dim_usize;
                    let q = quantize_mfp4g32_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32, 32u32, "MFP4G32")
                }
                GgufFormat::Mfp4Lloyd => {
                    let m = m_dim;
                    let k = k_dim_usize;
                    let q = quantize_mfp4g32_lloyd_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32Lloyd, 32u32, "MFP4G32Lloyd")
                }
                GgufFormat::Mfp4P => {
                    let m = m_dim;
                    let k = k_dim_usize;
                    let q = quantize_mfp4g32_p_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32P, 32u32, "MFP4G32P")
                }
                GgufFormat::Mfp4E8 => {
                    let m = m_dim;
                    let k = k_dim_usize;
                    let q = quantize_mfp4g32_e8_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32E8, 32u32, "MFP4G32E8")
                }
                GgufFormat::Mfp4E8Soa => {
                    let m = m_dim;
                    let k = k_dim_usize;
                    let q = quantize_mfp4g32_e8_soa_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32E8SOA, 32u32, "MFP4G32E8SOA")
                }
                GgufFormat::Mfp3E8 => {
                    let m = m_dim;
                    let k = k_dim_usize;
                    let q = quantize_mfp3g32_e8_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP3G32E8, 32u32, "MFP3G32E8")
                }
                GgufFormat::Mfp2E8 => {
                    let m = m_dim;
                    let k = k_dim_usize;
                    let q = quantize_mfp2g32_e8_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP2G32E8, 32u32, "MFP2G32E8")
                }
                GgufFormat::Ternary => {
                    // Direct low-bit: scale-only ternary g128, 34 B per 128 (2.125 bpw).
                    let q = quantize_tq2g128(&f32_data);
                    (q, QuantType::TQ2G128, 128u32, "TQ2G128")
                }
                GgufFormat::Binary => {
                    // Direct low-bit: scale-only binary g128, 18 B per 128 (1.14 bpw).
                    let q = quantize_bq1g128(&f32_data);
                    (q, QuantType::BQ1G128, 128u32, "BQ1G128")
                }
            }
        } else {
            // K not divisible by 256 — fall back to HFQ4-G128 (no rotation).
            // This branch fires for the rare ragged dim; ignores --format
            // (no G128 variant of mq4/mq6 exists).
            let f32_data = gguf_tensor_to_f32_hf(info, raw, &shape, is_norm, arch_id, vreorder);
            let m = m_dim;
            let k = k_dim_usize;
            let q = quantize_hfq4g128_2d(&f32_data, m, k);
            quant_params += n_elements as u64;
            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
        };

        total_bytes_out += data.len() as u64;
        eprintln!(
            "  {label:>9}: {} → {} {:?} ({} src={:?}, {:.1} KB → {:.1} KB)",
            info.name,
            out_name,
            info.shape,
            n_elements,
            info.dtype,
            raw.len() as f64 / 1024.0,
            data.len() as f64 / 1024.0,
        );

        // AWQ 侧车命名：把结尾的 `.weight` 换成 `.awq_scale.weight`
        // （与 `pipeline.rs` 的 safetensors 管线、以及运行时
        // `hfq.rs::awq_scale_f32_bytes` 的查找规则严格一致）。
        // 必须在 push 之前算好名字 —— `out_name` 会被 push 消费掉。
        let awq_sidecar_name = if awq_sidecar.is_some() {
            Some(match out_name.strip_suffix(".weight") {
                Some(stem) => format!("{stem}.awq_scale.weight"),
                None => format!("{out_name}.awq_scale.weight"),
            })
        } else {
            None
        };

        hfq_tensors.push(HfqTensor {
            name: out_name,
            quant_type,
            shape,
            group_size,
            data,
            spilled_len: 0,
        });

        if let (Some(scales), Some(sidecar_name)) = (awq_sidecar.take(), awq_sidecar_name) {
            let bytes = awq_scales_to_f16_bytes(&scales);
            eprintln!(
                "    AWQ:      {} [{}] (1D F16, {} B)",
                sidecar_name,
                scales.len(),
                bytes.len()
            );
            hfq_tensors.push(HfqTensor {
                name: sidecar_name,
                quant_type: QuantType::F16,
                shape: vec![scales.len() as u32],
                group_size: 0,
                data: bytes,
                spilled_len: 0,
            });
        }
    }

    if skipped_mtp > 0 {
        eprintln!(
            "  MTP tensors:    {skipped_mtp} (excluded; trunk only)"
        );
    }

    // ── `.mtp` 侧车写出（arch_id = 21 = QWEN35_MTP_HEAD）────────────────────
    if let Some(mtp_path) = mtp_out {
        if mtp_tensors.is_empty() {
            eprintln!(
                "warning: --mtp-out given but the checkpoint carries no packable MTP tensors \
                 (nextn_predict_layers={n_nextn}); nothing written"
            );
        } else {
            if !mtp_unmapped.is_empty() {
                eprintln!(
                    "warning: {} MTP tensor(s) had no canonical slot and were dropped: {:?}",
                    mtp_unmapped.len(),
                    mtp_unmapped
                );
            }
            let pick = |k: &str| config_json.get(k).cloned().unwrap_or(serde_json::Value::Null);
            let mtp_meta = serde_json::json!({
                "arch": "qwen35_mtp_head",
                "arch_id": 21u32,
                "source_model": input.display().to_string(),
                "source": "gguf",
                "n_embd": pick("hidden_size"),
                "n_layer": n_layers,
                "nextn_predict_layers": n_nextn,
                "n_head": pick("num_attention_heads"),
                "n_head_kv": pick("num_key_value_heads"),
                "n_embd_head": pick("head_dim"),
                "n_ff": pick("intermediate_size"),
                "ffn_kind": "dense",
                "num_experts": 0,
                "num_experts_per_tok": 0,
                "moe_intermediate_size": 0,
                "shared_expert_intermediate_size": 0,
                "norm_topk_prob": true,
                "vocab_size": pick("vocab_size"),
                "rope_theta": pick("rope_theta"),
                "rms_norm_eps": pick("rms_norm_eps"),
                "shared_embed_with_trunk": true,
                "shared_lm_head_with_trunk": true,
                "shared_output_norm_with_trunk": false,
                "tie_word_embeddings": true,
                "weight_quant": "MQ4G256",
                "has_compressed_lm_head_draft": false,
                "compressed_vocab_size": 0,
                "config_text_config": {
                    "partial_rotary_factor": config_json
                        .get("partial_rotary_factor")
                        .cloned()
                        .unwrap_or(serde_json::json!(0.25)),
                },
            });
            let mtp_meta_json = serde_json::to_string(&mtp_meta)?;
            write_hfq(mtp_path, 21u32, &mtp_meta_json, &mtp_tensors, None)?;
            eprintln!(
                "\nWrote MTP head: {} ({} tensors)",
                mtp_path.display(),
                mtp_tensors.len()
            );
        }
    }
    eprintln!("\n=== GGUF → MQ4 Summary ===");
    eprintln!("  Tensors:        {}", hfq_tensors.len());
    eprintln!("  Total params:   {total_params}");
    eprintln!(
        "  Quant'd params: {quant_params} ({:.1}%)",
        100.0 * quant_params as f64 / total_params as f64
    );
    eprintln!("  Input size:     {:.1} MB", total_bytes_in as f64 / 1e6);
    eprintln!(
        "  Output size:    {:.1} MB ({:.1}% of input)",
        total_bytes_out as f64 / 1e6,
        100.0 * total_bytes_out as f64 / total_bytes_in as f64,
    );

    write_hfq(output, arch_id, &metadata_json, &hfq_tensors, None)?;
    eprintln!("\nWrote: {}", output.display());
    Ok(())
}

pub(crate) fn dequantize_hfq_q8f16(data: &[u8], n_elements: usize) -> Result<Vec<f32>, String> {
    let n_blocks = n_elements.div_ceil(32);
    let expected = n_blocks * 34;
    if data.len() != expected {
        return Err(format!(
            "Q8F16 byte size {} != {expected} for {n_elements} elements",
            data.len()
        ));
    }
    let mut out = vec![0.0f32; n_elements];
    for b in 0..n_blocks {
        let off = b * 34;
        let scale = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
        let start = b * 32;
        let end = (start + 32).min(n_elements);
        for i in start..end {
            out[i] = (data[off + 2 + i - start] as i8) as f32 * scale;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{gguf_arch_is_moe_like, gguf_format_is_dense_only_mq_v2, GgufFormat};

    #[test]
    fn dense_only_mq_v2_predicate_covers_four_formats() {
        assert!(gguf_format_is_dense_only_mq_v2(GgufFormat::Mq6V2));
        assert!(gguf_format_is_dense_only_mq_v2(GgufFormat::Mq5V2));
        assert!(gguf_format_is_dense_only_mq_v2(GgufFormat::Mq3V2));
        assert!(gguf_format_is_dense_only_mq_v2(GgufFormat::Mq2V2));
        // siblings stay admitted for MoE
        assert!(!gguf_format_is_dense_only_mq_v2(GgufFormat::Mq6));
        assert!(!gguf_format_is_dense_only_mq_v2(GgufFormat::Mq5));
        assert!(!gguf_format_is_dense_only_mq_v2(GgufFormat::Mq3));
        assert!(!gguf_format_is_dense_only_mq_v2(GgufFormat::Mq2));
        assert!(!gguf_format_is_dense_only_mq_v2(GgufFormat::Mq4V2));
        assert!(!gguf_format_is_dense_only_mq_v2(GgufFormat::Mq4C));
        assert!(!gguf_format_is_dense_only_mq_v2(GgufFormat::Hfq4));
    }

    #[test]
    fn moe_like_archs_reject_mq_v2_dense_accepted() {
        // Qwen MoE + other MoE-class arches
        for arch in [6u32, 9, 10, 11, 12, 13] {
            assert!(gguf_arch_is_moe_like(arch), "arch {arch}");
            for fmt in [
                GgufFormat::Mq6V2,
                GgufFormat::Mq5V2,
                GgufFormat::Mq3V2,
                GgufFormat::Mq2V2,
            ] {
                assert!(
                    gguf_format_is_dense_only_mq_v2(fmt) && gguf_arch_is_moe_like(arch),
                    "must reject {fmt:?} on arch {arch}"
                );
            }
        }
        // dense GGUF arches unchanged (qwen3.5/3.8 dense=5, llama=1)
        for arch in [1u32, 5, 7, 8, 14] {
            assert!(!gguf_arch_is_moe_like(arch), "arch {arch} is dense");
            assert!(
                !(gguf_format_is_dense_only_mq_v2(GgufFormat::Mq6V2)
                    && gguf_arch_is_moe_like(arch))
            );
        }
    }
}
