// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! `hipfire offload-bench`: measure **this host's** GPU share for a split offload
//! step (`memory.offload_exec=passback`, see
//! [`hipfire_dispatch::offload_split`]).
//!
//! The share a pass-back step hands to the GPU is a property of the host (link
//! width, DRAM peak, core count, AVX2), not of the engine, so the engine
//! schedules it from its own arm timings and never bakes in a constant. This
//! routine gives an end user their own host's numbers, a pinned override for a
//! machine the scheduler gets wrong, and the evidence a bug report needs.
//!
//! It is a raw device benchmark: no model, no daemon, and it must not run
//! alongside one — a concurrent decode makes both engine rates meaningless. The
//! CLI refuses while a daemon pid file names a live process (see
//! `offload_bench_command`).
//!
//! The routines live here rather than in the CLI because the CLI already depends
//! on `hipfire-runtime`, and this needs `hipfire-dispatch` (already a dependency)
//! to name the probe primitives. No new dependency is added by either path.

use rdna_compute::DType;
use rdna_compute::Gpu;

use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::offload_split::{probe_synthetic, SplitCalibration};

/// The dense Qwen formats this repo ships, with their `.hfq` wire `quant_type`
/// (`docs/quant-formats/qt-register.txt`) so `--format` accepts either spelling.
///
/// The Lloyd variants are deliberately absent: their GEMV arms need the
/// per-tensor codebook LUTs, which a synthetic weight cannot carry, and the
/// unrotated MFP4/PARO families have no CPU decoder at all.
const FORMATS: &[(&str, u8, DType)] = &[
    ("mq4g256", 13, DType::MQ4G256),
    ("mq4g256v2", 44, DType::MQ4G256V2),
    ("mq4cg256", 45, DType::MQ4CG256),
    ("mq6g256", 15, DType::MQ6G256),
    ("mq6g256v2", 47, DType::MQ6G256V2),
    ("mq5g256v2", 48, DType::MQ5G256V2),
    ("mq3g256v2", 49, DType::MQ3G256V2),
    ("hfq4g256", 6, DType::HFQ4G256),
    ("q8_0", 3, DType::Q8_0),
];

/// Default activation width: the 9B dense trunk's hidden size, which is also the
/// `k` of most of its projections.
pub const DEFAULT_K: usize = 5120;
/// Default weight size per format: big enough that the copies and the launch
/// overhead do not dominate a single pass.
pub const DEFAULT_BUFFER_MB: usize = 192;
pub const DEFAULT_REPS: usize = 5;

#[derive(Clone, Debug)]
pub struct CalibrateOptions {
    /// A quant type (`"mq4"`, `"mq4g256v2"`, …) or a `.hfq` wire `quant_type`
    /// number as text; `None` = the default matrix (the dense Qwen formats this
    /// repo ships).
    pub format: Option<String>,
    /// Activation width; `None` = [`DEFAULT_K`].
    pub k: Option<usize>,
    pub buffer_mb: usize,
    pub reps: usize,
}

#[derive(Clone, Debug)]
pub struct CalibrationReport {
    pub arch: String,
    pub format: String,
    pub m: usize,
    pub k: usize,
    pub bytes: usize,
    pub cpu_bytes_per_s: f64,
    pub gpu_bytes_per_s: f64,
    pub share: f64,
    /// `(r_gpu + r_cpu) / max(r_gpu, r_cpu)` — the two-engine bound: what the
    /// CPUs' rate adds on top of the faster engine's, which is the most a split
    /// can buy on this host and link.
    pub predicted_speedup: f64,
}

/// Resolve `--format` to a `DType`: a name from [`FORMATS`] (case-insensitive) or
/// that entry's `quant_type` spelled in decimal.
fn resolve_format(raw: &str) -> Option<DType> {
    let needle = raw.trim().to_ascii_lowercase();
    FORMATS
        .iter()
        .find(|(name, qt, _)| *name == needle || qt.to_string() == needle)
        .map(|(_, _, dtype)| *dtype)
}

/// The formats `--format` can name, for the usage error.
pub fn resolvable_formats() -> Vec<String> {
    FORMATS
        .iter()
        .map(|(name, qt, _)| format!("{name} (qt={qt})"))
        .collect()
}

fn hip_err(msg: String) -> hip_bridge::HipError {
    hip_bridge::HipError::new(0, &msg)
}

/// Measure every selected format and return one report each.
///
/// Builds its own `DispatchCtx`; the caller only needs a `Gpu`. `m` is derived
/// from `buffer_mb` and the format's own row stride, so every format is measured
/// over roughly the same weight bytes and the reported rates are comparable.
pub fn calibrate(
    gpu: &mut Gpu,
    opts: &CalibrateOptions,
) -> Result<Vec<CalibrationReport>, hip_bridge::HipError> {
    let formats: Vec<(&str, DType)> = match opts.format.as_deref() {
        Some(raw) => {
            let dtype = resolve_format(raw).ok_or_else(|| {
                hip_err(format!(
                    "unknown offload-bench format '{raw}'; known: {}",
                    resolvable_formats().join(", ")
                ))
            })?;
            let name = FORMATS
                .iter()
                .find(|(_, _, d)| *d == dtype)
                .map(|(name, _, _)| *name)
                .unwrap_or("?");
            vec![(name, dtype)]
        }
        None => FORMATS.iter().map(|(name, _, d)| (*name, *d)).collect(),
    };
    let k = opts.k.unwrap_or(DEFAULT_K);
    let reps = opts.reps.max(1);
    let arch = gpu.arch.clone();
    let ctx = DispatchCtx::new(gpu);
    let mut reports = Vec::with_capacity(formats.len());
    for (name, dtype) in formats {
        let row_bytes = hipfire_dispatch::row_bytes_for(dtype, k).ok_or_else(|| {
            hip_err(format!(
                "offload-bench: no CPU decoder for {dtype:?}; it can never be split"
            ))
        })?;
        if row_bytes == 0 || k == 0 {
            return Err(hip_err(format!(
                "offload-bench: k={k} yields a zero row stride for {dtype:?} \
                 (needs k to be a whole number of groups)"
            )));
        }
        let bytes = opts.buffer_mb.max(1) * (1 << 20);
        let m = bytes.div_ceil(row_bytes);
        let SplitCalibration {
            cpu_bytes_per_s,
            gpu_bytes_per_s,
            share,
        } = probe_synthetic(gpu, &ctx, dtype, m, k, reps)
            .map_err(|e| hip_err(format!("offload-bench probe: {e}")))?;
        let fastest = gpu_bytes_per_s.max(cpu_bytes_per_s);
        reports.push(CalibrationReport {
            arch: arch.clone(),
            format: name.to_string(),
            m,
            k,
            bytes: m * row_bytes,
            cpu_bytes_per_s,
            gpu_bytes_per_s,
            share,
            predicted_speedup: if fastest > 0.0 {
                (gpu_bytes_per_s + cpu_bytes_per_s) / fastest
            } else {
                1.0
            },
        });
    }
    Ok(reports)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_resolution_accepts_names_and_wire_numbers() {
        assert_eq!(resolve_format("mq4g256v2"), Some(DType::MQ4G256V2));
        assert_eq!(resolve_format(" MQ4G256V2 "), Some(DType::MQ4G256V2));
        assert_eq!(resolve_format("44"), Some(DType::MQ4G256V2));
        assert_eq!(resolve_format("q8_0"), Some(DType::Q8_0));
        assert_eq!(resolve_format("13"), Some(DType::MQ4G256));
        assert_eq!(resolve_format("banana"), None);
        // Every resolvable name round-trips through its own wire number.
        for (name, qt, dtype) in FORMATS {
            assert_eq!(resolve_format(name), Some(*dtype));
            assert_eq!(resolve_format(&qt.to_string()), Some(*dtype));
        }
        // The matrix must be reachable by `--format` and decodable on the CPU: a
        // format without a CPU decoder could never be split. (Whether this host
        // has the vector row dot is not asserted — that is a property of the
        // CPU, not of the format.)
        for (name, _, dtype) in FORMATS {
            assert!(
                hipfire_dispatch::cpu_quant_for(*dtype).is_some(),
                "{name} has no CPU decoder"
            );
            assert!(hipfire_dispatch::row_bytes_for(*dtype, DEFAULT_K).is_some());
        }
    }
}
