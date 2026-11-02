// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Developer-only startup alias/baseline eligibility probe of the
//! `qwen4_qsa_code_check` loaded-code registry (`HIPFIRE_QWEN4_QSA_CODE_CHECK=1`).
//!
//! `HIPFIRE_QWEN4_QSA_CODE_CHECK=1 qsa_code_check_selftest`
//!
//! Needs `--features lab` (the `[[example]]` entry is gated like the other
//! QSA probes; nothing here uses a lab-only API):
//! `cargo build --release -p rdna-compute --features lab --example qsa_code_check_selftest`.
//!
//! The switch is read through the repo's process-config resolver
//! (`hipfire_config::process_value`, the table `FeatureFlags::from_active_config`
//! reads) and must be exactly `1`; anything else exits 2 before the GPU is
//! touched. The visible device (`Gpu::init`, honouring `HIPFIRE_DEVICES`) then
//! loads modules through the normal route preload (`begin_route_kernel_load`
//! with a manual closed plan of existing product images and symbols,
//! `ensure_route_modules_preloaded`, `finish_route_kernel_load`), so the
//! registry sees the same registrations a real run makes. No QSA (or any)
//! kernel is launched and no operand tensor is allocated:
//! - gfx1151: `qsa_select_bf16_pm_gfx1151`
//!   (`indexed_attention_select_scores_rows16_bf16_pm_gfx1151` and `..._rows32_bf16_pm_gfx1151`),
//!   `qsa_select_pm_gfx1151`, `spec_sampling_rows_parallel_pm_gfx1151`,
//!   `spec_accept_ordered_pm_gfx1151`, `mq_i4_moe_gemv_pm_gfx1151`;
//! - gfx1201: the `spec_*_pm_gfx1201` pair and `mq_i4_moe_gemv_pm_gfx1201`;
//! - gfx1100: `gemm_mq4g256v2_residual_iu4_pm_gfx1100` (the `_set`, `_add` and
//!   `gate_up_silu` symbols) and the EP tile mask / pull / pull-N8 PM modules;
//! - other arches: no PM image exists in `kernels.rs`; the run fails explicitly
//!   with no modules loaded;
//! - one cheap hipcc JIT module (`qsa_code_check_selftest_jit`) through
//!   `ensure_kernel_public`, before the route is sealed; a failed compile is
//!   reported and ignored.
//!
//! Then `rdna_compute::qsa_code_check_self_test()` (a CPU sample of the loaded
//! code against the registration baselines; no HIP/HSA call, never consumes the
//! once-only HIP 700 slot) is printed. Exit 0 only for `ranges > 0`,
//! `differ == 0` and a report that was not skipped; `unavailable` reasons stay
//! in the printed line. Exit 1 for any other outcome, 2 for the missing switch.

use rdna_compute::kernel_registry::{PlannedKernel, RouteKernelPlan};
use rdna_compute::{kernels, Gpu};

type Result<T, E = String> = std::result::Result<T, E>;
fn err(e: impl std::fmt::Debug) -> String {
    format!("{e:?}")
}

const SWITCH: &str = "HIPFIRE_QWEN4_QSA_CODE_CHECK";
const PREFIX: &str = "hipfire: code-check self-test: ";
const JIT_MODULE: &str = "qsa_code_check_selftest_jit";
const JIT_SYMBOL: &str = "qsa_code_check_selftest_noop";
const JIT_SRC: &str = "#include <hip/hip_runtime.h>\n\
    extern \"C\" __global__ void qsa_code_check_selftest_noop(int* out) { if (out != nullptr) { out[0] = 1; } }\n";

/// The parsed `N ranges, E equal, D differ, U unavailable` head of the line.
#[derive(Debug, PartialEq, Eq)]
struct Summary {
    ranges: usize,
    equal: usize,
    differ: usize,
    unavailable: usize,
}

/// What `qsa_code_check_self_test()` returned.
#[derive(Debug, PartialEq, Eq)]
enum Report {
    Sampled(Summary),
    Skipped(String),
}

/// Strict parse of the one-line report: the `hipfire: code-check self-test: `
/// prefix, then either `skipped: ...` or four `<count> <label>` fields in the
/// order ranges, equal, differ, unavailable up to the first `;` (later
/// segments hold module names and reasons and are never parsed).
fn parse_report(report: &str) -> Result<Report> {
    let line = report.strip_suffix('\n').unwrap_or(report);
    if line.contains('\n') {
        return Err(format!("expected one report line, got {report:?}"));
    }
    let rest = line
        .strip_prefix(PREFIX)
        .ok_or_else(|| format!("unexpected report prefix: {line:?}"))?;
    if let Some(why) = rest.strip_prefix("skipped") {
        return Ok(Report::Skipped(
            why.trim_start_matches(':').trim().to_owned(),
        ));
    }
    let head = rest.split(';').next().unwrap_or(rest);
    let fields: Vec<&str> = head.split(", ").collect();
    let [ranges, equal, differ, unavailable] = fields[..] else {
        return Err(format!("expected four summary fields, got {head:?}"));
    };
    let count = |field: &str, label: &str| -> Result<usize> {
        field
            .strip_suffix(label)
            .and_then(|n| n.strip_suffix(' '))
            .and_then(|n| n.parse::<usize>().ok())
            .ok_or_else(|| format!("bad summary field {field:?} (want '<count> {label}')"))
    };
    let s = Summary {
        ranges: count(ranges, "ranges")?,
        equal: count(equal, "equal")?,
        differ: count(differ, "differ")?,
        unavailable: count(unavailable, "unavailable")?,
    };
    if s.equal.checked_add(s.differ) != Some(s.ranges) {
        return Err(format!(
            "inconsistent summary: {s:?} (equal + differ != ranges)"
        ));
    }
    Ok(Report::Sampled(s))
}

/// `Ok(())` iff the report proves a real, unskipped, corruption-free sample.
fn verdict(report: &Report) -> Result<()> {
    match report {
        Report::Skipped(why) => Err(format!("self-test skipped: {why}")),
        Report::Sampled(s) if s.ranges == 0 => {
            Err("self-test sampled 0 ranges (no usable alias/baseline)".into())
        }
        Report::Sampled(s) if s.differ != 0 => Err(format!(
            "{} range(s) differ from the registration baseline",
            s.differ
        )),
        Report::Sampled(_) => Ok(()),
    }
}

/// One embedded product module of the plan: name, pinned image, the symbols
/// the route requests (a missing symbol fails the preload).
type Pm = (&'static str, &'static [u8], &'static [&'static str]);

const PM_GFX1151: &[Pm] = &[
    (
        "qsa_select_bf16_pm_gfx1151",
        kernels::QSA_SELECT_BF16_PM_GFX1151,
        &[
            "indexed_attention_select_scores_rows16_bf16_pm_gfx1151",
            "indexed_attention_select_scores_rows32_bf16_pm_gfx1151",
        ],
    ),
    (
        "qsa_select_pm_gfx1151",
        kernels::QSA_SELECT_PM_GFX1151,
        &[
            "indexed_attention_select_scores_rows16_f32_pm_gfx1151",
            "indexed_attention_select_scores_rows32_f32_pm_gfx1151",
            "indexed_attention_select_from_scores_pm_gfx1151",
        ],
    ),
    (
        "spec_sampling_rows_parallel_pm_gfx1151",
        kernels::SPEC_SAMPLING_ROWS_PARALLEL_PM_GFX1151,
        &["spec_cpu_ar_rows_parallel_f32_pm_gfx1151"],
    ),
    (
        "spec_accept_ordered_pm_gfx1151",
        kernels::SPEC_ACCEPT_ORDERED_PM_GFX1151,
        &["spec_accept_ordered_u32_pm_gfx1151"],
    ),
    (
        "mq_i4_moe_gemv_pm_gfx1151",
        kernels::MQ_I4_MOE_GEMV_PM_GFX1151,
        MQ_I4_MOE_SYMBOLS,
    ),
    (
        "mq_i4_moe_gemv_c1_pm_gfx1151",
        kernels::MQ_I4_MOE_GEMV_C1_PM_GFX1151,
        &rdna_compute::slab_moe::MQ_I4_MOE_GEMV_C1_SYMBOLS,
    ),
];

const PM_GFX1201: &[Pm] = &[
    (
        "spec_sampling_rows_parallel_pm_gfx1201",
        kernels::SPEC_SAMPLING_ROWS_PARALLEL_PM_GFX1201,
        &["spec_cpu_ar_rows_parallel_f32_pm_gfx1201"],
    ),
    (
        "spec_accept_ordered_pm_gfx1201",
        kernels::SPEC_ACCEPT_ORDERED_PM_GFX1201,
        &["spec_accept_ordered_u32_pm_gfx1201"],
    ),
    (
        "mq_i4_moe_gemv_pm_gfx1201",
        kernels::MQ_I4_MOE_GEMV_PM_GFX1201,
        MQ_I4_MOE_SYMBOLS,
    ),
];

/// gfx1100: the residual/gate-up IU4 GEMM module (one embedded image, the
/// three symbols its product launchers request) and the EP tile mask / pull /
/// pull-N8 builder images (module key == device symbol, as the product
/// loaders name them).
const PM_GFX1100: &[Pm] = &[
    (
        "gemm_mq4g256v2_residual_iu4_pm_gfx1100",
        kernels::GEMM_MQ4G256V2_RESIDUAL_IU4_PM_GFX1100,
        &[
            "gemm_mq4g256v2_residual_iu4_pm_set_gfx1100",
            "gemm_mq4g256v2_residual_iu4_pm_add_gfx1100",
            "gemm_mq4g256v2_gate_up_silu_iu4_pm_gfx1100",
        ],
    ),
    (
        rdna_compute::moe::EP_TILE_MASK_PM_GFX1100_SYMBOL,
        kernels::EP_TILE_MASK_PM_GFX1100,
        &[rdna_compute::moe::EP_TILE_MASK_PM_GFX1100_SYMBOL],
    ),
    (
        rdna_compute::moe::EP_TILE_PULL_PM_GFX1100_SYMBOL,
        kernels::EP_TILE_PULL_PM_GFX1100,
        &[rdna_compute::moe::EP_TILE_PULL_PM_GFX1100_SYMBOL],
    ),
    (
        rdna_compute::moe::EP_TILE_PULL_N8_PM_GFX1100_SYMBOL,
        kernels::EP_TILE_PULL_N8_PM_GFX1100,
        &[rdna_compute::moe::EP_TILE_PULL_N8_PM_GFX1100_SYMBOL],
    ),
];

const MQ_I4_MOE_SYMBOLS: &[&str] = &[
    rdna_compute::slab_moe::MQ_I4_MOE_GEMV_SYMBOLS[0][0],
    rdna_compute::slab_moe::MQ_I4_MOE_GEMV_SYMBOLS[0][1],
    rdna_compute::slab_moe::MQ_I4_MOE_GEMV_SYMBOLS[1][0],
    rdna_compute::slab_moe::MQ_I4_MOE_GEMV_SYMBOLS[1][1],
];

/// The manual closed plan of `arch`'s embedded PM modules (the product images
/// and symbols of the QSA, spec-sampling and slab MoE routes).
fn pm_plan(arch: &str) -> Result<(RouteKernelPlan, Vec<&'static str>)> {
    let set = match arch {
        "gfx1151" => PM_GFX1151,
        "gfx1201" => PM_GFX1201,
        "gfx1100" => PM_GFX1100,
        other => {
            return Err(format!(
                "no embedded PM plan is cataloged for {other} in this example (kernels.rs has PM images only for \
                 gfx1100, gfx1151 and gfx1201); no modules loaded"
            ))
        }
    };
    let entries = set
        .iter()
        .map(|&(module, image, symbols)| PlannedKernel::Embedded {
            module,
            image,
            radiowave_json: None,
            symbols,
        })
        .collect();
    Ok((
        RouteKernelPlan { entries },
        set.iter().map(|m| m.0).collect(),
    ))
}

fn run() -> Result<i32> {
    if hipfire_config::process_value(SWITCH).as_deref() != Some("1") {
        eprintln!("{SWITCH}=1 is required (exactly `1`); refusing to run");
        return Ok(2);
    }
    let mut gpu = Gpu::init().map_err(err)?;
    let arch = gpu.arch.clone();
    eprintln!("device arch {arch}");
    let (plan, modules) = pm_plan(&arch)?;

    // The JIT module loads lazily while no route is planned (a sealed route
    // refuses unplanned kernels), so it goes first.
    match gpu.ensure_kernel_public(JIT_MODULE, JIT_SRC, JIT_SYMBOL) {
        Ok(()) => eprintln!("loaded {JIT_MODULE} (hipcc JIT)"),
        Err(e) => eprintln!("note: hipcc JIT module {JIT_MODULE} not loaded: {e:?}"),
    }

    // Normal route preload: loads, registers and binds every planned symbol
    // (no kernel is launched, no operand tensor allocated); success means
    // every module and symbol is loaded.
    gpu.begin_route_kernel_load(plan, 1, None).map_err(err)?;
    gpu.ensure_route_modules_preloaded().map_err(err)?;
    gpu.finish_route_kernel_load().map_err(err)?;
    for m in &modules {
        eprintln!("loaded {m} (embedded PM)");
    }

    let line = rdna_compute::qsa_code_check_self_test();
    eprint!("{line}");
    let outcome = parse_report(&line).and_then(|r| verdict(&r));
    // Modules stay registered until here; `Gpu` teardown unregisters them.
    drop(gpu);
    match outcome {
        Ok(()) => Ok(0),
        Err(why) => {
            eprintln!("FAIL: {why}");
            Ok(1)
        }
    }
}

fn main() {
    std::process::exit(match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOTE: &str =
        "; note: differ = persistent CPU-visible loaded-code corruption at this sample";

    #[test]
    fn parses_clean_line() {
        let l = format!("{PREFIX}12 ranges, 12 equal, 0 differ, 0 unavailable{NOTE}\n");
        let r = parse_report(&l).unwrap();
        assert_eq!(
            r,
            Report::Sampled(Summary {
                ranges: 12,
                equal: 12,
                differ: 0,
                unavailable: 0
            })
        );
        assert!(verdict(&r).is_ok());
    }

    #[test]
    fn unavailable_reasons_and_module_names_are_not_parsed() {
        let l = format!(
            "{PREFIX}4 ranges, 4 equal, 0 differ, 2 unavailable; unavailable (not checked): module 'a; 9 ranges, 1 equal' (handle 0x1): x{NOTE}\n"
        );
        let r = parse_report(&l).unwrap();
        assert_eq!(
            r,
            Report::Sampled(Summary {
                ranges: 4,
                equal: 4,
                differ: 0,
                unavailable: 2
            })
        );
        assert!(verdict(&r).is_ok());
    }

    #[test]
    fn differ_zero_ranges_and_skipped_fail() {
        let d = parse_report(&format!(
            "{PREFIX}3 ranges, 2 equal, 1 differ, 0 unavailable; first differing range: x{NOTE}\n"
        ))
        .unwrap();
        assert!(verdict(&d).unwrap_err().contains("differ"));
        let z = parse_report(&format!(
            "{PREFIX}0 ranges, 0 equal, 0 differ, 3 unavailable{NOTE}\n"
        ))
        .unwrap();
        assert!(verdict(&z).unwrap_err().contains("0 ranges"));
        let s = parse_report(&format!("{PREFIX}skipped: registry busy (registration or unload in progress; no range checked)\n")).unwrap();
        assert!(matches!(&s, Report::Skipped(w) if w.starts_with("registry busy")));
        assert!(verdict(&s).unwrap_err().contains("skipped"));
        let p = parse_report(&format!(
            "{PREFIX}skipped: registry poisoned (no range checked)\n"
        ))
        .unwrap();
        assert!(verdict(&p).is_err());
    }

    #[test]
    fn malformed_lines_are_rejected() {
        let ok = "1 ranges, 1 equal, 0 differ, 0 unavailable";
        for l in [
            String::new(),
            "hipfire: code-check after HIP 700: 1 ranges, 1 equal, 0 differ, 0 unavailable\n"
                .to_owned(),
            format!("{PREFIX}1 ranges, 1 equal, 0 differ\n"),
            format!("{PREFIX}{ok}, 1 extra\n"),
            format!("{PREFIX}x ranges, 1 equal, 0 differ, 0 unavailable\n"),
            format!("{PREFIX}1 range, 1 equal, 0 differ, 0 unavailable\n"),
            format!("{PREFIX}2 ranges, 1 equal, 0 differ, 0 unavailable\n"),
            format!("{PREFIX}{ok}\n{PREFIX}{ok}\n"),
        ] {
            assert!(parse_report(&l).is_err(), "{l:?}");
        }
    }
}
