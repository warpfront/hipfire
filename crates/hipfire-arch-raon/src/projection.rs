// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Raon projection policy. Owner: slice A.
//!
//! The projections themselves are `Step::BoundaryProject` ops
//! (`hipfire_dispatch::pipeline::boundary_project`): every GEMM output is
//! rounded to BF16, the reference model's boundary. What stays here is the
//! Raon decision the program builder resolves into each op's `RowsRoute`:
//! which verify blocks run row-exactly ([`rows_exact`]).

use hipfire_runtime::llama::WeightTensor;
use rdna_compute::DType;

/// Whether `w` is an MQ8G256 attention-class weight (device layout B).
pub fn is_mq8(w: &WeightTensor) -> bool {
    w.gpu_dtype == DType::MQ8G256
}

/// Widest block the row-exact verify route takes (a DFlash2 window).
pub const ROWS_EXACT_MAX: usize = 16;

/// Blocks up to these rows take the row-exact route by default (`auto`): the
/// widest row count at which it is no slower than the batched route, from the
/// verify cost tables (`raon_dflash_oracle costs`, graph decode, beta
/// 365862651c; receipts `raon/dflash2/evidence/b37` R9700, `b34x` / `b38x`
/// 7900 XTX). R9700 2/3/4/5 rows: 11.68/13.24/15.08/17.12 ms row-exact vs
/// 13.97/14.99/15.99/16.87 batched; 7900 XTX 2/9/10 rows: 10.90/22.47/24.65 vs
/// 18.36/23.87/24.38. Unlisted arches are unmeasured: batched.
const ROWS_EXACT_AUTO: &[(&str, usize)] = &[("gfx1201", 4), ("gfx1100", 9)];

/// `HIPFIRE_RAON_VERIFY_ROWEXACT`: unset / `auto` = [`ROWS_EXACT_AUTO`] for this
/// arch; `1` = every block of `2..=ROWS_EXACT_MAX` rows; `0` = never.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RowsExactMode {
    Auto,
    On,
    Off,
}

static ROWS_EXACT_MODE: std::sync::LazyLock<RowsExactMode> = std::sync::LazyLock::new(|| {
    match hipfire_config::developer_var("HIPFIRE_RAON_VERIFY_ROWEXACT")
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .as_deref()
    {
        None | Some("") | Some("auto") => RowsExactMode::Auto,
        Some("1" | "on" | "true") => RowsExactMode::On,
        Some("0" | "off" | "false") => RowsExactMode::Off,
        Some(other) => {
            eprintln!(
                "raon1_mini: HIPFIRE_RAON_VERIFY_ROWEXACT={other} is not auto|0|1; using auto"
            );
            RowsExactMode::Auto
        }
    }
});

/// Widest block that takes the row-exact route on `arch` (0: none).
pub fn rows_exact_max(arch: &str) -> usize {
    match *ROWS_EXACT_MODE {
        RowsExactMode::Off => 0,
        RowsExactMode::On => ROWS_EXACT_MAX,
        RowsExactMode::Auto => ROWS_EXACT_AUTO
            .iter()
            .find(|(a, _)| *a == arch)
            .map_or(0, |&(_, max)| max),
    }
}

/// Whether a forward of `rows` rows on `arch` runs the row-exact multi-row
/// route: every row bit-identical to the decode step's single row (Q8_0:
/// `gemv_q8_0_rows_exact`, MoE `RowsExact`, MLA `DecodeExact`, dense layer 0
/// rows-exact, F32/qt44 projections per row). The decision depends on `rows`
/// and the arch only, so every op of one forward agrees.
pub fn rows_exact(arch: &str, rows: usize) -> bool {
    rows >= 2 && rows <= rows_exact_max(arch)
}
