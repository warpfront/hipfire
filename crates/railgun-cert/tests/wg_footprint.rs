// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Workgroup write footprints (`peacemaker_ir::passes::wg_footprint`) of the
//! real MegaM0 P0 standalone bodies in `tests/fixtures/task_graph/standalone`.
use std::path::PathBuf;

use peacemaker_ir::inst::{Arch, Frontend};
use peacemaker_ir::kernarg::KernelStatus;
use peacemaker_ir::passes::kernargs;
use peacemaker_ir::passes::wg_footprint::{analyze, ArgWgClass, WgFacts, WgLaunch};
use peacemaker_lift::{lift_object, Options};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/task_graph/standalone")
}

fn object(prefix: &str) -> Vec<u8> {
    let entry = std::fs::read_dir(dir())
        .expect("fixtures")
        .map(|e| e.expect("entry").path())
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(prefix))
        })
        .expect("standalone object");
    std::fs::read(entry).expect("object bytes")
}

/// `(template id, standalone object prefix, symbol, block)`: MegaM0 rows-6 pools.
const TEMPLATES: &[(u32, &str, &str, u32)] = &[
    (8, "6d0492fc", "gemv_bf16_xf32_x4_rows_otile_r6", 32),
    (10, "31db469d", "mq_rotate_x", 32),
    (11, "b5b73fdd", "gemm_mq6g256v2_f32_rows_x4_pk_r6_w4_s1", 32),
    (
        17,
        "7dbb0485",
        "moe_gate_up_unscatter_silu_rotate128_top10",
        32,
    ),
    (
        6,
        "b32b3e78",
        "gated_delta_gate_rotate_bf16_f32_batched",
        64,
    ),
    (7, "6d0492fc", "gemv_bf16_xf32_k4_rows_otile_r6", 128),
    (12, "b5b73fdd", "gemm_mq6g256v2_f32_rows_pk_r6_w4_s4", 128),
    (
        16,
        "07782644",
        "gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4",
        128,
    ),
    (
        18,
        "8c49fcc0",
        "gemv_mq4g128v2p_moe_down_top10_dni_k640",
        128,
    ),
    (4, "b32b3e78", "gated_delta_conv_bf16_f32_batched_k4", 256),
    (9, "1ee5877c", "hyper_read_up_fused_f32_sx_t1_k320", 256),
    (13, "95774460", "ar_hc_write_s_f32s_gfx1201", 256),
    (14, "6681f8e9", "zero_f32", 256),
    (15, "d3ee141d", "moe_route_scatter_top10_f32", 256),
];

fn facts(prefix: &str, symbol: &str, block: u32) -> (KernelStatus, WgFacts) {
    let bytes = object(prefix);
    let lifted = lift_object(
        &bytes,
        Options {
            frontend: Frontend::Hipcc,
        },
    )
    .expect("lift");
    let kernel = lifted
        .program
        .kernels
        .iter()
        .find(|k| k.symbol.0 == symbol)
        .expect("kernel");
    let a1 = kernargs::analyze(kernel, Arch::Gfx1201).status;
    let sealed = a1 != KernelStatus::Proven;
    let r = analyze(kernel, Arch::Gfx1201, sealed, WgLaunch { block_x: block });
    (a1, r)
}

#[test]
fn dump_real_bodies() {
    for &(t, prefix, symbol, block) in TEMPLATES {
        if std::env::var("WG_ONLY").is_ok_and(|o| o != symbol) {
            continue;
        }
        let (a1, r) = facts(prefix, symbol, block);
        println!("== t{t} {symbol} b{block}: A1 {a1:?}, {:?}", r.applicable);
        for a in &r.args {
            match &a.class {
                ArgWgClass::Disjoint(d) => println!(
                    "   arg@{}: DISJOINT axes {:?} bounds {:?} requires {:?} sites {}",
                    a.arg_offset, d.axes, d.bounds, d.requires, d.sites
                ),
                other => println!("   arg@{}: {other:?}", a.arg_offset),
            }
        }
    }
}

fn class_of(prefix: &str, symbol: &str, block: u32, offset: u32) -> ArgWgClass {
    let (_, r) = facts(prefix, symbol, block);
    assert_eq!(r.applicable, Ok(()), "{symbol}");
    r.arg(offset)
        .unwrap_or_else(|| panic!("{symbol}@{offset}"))
        .clone()
}

fn disjoint(c: &ArgWgClass) -> &peacemaker_ir::passes::wg_footprint::WgDisjoint {
    match c {
        ArgWgClass::Disjoint(d) => d,
        other => panic!("{other:?}"),
    }
}

/// `zero_f32` (t14, block 256): `P + 4 * (wg_x * blockDim + lane)` with the block
/// read from the hidden group size dword: disjoint along x under the recorded
/// requirement and tile bound.
#[test]
fn zero_f32_is_disjoint_along_x_under_the_group_size_requirement() {
    let c = class_of("6681f8e9", "zero_f32", 256, 0);
    let d = disjoint(&c);
    assert_eq!(d.axes, [0]);
    assert_eq!(d.block_x, 256);
    assert_eq!(d.bounds[0], 1 << 24);
    assert_eq!(d.requires.len(), 1);
    assert_eq!(
        (
            d.requires[0].offset,
            d.requires[0].mask,
            d.requires[0].value
        ),
        (28, 0xffff, 256)
    );
}

/// The receipt record of that fact.
#[test]
fn zero_f32_record_carries_block_bound_and_requirement() {
    let bytes = object("6681f8e9");
    let lifted = lift_object(
        &bytes,
        Options {
            frontend: Frontend::Hipcc,
        },
    )
    .expect("lift");
    let kernel = lifted
        .program
        .kernels
        .iter()
        .find(|k| k.symbol.0 == "zero_f32")
        .expect("kernel");
    let a1 = kernargs::analyze(kernel, Arch::Gfx1201);
    let recs = railgun_cert::wg_footprint_records(kernel, Arch::Gfx1201, &a1, Some(256));
    assert_eq!(recs.len(), 1);
    let r = &recs[0];
    assert_eq!(
        (r.record_offset, r.axes.clone(), r.block_x),
        (0, vec!["x".to_owned()], 256)
    );
    assert_eq!(
        (r.bounds[0].axis.as_str(), r.bounds[0].bound),
        ("x", 1 << 24)
    );
    assert_eq!(
        (
            r.requires[0].offset,
            r.requires[0].mask,
            r.requires[0].value
        ),
        (28, 0xffff, 256)
    );
}

/// `gated_delta_gate_rotate` (t6, block 64): the b128 stores through the pointers
/// at +0 and +24 write `1024 * wg_x + 16 * lane` (hand-derived by reading the
/// disassembly); the 8 b32 stores through +32 are restricted by an EXEC lane
/// guard (only lanes < 32) the prover does not track, so no fact.
#[test]
fn gate_rotate_proves_two_of_three_written_arguments() {
    let sym = "gated_delta_gate_rotate_bf16_f32_batched";
    for off in [0, 24] {
        let c = class_of("b32b3e78", sym, 64, off);
        assert_eq!(disjoint(&c).axes, [0], "+{off}");
    }
    assert!(matches!(
        class_of("b32b3e78", sym, 64, 32),
        ArgWgClass::Unproven(_)
    ));
}

/// The real conflicts MegaM0 hit (j8250): `gemv_k4_otile` +16 and the o4 MoE
/// gate/up `+32` are NOT provable by this analysis (column-major `c * M + row`
/// output guarded by `row < M`, and a row-major `row * N + col` output guarded by
/// `col < N`: both need a guard-bounded radix argument the prover lacks).
#[test]
fn the_j8250_conflicts_are_still_refused() {
    assert!(matches!(
        class_of("6d0492fc", "gemv_bf16_xf32_k4_rows_otile_r6", 128, 16),
        ArgWgClass::Unproven(_)
    ));
    assert!(matches!(
        class_of(
            "07782644",
            "gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4",
            128,
            32
        ),
        ArgWgClass::Unproven(_)
    ));
}
