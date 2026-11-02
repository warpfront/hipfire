// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Arbitrary-operand gfx1100 probe of the epilogue's complete IEEE divide DAG, for operand pairs that
//! need not arise from the BF16 gate domain (B4-C divide audit). It is a sibling of
//! `gfx1100_epilogue_intrinsic_probe` and leaves that probe, its code object and its schema untouched.
//!
//! The divide steps are the verbatim `gemm_uk/epilogue.rs::Epilogue::silu_dense_with` steps 15..25
//! (`v_div_scale_f32` x2, `v_rcp_f32`, `v_fma_f32`, `v_fmac_f32`, `v_mul_f32`, `v_fma_f32`,
//! `v_fmac_f32`, `v_fma_f32`, `s_mov vcc_lo` + `v_div_fmas_f32`, `v_div_fixup_f32`) for `n = 1`,
//! `DivNum::Gate`: numerator register `g` (the epilogue's gate), denominator register `t0` (the
//! epilogue's `1 + r`), same temporaries `t1..t5`, same operand order. A `v_mov_b32` tap after each
//! step copies the intermediate into the record; taps never change an operand.
//!
//! Kernel `c24_divide_cases_probe_gfx1100` (wave32, 32 threads per workgroup, no LDS). Kernarg (24 B):
//! `cases@0` (read-only, `n` pairs of u32 `{numerator_bits, denominator_bits}`), `out@8` (write-only,
//! `n * 64` B), `n@16` (u32, multiple of 32). Thread `i` writes the 16-word record:
//!
//! | word | contents |
//! |---|---|
//! | 0, 1 | numerator, denominator bits (echo) |
//! | 2 | `v_div_scale_f32 t1, null, den, den, num` |
//! | 3 | `v_div_scale_f32 t2, s_num, num, den, num` |
//! | 4 | `v_rcp_f32 t3, t1` |
//! | 5 | `v_fma_f32 t4, -t1, t3, 1.0` |
//! | 6 | `v_fmac_f32 t3, t4, t3` |
//! | 7 | `v_mul_f32 t4, t2, t3` |
//! | 8 | `v_fma_f32 t5, -t1, t4, t2` |
//! | 9 | `v_fmac_f32 t4, t5, t3` |
//! | 10 | `v_fma_f32 t5, -t1, t4, t2` |
//! | 11 | `v_div_fmas_f32 t5, t5, t3, t4` |
//! | 12 | `v_div_fixup_f32 t5, t5, den, num` — **divide result** |
//! | 13 | the numerator-scale carry `s_num` (VCC input of `v_div_fmas`) as 0 / 1 for this lane |
//! | 14 | 0 |
//! | 15 | marker `0x44495650` |
//!
//! Usage: `cargo run -p peacemaker-isa --release --features lab --example gfx1100_divide_cases_probe -- --out-dir DIR`
//! writes `DIR/gfx1100_divide_cases_probe.{s,proof.json,co,hxaco,manifest.json}` (native assembler).

use peacemaker_isa::{
    insn::{Instruction, MemoryClass},
    kernels::common::{add64, lit, mem, op, s, smem, sop, sr, v, vr},
    native,
    plan::Access,
    reg::{Kind, Live},
    Arch, Builder, KernargLayout, KernelSpec, RegPlan,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::PathBuf};

type R = Result<(), String>;

const SYMBOL: &str = "c24_divide_cases_probe_gfx1100";
const STEM: &str = "gfx1100_divide_cases_probe";
const KERNARG_BYTES: u32 = 24;
const WORKGROUP: u16 = 32;
const RECORD_WORDS: u32 = 16;
const MARKER: u32 = 0x4449_5650;
const EXIT: &str = ".Lc24_divcases_probe_exit";
const USAGE: &str = "usage: gfx1100_divide_cases_probe --out-dir DIR";
const SCHEMA: &str = "consumer24.gfx1100.divide-cases.v1";

const RECORD_FIELDS: [&str; 16] = [
    "numerator_bits",
    "denominator_bits",
    "div_scale_den",
    "div_scale_num",
    "rcp",
    "fma_one_minus_d_r",
    "fmac_r_refine",
    "mul_q0",
    "fma_residual0",
    "fmac_q1",
    "fma_residual1",
    "div_fmas",
    "divide_result",
    "numerator_scale_carry",
    "zero",
    "marker",
];

const KARG: u8 = 0;
const WGX: u8 = 2;
const WG32: u8 = 3;
const GATES_PTR: u8 = 4;
const OUT_PTR: u8 = 6;
const M_NUM: u8 = 8; // lane-mask SGPR inside the 8-wide `masks` range
const NREG: u8 = 16;
const TMP: u8 = 17;
// v0 lane, v1 case byte offset, v2 numerator, v3 denominator, v4 record byte offset.
const LANE: u8 = 0;
const VIN: u8 = 1;
const NUM: u8 = 2;
const DEN: u8 = 3;
const VOUT: u8 = 4;
const T: u8 = 8; // t0 = denominator copy, t1..t5 divide temporaries
const REC: u8 = 16; // record words v16..v31
const VGPRS: u16 = 32;

fn plan() -> Result<RegPlan, String> {
    let mut regs = RegPlan::new(VGPRS, 24)?;
    for (name, kind, base, len) in [
        ("kernarg", Kind::S, KARG, 2),
        ("workgroup_x", Kind::S, WGX, 1),
        ("workgroup_base", Kind::S, WG32, 1),
        ("pointers", Kind::S, GATES_PTR, 4),
        ("masks", Kind::S, M_NUM, 8),
        ("count", Kind::S, NREG, 1),
        ("temp", Kind::S, TMP, 1),
        ("lane_and_io", Kind::V, 0, 8),
        ("divide_temps", Kind::V, T, 8),
        ("record_lo", Kind::V, REC, 8),
        ("record_hi", Kind::V, REC + 8, 8),
    ] {
        regs.add_range(name, kind, base, len, Live::Whole)?;
    }
    Ok(regs)
}

fn branch(b: &mut Builder, text: &str) -> R {
    b.control(Instruction::new(text, vec![], vec![]))
}
fn tap(b: &mut Builder, word: u8, src: u8) -> R {
    op(
        b,
        format!("v_mov_b32_e32 v{}, v{src}", REC + word),
        &[v(REC + word)],
        &[v(src)],
    )
}

fn prologue(b: &mut Builder) -> R {
    smem(b, GATES_PTR, 4, KARG, 0)?;
    smem(b, NREG, 1, KARG, 16)?;
    sop(b, format!("s_lshl_b32 s{WG32}, s{WGX}, 5"), &[WG32], &[WGX])?;
    sop(
        b,
        format!("s_cmp_ge_u32 s{WG32}, s{NREG}"),
        &[],
        &[WG32, NREG],
    )?;
    branch(b, &format!("s_cbranch_scc1 {EXIT}"))?;
    // cases += 8 * 32 * wg; out += 64 * 32 * wg.
    sop(b, format!("s_lshl_b32 s{TMP}, s{WG32}, 3"), &[TMP], &[WG32])?;
    add64(b, GATES_PTR, GATES_PTR, TMP)?;
    sop(b, format!("s_lshl_b32 s{TMP}, s{WG32}, 6"), &[TMP], &[WG32])?;
    add64(b, OUT_PTR, OUT_PTR, TMP)?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{VIN}, 3, v{LANE}"),
        &[v(VIN)],
        &[v(LANE)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{VOUT}, 6, v{LANE}"),
        &[v(VOUT)],
        &[v(LANE)],
    )?;
    mem(
        b,
        format!(
            "global_load_b64 v[{NUM}:{DEN}], v{VIN}, s[{GATES_PTR}:{}]",
            GATES_PTR + 1
        ),
        &[vr(NUM, 2)],
        &[v(VIN), sr(GATES_PTR, 2)],
        MemoryClass::VmemLoad,
    )?;
    tap(b, 0, NUM)?;
    tap(b, 1, DEN)?;
    // t0 = the epilogue's denominator register.
    op(b, format!("v_mov_b32_e32 v{T}, v{DEN}"), &[v(T)], &[v(DEN)])
}

/// `silu_dense_with` steps 15..25 (`DivNum::Gate`, n = 1) with `g = v{NUM}`, `t0 = v{T}`.
fn divide_dag(b: &mut Builder) -> R {
    let t = |i: u8| T + i;
    let g = NUM;
    let m_num = M_NUM;
    op(
        b,
        format!(
            "v_div_scale_f32 v{0}, null, v{1}, v{1}, v{2}",
            t(1),
            t(0),
            g
        ),
        &[v(t(1))],
        &[v(t(0)), v(g)],
    )?;
    tap(b, 2, t(1))?;
    op(
        b,
        format!(
            "v_div_scale_f32 v{0}, s{1}, v{2}, v{3}, v{2}",
            t(2),
            m_num,
            g,
            t(0)
        ),
        &[v(t(2)), s(m_num)],
        &[v(t(0)), v(g)],
    )?;
    tap(b, 3, t(2))?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{}, 0, 1, s{m_num}", REC + 13),
        &[v(REC + 13)],
        &[s(m_num)],
    )?;
    op(
        b,
        format!("v_rcp_f32_e32 v{}, v{}", t(3), t(1)),
        &[v(t(3))],
        &[v(t(1))],
    )?;
    tap(b, 4, t(3))?;
    op(
        b,
        format!("v_fma_f32 v{}, -v{}, v{}, 1.0", t(4), t(1), t(3)),
        &[v(t(4))],
        &[v(t(1)), v(t(3))],
    )?;
    tap(b, 5, t(4))?;
    op(
        b,
        format!("v_fmac_f32_e32 v{0}, v{1}, v{0}", t(3), t(4)),
        &[v(t(3))],
        &[v(t(3)), v(t(4))],
    )?;
    tap(b, 6, t(3))?;
    op(
        b,
        format!("v_mul_f32_e32 v{}, v{}, v{}", t(4), t(2), t(3)),
        &[v(t(4))],
        &[v(t(2)), v(t(3))],
    )?;
    tap(b, 7, t(4))?;
    op(
        b,
        format!("v_fma_f32 v{}, -v{}, v{}, v{}", t(5), t(1), t(4), t(2)),
        &[v(t(5))],
        &[v(t(1)), v(t(4)), v(t(2))],
    )?;
    tap(b, 8, t(5))?;
    op(
        b,
        format!("v_fmac_f32_e32 v{}, v{}, v{}", t(4), t(5), t(3)),
        &[v(t(4))],
        &[v(t(4)), v(t(5)), v(t(3))],
    )?;
    tap(b, 9, t(4))?;
    op(
        b,
        format!("v_fma_f32 v{}, -v{}, v{}, v{}", t(5), t(1), t(4), t(2)),
        &[v(t(5))],
        &[v(t(1)), v(t(4)), v(t(2))],
    )?;
    tap(b, 10, t(5))?;
    op(b, format!("s_mov_b32 vcc_lo, s{m_num}"), &[], &[s(m_num)])?;
    op(
        b,
        format!("v_div_fmas_f32 v{0}, v{0}, v{1}, v{2}", t(5), t(3), t(4)),
        &[v(t(5))],
        &[v(t(5)), v(t(3)), v(t(4))],
    )?;
    tap(b, 11, t(5))?;
    op(
        b,
        format!("v_div_fixup_f32 v{0}, v{0}, v{1}, v{2}", t(5), t(0), g),
        &[v(t(5))],
        &[v(t(5)), v(t(0)), v(g)],
    )?;
    tap(b, 12, t(5))
}

fn emit() -> Result<peacemaker_isa::Emitted, String> {
    let kernargs = KernargLayout::new(KERNARG_BYTES)
        .pointer_access("cases", 0, Access::ReadOnly)
        .pointer_access("out", 8, Access::WriteOnly)
        .hidden("n", 16, 4, "by_value");
    let spec = KernelSpec {
        kernel_id: "c24_divide_cases_probe".into(),
        variant: "gfx1100".into(),
        arch: Arch::Gfx1100,
        symbol: SYMBOL.into(),
        kernargs,
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: WORKGROUP,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(spec, plan()?);
    prologue(&mut b)?;
    divide_dag(&mut b)?;
    op(
        &mut b,
        format!("v_mov_b32_e32 v{}, 0", REC + 14),
        &[v(REC + 14)],
        &[],
    )?;
    op(
        &mut b,
        format!("v_mov_b32_e32 v{}, {}", REC + 15, lit(MARKER)),
        &[v(REC + 15)],
        &[],
    )?;
    for i in 0..4u8 {
        let first = REC + 4 * i;
        let off = if i == 0 {
            String::new()
        } else {
            format!(" offset:{}", 16 * u32::from(i))
        };
        mem(
            &mut b,
            format!(
                "global_store_b128 v{VOUT}, v[{first}:{}], s[{OUT_PTR}:{}]{off}",
                first + 3,
                OUT_PTR + 1
            ),
            &[],
            &[v(VOUT), vr(first, 4), sr(OUT_PTR, 2)],
            MemoryClass::VmemStore,
        )?;
    }
    b.wait_all()?;
    b.label(EXIT)?;
    branch(&mut b, "s_endpgm")?;
    b.finish()
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut out_dir = None;
    while let Some(flag) = args.next() {
        let value = args.next().ok_or(USAGE)?;
        match flag.as_str() {
            "--out-dir" => out_dir = Some(PathBuf::from(value)),
            _ => return Err(USAGE.into()),
        }
    }
    let dir = out_dir.ok_or(USAGE)?;
    fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let emitted = emit()?;
    let text = &emitted.s_text;
    let begin = text
        .find(&format!("{SYMBOL}:\n"))
        .ok_or("symbol label missing from source")?
        + SYMBOL.len()
        + 2;
    let end = text
        .find(&format!(".L{SYMBOL}_end:"))
        .ok_or("end label missing from source")?;
    let listing: Vec<&str> = text[begin..end]
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let mut census: BTreeMap<&str, u32> = BTreeMap::new();
    for l in &listing {
        if let Some(m) = l.split_whitespace().next() {
            if !m.ends_with(':') {
                *census.entry(m).or_insert(0) += 1
            }
        }
    }
    let co = native::assemble(text, Arch::Gfx1100)?;
    let hxaco = native::bundle(&co, Arch::Gfx1100, native::DEFAULT_HOST_TARGET);
    let files = [
        (format!("{STEM}.s"), text.as_bytes().to_vec()),
        (
            format!("{STEM}.proof.json"),
            serde_json::to_vec_pretty(&emitted.proof).map_err(|e| e.to_string())?,
        ),
        (format!("{STEM}.co"), co.clone()),
        (format!("{STEM}.hxaco"), hxaco.clone()),
    ];
    for (name, bytes) in &files {
        fs::write(dir.join(name), bytes).map_err(|e| format!("{name}: {e}"))?;
    }
    let manifest = json!({
        "schema": "consumer24.gfx1100.divide-cases-build.v1", "probe_schema": SCHEMA,
        "symbol": SYMBOL, "arch": "gfx1100",
        "dag_source": "crates/peacemaker-isa/src/kernels/gemm_uk/epilogue.rs::Epilogue::silu_dense_with steps 15..25 (DivNum::Gate, n = 1)",
        "dag_source_sha256": sha(include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/kernels/gemm_uk/epilogue.rs"))),
        "generator_sha256": sha(include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/gfx1100_divide_cases_probe.rs"))),
        "s_sha256": sha(text.as_bytes()), "co_sha256": sha(&co), "hxaco_sha256": sha(&hxaco),
        "builder_crate_version": emitted.proof.builder_crate_version, "builder_git_sha": emitted.proof.builder_git_sha,
        "kernarg_bytes": KERNARG_BYTES, "workgroup_size": WORKGROUP, "record_words": RECORD_WORDS, "record_fields": RECORD_FIELDS,
        "marker": MARKER, "kernarg": {"cases": 0, "out": 8, "n": 16},
        "denorm": {"float_denorm_mode_32": 3, "float_denorm_mode_16_64": 3, "source": "peacemaker-isa emit.rs descriptor (same as every builder kernel)"},
        "vgprs": VGPRS, "shape": emitted.shape, "instructions": listing.len(), "census": census, "listing": listing,
    });
    let bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?;
    let path = dir.join(format!("{STEM}.manifest.json"));
    fs::write(&path, &bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    println!(
        "{}",
        json!({
            "manifest": path, "manifest_sha256": sha(&bytes), "co": dir.join(format!("{STEM}.co")), "co_sha256": sha(&co),
            "hxaco_sha256": sha(&hxaco), "symbol": SYMBOL, "instructions": listing.len(),
        })
    );
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("gfx1100_divide_cases_probe: {e}");
        std::process::exit(1);
    }
}
