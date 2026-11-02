// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Arbitrary-operand gfx1100 probe of the four epilogue intrinsics the exact K2 oracle consumes
//! (B4-C key closure): `exp2(arg)`, `ldexp(value, exponent)`, `divide(numerator, denominator)` and
//! `mul(lhs, rhs)` (the final `Finish::MulUp` product `q * up`, `lhs` = `q` in src0, `rhs` = `up` in vsrc1).
//! Every operand is a raw 32-bit pattern (no normalisation: NaN payloads, infinities, zeros, subnormals,
//! any `i32` exponent; `inf * 0` and `0 * inf` are distinct keys, as are both operand orders of any NaN pair).
//! It is a sibling of `gfx1100_epilogue_intrinsic_probe` and
//! `gfx1100_divide_cases_probe` and leaves both, their code objects and their schemas untouched.
//! No candidate K2 code participates: the instruction forms are copied from
//! `gemm_uk/epilogue.rs::Epilogue::silu_dense_with` (`v_exp_f32_e32 t0, t0` for step 7,
//! `v_ldexp_f32 t0, t0, t2` for step 9, steps 15..25 for the complete IEEE divide DAG with
//! `DivNum::Gate`, `n = 1`: numerator register `g`, denominator register `t0`, and
//! `v_mul_f32_e32 dst, q, up` from step 26 with `Finish::MulUp`).
//!
//! Kernel `c24_intrinsic_keys_probe_gfx1100` (wave32, 32 threads per workgroup, no LDS). The four
//! operations are independent: each thread owns one *index* `i` and runs the four operations on four
//! separate operand sets with separate registers and no data flowing between them (the exp2 result never
//! feeds ldexp, the divide result never feeds mul, etc.). Lists shorter than the longest are padded on the
//! host with benign operands whose records the observer discards. Kernarg (24 B): `keys@0` (read-only,
//! `n` x 32 B), `out@8` (write-only, `n * 128` B), `n@16` (u32, multiple of 32).
//!
//! Key record (8 words): `exp2_arg, ldexp_value, ldexp_exponent, divide_numerator, divide_denominator,
//! mul_lhs, mul_rhs, 0`.
//! Output record (32 words, thread `i` at `out + 128 * i`):
//!
//! | word | contents |
//! |---|---|
//! | 0 | exp2 operand (echo) |
//! | 1 | `v_exp_f32 a, a` — **exp2 result** |
//! | 2, 3 | ldexp value, exponent (echo) |
//! | 4 | `v_ldexp_f32 r, r, e` — **ldexp result** |
//! | 5, 6 | divide numerator, denominator (echo) |
//! | 7 | `v_div_scale_f32 t1, null, den, den, num` |
//! | 8 | `v_div_scale_f32 t2, s_num, num, den, num` |
//! | 9 | `v_rcp_f32 t3, t1` |
//! | 10 | `v_fma_f32 t4, -t1, t3, 1.0` |
//! | 11 | `v_fmac_f32 t3, t4, t3` |
//! | 12 | `v_mul_f32 t4, t2, t3` |
//! | 13 | `v_fma_f32 t5, -t1, t4, t2` |
//! | 14 | `v_fmac_f32 t4, t5, t3` |
//! | 15 | `v_fma_f32 t5, -t1, t4, t2` |
//! | 16 | `v_div_fmas_f32 t5, t5, t3, t4` |
//! | 17 | `v_div_fixup_f32 t5, t5, den, num` — **divide result** |
//! | 18 | the numerator-scale carry `s_num` (VCC input of `v_div_fmas`) as 0 / 1 for this lane |
//! | 19 | index `i` (workgroup base + lane) |
//! | 20, 21 | mul lhs (`q`), rhs (`up`) (echo) |
//! | 22 | `v_mul_f32_e32 r, lhs, rhs` (`Finish::MulUp`: src0 = lhs, vsrc1 = rhs) — **mul result** |
//! | 23..30 | 0 |
//! | 31 | marker `0x4b455950` |
//!
//! Usage: `cargo run -p peacemaker-isa --release --features lab --example gfx1100_intrinsic_keys_probe -- --out-dir DIR`
//! writes `DIR/gfx1100_intrinsic_keys_probe.{s,proof.json,co,hxaco,manifest.json}` (native assembler).
//! Run the code object with `gfx1100_intrinsic_keys_observe` (hipfire-runtime example) on a key file
//! of schema `consumer24.gfx1100.intrinsic-keys.v1`.

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

const SYMBOL: &str = "c24_intrinsic_keys_probe_gfx1100";
const STEM: &str = "gfx1100_intrinsic_keys_probe";
const KERNARG_BYTES: u32 = 24;
const WORKGROUP: u16 = 32;
const KEY_WORDS: u32 = 8;
const RECORD_WORDS: u32 = 32;
const MARKER: u32 = 0x4b45_5950;
const EXIT: &str = ".Lc24_intrinsic_keys_probe_exit";
const USAGE: &str = "usage: gfx1100_intrinsic_keys_probe --out-dir DIR";
const KEYS_SCHEMA: &str = "consumer24.gfx1100.intrinsic-keys.v1";
const OBSERVATIONS_SCHEMA: &str = "consumer24.gfx1100.intrinsic-observations.v1";

const RECORD_FIELDS: [&str; 32] = [
    "exp2_arg",
    "exp2_result",
    "ldexp_value",
    "ldexp_exponent",
    "ldexp_result",
    "divide_numerator",
    "divide_denominator",
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
    "index",
    "mul_lhs",
    "mul_rhs",
    "mul_result",
    "zero23",
    "zero24",
    "zero25",
    "zero26",
    "zero27",
    "zero28",
    "zero29",
    "zero30",
    "marker",
];

const KARG: u8 = 0;
const WGX: u8 = 2;
const WG32: u8 = 3;
const KEYS_PTR: u8 = 4;
const OUT_PTR: u8 = 6;
const M_NUM: u8 = 8; // lane-mask SGPR inside the 8-wide `masks` range
const NREG: u8 = 16;
const TMP: u8 = 17;
// v0 lane, v1 key byte offset, v2 record byte offset.
const LANE: u8 = 0;
const VIN: u8 = 1;
const VOUT: u8 = 2;
// Loaded key record: v8 exp2 arg, v9 ldexp value, v10 ldexp exponent, v11 numerator, v12 denominator, v13 mul lhs, v14 mul rhs, v15 padding.
const KEY: u8 = 8;
const EXP_ARG: u8 = KEY;
const LD_VAL: u8 = KEY + 1;
const LD_EXP: u8 = KEY + 2;
const NUM: u8 = KEY + 3;
const DEN: u8 = KEY + 4;
const MUL_LHS: u8 = KEY + 5;
const MUL_RHS: u8 = KEY + 6;
// Independent working registers: exp2 `a`, ldexp `r`/`e`, mul `r` (v16..v19), divide t0..t5 (v24..v31).
const EXP_W: u8 = 16;
const LD_W: u8 = 17;
const LD_E: u8 = 18;
const MUL_W: u8 = 19;
const T: u8 = 24;
const REC: u8 = 32; // record words v32..v63
const VGPRS: u16 = 64;

fn plan() -> Result<RegPlan, String> {
    let mut regs = RegPlan::new(VGPRS, 24)?;
    for (name, kind, base, len) in [
        ("kernarg", Kind::S, KARG, 2),
        ("workgroup_x", Kind::S, WGX, 1),
        ("workgroup_base", Kind::S, WG32, 1),
        ("pointers", Kind::S, KEYS_PTR, 4),
        ("masks", Kind::S, M_NUM, 8),
        ("count", Kind::S, NREG, 1),
        ("temp", Kind::S, TMP, 1),
        ("lane_and_io", Kind::V, 0, 8),
        ("key_in", Kind::V, KEY, 8),
        ("scalar_ops", Kind::V, EXP_W, 8),
        ("divide_temps", Kind::V, T, 8),
        ("record_0", Kind::V, REC, 8),
        ("record_1", Kind::V, REC + 8, 8),
        ("record_2", Kind::V, REC + 16, 8),
        ("record_3", Kind::V, REC + 24, 8),
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
    smem(b, KEYS_PTR, 4, KARG, 0)?;
    smem(b, NREG, 1, KARG, 16)?;
    sop(b, format!("s_lshl_b32 s{WG32}, s{WGX}, 5"), &[WG32], &[WGX])?;
    sop(
        b,
        format!("s_cmp_ge_u32 s{WG32}, s{NREG}"),
        &[],
        &[WG32, NREG],
    )?;
    branch(b, &format!("s_cbranch_scc1 {EXIT}"))?;
    // keys += 32 B * 32 * wg; out += 128 B * 32 * wg.
    sop(b, format!("s_lshl_b32 s{TMP}, s{WG32}, 5"), &[TMP], &[WG32])?;
    add64(b, KEYS_PTR, KEYS_PTR, TMP)?;
    sop(b, format!("s_lshl_b32 s{TMP}, s{WG32}, 7"), &[TMP], &[WG32])?;
    add64(b, OUT_PTR, OUT_PTR, TMP)?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{VIN}, 5, v{LANE}"),
        &[v(VIN)],
        &[v(LANE)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{VOUT}, 7, v{LANE}"),
        &[v(VOUT)],
        &[v(LANE)],
    )?;
    mem(
        b,
        format!(
            "global_load_b128 v[{KEY}:{}], v{VIN}, s[{KEYS_PTR}:{}]",
            KEY + 3,
            KEYS_PTR + 1
        ),
        &[vr(KEY, 4)],
        &[v(VIN), sr(KEYS_PTR, 2)],
        MemoryClass::VmemLoad,
    )?;
    mem(
        b,
        format!(
            "global_load_b128 v[{}:{}], v{VIN}, s[{KEYS_PTR}:{}] offset:16",
            KEY + 4,
            KEY + 7,
            KEYS_PTR + 1
        ),
        &[vr(KEY + 4, 4)],
        &[v(VIN), sr(KEYS_PTR, 2)],
        MemoryClass::VmemLoad,
    )?;
    // Raw echoes of every operand.
    tap(b, 0, EXP_ARG)?;
    tap(b, 2, LD_VAL)?;
    tap(b, 3, LD_EXP)?;
    tap(b, 5, NUM)?;
    tap(b, 6, DEN)?;
    tap(b, 20, MUL_LHS)?;
    tap(b, 21, MUL_RHS)?;
    // Index echo (workgroup base + lane).
    op(
        b,
        format!("v_add_nc_u32_e32 v{}, s{WG32}, v{LANE}", REC + 19),
        &[v(REC + 19)],
        &[s(WG32), v(LANE)],
    )
}

/// Epilogue step 7 (`v_exp_f32_e32 t0, t0`), isolated: the operand is the raw key bits.
fn exp2_op(b: &mut Builder) -> R {
    op(
        b,
        format!("v_mov_b32_e32 v{EXP_W}, v{EXP_ARG}"),
        &[v(EXP_W)],
        &[v(EXP_ARG)],
    )?;
    op(
        b,
        format!("v_exp_f32_e32 v{0}, v{0}", EXP_W),
        &[v(EXP_W)],
        &[v(EXP_W)],
    )?;
    tap(b, 1, EXP_W)
}

/// Epilogue step 9 (`v_ldexp_f32 t0, t0, t2`), isolated: value and exponent are the raw key bits.
fn ldexp_op(b: &mut Builder) -> R {
    op(
        b,
        format!("v_mov_b32_e32 v{LD_W}, v{LD_VAL}"),
        &[v(LD_W)],
        &[v(LD_VAL)],
    )?;
    op(
        b,
        format!("v_mov_b32_e32 v{LD_E}, v{LD_EXP}"),
        &[v(LD_E)],
        &[v(LD_EXP)],
    )?;
    op(
        b,
        format!("v_ldexp_f32 v{0}, v{0}, v{1}", LD_W, LD_E),
        &[v(LD_W)],
        &[v(LD_W), v(LD_E)],
    )?;
    tap(b, 4, LD_W)
}

/// `silu_dense_with` step 26 with `Finish::MulUp` (`v_mul_f32_e32 gate, q, up`), isolated and independent of the divide
/// result: `q` (src0) is the raw lhs key bits, `up` (vsrc1) the raw rhs key bits; the destination is a separate register.
fn mul_op(b: &mut Builder) -> R {
    op(
        b,
        format!("v_mul_f32_e32 v{}, v{}, v{}", MUL_W, MUL_LHS, MUL_RHS),
        &[v(MUL_W)],
        &[v(MUL_LHS), v(MUL_RHS)],
    )?;
    tap(b, 22, MUL_W)
}

/// `silu_dense_with` steps 15..25 (`DivNum::Gate`, n = 1) with `g = v{NUM}`, `t0 = v{T}` (copy of the denominator).
fn divide_dag(b: &mut Builder) -> R {
    let t = |i: u8| T + i;
    let g = NUM;
    let m_num = M_NUM;
    op(
        b,
        format!("v_mov_b32_e32 v{}, v{DEN}", t(0)),
        &[v(t(0))],
        &[v(DEN)],
    )?;
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
    tap(b, 7, t(1))?;
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
    tap(b, 8, t(2))?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{}, 0, 1, s{m_num}", REC + 18),
        &[v(REC + 18)],
        &[s(m_num)],
    )?;
    op(
        b,
        format!("v_rcp_f32_e32 v{}, v{}", t(3), t(1)),
        &[v(t(3))],
        &[v(t(1))],
    )?;
    tap(b, 9, t(3))?;
    op(
        b,
        format!("v_fma_f32 v{}, -v{}, v{}, 1.0", t(4), t(1), t(3)),
        &[v(t(4))],
        &[v(t(1)), v(t(3))],
    )?;
    tap(b, 10, t(4))?;
    op(
        b,
        format!("v_fmac_f32_e32 v{0}, v{1}, v{0}", t(3), t(4)),
        &[v(t(3))],
        &[v(t(3)), v(t(4))],
    )?;
    tap(b, 11, t(3))?;
    op(
        b,
        format!("v_mul_f32_e32 v{}, v{}, v{}", t(4), t(2), t(3)),
        &[v(t(4))],
        &[v(t(2)), v(t(3))],
    )?;
    tap(b, 12, t(4))?;
    op(
        b,
        format!("v_fma_f32 v{}, -v{}, v{}, v{}", t(5), t(1), t(4), t(2)),
        &[v(t(5))],
        &[v(t(1)), v(t(4)), v(t(2))],
    )?;
    tap(b, 13, t(5))?;
    op(
        b,
        format!("v_fmac_f32_e32 v{}, v{}, v{}", t(4), t(5), t(3)),
        &[v(t(4))],
        &[v(t(4)), v(t(5)), v(t(3))],
    )?;
    tap(b, 14, t(4))?;
    op(
        b,
        format!("v_fma_f32 v{}, -v{}, v{}, v{}", t(5), t(1), t(4), t(2)),
        &[v(t(5))],
        &[v(t(1)), v(t(4)), v(t(2))],
    )?;
    tap(b, 15, t(5))?;
    op(b, format!("s_mov_b32 vcc_lo, s{m_num}"), &[], &[s(m_num)])?;
    op(
        b,
        format!("v_div_fmas_f32 v{0}, v{0}, v{1}, v{2}", t(5), t(3), t(4)),
        &[v(t(5))],
        &[v(t(5)), v(t(3)), v(t(4))],
    )?;
    tap(b, 16, t(5))?;
    op(
        b,
        format!("v_div_fixup_f32 v{0}, v{0}, v{1}, v{2}", t(5), t(0), g),
        &[v(t(5))],
        &[v(t(5)), v(t(0)), v(g)],
    )?;
    tap(b, 17, t(5))
}

fn emit() -> Result<peacemaker_isa::Emitted, String> {
    let kernargs = KernargLayout::new(KERNARG_BYTES)
        .pointer_access("keys", 0, Access::ReadOnly)
        .pointer_access("out", 8, Access::WriteOnly)
        .hidden("n", 16, 4, "by_value");
    let spec = KernelSpec {
        kernel_id: "c24_intrinsic_keys_probe".into(),
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
    exp2_op(&mut b)?;
    ldexp_op(&mut b)?;
    divide_dag(&mut b)?;
    mul_op(&mut b)?;
    for w in 23..31u8 {
        op(
            &mut b,
            format!("v_mov_b32_e32 v{}, 0", REC + w),
            &[v(REC + w)],
            &[],
        )?;
    }
    op(
        &mut b,
        format!("v_mov_b32_e32 v{}, {}", REC + 31, lit(MARKER)),
        &[v(REC + 31)],
        &[],
    )?;
    for i in 0..8u8 {
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
        "schema": "consumer24.gfx1100.intrinsic-keys-build.v1", "keys_schema": KEYS_SCHEMA, "probe_schema": OBSERVATIONS_SCHEMA,
        "symbol": SYMBOL, "arch": "gfx1100",
        "dag_source": "crates/peacemaker-isa/src/kernels/gemm_uk/epilogue.rs::Epilogue::silu_dense_with step 7 (v_exp_f32), step 9 (v_ldexp_f32), steps 15..25 (DivNum::Gate, n = 1), step 26 (Finish::MulUp: v_mul_f32_e32 gate, q, up)",
        "dag_source_sha256": sha(include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/kernels/gemm_uk/epilogue.rs"))),
        "generator_sha256": sha(include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/gfx1100_intrinsic_keys_probe.rs"))),
        "s_sha256": sha(text.as_bytes()), "co_sha256": sha(&co), "hxaco_sha256": sha(&hxaco),
        "builder_crate_version": emitted.proof.builder_crate_version, "builder_git_sha": emitted.proof.builder_git_sha,
        "kernarg_bytes": KERNARG_BYTES, "workgroup_size": WORKGROUP, "key_words": KEY_WORDS, "record_words": RECORD_WORDS, "record_fields": RECORD_FIELDS,
        "marker": MARKER, "kernarg": {"keys": 0, "out": 8, "n": 16},
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
        eprintln!("gfx1100_intrinsic_keys_probe: {e}");
        std::process::exit(1);
    }
}
