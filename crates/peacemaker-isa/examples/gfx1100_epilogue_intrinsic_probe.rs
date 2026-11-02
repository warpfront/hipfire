// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Independent gfx1100 device probe of the BF16 SiLU epilogue's intrinsic DAG
//! (`gemm_uk/epilogue.rs::Epilogue::silu_dense_with`, `DivNum::Gate`, `Finish::MulUp`), the
//! device side of `mq_i4_reference::EpilogueIntrinsics::{exp2, ldexp, divide}` (B4-C).
//!
//! It shares no code with any candidate K2 emitter. The instruction sequence is a verbatim,
//! single-element copy of the 26 `silu_dense_with` steps (same mnemonics, operand order,
//! literals and temporaries), and a `v_mov_b32` tap after the step that defines each
//! intermediate copies it into the per-gate output record. Taps never change an operand.
//! The same kernel also runs the unmodified `Epilogue::silu_dense` on a second copy of the
//! gate (up = 1.0) and stores that result as word 14, so the observer proves at run time that
//! the tapped copy equals the real emitter for every input (`word13 == word14`).
//!
//! Kernel `c24_epilogue_intrinsic_probe_gfx1100` (wave32, 32 threads per workgroup, no LDS,
//! no barrier). Kernarg (24 bytes): `gates@0` (read-only, `n` u32 raw f32 gate bits), `out@8`
//! (write-only, `n * 64` bytes), `n@16` (u32, a multiple of 32; workgroups at or beyond
//! `n / 32` return). Thread `i` reads `gates[i]` and writes the 16-word record `out[i]`:
//!
//! | word | contents |
//! |---|---|
//! | 0 | gate bits `g` (echo) |
//! | 1 | `ph = (-log2e) * g` |
//! | 2 | `pl = fma(-log2e, g, -ph)` then `fmac(-log2e_lo, g)` |
//! | 3 | `e = v_rndne(ph)` |
//! | 4 | `ph - e` |
//! | 5 | `a = (ph - e) + pl` — **exp2 operand** |
//! | 6 | `v_exp_f32(a)` — **exp2 result** / ldexp operand |
//! | 7 | `v_cvt_i32_f32(e)` — **ldexp exponent** |
//! | 8 | `v_ldexp_f32(exp2 result, exponent)` — **ldexp result** |
//! | 9 | `r` after the `-g < -103.28` underflow `v_cndmask` |
//! | 10 | `r` after the `-g > 88.72` overflow `v_cndmask` |
//! | 11 | `den = 1 + r` — **divide denominator** (numerator is `g`) |
//! | 12 | `v_div_fixup` result — **divide result** (the full `v_div_scale`..`v_div_fixup` DAG) |
//! | 13 | `q * 1.0` (`Finish::MulUp` with up = 1.0) |
//! | 14 | the real `Epilogue::silu_dense(.., 1)` result on the same gate, up = 1.0 |
//! | 15 | marker `0x50524f42` (an unwritten record keeps the host's `0xa5` poison) |
//!
//! Usage: `cargo run -p peacemaker-isa --release --features lab --example gfx1100_epilogue_intrinsic_probe -- --out-dir DIR`
//! writes `DIR/gfx1100_epilogue_intrinsic_probe.{s,proof.json,co,hxaco,manifest.json}`
//! (native assembler and linker: no ROCm tools, no `toolchain` feature). The manifest binds
//! the code object / bundle / source sha256, the instruction listing, the denorm modes and the
//! shared input-set hashes; the observer (`hipfire-runtime` example
//! `gfx1100_epilogue_intrinsic_observe`) refuses a code object whose hash differs from it.

use peacemaker_isa::{
    insn::{Instruction, MemoryClass},
    kernels::{
        common::{add64, lit, mem, op, s, smem, sop, sr, v, vr},
        gemm_uk::Epilogue,
    },
    native,
    plan::Access,
    reg::{Kind, Live},
    Arch, Builder, KernargLayout, KernelSpec, RegPlan,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::PathBuf};

#[path = "gfx1100_epilogue_probe/gates.rs"]
mod gates;

type R = Result<(), String>;

const SYMBOL: &str = "c24_epilogue_intrinsic_probe_gfx1100";
const STEM: &str = "gfx1100_epilogue_intrinsic_probe";
const KERNARG_BYTES: u32 = 24;
const WORKGROUP: u16 = 32;
const RECORD_WORDS: u32 = 16;
const MARKER: u32 = 0x5052_4f42;
const EXIT: &str = ".Lc24_epi_probe_exit";
const USAGE: &str = "usage: gfx1100_epilogue_intrinsic_probe --out-dir DIR";

const RECORD_FIELDS: [&str; 16] = [
    "gate_bits",
    "ph",
    "pl",
    "e_rndne",
    "ph_minus_e",
    "exp2_arg",
    "exp2_result",
    "ldexp_exponent",
    "ldexp_result",
    "r_after_underflow_select",
    "r_after_overflow_select",
    "divide_den",
    "divide_result",
    "final_mul_up1",
    "silu_dense_reference",
    "marker",
];

// SGPRs: s0:1 kernarg pointer; the gfx11 workgroup id x is s2 (user_sgpr_count = 2).
const KARG: u8 = 0;
const WGX: u8 = 2;
const WG32: u8 = 3;
const GATES_PTR: u8 = 4;
const OUT_PTR: u8 = 6;
/// Reference `silu_dense` lane masks (under s8, over s10, num s12) and the tapped copy's (s16, s18, s20).
const REF_MASKS: u8 = 8;
const MY_MASKS: u8 = 16;
const NREG: u8 = 24;
const TMP: u8 = 25;
// VGPRs: v0 lane, v1 gate byte offset, v2 gate (the tapped copy's `g`), v3 record byte offset,
// v4 up = 1.0, v5 the reference copy's gate.
const LANE: u8 = 0;
const VIN: u8 = 1;
const G: u8 = 2;
const VOUT: u8 = 3;
const UP: u8 = 4;
const GREF: u8 = 5;
const T: u8 = 8; // tapped copy temporaries t0..t5 = v8..v13
const RT: u8 = 16; // reference temporaries v16..v21
const REC: u8 = 24; // record words v24..v39
const VGPRS: u16 = 40;

fn plan() -> Result<RegPlan, String> {
    let mut regs = RegPlan::new(VGPRS, 32)?;
    for (name, kind, base, len) in [
        ("kernarg", Kind::S, KARG, 2),
        ("workgroup_x", Kind::S, WGX, 1),
        ("workgroup_base", Kind::S, WG32, 1),
        ("pointers", Kind::S, GATES_PTR, 4),
        ("reference_masks", Kind::S, REF_MASKS, 8),
        ("tapped_masks", Kind::S, MY_MASKS, 8),
        ("count", Kind::S, NREG, 1),
        ("temp", Kind::S, TMP, 1),
        ("lane_and_io", Kind::V, 0, 8),
        ("tapped_temps", Kind::V, T, 8),
        ("reference_temps", Kind::V, RT, 8),
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

/// `v{REC+word} = v{src}` (bit copy).
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
    // gates += 4 * 32 * wg; out += 64 * 32 * wg.
    sop(b, format!("s_lshl_b32 s{TMP}, s{WG32}, 2"), &[TMP], &[WG32])?;
    add64(b, GATES_PTR, GATES_PTR, TMP)?;
    sop(b, format!("s_lshl_b32 s{TMP}, s{WG32}, 6"), &[TMP], &[WG32])?;
    add64(b, OUT_PTR, OUT_PTR, TMP)?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{VIN}, 2, v{LANE}"),
        &[v(VIN)],
        &[v(LANE)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{VOUT}, 6, v{LANE}"),
        &[v(VOUT)],
        &[v(LANE)],
    )?;
    op(b, format!("v_mov_b32_e32 v{UP}, 1.0"), &[v(UP)], &[])?;
    mem(
        b,
        format!(
            "global_load_b32 v{G}, v{VIN}, s[{GATES_PTR}:{}]",
            GATES_PTR + 1
        ),
        &[v(G)],
        &[v(VIN), sr(GATES_PTR, 2)],
        MemoryClass::VmemLoad,
    )?;
    // The reference copy and the echo.
    op(
        b,
        format!("v_mov_b32_e32 v{GREF}, v{G}"),
        &[v(GREF)],
        &[v(G)],
    )?;
    tap(b, 0, G)
}

/// The 26 `Epilogue::silu_dense_with(DivNum::Gate, Finish::MulUp)` steps for one element
/// (`n = 1`, `gate = v{G}`, `up = v{UP}`), in step order, each followed by its tap.
fn tapped_dag(b: &mut Builder) -> R {
    let t = |i: u8| T + i;
    let (m_under, m_over, m_num) = (MY_MASKS, MY_MASKS + 2, MY_MASKS + 4);
    let g = G;
    // ph = RN(-log2e * g); pl = fma(-log2e, g, -ph); pl = fma(-log2e_lo, g, pl)
    op(
        b,
        format!("v_mul_f32_e32 v{}, 0xbfb8aa3b, v{g}", t(0)),
        &[v(t(0))],
        &[v(g)],
    )?;
    tap(b, 1, t(0))?;
    op(
        b,
        format!("v_fma_f32 v{}, 0xbfb8aa3b, v{g}, -v{}", t(1), t(0)),
        &[v(t(1))],
        &[v(g), v(t(0))],
    )?;
    op(
        b,
        format!("v_fmac_f32_e32 v{}, 0xb2a5705f, v{g}", t(1)),
        &[v(t(1))],
        &[v(t(1)), v(g)],
    )?;
    tap(b, 2, t(1))?;
    // e = rndne(ph); a = (ph - e) + pl; r = ldexp(exp2(a), int(e))
    op(
        b,
        format!("v_rndne_f32_e32 v{}, v{}", t(2), t(0)),
        &[v(t(2))],
        &[v(t(0))],
    )?;
    tap(b, 3, t(2))?;
    op(
        b,
        format!("v_sub_f32_e32 v{0}, v{0}, v{1}", t(0), t(2)),
        &[v(t(0))],
        &[v(t(0)), v(t(2))],
    )?;
    tap(b, 4, t(0))?;
    op(
        b,
        format!("v_add_f32_e32 v{0}, v{0}, v{1}", t(0), t(1)),
        &[v(t(0))],
        &[v(t(0)), v(t(1))],
    )?;
    tap(b, 5, t(0))?;
    op(
        b,
        format!("v_exp_f32_e32 v{0}, v{0}", t(0)),
        &[v(t(0))],
        &[v(t(0))],
    )?;
    tap(b, 6, t(0))?;
    op(
        b,
        format!("v_cvt_i32_f32_e32 v{0}, v{0}", t(2)),
        &[v(t(2))],
        &[v(t(2))],
    )?;
    tap(b, 7, t(2))?;
    op(
        b,
        format!("v_ldexp_f32 v{0}, v{0}, v{1}", t(0), t(2)),
        &[v(t(0))],
        &[v(t(0)), v(t(2))],
    )?;
    tap(b, 8, t(0))?;
    // r = -g < -103.28 ? 0 : r; r = -g > 88.72 ? +inf : r
    op(
        b,
        format!("v_cmp_nlt_f32_e64 s{m_under}, 0x42ce8ed0, v{g}"),
        &[s(m_under)],
        &[v(g)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{0}, 0, v{0}, s{1}", t(0), m_under),
        &[v(t(0))],
        &[v(t(0)), s(m_under)],
    )?;
    tap(b, 9, t(0))?;
    op(
        b,
        format!("v_cmp_ngt_f32_e64 s{m_over}, 0xc2b17218, v{g}"),
        &[s(m_over)],
        &[v(g)],
    )?;
    op(
        b,
        format!(
            "v_cndmask_b32_e64 v{0}, 0x7f800000, v{0}, s{1}",
            t(0),
            m_over
        ),
        &[v(t(0))],
        &[v(t(0)), s(m_over)],
    )?;
    tap(b, 10, t(0))?;
    // d = 1 + r; q = g / d (div_scale, rcp, three fma refinements, fmas, fixup)
    op(
        b,
        format!("v_add_f32_e32 v{0}, 1.0, v{0}", t(0)),
        &[v(t(0))],
        &[v(t(0))],
    )?;
    tap(b, 11, t(0))?;
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
    op(
        b,
        format!("v_rcp_f32_e32 v{}, v{}", t(3), t(1)),
        &[v(t(3))],
        &[v(t(1))],
    )?;
    op(
        b,
        format!("v_fma_f32 v{}, -v{}, v{}, 1.0", t(4), t(1), t(3)),
        &[v(t(4))],
        &[v(t(1)), v(t(3))],
    )?;
    op(
        b,
        format!("v_fmac_f32_e32 v{0}, v{1}, v{0}", t(3), t(4)),
        &[v(t(3))],
        &[v(t(3)), v(t(4))],
    )?;
    op(
        b,
        format!("v_mul_f32_e32 v{}, v{}, v{}", t(4), t(2), t(3)),
        &[v(t(4))],
        &[v(t(2)), v(t(3))],
    )?;
    op(
        b,
        format!("v_fma_f32 v{}, -v{}, v{}, v{}", t(5), t(1), t(4), t(2)),
        &[v(t(5))],
        &[v(t(1)), v(t(4)), v(t(2))],
    )?;
    op(
        b,
        format!("v_fmac_f32_e32 v{}, v{}, v{}", t(4), t(5), t(3)),
        &[v(t(4))],
        &[v(t(4)), v(t(5)), v(t(3))],
    )?;
    op(
        b,
        format!("v_fma_f32 v{}, -v{}, v{}, v{}", t(5), t(1), t(4), t(2)),
        &[v(t(5))],
        &[v(t(1)), v(t(4)), v(t(2))],
    )?;
    op(b, format!("s_mov_b32 vcc_lo, s{m_num}"), &[], &[s(m_num)])?;
    op(
        b,
        format!("v_div_fmas_f32 v{0}, v{0}, v{1}, v{2}", t(5), t(3), t(4)),
        &[v(t(5))],
        &[v(t(5)), v(t(3)), v(t(4))],
    )?;
    op(
        b,
        format!("v_div_fixup_f32 v{0}, v{0}, v{1}, v{2}", t(5), t(0), g),
        &[v(t(5))],
        &[v(t(5)), v(t(0)), v(g)],
    )?;
    tap(b, 12, t(5))?;
    // h = q * u
    op(
        b,
        format!("v_mul_f32_e32 v{}, v{}, v{}", g, t(5), UP),
        &[v(g)],
        &[v(t(5)), v(UP)],
    )?;
    tap(b, 13, g)
}

fn emit() -> Result<peacemaker_isa::Emitted, String> {
    let kernargs = KernargLayout::new(KERNARG_BYTES)
        .pointer_access("gates", 0, Access::ReadOnly)
        .pointer_access("out", 8, Access::WriteOnly)
        .hidden("n", 16, 4, "by_value");
    let spec = KernelSpec {
        kernel_id: "c24_epilogue_intrinsic_probe".into(),
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
    tapped_dag(&mut b)?;
    // The unmodified emitter on the second copy: h lands in v{GREF}.
    Epilogue::silu_dense(&mut b, GREF, UP, RT, REF_MASKS, 1)?;
    tap(&mut b, 14, GREF)?;
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
    // Instruction listing of the kernel body (labels and directives excluded).
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
    let (finite, exceptions) = (gates::finite_bf16(), gates::exceptional());
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
        "schema": "consumer24.gfx1100.epilogue-probe-build.v1",
        "probe_schema": gates::SCHEMA,
        "symbol": SYMBOL, "arch": "gfx1100",
        "dag_source": "crates/peacemaker-isa/src/kernels/gemm_uk/epilogue.rs::Epilogue::silu_dense_with(DivNum::Gate, Finish::MulUp)",
        "dag_source_sha256": sha(include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/kernels/gemm_uk/epilogue.rs"))),
        "generator_sha256": sha(include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/gfx1100_epilogue_intrinsic_probe.rs"))),
        "gates_source_sha256": sha(include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/gfx1100_epilogue_probe/gates.rs"))),
        "s_sha256": sha(text.as_bytes()), "co_sha256": sha(&co), "hxaco_sha256": sha(&hxaco),
        "builder_crate_version": emitted.proof.builder_crate_version, "builder_git_sha": emitted.proof.builder_git_sha,
        "kernarg_bytes": KERNARG_BYTES, "workgroup_size": WORKGROUP, "record_words": RECORD_WORDS, "record_fields": RECORD_FIELDS,
        "marker": MARKER, "kernarg": {"gates": 0, "out": 8, "n": 16},
        "denorm": {"float_denorm_mode_32": 3, "float_denorm_mode_16_64": 3, "source": "peacemaker-isa emit.rs descriptor (same as every builder kernel)"},
        "vgprs": VGPRS, "shape": emitted.shape, "instructions": listing.len(), "census": census, "listing": listing,
        "gate_sets": {
            "finite_bf16": {"count": finite.len(), "sha256_le_u32": gates::list_sha256(&finite)},
            "exceptions": {"count": exceptions.len(), "sha256_le_u32": gates::list_sha256(&exceptions)},
        },
    });
    let bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?;
    let path = dir.join(format!("{STEM}.manifest.json"));
    fs::write(&path, &bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    println!(
        "{}",
        json!({
            "manifest": path, "manifest_sha256": sha(&bytes), "co": dir.join(format!("{STEM}.co")), "co_sha256": sha(&co),
            "hxaco_sha256": sha(&hxaco), "symbol": SYMBOL, "instructions": listing.len(),
            "finite_bf16": finite.len(), "exceptions": exceptions.len(),
        })
    );
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("gfx1100_epilogue_intrinsic_probe: {e}");
        std::process::exit(1);
    }
}
