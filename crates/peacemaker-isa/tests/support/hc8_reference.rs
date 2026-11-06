// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Independent reference for the gfx1201 `hc8_wmma_gfx1201` PM kernels. Shared by the CPU byte oracle
//! (`tests/hc8_wmma_gfx1201.rs`) and the hardware byte oracle example (`#[path]` include).
//!
//! NOTHING here uses the compressed-weight generator's instructions, tiles, LDS fold or decode: the reference takes
//! host-decoded BF16 weights in plain row-major order, issues `v_wmma_f32_16x16x16_bf16` directly from its own
//! fragment loads and reproduces the kernel's frozen numerics contract:
//!
//! * per output tile (16 weight rows x 16 columns) and per K slice `ki in 0..KW` (`K/KW` contiguous K elements),
//!   an accumulator starting at +0 chains one WMMA per K16 step in ascending K (`C = 0` inline on the first);
//! * the KW partials fold sequentially ascending, `f = p0; f = f + p1; ...; f = f + p_{KW-1}` (`v_add_f32 f, f, p`);
//! * `set`: `y = f * s[m]` (one `v_mul_f32`, `f` is src0); `act`: `x = f * s[m]`, then the pinned hipcc `hc_act` VALU
//!   transcription (`rt(x)`, `rt(act_scale * x)`, `exp`, `1 + e`, the full `v_div_scale/rcp/fma/div_fmas/div_fixup`
//!   division, `rt`), the same DAG `lane_rows_bf16_gfx1201` (`emit_ref_fold`) pins against the hipcc object.
//!
//! ABI of the reference entry (same 48 bytes as the kernel, `q` differs):
//! `q@0` u64 row-major decoded BF16 `[M][K]` (u16 little endian), `s@8` f32 `[M]`, `x@16` f32 `[rows][K]`
//! (BF16-exact; the high 16 bits are the BF16), `y@24` f32 `[rows][M]`, `M@32`, `K@36` (ignored: K is compile-time),
//! `rows@40` u32, `act_scale@44` f32. Launch: block `REFERENCE_BLOCK` (one wave), grid `reference_grid(M, rows)`;
//! grid Y is the 16-column tile. Dead columns (`n >= rows`) load nothing (zero B) and store nothing.
//!
//! Semantics provenance: the emulator's BF16 WMMA is hardware-qualified (float_edges.md: 0 mismatches over
//! 7,437,056 captured gfx1201 words); `v_cvt_pk_f32_fp8` NaN sign/payload and the `act` exceptional domains are
//! emulator-modeled only: the hardware byte oracle is their receipt.
#![allow(dead_code)]

use peacemaker_isa::insn::{Instruction, MemoryClass};
use peacemaker_isa::kernels::common::{lit, mem as mem_op, op, s, smem, sop, sr, v, vr};
use peacemaker_isa::kernels::hc8_wmma_gfx1201::{Fmt, Geo};
use peacemaker_isa::ledger::Counter;
use peacemaker_isa::plan::Access;
use peacemaker_isa::reg::{Kind, Live};
use peacemaker_isa::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

pub const REFERENCE_BLOCK: u32 = 32;
/// Bytes of the reference kernarg segment (same layout as the kernel).
pub const REFERENCE_KERNARG_BYTES: usize = 48;

type R = Result<(), String>;

/// `hc8_ref_kw{KW}_k{K}_{set|act}`.
pub fn reference_symbol(geo: Geo, k: u32) -> String {
    format!(
        "hc8_ref_kw{}_k{}_{}",
        geo.kw,
        k,
        if geo.act { "act" } else { "set" }
    )
}

/// `(M / 16, ceil(rows / 16))`: grid X is the 16-row weight tile, grid Y the 16-column tile.
pub fn reference_grid(m: u32, rows: u32) -> (u32, u32) {
    (m / 16, rows.div_ceil(16))
}

/// The reference module source for every `(geo, k)` (de-duplicated by symbol). Returns the emitted entries and the
/// assembler text for `peacemaker_isa::native::assemble(&text, Arch::Gfx1201)`.
pub fn reference_module(entries: &[(Geo, u32)]) -> Result<(Vec<Emitted>, String), String> {
    let mut seen: Vec<String> = Vec::new();
    let mut emitted = Vec::new();
    for &(geo, k) in entries {
        let sym = reference_symbol(geo, k);
        if seen.contains(&sym) {
            continue;
        }
        seen.push(sym);
        emitted.push(emit_reference(geo, k)?);
    }
    let (text, _) = peacemaker_isa::kernels::iu4_gemm::module(&emitted, "hc8_reference_module")?;
    Ok((emitted, text))
}

// ------------------------------------------------------------------------------------------------ host decode

fn pow2(e: i32) -> f32 {
    f32::from_bits(((127 + e) as u32) << 23)
}

/// Exact host decode of raw 8-bit weights to BF16 bits (row-major order is preserved). Independent of the PM decode.
///
/// * `E4M3` (OCP, bias 7, no infinity): `e == 0` -> `m * 2^-9`; else `(8 + m) * 2^(e-10)`; `S.1111.111` -> NaN with
///   the sign preserved and quiet bit set (`sign | 0x7fc0`, the emulator's `fp8_to_f32` model of
///   `v_cvt_pk_f32_fp8`; the hardware receipt is the byte oracle). Zero keeps its sign.
/// * `I8`: two's complement, exact (`-128` -> `0xc300`; `0` -> `+0`).
///
/// Every value is exactly a BF16, so the result is the high half of the f32.
pub fn decode_weights(fmt: Fmt, q: &[u8]) -> Vec<u16> {
    q.iter()
        .map(|&b| match fmt {
            Fmt::E4M3 => {
                let sign = u16::from(b & 0x80) << 8;
                let e = i32::from((b >> 3) & 15);
                let m = u32::from(b & 7);
                if e == 15 && m == 7 {
                    return sign | 0x7fc0;
                }
                let mag = if e == 0 {
                    m as f32 * pow2(-9)
                } else {
                    (8 + m) as f32 * pow2(e - 10)
                };
                sign | (mag.to_bits() >> 16) as u16
            }
            Fmt::I8 => ((b as i8) as f32).to_bits().wrapping_shr(16) as u16,
        })
        .collect()
}

/// BF16 bits of an f32 that is BF16-exact (the high 16 bits; the kernel's contract).
pub fn bf16_bits(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

/// Pure-Rust model of the reference numerics (set epilogue): the hardware-qualified BF16 WMMA dot of the emulator's
/// `mma::wmma_f32_bf16` in logical K order, chained per K slice, folded sequentially ascending, then `f * s[m]`.
/// Returns the `y[n][m]` words, `rows * m` of them. NaN payload selection of the host `+`/`*` is not the
/// interpreter's: compare NaN results by NaN-ness only.
pub fn host_model_set(
    w: &[u16],
    m: usize,
    k: usize,
    x: &[f32],
    rows: usize,
    kw: usize,
    sc: &[f32],
) -> Vec<u32> {
    assert_eq!(w.len(), m * k);
    assert_eq!(x.len(), rows * k);
    assert_eq!(k % (16 * kw), 0);
    let steps = k / 16 / kw;
    let mut out = vec![0u32; rows * m];
    for n in 0..rows {
        let b: Vec<u16> = x[n * k..(n + 1) * k]
            .iter()
            .map(|&f| bf16_bits(f))
            .collect();
        for r in 0..m {
            let a = &w[r * k..(r + 1) * k];
            let mut fold = 0f32;
            for ki in 0..kw {
                let mut acc = 0u32;
                for t in 0..steps {
                    let k0 = (ki * steps + t) * 16;
                    acc = peacemaker_ir::emu::mma::wmma_f32_bf16(
                        peacemaker_ir::Arch::Gfx1201,
                        &a[k0..k0 + 16],
                        &b[k0..k0 + 16],
                        acc,
                    );
                }
                let p = f32::from_bits(acc);
                fold = if ki == 0 { p } else { fold + p };
            }
            out[n * m + r] = (fold * sc[r]).to_bits();
        }
    }
    out
}

// ------------------------------------------------------------------------------------------------ emitter

fn ref_spec(symbol: &str, variant: &str, kernargs: KernargLayout) -> KernelSpec {
    KernelSpec {
        kernel_id: "hc8_wmma_reference".into(),
        variant: variant.into(),
        arch: Arch::Gfx1201,
        symbol: symbol.into(),
        kernargs,
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: true,
        workgroup_size: REFERENCE_BLOCK,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    }
}

const VGPRS: u8 = 64;
const SGPR_END: u8 = 56;

fn ref_plan() -> Result<RegPlan, String> {
    let mut p = RegPlan::new(u16::from(VGPRS), 104)?;
    p.s::<2>("kernarg_ptr", 0, Live::Whole)?;
    for base in (8u8..SGPR_END).step_by(8) {
        p.add_range(&format!("s_blk{base}"), Kind::S, base, 8, Live::Whole)?;
    }
    for base in (0u8..VGPRS).step_by(8) {
        p.add_range(&format!("v_blk{base}"), Kind::V, base, 8, Live::Whole)?;
    }
    Ok(p)
}

// SGPRs.
const S_Q: u8 = 8; // s8:9 q
const S_S: u8 = 10; // s10:11 scale
const S_X: u8 = 12; // s12:13 x
const S_Y: u8 = 14; // s14:15 y
const S_M: u8 = 16;
const S_ROWS: u8 = 18;
const S_ACT: u8 = 19;
const S_MT: u8 = 20;
const S_CT: u8 = 21;
const S_AB: u8 = 22; // mt * 16 * K * 2
const S_XB: u8 = 23; // ct * 16 * K * 4
const S_CT16: u8 = 24;
const S_MTB: u8 = 25; // mt * 16 * 4
const S_M4: u8 = 26; // M * 4
const S_EM: u8 = 32; // s32:33 live-column mask
const MASK0: u8 = 48;
const MASK1: u8 = 50;

// VGPRs.
const V_TID: u8 = 0;
const V_AOFF: u8 = 1;
const V_XOFF: u8 = 2;
const V_R: u8 = 3;
const V_H: u8 = 4;
const V_N: u8 = 5;
const V_YOFF: u8 = 6;
const V_T: u8 = 7;
const F: u8 = 8; // folded accumulator v8..15
const C: u8 = 16; // current slice chain v16..23
const RAW: u8 = 32; // raw x f32 v32..39
const PK0: u8 = 48; // pack temps v48,v49
const T0: u8 = 50; // act temps v50..55
const SV: u8 = 56; // s[m] v56..63
const SETS: [(u8, u8); 2] = [(24, 28), (40, 44)]; // (A frag v..+3, B frag v..+3)

fn valu(b: &mut Builder, text: String, defs: &[u8], uses: &[u8]) -> R {
    op(
        b,
        text,
        &defs.iter().map(|&n| v(n)).collect::<Vec<_>>(),
        &uses.iter().map(|&n| v(n)).collect::<Vec<_>>(),
    )
}

fn exec_all(b: &mut Builder) -> R {
    op(b, "s_mov_b32 exec_lo, -1", &[], &[])
}

fn exec_live(b: &mut Builder) -> R {
    op(b, format!("s_mov_b32 exec_lo, s{S_EM}"), &[], &[s(S_EM)])
}

fn off(bytes: u32) -> String {
    if bytes == 0 {
        String::new()
    } else {
        format!(" offset:{bytes}")
    }
}

/// BF16 round trip of the f32 in `v{val}` on three scratch VGPRs (finite: `(u + ((u >> 16) & 1) + 0x7fff) &
/// 0xffff0000`; exponent field all ones: `low16(u) == 0 ? u : u | 0x10000`, then the mask). Same transcription as the
/// `lane_rows_bf16_gfx1201` reference.
fn round_trip(b: &mut Builder, val: u8, t: [u8; 3]) -> R {
    let [t0, t1, t2] = t;
    valu(b, format!("v_bfe_u32 v{t0}, v{val}, 16, 1"), &[t0], &[val])?;
    valu(
        b,
        format!("v_add_nc_u32_e32 v{t0}, v{val}, v{t0}"),
        &[t0],
        &[val, t0],
    )?;
    valu(
        b,
        format!("v_add_nc_u32_e32 v{t0}, 0x7fff, v{t0}"),
        &[t0],
        &[t0],
    )?;
    valu(
        b,
        format!("v_and_b32_e32 v{t1}, 0xffff, v{val}"),
        &[t1],
        &[val],
    )?;
    valu(
        b,
        format!("v_or_b32_e32 v{t2}, 0x10000, v{val}"),
        &[t2],
        &[val],
    )?;
    op(
        b,
        format!("v_cmp_eq_u32_e64 s{MASK0}, 0, v{t1}"),
        &[sr(MASK0, 2)],
        &[v(t1)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{t1}, v{t2}, v{val}, s{MASK0}"),
        &[v(t1)],
        &[v(t2), v(val), sr(MASK0, 2)],
    )?;
    valu(
        b,
        format!("v_and_b32_e32 v{t2}, 0x7f800000, v{val}"),
        &[t2],
        &[val],
    )?;
    op(
        b,
        format!("v_cmp_eq_u32_e64 s{MASK1}, 0x7f800000, v{t2}"),
        &[sr(MASK1, 2)],
        &[v(t2)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{t0}, v{t0}, v{t1}, s{MASK1}"),
        &[v(t0)],
        &[v(t0), v(t1), sr(MASK1, 2)],
    )?;
    valu(
        b,
        format!("v_and_b32_e32 v{val}, 0xffff0000, v{t0}"),
        &[val],
        &[t0],
    )
}

fn alu_wait(b: &mut Builder, which: &str) -> R {
    b.push(Instruction::new(
        format!("s_wait_alu {which}"),
        vec![],
        vec![],
    ))
}

/// The hc_act tail on `v{out}` (already `fold * s[m]`): the pinned hipcc VALU transcription. `act_scale` is `s{S_ACT}`.
fn act_tail(b: &mut Builder, out: u8) -> R {
    let [t0, t1, t2, t3, t4, t5] = [T0, T0 + 1, T0 + 2, T0 + 3, T0 + 4, T0 + 5];
    round_trip(b, out, [t0, t1, t2])?;
    op(
        b,
        format!("v_mul_f32_e32 v{out}, s{S_ACT}, v{out}"),
        &[v(out)],
        &[v(out), s(S_ACT)],
    )?;
    round_trip(b, out, [t0, t1, t2])?;
    valu(
        b,
        format!("v_mul_f32_e32 v{t0}, 0xbfb8aa3b, v{out}"),
        &[t0],
        &[out],
    )?;
    valu(
        b,
        format!("v_xor_b32_e32 v{t1}, 0x80000000, v{t0}"),
        &[t1],
        &[t0],
    )?;
    valu(b, format!("v_rndne_f32_e32 v{t2}, v{t0}"), &[t2], &[t0])?;
    valu(
        b,
        format!("v_fma_f32 v{t1}, v{out}, 0xbfb8aa3b, v{t1}"),
        &[t1],
        &[out, t1],
    )?;
    valu(
        b,
        format!("v_sub_f32_e32 v{t0}, v{t0}, v{t2}"),
        &[t0],
        &[t0, t2],
    )?;
    valu(
        b,
        format!("v_fma_f32 v{t1}, v{out}, 0xb2a5705f, v{t1}"),
        &[t1],
        &[out, t1],
    )?;
    valu(
        b,
        format!("v_add_f32_e32 v{t0}, v{t0}, v{t1}"),
        &[t0],
        &[t0, t1],
    )?;
    valu(b, format!("v_cvt_i32_f32_e32 v{t3}, v{t2}"), &[t3], &[t2])?;
    valu(b, format!("v_exp_f32_e32 v{t0}, v{t0}"), &[t0], &[t0])?;
    valu(
        b,
        format!("v_ldexp_f32 v{t0}, v{t0}, v{t3}"),
        &[t0],
        &[t0, t3],
    )?;
    // keep e when !(x > 0x42ce8ed0); +Inf when x < 0xc2b17218 (both unordered-true).
    op(
        b,
        format!("v_cmp_nlt_f32_e64 s{MASK0}, 0x42ce8ed0, v{out}"),
        &[sr(MASK0, 2)],
        &[v(out)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{t0}, 0, v{t0}, s{MASK0}"),
        &[v(t0)],
        &[v(t0), sr(MASK0, 2)],
    )?;
    op(
        b,
        format!("v_cmp_ngt_f32_e64 s{MASK1}, 0xc2b17218, v{out}"),
        &[sr(MASK1, 2)],
        &[v(out)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{t0}, 0x7f800000, v{t0}, s{MASK1}"),
        &[v(t0)],
        &[v(t0), sr(MASK1, 2)],
    )?;
    // y = x / (1 + exp(-x)).
    let [w0, w1, w2, w3, w4, w5] = [t0, t1, t2, t3, t4, t5];
    valu(b, format!("v_add_f32_e32 v{w0}, 1.0, v{w0}"), &[w0], &[w0])?;
    valu(
        b,
        format!("v_div_scale_f32 v{w1}, null, v{w0}, v{w0}, v{out}"),
        &[w1],
        &[w0, out],
    )?;
    valu(b, format!("v_rcp_f32_e32 v{w2}, v{w1}"), &[w2], &[w1])?;
    valu(
        b,
        format!("v_xor_b32_e32 v{w1}, 0x80000000, v{w1}"),
        &[w1],
        &[w1],
    )?;
    valu(
        b,
        format!("v_fma_f32 v{w3}, v{w1}, v{w2}, 1.0"),
        &[w3],
        &[w1, w2],
    )?;
    valu(
        b,
        format!("v_fmac_f32_e32 v{w2}, v{w3}, v{w2}"),
        &[w2],
        &[w2, w3],
    )?;
    valu(
        b,
        format!("v_div_scale_f32 v{w3}, vcc_lo, v{out}, v{w0}, v{out}"),
        &[w3],
        &[out, w0],
    )?;
    valu(
        b,
        format!("v_mul_f32_e32 v{w4}, v{w3}, v{w2}"),
        &[w4],
        &[w3, w2],
    )?;
    valu(
        b,
        format!("v_fma_f32 v{w5}, v{w1}, v{w4}, v{w3}"),
        &[w5],
        &[w1, w4, w3],
    )?;
    valu(
        b,
        format!("v_fmac_f32_e32 v{w4}, v{w5}, v{w2}"),
        &[w4],
        &[w4, w5, w2],
    )?;
    valu(
        b,
        format!("v_fmac_f32_e32 v{w3}, v{w1}, v{w4}"),
        &[w3],
        &[w3, w1, w4],
    )?;
    alu_wait(b, "depctr_va_vcc(0)")?;
    valu(
        b,
        format!("v_div_fmas_f32 v{w1}, v{w3}, v{w2}, v{w4}"),
        &[w1],
        &[w3, w2, w4],
    )?;
    valu(
        b,
        format!("v_div_fixup_f32 v{w0}, v{w1}, v{w0}, v{out}"),
        &[w0],
        &[w1, w0, out],
    )?;
    valu(b, format!("v_mov_b32_e32 v{out}, v{w0}"), &[out], &[w0])?;
    round_trip(b, out, [t0, t1, t2])
}

/// One reference entry: `K` compile-time, `geo.kw` and `geo.act` select the numerics (`mw`, `nt`, `fmt` do not
/// change a single output bit). See the module doc for the ABI and launch.
pub fn emit_reference(geo: Geo, k: u32) -> Result<Emitted, String> {
    let kw = geo.kw;
    if kw == 0 || k == 0 || k % (32 * kw) != 0 {
        return Err(format!(
            "reference: K={k} must be a multiple of 32*KW={}",
            32 * kw
        ));
    }
    let steps = k / 16 / kw; // K16 steps per slice
    let kernargs = KernargLayout::new(48)
        .pointer_access("q", 0, Access::ReadOnly)
        .pointer_access("s", 8, Access::ReadOnly)
        .pointer_access("x", 16, Access::ReadOnly)
        .pointer_access("y", 24, Access::WriteOnly)
        .hidden("M", 32, 4, "by_value")
        .hidden("K", 36, 4, "by_value")
        .hidden("rows", 40, 4, "by_value")
        .hidden("act_scale", 44, 4, "by_value");
    let mut bld = Builder::new(
        ref_spec(
            &reference_symbol(geo, k),
            &format!("kw{kw}_k{k}_{}", if geo.act { "act" } else { "set" }),
            kernargs,
        ),
        ref_plan()?,
    );
    bld.enable_delay_alu();
    let b = &mut bld;

    // Kernarg: q,s,x,y -> s8..s15; M,K,rows,act_scale -> s16..s19.
    smem(b, S_Q, 8, 0, 0)?;
    smem(b, S_M, 4, 0, 32)?;
    b.wait_all()?;
    sop(b, format!("s_mov_b32 s{S_MT}, ttmp9"), &[S_MT], &[])?;
    sop(b, format!("s_and_b32 s{S_CT}, ttmp7, 0xffff"), &[S_CT], &[])?;
    sop(
        b,
        format!("s_mul_i32 s{S_AB}, s{S_MT}, {}", lit(32 * k)),
        &[S_AB],
        &[S_MT],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{S_XB}, s{S_CT}, {}", lit(64 * k)),
        &[S_XB],
        &[S_CT],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{S_CT16}, s{S_CT}, 4"),
        &[S_CT16],
        &[S_CT],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{S_MTB}, s{S_MT}, 6"),
        &[S_MTB],
        &[S_MT],
    )?;
    sop(b, format!("s_lshl_b32 s{S_M4}, s{S_M}, 2"), &[S_M4], &[S_M])?;
    // High words of the wave32 masks are never written.
    sop(b, format!("s_mov_b32 s{}, 0", S_EM + 1), &[S_EM + 1], &[])?;
    sop(b, format!("s_mov_b32 s{}, 0", MASK0 + 1), &[MASK0 + 1], &[])?;
    sop(b, format!("s_mov_b32 s{}, 0", MASK1 + 1), &[MASK1 + 1], &[])?;

    // Lane geometry: r = lane & 15 (A row / B column), h = lane >> 4.
    valu(
        b,
        format!("v_and_b32_e32 v{V_R}, 15, v{V_TID}"),
        &[V_R],
        &[V_TID],
    )?;
    valu(
        b,
        format!("v_lshrrev_b32_e32 v{V_H}, 4, v{V_TID}"),
        &[V_H],
        &[V_TID],
    )?;
    // A byte offset of the lane: (mt*16 + r)*K*2 + h*8 (chunk of 4 bf16 at k = 4h).
    valu(
        b,
        format!("v_mul_u32_u24_e32 v{V_AOFF}, {}, v{V_R}", lit(2 * k)),
        &[V_AOFF],
        &[V_R],
    )?;
    valu(
        b,
        format!("v_lshlrev_b32_e32 v{V_T}, 3, v{V_H}"),
        &[V_T],
        &[V_H],
    )?;
    valu(
        b,
        format!("v_add_nc_u32_e32 v{V_AOFF}, v{V_AOFF}, v{V_T}"),
        &[V_AOFF],
        &[V_AOFF, V_T],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{V_AOFF}, s{S_AB}, v{V_AOFF}"),
        &[v(V_AOFF)],
        &[s(S_AB), v(V_AOFF)],
    )?;
    // X byte offset: (ct*16 + r)*K*4 + h*16 (4 f32 at k = 4h).
    valu(
        b,
        format!("v_mul_u32_u24_e32 v{V_XOFF}, {}, v{V_R}", lit(4 * k)),
        &[V_XOFF],
        &[V_R],
    )?;
    valu(
        b,
        format!("v_lshlrev_b32_e32 v{V_T}, 4, v{V_H}"),
        &[V_T],
        &[V_H],
    )?;
    valu(
        b,
        format!("v_add_nc_u32_e32 v{V_XOFF}, v{V_XOFF}, v{V_T}"),
        &[V_XOFF],
        &[V_XOFF, V_T],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{V_XOFF}, s{S_XB}, v{V_XOFF}"),
        &[v(V_XOFF)],
        &[s(S_XB), v(V_XOFF)],
    )?;
    // Column n = ct*16 + r and the live-column mask (rows > n).
    op(
        b,
        format!("v_add_nc_u32_e32 v{V_N}, s{S_CT16}, v{V_R}"),
        &[v(V_N)],
        &[s(S_CT16), v(V_R)],
    )?;
    op(
        b,
        format!("v_cmp_gt_u32_e64 s{S_EM}, s{S_ROWS}, v{V_N}"),
        &[sr(S_EM, 2)],
        &[s(S_ROWS), v(V_N)],
    )?;
    // Dead columns keep zero raw X (masked loads never write them), hence B = 0.
    for i in 0..8u8 {
        valu(b, format!("v_mov_b32_e32 v{}, 0", RAW + i), &[RAW + i], &[])?;
    }

    for ki in 0..kw {
        let bank = if ki == 0 { F } else { C };
        for t in 0..steps {
            let k0 = (ki * steps + t) * 16;
            let (a, bb) = SETS[(t & 1) as usize];
            // A fragment: bf16 k = k0+4h..+3 (VGPR 0,1) and k0+8+4h..+3 (VGPR 2,3).
            for (i, extra) in [0u32, 16].into_iter().enumerate() {
                let d = a + 2 * i as u8;
                mem_op(
                    b,
                    format!(
                        "global_load_b64 v[{d}:{}], v{V_AOFF}, s[{S_Q}:{}]{}",
                        d + 1,
                        S_Q + 1,
                        off(k0 * 2 + extra)
                    ),
                    &[vr(d, 2)],
                    &[v(V_AOFF), sr(S_Q, 2)],
                    MemoryClass::VmemLoad,
                )?;
            }
            // B fragment source: x[n][k0+4h..+3] and x[n][k0+8+4h..+3], live columns only.
            exec_live(b)?;
            for (i, extra) in [0u32, 32].into_iter().enumerate() {
                let d = RAW + 4 * i as u8;
                mem_op(
                    b,
                    format!(
                        "global_load_b128 v[{d}:{}], v{V_XOFF}, s[{S_X}:{}]{}",
                        d + 3,
                        S_X + 1,
                        off(k0 * 4 + extra)
                    ),
                    &[vr(d, 4)],
                    &[v(V_XOFF), sr(S_X, 2)],
                    MemoryClass::VmemLoad,
                )?;
            }
            exec_all(b)?;
            b.wait(Counter::Load, 0)?;
            // Pack the high halves of consecutive f32 into BF16 pairs (lo = even k).
            for p in 0..4u8 {
                let (x0, x1) = (RAW + 2 * p, RAW + 2 * p + 1);
                valu(
                    b,
                    format!("v_lshrrev_b32_e32 v{PK0}, 16, v{x0}"),
                    &[PK0],
                    &[x0],
                )?;
                valu(
                    b,
                    format!("v_and_b32_e32 v{}, 0xffff0000, v{x1}", PK0 + 1),
                    &[PK0 + 1],
                    &[x1],
                )?;
                valu(
                    b,
                    format!("v_or_b32_e32 v{}, v{PK0}, v{}", bb + p, PK0 + 1),
                    &[bb + p],
                    &[PK0, PK0 + 1],
                )?;
            }
            let first = t == 0;
            let mut uses = vec![vr(a, 4), vr(bb, 4)];
            let cop = if first {
                "0".to_string()
            } else {
                uses.push(vr(bank, 8));
                format!("v[{bank}:{}]", bank + 7)
            };
            b.push(Instruction::new(
                format!(
                    "v_wmma_f32_16x16x16_bf16 v[{bank}:{}], v[{a}:{}], v[{bb}:{}], {cop}",
                    bank + 7,
                    a + 3,
                    bb + 3
                ),
                vec![vr(bank, 8)],
                uses,
            ))?;
        }
        if ki > 0 {
            for j in 0..8u8 {
                valu(
                    b,
                    format!("v_add_f32_e32 v{}, v{}, v{}", F + j, F + j, C + j),
                    &[F + j],
                    &[F + j, C + j],
                )?;
            }
        }
    }

    // Epilogue. D[row,col] -> lane (row>>3)*16 + col, VGPR row & 7: m = mt*16 + h*8 + j.
    valu(
        b,
        format!("v_lshlrev_b32_e32 v{V_T}, 5, v{V_H}"),
        &[V_T],
        &[V_H],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{V_T}, s{S_MTB}, v{V_T}"),
        &[v(V_T)],
        &[s(S_MTB), v(V_T)],
    )?;
    for (i, extra) in [0u32, 16].into_iter().enumerate() {
        let d = SV + 4 * i as u8;
        mem_op(
            b,
            format!(
                "global_load_b128 v[{d}:{}], v{V_T}, s[{S_S}:{}]{}",
                d + 3,
                S_S + 1,
                off(extra)
            ),
            &[vr(d, 4)],
            &[v(V_T), sr(S_S, 2)],
            MemoryClass::VmemLoad,
        )?;
    }
    b.wait(Counter::Load, 0)?;
    for j in 0..8u8 {
        valu(
            b,
            format!("v_mul_f32_e32 v{}, v{}, v{}", F + j, F + j, SV + j),
            &[F + j],
            &[F + j, SV + j],
        )?;
    }
    if geo.act {
        for j in 0..8u8 {
            act_tail(b, F + j)?;
        }
    }
    op(
        b,
        format!("v_mul_u32_u24_e32 v{V_YOFF}, s{S_M4}, v{V_N}"),
        &[v(V_YOFF)],
        &[s(S_M4), v(V_N)],
    )?;
    valu(
        b,
        format!("v_add_nc_u32_e32 v{V_YOFF}, v{V_YOFF}, v{V_T}"),
        &[V_YOFF],
        &[V_YOFF, V_T],
    )?;
    exec_live(b)?;
    for j in 0..8u8 {
        mem_op(
            b,
            format!(
                "global_store_b32 v{V_YOFF}, v{}, s[{S_Y}:{}]{}",
                F + j,
                S_Y + 1,
                off(4 * u32::from(j))
            ),
            &[],
            &[v(V_YOFF), v(F + j), sr(S_Y, 2)],
            MemoryClass::VmemStore,
        )?;
    }
    b.wait_all()?;
    op(b, "s_endpgm", &[], &[])?;
    bld.finish()
}
