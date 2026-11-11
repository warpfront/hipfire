// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The `mq6_i8_gemm` generator as public building blocks.
//!
//! [`super::emit`], [`super::kloop::emit`] and [`super::prologue::emit`] are written over exactly
//! these functions, so a composition that mirrors them (the `.rip` twin
//! `kernels/mq6_i8_gemm.rip`) re-emits their bytes, and a composition that differs (a wait moved,
//! the split barrier signalled elsewhere, a decode placed between other WMMAs, another VOPD
//! pairing) is checked by the same builder and, for the LDS rings, by the same checked driver as
//! the production kernel.
//!
//! Unlike `iu4_gemm::blocks` there is no ring-owning block here. The split is finer:
//!
//! - **LDS ring operations stay with the caller.** The blocks below that touch LDS only *return*
//!   [`Instruction`]s ([`store_w_insns`], [`store_a_insns`], [`meta_store_insns`],
//!   [`step_load_insns`], [`fold_token_load_insn`], [`fold_sz_load_insn`]); the caller issues them
//!   through its typed or driver rings (`ds_store`, `ds_load_cur`), so every ring transition, wait
//!   and split-barrier position is the caller's statement, checked by the typed core or the
//!   runtime driver.
//! - **Everything else emits straight into the builder** (fetches, decode, WMMA steps, the fold,
//!   the seed fill, the prologue's scalar and lane set-up, the epilogue).
//!
//! The blocks follow the spec's [`Tile`](super::Tile): the staging rounds (`a_rounds` / `w_rounds`
//! instructions of [`store_a_insns`], one decode of [`decode_insns`] and one weight store per
//! weight round), the LDS layout behind every offset, the raster shifts of the prologue and the
//! epilogue's coordinates. A composition reads `Tile::w_rounds` (the `.rip` twin: `g.w_rounds()`)
//! to queue the second decode / store only on the tiles that have a second weight round.
//!
//! The kernel order, as `emit` composes it:
//!
//! ```text
//! lds = declare four two-buffer rings (A, W, DS, SZ) over the eight slots of `Gen::slot`
//! end = wg.exit(END);  s_setprio 1 (banded raster)
//! front; exit_if(outside 0); exit_if(outside 1); scalars; lanes
//! fetch0; zero_acc; stage block 0 (decode, store, fetch, meta); signal(prime ..) ; fetch ; seed_fill; wait_arrived
//! K-loop: label K_BEGIN; skip if no full trip { loop_carried { block m0; block m1; trip_end } };
//!         peeled blocks t0, t1   (block = phase A, B1 split barrier, phase B, fold, B2 split barrier)
//! label EPI; epilogue; wg.end(end)
//! ```
use super::{fold, prologue, publish};
use crate::kernels::common::{op, s, sr, vr};
use crate::{insn::Instruction, reg::Live, Builder, V};

pub use super::{Epi, Gen, Raster, Spec, END, ENTRY, EPI, K_BEGIN, K_LOOP, K_LOOP_END};

/// `.Lmq6i8_<tag>`: the label a block named `tag` places.
pub fn label(tag: &str) -> String {
    super::kloop::label(tag)
}

/// The label of the peeled group's block `h` (`.Lmq6i8_t<h>`).
pub fn tail_label(h: usize) -> String {
    super::kloop::tail_label(h)
}

fn plane(what: &str, h: usize) -> Result<(), String> {
    if h > 1 {
        return Err(format!("{what}: plane {h} is not 0 or 1"));
    }
    Ok(())
}

/// The compare of the trip counter (`K/256 - 1`, from the `K` kernel argument only) with zero:
/// `cmp` is `s_cmp_eq_u32` (no full trip) or `s_cmp_lg_u32` (another trip), or another scalar
/// compare mnemonic of the 32-bit family (`s_cmp_{eq,lg,lt,le,gt,ge}_{u32,i32}`); anything else is
/// refused, so the text cannot smuggle in another instruction. The caller consumes it with
/// `scmp_wg_uniform`.
pub fn trips_cmp_insn(g: &Gen, cmp: &str) -> Result<Instruction, String> {
    let ok = ["eq", "lg", "lt", "le", "gt", "ge"].iter().any(|c| {
        ["u32", "i32"]
            .iter()
            .any(|t| cmp == format!("s_cmp_{c}_{t}"))
    });
    if !ok {
        return Err(format!(
            "trips_cmp: `{cmp}` is not a 32-bit scalar compare (s_cmp_eq_u32, s_cmp_lg_u32, ...)"
        ));
    }
    Ok(Instruction::new(
        format!("{cmp} s{}, 0", g.trips),
        vec![],
        vec![s(g.trips)],
    ))
}

/// `s_add_nc_u64` of the K-group stride onto activation descriptor `which` (`0` even block, `1`
/// odd block).
pub fn step_descriptor(b: &mut Builder, g: &Gen, which: usize) -> Result<(), String> {
    plane("step_descriptor", which)?;
    let srd = g.srd_a[which];
    op(
        b,
        format!(
            "s_add_nc_u64 s[{0}:{1}], s[{0}:{1}], s[{2}:{3}]",
            srd,
            srd + 1,
            g.step2,
            g.step2 + 1
        ),
        &[sr(srd, 2)],
        &[sr(srd, 2), sr(g.step2, 2)],
    )
}

/// The K-loop's trip end: the odd-block descriptor and the group offsets move on after every read
/// of them in the trip, and the trip counter drops by one.
pub fn trip_end(b: &mut Builder, g: &Gen) -> Result<(), String> {
    step_descriptor(b, g, 1)?;
    for r in [g.goff, g.goff1] {
        op(
            b,
            format!("s_add_co_i32 s{0}, s{0}, 0xc8", r),
            &[s(r)],
            &[s(r)],
        )?;
    }
    op(
        b,
        format!("s_add_co_i32 s{0}, s{0}, -1", g.trips),
        &[s(g.trips)],
        &[s(g.trips)],
    )
}

fn ring3_only(what: &str, g: &Gen) -> Result<(), String> {
    if g.depth() != 3 {
        return Err(format!("{what}: only a ring-depth-3 generator has it"));
    }
    Ok(())
}

/// Depth-3 group end: the group offsets move on one K group after every read of them in the group
/// (the activation descriptors step at the start of their block instead, see
/// [`step_descriptor`]).
pub fn group_end(b: &mut Builder, g: &Gen) -> Result<(), String> {
    ring3_only("group_end", g)?;
    for r in [g.goff, g.goff1] {
        op(
            b,
            format!("s_add_co_i32 s{0}, s{0}, 0xc8", r),
            &[s(r)],
            &[s(r)],
        )?;
    }
    Ok(())
}

/// Depth-3 trip counter: drops by one after the three groups of a loop trip.
pub fn trips_dec(b: &mut Builder, g: &Gen) -> Result<(), String> {
    ring3_only("trips_dec", g)?;
    op(
        b,
        format!("s_add_co_i32 s{0}, s{0}, -1", g.trips),
        &[s(g.trips)],
        &[s(g.trips)],
    )
}

/// Depth-3 K split, from the kernel argument only: with `n = K/256 - 1` full groups before the
/// peeled one, `q = n / 3` loop trips replace the trip counter (`g.trips`) and `r = n % 3` leading
/// groups go to scalar `tmp` (the arm selector, [`ring3_cmp_insn`]). The slot of slab `s` is
/// `(s - r) mod 3`, so the loop body (three groups) and the peeled group start at slot phase 0.
pub fn ring3_split(b: &mut Builder, g: &Gen) -> Result<(), String> {
    ring3_only("ring3_split", g)?;
    let (r, q, t) = (g.tmp, g.tmp + 1, g.tmp + 2);
    // q = floor(n * 0xAAAAAAAB / 2^33) is exact for every 32-bit n.
    op(
        b,
        format!("s_mul_hi_u32 s{q}, s{}, 0xaaaaaaab", g.trips),
        &[s(q)],
        &[s(g.trips)],
    )?;
    op(b, format!("s_lshr_b32 s{q}, s{q}, 1"), &[s(q)], &[s(q)])?;
    op(b, format!("s_mul_i32 s{t}, s{q}, 3"), &[s(t)], &[s(q)])?;
    op(
        b,
        format!("s_sub_co_i32 s{r}, s{}, s{t}", g.trips),
        &[s(r)],
        &[s(g.trips), s(t)],
    )?;
    op(
        b,
        format!("s_mov_b32 s{}, s{q}", g.trips),
        &[s(g.trips)],
        &[s(q)],
    )
}

/// The compare of the arm selector `r` ([`ring3_split`]) with an immediate: `cmp` as in
/// [`trips_cmp_insn`]. The caller consumes it with `scmp_wg_uniform`.
pub fn ring3_cmp_insn(g: &Gen, cmp: &str, imm: u32) -> Result<Instruction, String> {
    ring3_only("ring3_cmp", g)?;
    let ok = ["eq", "lg", "lt", "le", "gt", "ge"].iter().any(|c| {
        ["u32", "i32"]
            .iter()
            .any(|t| cmp == format!("s_cmp_{c}_{t}"))
    });
    if !ok || imm > 64 {
        return Err(format!(
            "ring3_cmp: `{cmp} {imm}` is not a 32-bit scalar compare with an inline constant"
        ));
    }
    Ok(Instruction::new(
        format!("{cmp} s{}, {imm}", g.tmp),
        vec![],
        vec![s(g.tmp)],
    ))
}

fn between(a: &str, b: &str) -> Live {
    Live::Between(a.into(), b.into())
}

/// The register lifetimes of one block: the fragment sets live from the block's label `entry` to
/// its fold label `fold`; the fold registers (`sc`, `z'`, products) from `fold` to the label `end`
/// that closes the block (the next block's, or the loop end).
pub fn declare_block_regs(
    b: &mut Builder,
    g: &Gen,
    entry: &str,
    fold: &str,
    end: &str,
) -> Result<(), String> {
    for (set, f) in g.f.into_iter().enumerate() {
        let names: [&'static str; 3] = if set == 0 {
            ["f0_w", "f0_a0", "f0_a1"]
        } else {
            ["f1_w", "f1_a0", "f1_a1"]
        };
        for (part, name) in names.into_iter().enumerate() {
            b.regs
                .v::<4>(name, f + 4 * part as u8, between(entry, fold))?;
        }
    }
    // Fold registers: row group 0's sc / z' octets and the products, over the fragment sets.
    b.regs.v::<8>("fold_sc", fold::SC, between(fold, end))?;
    b.regs.v::<8>("fold_zp", fold::ZP, between(fold, end))?;
    b.regs.v::<8>("fold_t", fold::T, between(fold, end))?;
    Ok(())
}

/// Raw activation chunks of K32 half `sb` of slab `slab` of the block whose descriptor is `srd`:
/// one b64 per staging round into `a_pf[r]`.
pub fn fetch_a(b: &mut Builder, g: &Gen, slab: u32, sb: u32, srd: u8) -> Result<(), String> {
    if slab > 1 || sb > 1 {
        return Err(format!(
            "fetch_a: slab {slab} / half {sb} out of range (0..2)"
        ));
    }
    publish::fetch_a(b, g, slab, sb, srd)
}

/// Raw weight dwords (both rounds) at group-relative byte offset `base` (`8 + 96 h + 48 slab`,
/// plus the group stride when fetching across a group boundary).
pub fn fetch_w(b: &mut Builder, g: &Gen, base: u32) -> Result<(), String> {
    publish::fetch_w(b, g, base)
}

/// Next block's token word (`d` on even lanes, `s` on odd lanes) and packed row header. `h`: the
/// parity of the CURRENT block; `hdr`: header byte offset inside the group of the next block.
pub fn fetch_meta(b: &mut Builder, g: &Gen, h: usize, hdr: u32) -> Result<(), String> {
    plane("fetch_meta", h)?;
    publish::fetch_meta(b, g, h, hdr)
}

/// Retire every outstanding memory operation (the builder's ledger-wide wait). The raw staging
/// registers are still sources of the previous half's LDS stores until they have read them.
pub fn wait_all(b: &mut Builder) -> Result<(), String> {
    b.wait_all()
}

/// The VALU instructions that decode weight round `r` (`0` or `1`): the six-bit codes of its raw
/// dwords spread into the two iu8 operands, left in the decode scratch (`e0` in the `o` pair, `e1`
/// in the `p` pair). The caller pushes them (the K-loop spreads them between WMMAs).
pub fn decode_insns(g: &Gen, r: usize) -> Result<Vec<Instruction>, String> {
    if r >= g.w_pf.len() {
        return Err(format!(
            "decode_round: round {r} out of range (0..{})",
            g.w_pf.len()
        ));
    }
    let mut insns = Vec::new();
    publish::decode_round(&mut insns, g, r)?;
    Ok(insns)
}

/// `scratch[0..8] = 0x4B400000`: the WMMA chain seed.
pub fn seed_fill(b: &mut Builder, g: &Gen) -> Result<(), String> {
    super::kloop::seed_fill(b, g)
}

/// The `i`-th (`0..8`, in [`fold::BUNDLE_ORDER`]) `v_wmma_i32_16x16x16_iu8` of K16 step `step`
/// (`0..4`) on fragment set `set`. With `first`, the accumulator seed is the decode scratch's
/// `0x4B400000` octet instead of the accumulator itself.
pub fn mma_insn(
    g: &Gen,
    step: u32,
    set: usize,
    first: bool,
    i: usize,
) -> Result<Instruction, String> {
    if step > 3 || set > 1 || i > 7 {
        return Err(format!(
            "mma: step {step} / set {set} / index {i} out of range (4 / 2 / 8)"
        ));
    }
    let f = g.f[set];
    let (sb, e) = (step >> 1, step & 1);
    let k = fold::BUNDLE_ORDER[i];
    let (nb, rg) = (k / 2, k % 2);
    let dst = V::<8>(g.cacc + 8 * k as u8);
    // Weight operand: data half `rg` sits in slot `rg ^ sb`; token operand: nb pair `nb >> 1`,
    // data half `nb & 1` in slot `(nb & 1) ^ e`.
    let wop = V::<2>(f + 2 * (rg as u8 ^ sb as u8));
    let xop = V::<2>(f + 4 + 4 * (nb as u8 >> 1) + 2 * ((nb as u8 & 1) ^ e as u8));
    Ok(if first {
        // Seed: the octet of `0x4B400000` words the previous block's tail (or the prologue) left in
        // the decode scratch.
        let seed = V::<8>(g.scratch);
        Instruction::new(
            format!(
                "v_wmma_i32_16x16x16_iu8 {}, {}, {}, {} neg_lo:[0,1,0]",
                dst.reg(),
                wop.reg(),
                xop.reg(),
                seed.reg()
            ),
            vec![dst.reg()],
            vec![wop.reg(), xop.reg(), seed.reg()],
        )
    } else {
        Instruction::new(
            format!(
                "v_wmma_i32_16x16x16_iu8 {}, {}, {}, {} neg_lo:[0,1,0]",
                dst.reg(),
                wop.reg(),
                xop.reg(),
                dst.reg()
            ),
            vec![dst.reg()],
            vec![wop.reg(), xop.reg(), dst.reg()],
        )
    })
}

/// Issue [`mma_insn`].
pub fn mma(
    b: &mut Builder,
    g: &Gen,
    step: u32,
    set: usize,
    first: bool,
    i: usize,
) -> Result<(), String> {
    b.push(mma_insn(g, step, set, first, i)?)
}

/// The `ds_load_b128`s of K16 step `step` (`sb * 2 + e`, `0..4`) into fragment set `set` from LDS
/// slot `slot`: the weight row-group pair (from the `W` ring), then the two token-group pairs
/// (from the `A` ring).
pub fn step_load_insns(
    g: &Gen,
    slot: usize,
    step: u32,
    set: usize,
) -> Result<[Instruction; 3], String> {
    if slot >= g.depth() || step > 3 || set > 1 {
        return Err(format!(
            "step_loads: slot {slot} / step {step} / set {set} out of range ({} / 4 / 2)",
            g.depth()
        ));
    }
    let f = g.f[set];
    let l = g.spec.tile.layout();
    let (wb, ab) = (l.w_slot(slot)?, l.a_slot(slot)?);
    let o = 512 * step;
    let w = publish::ds_load(
        format!(
            "ds_load_b128 v[{}:{}], v{}{}",
            f,
            f + 3,
            g.fr_w,
            super::ds_offset(wb + o)
        ),
        vr(f, 4),
        g.fr_w,
    );
    let a = |half: u32| {
        let dst = f + 4 + 4 * half as u8;
        publish::ds_load(
            format!(
                "ds_load_b128 v[{}:{}], v{}{}",
                dst,
                dst + 3,
                g.fr_a,
                super::ds_offset(ab + o + 2048 * half)
            ),
            vr(dst, 4),
            g.fr_a,
        )
    };
    Ok([w, a(0), a(1)])
}

/// The two LDS stores of weight round `r` (decoded by [`decode_insns`]) into slot `slot` of the
/// `W` ring.
pub fn store_w_insns(g: &Gen, slot: usize, r: usize) -> Result<[Instruction; 2], String> {
    publish::store_w_insns(g, slot, r)
}

/// The LDS stores of the raw activation chunks (from [`fetch_a`]) of K32 half `sb` into slot
/// `slot` of the `A` ring, one per activation round (`Tile::a_rounds`).
pub fn store_a_insns(g: &Gen, slot: usize, sb: u32) -> Result<Vec<Instruction>, String> {
    publish::store_a_insns(g, slot, sb)
}

/// The VALU part of the metadata publication: the next block's token word and row word, in the
/// decode scratch. Issued before the `DS` / `SZ` rings begin their write.
pub fn meta_valu(b: &mut Builder, g: &Gen) -> Result<(), String> {
    publish::meta_insns(b, g)
}

/// The two LDS stores of [`meta_valu`]'s words into plane `plane` (the ring's next index): the
/// token word into the `DS` ring, the row word into the `SZ` ring.
pub fn meta_store_insns(g: &Gen, plane: usize) -> Result<[Instruction; 2], String> {
    publish::meta_store_insns(g, plane)
}

/// Fold address set-up for plane `h`: the token `(d, float(s))` address (`msn` as the address
/// temporary).
pub fn fold_addr(b: &mut Builder, g: &Gen, h: usize) -> Result<(), String> {
    use crate::kernels::common::{lit, v};
    plane("fold_addr", h)?;
    op(
        b,
        format!(
            "v_add_nc_u32_e32 v{}, {}, v{}",
            g.msn,
            lit(g.spec.tile.layout().ds[h]),
            g.d_addr
        ),
        &[v(g.msn)],
        &[v(g.d_addr)],
    )
}

/// The `ds_load_2addr_b32` of token block `nb`'s `(d, float(s))` pair (from the `DS` ring, after
/// [`fold_addr`]).
pub fn fold_token_load_insn(g: &Gen, nb: usize) -> Result<Instruction, String> {
    if nb > 3 {
        return Err(format!("fold token load: block {nb} out of range (0..4)"));
    }
    let nb = nb as u32;
    let dst = g.tok + 2 * nb as u8;
    Ok(publish::ds_load(
        format!(
            "ds_load_2addr_b32 v[{}:{}], v{}{}",
            dst,
            dst + 1,
            g.msn,
            super::ds_offsets(16 * nb, 128 + 16 * nb)
        ),
        vr(dst, 2),
        g.msn,
    ))
}

/// A `ds_load_b128` of the row planes `sc` / `z'` of plane `h` (from the `SZ` ring) into `dst..+4`
/// at byte `rel` of the plane: `sc` octets at `0`, `16` (row group 0) and `64`, `80` (group 1);
/// `z'` at `512` plus the same.
pub fn fold_sz_load_insn(g: &Gen, h: usize, dst: u8, rel: u32) -> Result<Instruction, String> {
    plane("fold_sz_load", h)?;
    Ok(publish::ds_load(
        format!(
            "ds_load_b128 v[{}:{}], v{}{}",
            dst,
            dst + 3,
            g.sc_addr,
            super::ds_offset(g.spec.tile.layout().sz[h] + rel)
        ),
        vr(dst, 4),
        g.sc_addr,
    ))
}

/// `msn[nb] = -(12582912.0 + 32 * s[nb])` into `g.msn` (exact).
pub fn emit_msn(b: &mut Builder, g: &Gen, nb: usize) -> Result<(), String> {
    fold::emit_msn(b, g, nb)
}

/// Stage `stage` (`0..4`: the eight `mul`/`add` packets, then three groups of four `fmac` / `mul` /
/// `fmac` pairs) of fragment `k`'s fold (`k = nb * 2 + rg`, `0..8`), with the `sc` / `z'` octets
/// at `sc` / `zp`.
pub fn fold_stage(
    b: &mut Builder,
    g: &Gen,
    k: usize,
    stage: usize,
    sc: u8,
    zp: u8,
) -> Result<(), String> {
    fold::fold_stage(b, g, k, stage, sc, zp)
}

/// The number of stages of a fragment's fold ([`fold_stage`]).
pub const FOLD_STAGES: usize = fold::STAGES;

/// All four stages of fragment `k`'s fold.
pub fn fold_fragment(b: &mut Builder, g: &Gen, k: usize, sc: u8, zp: u8) -> Result<(), String> {
    fold::fragment(b, g, k, sc, zp)
}

/// Kernel arguments (`s_load_b256` x2) and the CTA raster: the tile origin (`rs`, `bs`).
pub fn prologue_front(b: &mut Builder, g: &Gen) -> Result<(), String> {
    prologue::front(b, g)
}

/// The scalar compare of the tile-outside exits: `0` tile rows start at/past `M`, `1` tile tokens
/// start at/past `N`. The caller consumes it with `scmp_wg_uniform` and `exit_if`.
pub fn tile_outside_insn(g: &Gen, which: usize) -> Result<Instruction, String> {
    prologue::outside_insn(g, which)
}

/// Scalars (row stride, trips, group offsets, wave index, odd-lane mask) and buffer descriptors.
pub fn prologue_scalars(b: &mut Builder, g: &Gen) -> Result<(), String> {
    prologue::scalars(b, g)
}

/// Hoisted lane offsets and fold addresses.
pub fn prologue_lanes(b: &mut Builder, g: &Gen) -> Result<(), String> {
    prologue::lanes(b, g)
}

/// Block 0's raw fetches: slab 0 K32 half 0, the weight dwords, the token word and row header.
pub fn prologue_fetch(b: &mut Builder, g: &Gen) -> Result<(), String> {
    prologue::fetch0(b, g)
}

/// Zero the float accumulators.
pub fn zero_acc(b: &mut Builder, g: &Gen) -> Result<(), String> {
    prologue::zero_acc(b, g)
}

/// Everything after the [`EPI`] label (the caller places the label): the output descriptor, the
/// lane masks and the masked stores of the spec's epilogue.
pub fn epilogue(b: &mut Builder, g: &Gen) -> Result<(), String> {
    super::epilogue::body(b, g)
}
