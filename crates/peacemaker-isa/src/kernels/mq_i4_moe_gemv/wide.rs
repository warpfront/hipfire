// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Wide twins of the K1 lane-union decode GEMV: the eight entries
//! `mq_i4_moe_gemv_{mq3i4,mq4sym}_{r256_gate_up,r128_down}_c{16,32}` (design `WideTwinGemvSol` §7.1/§7.3).
//!
//! The 96-byte kernarg ABI (the narrow entries': 88 bytes plus `n_experts@88` and a zero `reserved@92`), the slot/X/Y addressing and every arithmetic step are the narrow entries'
//! ([`super`]); only the schedule differs, so every `(slot, output row)` value stays bitwise the narrow
//! (C1) entry's:
//!
//!  * Launch: grid `[ceil(M/8), U_capacity, 1]`, block 256 (eight waves). `x` is the group of eight
//!    matrix-local output rows `8*wg_x ..`, `y` the distinct expert. Wave `w` owns the output row
//!    `8*wg_x + w` for the whole K traversal, across every slot of the expert (`n <= C`, `C = 16 | 32`):
//!    one wave32 per output row, the lane arithmetic and reduction tree of the narrow entry unchanged.
//!    A workgroup returns uniformly (before its first barrier) when its first row is `>= M`, its expert
//!    index is `>= U`, its expert id is `>= n_experts` (unsigned), `n == 0`, `n` exceeds the entry's capacity or
//!    `offsets[u+1] > 10*C`.
//!  * Per group (128 columns; `Mq4Sym` gate: 256) the wave loads and dequantizes its weights exactly once,
//!    outside every slot loop (no weight-class load sits inside a tile or slot branch: the census test
//!    asserts it, and the count is the same for C16 and C32). The expert's slots then run in tiles of eight:
//!    producer wave `p` loads the group's X slice of slot `8*t + p` (the lane's own `nw` elements, the same
//!    loads the narrow entry issues) and stores it to LDS, one workgroup barrier publishes the tile, every
//!    wave with a valid row consumes the tile's up-to-eight slots from LDS (`ds_load_b128`) into the
//!    slot's own accumulators (`4*C` gate chains / `C` down chains), and a second barrier retires the tile
//!    before the next fill. X therefore costs one global read per `(workgroup, slot, group)` instead of
//!    one per output row, and the tile is 4096 B (W3, MQ4 down) or 8192 B (MQ4 gate, G256) of static LDS.
//!    Tiles with `8*t >= n` are skipped uniformly (their barriers with them).
//!  * Tail waves (`row >= M`) never load weights, compute or store, but produce X and run every barrier.
//!  * Slot metadata is one distributed slot per hardware lane: lane `p` holds the X byte offset and the
//!    Y byte offset (`slot*y_slot_stride*4 + row_in_y*4`) of the expert's `p`-th slot (`p` clamped to
//!    `n-1`), read with `v_readlane_b32` where needed. No persistent per-slot offset banks and no LDS table.
//!  * Numerics (identical to the narrow entry; see its module docs): `Mq3I4` `w = RN(float(d) * float(2c-7))`,
//!    group dot `RN(w0*x0)` then `fma(w_j,x_j,.)`; `Mq4Sym` `w = fma(d, float(q), z)`, group dot start
//!    `RN(w1*x1)`, then `fma(w0,x0,.)`, then `j = 2..`; the group dot is added separately into `acc[g&3]`
//!    (gate) / the single chain (down); gate fold `(a0+a1)+(a2+a3)`; the 16,8,4,2,1 snapshot shuffle tree
//!    (`ds_bpermute_b32`, own+other, out-of-range lanes unchanged); lane 0 stores the original slot.
//!    The reduction is software-pipelined four slots at a time (independent chains interleaved), which
//!    does not change any chain's operand order.
//!
//! Resources (design §4 ceilings; the emitted counts are in the contract): gate `4*C` accumulators per lane
//! (C32: 128), down `C`; one `nw`-register weight group, one `nw`-register X vector; the plan is
//! checked in `u16` and refused above 256 VGPRs.
//!
//! `Cap::C8` is reserved internally for the new `_t8` path: one eight-slot
//! descriptor per grid-z tile, total `C` in 1..=64, fixed `U=80`. It allocates
//! only 32 gate accumulators (eight down), independent of total logical rows.
//! Descriptor pointers advance by 1928 bytes per tile; absolute slot IDs keep
//! X/Y at their original bases. Legacy C16/C32 entries retain their schedule.
//! Tiled offsets must fit both the physical 80-slot descriptor and `10*C`.
//! All clamped slot IDs are validated before reading the expert pointer table.
use super::*;
use crate::lds::Transition;

/// Output rows per workgroup (one wave32 each).
pub const M_BLOCK: u32 = 8;
/// Slots staged per X tile (one producer wave each).
pub const SLOT_TILE: u32 = 8;
/// Threads per workgroup.
pub const WORKGROUP_SIZE: u32 = M_BLOCK * 32;

// Extra SGPRs of the wide plan (the narrow bookkeeping occupies s0..s79).
const WAVE: u8 = 80;
const ROWB: u8 = 81;
const SSI: u8 = 82;
const XSL: u8 = 83;
/// `n_experts@88`: the extent of the `expert_weight_ptrs` table (the narrow entries' `SNEXP`, moved clear of `WAVE`).
const SNEXP_W: u8 = 84;
const SGPRS: u8 = 88;
const TILE: u8 = 85;

// A separate label namespace lets a rowtile and its singleton coexist in one image.
fn lbl(spec: Spec, name: &str) -> String {
    if spec.cap == Cap::C8 {
        format!(".Lmqi4moe_{}_t8_{name}", spec.tag())
    } else {
        super::lbl(spec, name)
    }
}

/// VGPR layout of a wide entry: the narrow [`Lay`] fields for the shared pieces (`xo` / `ya` are one lane
/// register each, `acc` holds `C * chains` accumulators) plus the LDS addressing registers.
#[derive(Clone, Copy, Debug)]
struct WLay {
    l: Lay,
    /// Lane index, wave index (vector), producer LDS store address, consumer LDS read base.
    lane: u8,
    wv: u8,
    xs: u8,
    xlds: u8,
    /// Four exchange registers of the interleaved shuffle tree.
    exch: u8,
    cap: u32,
    end: u16,
}

impl WLay {
    fn new(spec: Spec) -> Result<Self, String> {
        let (nw, chains, cap) = (spec.lane_elems(), spec.kind.chains(), spec.cap.slots());
        let mut n = 1u16;
        let mut take = |len: u16, align: u16| {
            n = n.div_ceil(align) * align;
            let base = n;
            n += len;
            base
        };
        let lane = take(1, 1);
        let wv = take(1, 1);
        let xs = take(1, 1);
        let xlds = take(1, 1);
        let caddr = take(1, 1);
        let saddr = take(1, 1);
        let xl = take(1, 1);
        let cl = take(1, 1);
        let shift = take(1, 1);
        let hl = take(1, 1);
        let xb = take(1, 1);
        let xa = take(1, 1);
        let pkt0 = take(1, 1);
        let pkt1 = take(1, 1);
        let d = take(1, 1);
        let z = take(1, 1);
        let dot = take(2, 1);
        let exch = take(4, 1);
        let shuf = take(5, 1);
        let w = take(u16::from(nw), 4);
        let x = take(8, 4);
        let xo = take(1, 1);
        let ya = take(1, 1);
        let acc = take(cap as u16 * u16::from(chains), 1);
        let end = n.div_ceil(8) * 8;
        if end > 256 {
            return Err(format!(
                "{}: {end} VGPRs exceed the 256 register file",
                spec.symbol()
            ));
        }
        let b = |v: u16| v as u8;
        let l = Lay {
            nw,
            chains,
            caddr: b(caddr),
            saddr: b(saddr),
            xl: b(xl),
            cl: b(cl),
            shift: b(shift),
            hl: b(hl),
            xb: b(xb),
            xa: b(xa),
            pkt0: b(pkt0),
            pkt1: b(pkt1),
            d: b(d),
            z: b(z),
            dot: b(dot),
            exch: b(exch),
            shuf: b(shuf),
            w: b(w),
            x: b(x),
            acc: b(acc),
            xo: b(xo),
            ya: b(ya),
            end: 0,
        };
        if u16::from(l.acc) + cap as u16 * u16::from(chains) > 256 {
            return Err("accumulator bank overflows the register file".into());
        }
        Ok(Self {
            l,
            lane: b(lane),
            wv: b(wv),
            xs: b(xs),
            xlds: b(xlds),
            exch: b(exch),
            cap,
            end,
        })
    }
    /// Bytes between two slots of the staged X tile: `nw` elements per lane over 32 lanes.
    fn pitch(&self) -> u32 {
        u32::from(self.l.nw) * 128
    }
    fn tiles(&self) -> u32 {
        self.cap / SLOT_TILE
    }
}

fn plan(arch: Arch, l: &WLay) -> Result<RegPlan, String> {
    let b = &l.l;
    let mut regs = RegPlan::new(l.end, u16::from(SGPRS))?;
    if !arch.gfx12() {
        for (name, base) in [("workgroup_id_x", WGX), ("workgroup_id_y", WGY)] {
            declare(&mut regs, name, RKind::S, base, 1)?;
        }
    }
    for (name, base, len) in [
        ("kernarg", KARG, 2),
        ("kernel_arguments_ptrs", WPTRS, 8),
        ("kernel_arguments_x_y_m_k", XROT, 8),
        ("kernel_arguments_u_c_div_src", SU, 4),
        ("kernel_arguments_strides", SXSTR, 2),
        ("row", ROW, 1),
        ("expert_index", UIDX, 1),
        ("expert_offsets_pair", OFF0, 2),
        ("expert_id", EID, 1),
        ("slot_count", NS, 1),
        ("table_pointer", TPTR, 2),
        ("expert_pointer", WPTR, 2),
        ("x_descriptor", SRDX, 4),
        ("row_pointer", ROW_PTR, 2),
        ("scale_pointer", SCALE_PTR, 2),
        ("groups", GROUPS, 1),
        ("row_bytes", ROW_BYTES, 1),
        ("codes_bytes", CODES_BYTES, 1),
        ("temp", TMP, 1),
        ("temp2", TMP2, 1),
        ("group_index", GI, 1),
        ("quad_end", QUADS, 1),
        ("group_code_offset", G_CODE, 1),
        ("group_x_offset", G_X, 1),
        ("group_scale_offset", G_SCALE, 1),
        ("y_select", YSEL, 2),
        ("row_in_y_bytes", ROWIN4, 1),
        ("x_stride_bytes", SXS4, 1),
        ("y_stride_bytes", SYS4, 1),
        ("slots_minus_one", NSM1, 1),
        ("gather_live", LIVE, 2),
        ("gather_mask", MASK, 2),
        ("shuffle_masks", SHM, 5),
        ("lane_temps", TL, 4),
        ("half_rows", MI, 1),
        ("row_minus_half", ROWM, 1),
        ("row_in_y", ROWIN, 1),
        ("wave", WAVE, 1),
        ("row_block", ROWB, 1),
        ("slot_index", SSI, 1),
        ("slot_x_offset", XSL, 1),
    ] {
        declare(&mut regs, name, RKind::S, base, len)?;
    }
    declare(
        &mut regs,
        "kernel_arguments_n_experts",
        RKind::S,
        SNEXP_W,
        1,
    )?;
    if l.cap == SLOT_TILE {
        declare(&mut regs, "row_tile", RKind::S, TILE, 1)?;
    }
    let chains = u16::from(b.chains);
    for (name, base, len) in [
        ("thread_id", LANE, 1),
        ("lane", l.lane, 1),
        ("wave_index", l.wv, 1),
        ("x_tile_store_address", l.xs, 1),
        ("x_tile_read_base", l.xlds, 1),
        ("code_address", b.caddr, 1),
        ("scale_address", b.saddr, 1),
        ("x_lane_offset", b.xl, 1),
        ("code_lane_offset", b.cl, 1),
        ("packet_shift", b.shift, 1),
        ("header_lane_offset", b.hl, 1),
        ("x_base", b.xb, 1),
        ("x_address", b.xa, 1),
        ("packet0", b.pkt0, 1),
        ("packet1", b.pkt1, 1),
        ("scale", b.d, 1),
        ("zero_point", b.z, 1),
        ("dots", b.dot, 2),
        ("exchange", l.exch, 4),
        ("shuffle_addresses", b.shuf, 5),
        ("weights", b.w, b.nw),
        ("x_values", b.x, 8),
        ("slot_x_offsets", b.xo, 1),
        ("slot_y_addresses", b.ya, 1),
    ] {
        declare(&mut regs, name, RKind::V, base, len)?;
    }
    // Accumulators: up to 128 registers, declared in <=255-wide pieces (`declare` takes u8 lengths).
    declare(
        &mut regs,
        "accumulators",
        RKind::V,
        b.acc,
        (l.cap as u16 * chains) as u8,
    )?;
    let top = u16::from(b.acc) + l.cap as u16 * chains;
    if l.end > top {
        declare(
            &mut regs,
            "granule_pad",
            RKind::V,
            top as u8,
            (l.end - top) as u8,
        )?;
    }
    Ok(regs)
}

fn slot_index_gate(b: &mut Builder, bound: u32, target: &str) -> R {
    sop(b, format!("s_cmp_gt_u32 s{NS}, {bound}"), &[], &[NS])?;
    branch(b, &format!("s_cbranch_scc0 {target}"))
}

/// `if row >= M skip to target` (wave-uniform).
fn row_gate(b: &mut Builder, target: &str) -> R {
    sop(b, format!("s_cmp_lt_u32 s{ROW}, s{SM}"), &[], &[ROW, SM])?;
    branch(b, &format!("s_cbranch_scc0 {target}"))
}

pub(super) fn emit(spec: Spec, net: Network) -> Result<Emitted, String> {
    let l = WLay::new(spec)?;
    let rowtiles = spec.cap == Cap::C8;
    let lds_bytes = SLOT_TILE * 32 * 4 * u32::from(spec.lane_elems());
    let kernargs = KernargLayout::new(KERNARG_BYTES)
        .pointer_access("expert_weight_ptrs", 0, Access::ReadOnly)
        .pointer_access("expert_ids", 8, Access::ReadOnly)
        .pointer_access("expert_offsets", 16, Access::ReadOnly)
        .pointer_access("slot_ids", 24, Access::ReadOnly)
        .pointer_access("Xrot", 32, Access::ReadOnly)
        .pointer_access("Y0", 40, Access::WriteOnly)
        .pointer_access("Y1", 48, Access::WriteOnly)
        .hidden("M", 56, 4, "by_value")
        .hidden("K", 60, 4, "by_value")
        .hidden("U", 64, 4, "by_value")
        .hidden("C", 68, 4, "by_value")
        .hidden("x_row_div", 72, 4, "by_value")
        .hidden("x_src_rows", 76, 4, "by_value")
        .hidden("x_row_stride", 80, 4, "by_value")
        .hidden("y_slot_stride", 84, 4, "by_value")
        .hidden("n_experts", 88, 4, "by_value")
        .hidden("reserved", 92, 4, "by_value");
    let kspec = KernelSpec {
        kernel_id: "mq_i4_moe_gemv".into(),
        variant: if rowtiles {
            format!("{}_t8", spec.tag())
        } else {
            spec.tag()
        },
        arch: spec.arch,
        symbol: if rowtiles {
            format!("{}_t8", spec.symbol())
        } else {
            spec.symbol()
        },
        kernargs,
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: true,
        workgroup_size: WORKGROUP_SIZE as u16,
        group_segment_fixed_size: lds_bytes,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan(spec.arch, &l)?);
    b.enable_delay_alu();
    if rowtiles {
        b.enable_workgroup_id_z();
    }
    let slot = b.lds_slot("x_tile", 0, lds_bytes)?;
    prologue(&mut b, spec, &l)?;
    b.wait_all()?;
    match spec.kind {
        Kind::GateUp => {
            // Groups strictly ascending; group g feeds chain g & 3: whole quads in the loop, then up to
            // three tail groups on chains 0, 1, 2 (the narrow entry's order).
            sop(
                &mut b,
                format!("s_and_b32 s{QUADS}, s{GROUPS}, -4"),
                &[QUADS],
                &[GROUPS],
            )?;
            sop(&mut b, format!("s_cmp_eq_u32 s{QUADS}, 0"), &[], &[QUADS])?;
            branch(&mut b, &format!("s_cbranch_scc1 {}", lbl(spec, "tail")))?;
            b.loop_(&lbl(spec, "quad"), |b| {
                for chain in 0..4 {
                    group(b, spec, &l, net, slot, chain, &format!("q{chain}"))?;
                }
                sop(
                    b,
                    format!("s_cmp_lt_u32 s{GI}, s{QUADS}"),
                    &[],
                    &[GI, QUADS],
                )?;
                branch(b, &format!("s_cbranch_scc1 {}", lbl(spec, "quad")))
            })?;
            b.label(&lbl(spec, "tail"))?;
            for chain in 0..3 {
                sop(
                    &mut b,
                    format!("s_cmp_lt_u32 s{GI}, s{GROUPS}"),
                    &[],
                    &[GI, GROUPS],
                )?;
                branch(&mut b, &format!("s_cbranch_scc0 {}", lbl(spec, "reduce")))?;
                group(&mut b, spec, &l, net, slot, chain, &format!("t{chain}"))?;
            }
        }
        Kind::Down => {
            sop(&mut b, format!("s_cmp_eq_u32 s{GROUPS}, 0"), &[], &[GROUPS])?;
            branch(&mut b, &format!("s_cbranch_scc1 {}", lbl(spec, "reduce")))?;
            b.loop_(&lbl(spec, "group"), |b| {
                group(b, spec, &l, net, slot, 0, "g")?;
                sop(
                    b,
                    format!("s_cmp_lt_u32 s{GI}, s{GROUPS}"),
                    &[],
                    &[GI, GROUPS],
                )?;
                branch(b, &format!("s_cbranch_scc1 {}", lbl(spec, "group")))
            })?;
        }
    }
    b.label(&lbl(spec, "reduce"))?;
    epilogue(&mut b, spec, &l)?;
    b.finish()
}

/// Stage the expert's slot IDs once, clamping duplicate lanes to its last live slot.
fn load_slots(b: &mut Builder, l: &WLay) -> R {
    let (sid, p) = (l.l.x, l.l.x + 7);
    valu(
        b,
        &format!("v_min_u32_e32 v{p}, s{NSM1}, v{}", l.lane),
        &[p],
        &[l.lane],
        &[NSM1],
    )?;
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{p}, s{OFF0}, v{p}"),
        &[p],
        &[p],
        &[OFF0],
    )?;
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{p}, 2, v{p}"),
        &[p],
        &[p],
        &[],
    )?;
    gload(b, "global_load_b32", sid, p, SLOTS, "")
}

/// Kernarg loads, the work-item-derived LDS addresses (entry block: the M7 LDS bound executes it), the
/// guards, the expert's pointer, the wave's row bookkeeping, per-lane address constants, zeroed
/// accumulators and the lane-distributed slot metadata.
fn prologue(b: &mut Builder, spec: Spec, l: &WLay) -> R {
    let a = b.spec.arch;
    let exit = lbl(spec, "exit");
    let ll = &l.l;
    let nw = ll.nw;
    smem(b, WPTRS, 8, KARG, 0)?;
    smem(b, XROT, 8, KARG, 0x20)?;
    smem(b, SU, 4, KARG, 0x40)?;
    smem(b, SXSTR, 2, KARG, 0x50)?;
    smem(b, SNEXP_W, 1, KARG, 0x58)?;
    // LDS addresses, all functions of the work-item id alone and defined once (before the first branch):
    //   producer store: wave*pitch + lane*16 (`[slot=wave][half][lane][4 f32]`), consumer base: lane*16.
    valu(
        b,
        &format!("v_and_b32_e32 v{}, 31, v{LANE}", l.lane),
        &[l.lane],
        &[LANE],
        &[],
    )?;
    valu(
        b,
        &format!("v_lshrrev_b32_e32 v{}, 5, v{LANE}", l.wv),
        &[l.wv],
        &[LANE],
        &[],
    )?;
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{}, 4, v{}", l.xlds, l.lane),
        &[l.xlds],
        &[l.lane],
        &[],
    )?;
    if nw == 8 {
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{}, 10, v{}", l.xs, l.wv),
            &[l.xs],
            &[l.wv],
            &[],
        )?;
        valu(
            b,
            &format!("v_lshl_add_u32 v{0}, v{1}, 4, v{0}", l.xs, l.lane),
            &[l.xs],
            &[l.lane, l.xs],
            &[],
        )?;
    } else {
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{}, 4, v{LANE}", l.xs),
            &[l.xs],
            &[LANE],
            &[],
        )?;
    }
    op(
        b,
        format!("v_readfirstlane_b32 s{WAVE}, v{}", l.wv),
        &[s(WAVE)],
        &[v(l.wv)],
    )?;
    // Workgroup ids: gfx12 `ttmp9` (x) and `ttmp7[15:0]` (y); gfx11 the first two system SGPRs after the user SGPRs.
    match a.workgroup_ids(2)? {
        WgIds {
            x: WgIdSource::Ttmp9,
            y: WgIdSource::Ttmp7,
        } => {
            sop(b, format!("s_mov_b32 s{ROWB}, ttmp9"), &[ROWB], &[])?;
            sop(b, format!("s_and_b32 s{UIDX}, ttmp7, 0xffff"), &[UIDX], &[])?;
        }
        WgIds {
            x: WgIdSource::Sgpr(x),
            y: WgIdSource::Sgpr(y),
        } if (x, y) == (WGX, WGY) => {
            sop(b, format!("s_mov_b32 s{ROWB}, s{WGX}"), &[ROWB], &[WGX])?;
            sop(b, format!("s_mov_b32 s{UIDX}, s{WGY}"), &[UIDX], &[WGY])?;
        }
        ids => {
            return Err(format!(
                "mq_i4_moe_gemv wide: unexpected workgroup id sources {ids:?}"
            ))
        }
    }
    if spec.cap == Cap::C8 {
        // gfx1201 packs y/z into the low/high halves of ttmp7.
        sop(b, format!("s_lshr_b32 s{TILE}, ttmp7, 16"), &[TILE], &[])?;
        sop(b, format!("s_cmp_eq_u32 s{SC}, 0"), &[], &[SC])?;
        branch(b, &format!("s_cbranch_scc1 {exit}"))?;
        sop(b, format!("s_cmp_gt_u32 s{SC}, 64"), &[], &[SC])?;
        branch(b, &format!("s_cbranch_scc1 {exit}"))?;
        sop(b, format!("s_cmp_eq_u32 s{SU}, {}", lit(80)), &[], &[SU])?;
        branch(b, &format!("s_cbranch_scc0 {exit}"))?;
        sop(b, format!("s_lshl_b32 s{TMP}, s{TILE}, 3"), &[TMP], &[TILE])?;
        sop(b, format!("s_cmp_ge_u32 s{TMP}, s{SC}"), &[], &[TMP, SC])?;
        branch(b, &format!("s_cbranch_scc1 {exit}"))?;
        sop(b, format!("s_cmp_eq_u32 s{SDIV}, 0"), &[], &[SDIV])?;
        branch(b, &format!("s_cbranch_scc1 {exit}"))?;
        sop(b, format!("s_cmp_eq_u32 s{SK}, 0"), &[], &[SK])?;
        branch(b, &format!("s_cbranch_scc1 {exit}"))?;
        let mask = if spec.lane_elems() == 8 { 255 } else { 127 };
        sop(
            b,
            format!("s_and_b32 s{TMP}, s{SK}, {}", lit(mask)),
            &[TMP],
            &[SK],
        )?;
        sop(b, format!("s_cmp_eq_u32 s{TMP}, 0"), &[], &[TMP])?;
        branch(b, &format!("s_cbranch_scc0 {exit}"))?;
        sop(
            b,
            format!("s_mul_i32 s{TMP}, s{TILE}, {}", lit(1928)),
            &[TMP],
            &[TILE],
        )?;
        for ptr in [EIDS, OFFS, SLOTS] {
            add64(b, ptr, ptr, TMP)?;
        }
    }
    sop(
        b,
        format!("s_lshl_b32 s{ROWB}, s{ROWB}, 3"),
        &[ROWB],
        &[ROWB],
    )?;
    sop(b, format!("s_cmp_ge_u32 s{ROWB}, s{SM}"), &[], &[ROWB, SM])?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    sop(b, format!("s_cmp_ge_u32 s{UIDX}, s{SU}"), &[], &[UIDX, SU])?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    // offsets[u], offsets[u+1] and the expert id; the expert's tensor pointer from the table.
    sop(b, format!("s_lshl_b32 s{TMP}, s{UIDX}, 2"), &[TMP], &[UIDX])?;
    add64(b, TPTR, OFFS, TMP)?;
    smem(b, OFF0, 2, TPTR, 0)?;
    add64(b, TPTR, EIDS, TMP)?;
    smem(b, EID, 1, TPTR, 0)?;
    sop(
        b,
        format!("{} s{NS}, s{OFF1}, s{OFF0}", s_sub_i32(a)),
        &[NS],
        &[OFF1, OFF0],
    )?;
    sop(b, format!("s_cmp_eq_u32 s{NS}, 0"), &[], &[NS])?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    sop(b, format!("s_cmp_gt_u32 s{NS}, {}", l.cap), &[], &[NS])?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    if spec.cap == Cap::C8 {
        sop(b, format!("s_mov_b32 s{TMP}, {}", lit(80)), &[TMP], &[])?;
        // Bound both offsets: subtraction can otherwise wrap from UINT_MAX
        // to a small apparently live count.
        sop(
            b,
            format!("s_cmp_gt_u32 s{OFF0}, s{TMP}"),
            &[],
            &[OFF0, TMP],
        )?;
        branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    } else {
        sop(b, format!("s_mul_i32 s{TMP}, s{SC}, 10"), &[TMP], &[SC])?;
    }
    sop(
        b,
        format!("s_cmp_gt_u32 s{OFF1}, s{TMP}"),
        &[],
        &[OFF1, TMP],
    )?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    if spec.cap == Cap::C8 {
        // The final tile may contain fewer than eight logical rows: its
        // physical slot capacity does not enlarge the logical 10*C domain.
        sop(b, format!("s_mul_i32 s{TMP}, s{SC}, 10"), &[TMP], &[SC])?;
        sop(
            b,
            format!("s_cmp_gt_u32 s{OFF1}, s{TMP}"),
            &[],
            &[OFF1, TMP],
        )?;
        branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    }
    // Expert id range check before the table index (the narrow entries' guard): an id at or past `n_experts`
    // (unsigned, so a negative id too) exits without a load or a store.
    sop(
        b,
        format!("s_cmp_ge_u32 s{EID}, s{SNEXP_W}"),
        &[],
        &[EID, SNEXP_W],
    )?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    if spec.cap == Cap::C8 {
        sop(
            b,
            format!("{} s{NSM1}, s{NS}, -1", s_add_i32(a)),
            &[NSM1],
            &[NS],
        )?;
        load_slots(b, l)?;
        // Every wave sees the same complete slot list, including clamped
        // duplicate lanes. Reject uniformly before the table or any barrier.
        valu(
            b,
            &format!("v_cmp_le_u32_e32 vcc_lo, s{TMP}, v{}", ll.x),
            &[],
            &[ll.x],
            &[TMP],
        )?;
        branch(b, &format!("s_cbranch_vccnz {exit}"))?;
    }
    sop(b, format!("s_lshl_b32 s{TMP}, s{EID}, 3"), &[TMP], &[EID])?;
    add64(b, TPTR, WPTRS, TMP)?;
    smem(b, WPTR, 2, TPTR, 0)?;
    // This wave's row: 8*wg_x + wave (bookkeeping below follows it; a tail wave's pointers are never dereferenced).
    sop(
        b,
        format!("{} s{ROW}, s{ROWB}, s{WAVE}", s_add_i32(a)),
        &[ROW],
        &[ROWB, WAVE],
    )?;
    match (spec.codec, spec.kind) {
        (Codec::Mq3I4, _) => {
            sop(
                b,
                format!("s_lshr_b32 s{GROUPS}, s{SK}, 7"),
                &[GROUPS],
                &[SK],
            )?;
            sop(
                b,
                format!("s_lshr_b32 s{ROW_BYTES}, s{SK}, 3"),
                &[ROW_BYTES],
                &[SK],
            )?;
            sop(
                b,
                format!("s_mul_i32 s{ROW_BYTES}, s{ROW_BYTES}, 3"),
                &[ROW_BYTES],
                &[ROW_BYTES],
            )?;
            sop(
                b,
                format!("s_mul_i32 s{CODES_BYTES}, s{SM}, s{ROW_BYTES}"),
                &[CODES_BYTES],
                &[SM, ROW_BYTES],
            )?;
            sop(
                b,
                format!("s_mul_i32 s{TMP}, s{ROW}, s{ROW_BYTES}"),
                &[TMP],
                &[ROW, ROW_BYTES],
            )?;
            add64(b, ROW_PTR, WPTR, TMP)?;
            sop(
                b,
                format!("s_mul_i32 s{TMP2}, s{ROW}, s{GROUPS}"),
                &[TMP2],
                &[ROW, GROUPS],
            )?;
            sop(
                b,
                format!("s_lshl_b32 s{TMP2}, s{TMP2}, 1"),
                &[TMP2],
                &[TMP2],
            )?;
            sop(
                b,
                format!("{} s{TMP2}, s{TMP2}, s{CODES_BYTES}", s_add_i32(a)),
                &[TMP2],
                &[TMP2, CODES_BYTES],
            )?;
            add64(b, SCALE_PTR, WPTR, TMP2)?;
        }
        (Codec::Mq4Sym, kind) => {
            let (shift, stride) = if kind == Kind::GateUp {
                (8, 136)
            } else {
                (7, 68)
            };
            sop(
                b,
                format!("s_lshr_b32 s{GROUPS}, s{SK}, {shift}"),
                &[GROUPS],
                &[SK],
            )?;
            sop(
                b,
                format!("s_mul_i32 s{ROW_BYTES}, s{GROUPS}, {}", lit(stride)),
                &[ROW_BYTES],
                &[GROUPS],
            )?;
            sop(
                b,
                format!("s_mul_i32 s{TMP}, s{ROW}, s{ROW_BYTES}"),
                &[TMP],
                &[ROW, ROW_BYTES],
            )?;
            add64(b, ROW_PTR, WPTR, TMP)?;
        }
    }
    // Y pointer and the row's byte offset inside a slot's output row.
    match spec.kind {
        Kind::GateUp => {
            sop(b, format!("s_lshr_b32 s{MI}, s{SM}, 1"), &[MI], &[SM])?;
            sop(
                b,
                format!("{} s{ROWM}, s{ROW}, s{MI}", s_sub_i32(a)),
                &[ROWM],
                &[ROW, MI],
            )?;
            sop(b, format!("s_cmp_lt_u32 s{ROW}, s{MI}"), &[], &[ROW, MI])?;
            op(
                b,
                format!(
                    "s_cselect_b64 s[{YSEL}:{}], s[{Y0}:{}], s[{Y1}:{}]",
                    YSEL + 1,
                    Y0 + 1,
                    Y1 + 1
                ),
                &[sr(YSEL, 2)],
                &[sr(Y0, 2), sr(Y1, 2)],
            )?;
            sop(
                b,
                format!("s_cselect_b32 s{ROWIN}, s{ROW}, s{ROWM}"),
                &[ROWIN],
                &[ROW, ROWM],
            )?;
            sop(
                b,
                format!("s_lshl_b32 s{ROWIN4}, s{ROWIN}, 2"),
                &[ROWIN4],
                &[ROWIN],
            )?;
        }
        Kind::Down => {
            sop(b, format!("s_mov_b32 s{YSEL}, s{Y0}"), &[YSEL], &[Y0])?;
            sop(
                b,
                format!("s_mov_b32 s{}, s{}", YSEL + 1, Y0 + 1),
                &[YSEL + 1],
                &[Y0 + 1],
            )?;
            sop(
                b,
                format!("s_lshl_b32 s{ROWIN4}, s{ROW}, 2"),
                &[ROWIN4],
                &[ROW],
            )?;
        }
    }
    sop(
        b,
        format!("s_lshl_b32 s{SXS4}, s{SXSTR}, 2"),
        &[SXS4],
        &[SXSTR],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{SYS4}, s{SYSTR}, 2"),
        &[SYS4],
        &[SYSTR],
    )?;
    if spec.cap != Cap::C8 {
        sop(
            b,
            format!("{} s{NSM1}, s{NS}, -1", s_add_i32(a)),
            &[NSM1],
            &[NS],
        )?;
    }
    // X descriptor: base Xrot, x_src_rows*x_row_stride*4 records (rows past it read 0).
    sop(b, format!("s_mov_b32 s{SRDX}, s{XROT}"), &[SRDX], &[XROT])?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{}", SRDX + 1, XROT + 1),
        &[SRDX + 1],
        &[XROT + 1],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{TMP}, s{SXSRC}, s{SXS4}"),
        &[TMP],
        &[SXSRC, SXS4],
    )?;
    srd_tail(b, SRDX, Some(TMP))?;
    sop(b, format!("s_mov_b32 s{GI}, 0"), &[GI], &[])?;
    // Per-lane constants: global X byte offset within a group (16*l, or 32*l for the 8-element lane),
    // the code byte offset of the lane's packet within a group and its extras.
    let xshift = if nw == 8 { 5 } else { 4 };
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{}, {xshift}, v{}", ll.xl, l.lane),
        &[ll.xl],
        &[l.lane],
        &[],
    )?;
    match (spec.codec, spec.kind) {
        (Codec::Mq3I4, _) => {
            // (3l) >> 1 and the W3 packet's bit shift 4*(l & 1).
            valu(
                b,
                &format!("v_mul_u32_u24_e32 v{}, 3, v{}", ll.cl, l.lane),
                &[ll.cl],
                &[l.lane],
                &[],
            )?;
            valu(
                b,
                &format!("v_lshrrev_b32_e32 v{0}, 1, v{0}", ll.cl),
                &[ll.cl],
                &[ll.cl],
                &[],
            )?;
            valu(
                b,
                &format!("v_and_b32_e32 v{}, 1, v{}", ll.shift, l.lane),
                &[ll.shift],
                &[l.lane],
                &[],
            )?;
            valu(
                b,
                &format!("v_lshlrev_b32_e32 v{0}, 2, v{0}", ll.shift),
                &[ll.shift],
                &[ll.shift],
                &[],
            )?;
        }
        (Codec::Mq4Sym, Kind::GateUp) => {
            // Payload word 4*l; header half (bytes 0 or 4) 4*(l >> 4).
            valu(
                b,
                &format!("v_lshlrev_b32_e32 v{}, 2, v{}", ll.cl, l.lane),
                &[ll.cl],
                &[l.lane],
                &[],
            )?;
            valu(
                b,
                &format!("v_lshrrev_b32_e32 v{}, 4, v{}", ll.hl, l.lane),
                &[ll.hl],
                &[l.lane],
                &[],
            )?;
            valu(
                b,
                &format!("v_lshlrev_b32_e32 v{0}, 2, v{0}", ll.hl),
                &[ll.hl],
                &[ll.hl],
                &[],
            )?;
        }
        (Codec::Mq4Sym, Kind::Down) => valu(
            b,
            &format!("v_lshlrev_b32_e32 v{}, 1, v{}", ll.cl, l.lane),
            &[ll.cl],
            &[l.lane],
            &[],
        )?,
    }
    for k in 0..(l.cap as u16 * u16::from(ll.chains)) as u8 {
        valu(
            b,
            &format!("v_mov_b32_e32 v{}, 0", ll.acc + k),
            &[ll.acc + k],
            &[],
            &[],
        )?;
    }
    // Slot metadata, one slot per hardware lane: lane p loads slot position min(p, n-1) of the expert (never
    // past its slots), derives the X byte offset (slot / x_row_div * x_row_stride * 4) and the Y byte offset
    // (slot * y_slot_stride * 4 + row-in-Y * 4). Slot positions are then read with `v_readlane_b32`.
    let (sid, t0) = (ll.x, ll.x + 2);
    if spec.cap != Cap::C8 {
        load_slots(b, l)?;
    }
    gather_offset(
        b,
        ll.xo,
        sid,
        Some(SDIV),
        1,
        LIVE,
        GatherTemps {
            v: [t0, t0 + 1, t0 + 2, t0 + 3],
            mask: MASK,
        },
    )?;
    valu(
        b,
        &format!("v_mul_lo_u32 v{0}, v{0}, s{SXS4}", ll.xo),
        &[ll.xo],
        &[ll.xo],
        &[SXS4],
    )?;
    valu(
        b,
        &format!("v_mul_lo_u32 v{}, v{sid}, s{SYS4}", ll.ya),
        &[ll.ya],
        &[sid],
        &[SYS4],
    )?;
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{0}, s{ROWIN4}, v{0}", ll.ya),
        &[ll.ya],
        &[ll.ya],
        &[ROWIN4],
    )?;
    // The X registers double as the temporaries above; every lane of the producer/consumer path reads them
    // (a producer wave without a slot skips the store), so leave them defined.
    for k in 0..8u8 {
        valu(
            b,
            &format!("v_mov_b32_e32 v{}, 0", ll.x + k),
            &[ll.x + k],
            &[],
            &[],
        )?
    }
    Ok(())
}

fn range(base: u8, len: u8) -> String {
    format!("v[{base}:{}]", u16::from(base) + u16::from(len) - 1)
}
fn off(bytes: u32) -> String {
    if bytes == 0 {
        String::new()
    } else {
        format!(" offset:{bytes}")
    }
}

/// One group of this wave's row: weights loaded and dequantized once, then every tile of eight slots staged
/// through LDS and applied to the slots' accumulators. Chain `chain` (always 0 for down) takes the group partial.
fn group(
    b: &mut Builder,
    spec: Spec,
    l: &WLay,
    net: Network,
    slot: usize,
    chain: u8,
    inst: &str,
) -> R {
    let arch = b.spec.arch;
    let ll = &l.l;
    let nw = ll.nw;
    match spec.codec {
        Codec::Mq3I4 => {
            sop(
                b,
                format!("s_mul_i32 s{G_CODE}, s{GI}, 48"),
                &[G_CODE],
                &[GI],
            )?;
            sop(b, format!("s_lshl_b32 s{G_X}, s{GI}, 9"), &[G_X], &[GI])?;
            sop(
                b,
                format!("s_lshl_b32 s{G_SCALE}, s{GI}, 1"),
                &[G_SCALE],
                &[GI],
            )?;
        }
        Codec::Mq4Sym => {
            sop(
                b,
                format!(
                    "s_mul_i32 s{G_CODE}, s{GI}, {}",
                    lit(if nw == 8 { 136 } else { 68 })
                ),
                &[G_CODE],
                &[GI],
            )?;
            sop(
                b,
                format!("s_lshl_b32 s{G_X}, s{GI}, {}", if nw == 8 { 10 } else { 9 }),
                &[G_X],
                &[GI],
            )?;
        }
    }
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{}, s{G_CODE}, v{}", ll.caddr, ll.cl),
        &[ll.caddr],
        &[ll.cl],
        &[G_CODE],
    )?;
    match (spec.codec, spec.kind) {
        (Codec::Mq3I4, _) => valu(
            b,
            &format!("v_mov_b32_e32 v{}, s{G_SCALE}", ll.saddr),
            &[ll.saddr],
            &[],
            &[G_SCALE],
        )?,
        (Codec::Mq4Sym, Kind::GateUp) => valu(
            b,
            &format!("v_add_nc_u32_e32 v{}, s{G_CODE}, v{}", ll.saddr, ll.hl),
            &[ll.saddr],
            &[ll.hl],
            &[G_CODE],
        )?,
        (Codec::Mq4Sym, Kind::Down) => valu(
            b,
            &format!("v_mov_b32_e32 v{}, s{G_CODE}", ll.saddr),
            &[ll.saddr],
            &[],
            &[G_CODE],
        )?,
    }
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{}, s{G_X}, v{}", ll.xb, ll.xl),
        &[ll.xb],
        &[ll.xl],
        &[G_X],
    )?;
    // Weight loads: once per group, by valid waves only, before any tile or slot branch.
    let wdone = lbl(spec, &format!("{inst}_wdone"));
    row_gate(b, &wdone)?;
    match (spec.codec, spec.kind) {
        (Codec::Mq3I4, _) => {
            match net {
                Network::Reference => {
                    gload(b, "global_load_u8", ll.pkt0, ll.caddr, ROW_PTR, "")?;
                    gload(b, "global_load_u8", ll.pkt1, ll.caddr, ROW_PTR, " offset:1")?;
                }
                Network::Optimized => gload(b, "global_load_u16", ll.pkt0, ll.caddr, ROW_PTR, "")?,
            }
            gload(b, "global_load_u16", ll.d, ll.saddr, SCALE_PTR, "")?;
        }
        (Codec::Mq4Sym, Kind::GateUp) => {
            gload(
                b,
                "global_load_b32",
                ll.pkt0,
                ll.caddr,
                ROW_PTR,
                " offset:8",
            )?;
            gload(b, "global_load_b32", ll.z, ll.saddr, ROW_PTR, "")?;
        }
        (Codec::Mq4Sym, Kind::Down) => {
            gload(
                b,
                "global_load_u16",
                ll.pkt0,
                ll.caddr,
                ROW_PTR,
                " offset:4",
            )?;
            gload(b, "global_load_b32", ll.z, ll.saddr, ROW_PTR, "")?;
        }
    }
    b.label(&wdone)?;
    let gend = lbl(spec, &format!("{inst}_gend"));
    let pitch = l.pitch();
    for t in 0..l.tiles() {
        let tag = |name: &str| lbl(spec, &format!("{inst}_t{t}_{name}"));
        if t > 0 {
            slot_index_gate(b, SLOT_TILE * t, &gend)?
        }
        // Fill: producer wave `w` stages slot position 8t + w (when it exists): the lane's own X elements,
        // loaded exactly as the narrow entry loads them, stored to LDS as unchanged bits.
        sop(
            b,
            format!(
                "{} s{SSI}, s{WAVE}, {}",
                s_add_i32(arch),
                lit(SLOT_TILE * t)
            ),
            &[SSI],
            &[WAVE],
        )?;
        let (pdone, sdone) = (tag("pdone"), tag("sdone"));
        sop(b, format!("s_cmp_gt_u32 s{NS}, s{SSI}"), &[], &[NS, SSI])?;
        branch(b, &format!("s_cbranch_scc0 {pdone}"))?;
        op(
            b,
            format!("v_readlane_b32 s{XSL}, v{}, s{SSI}", ll.xo),
            &[s(XSL)],
            &[v(ll.xo), s(SSI)],
        )?;
        valu(
            b,
            &format!("v_add_nc_u32_e32 v{}, s{XSL}, v{}", ll.xa, ll.xb),
            &[ll.xa],
            &[ll.xb],
            &[XSL],
        )?;
        bload(b, 4, ll.x, ll.xa, SRDX, 0)?;
        if nw == 8 {
            bload(b, 4, ll.x + 4, ll.xa, SRDX, 16)?
        }
        b.label(&pdone)?;
        b.wait_all()?;
        sop(b, format!("s_cmp_gt_u32 s{NS}, s{SSI}"), &[], &[NS, SSI])?;
        branch(b, &format!("s_cbranch_scc0 {sdone}"))?;
        for h in 0..nw / 4 {
            b.ds_store(
                slot,
                Instruction::new(
                    format!(
                        "ds_store_b128 v{}, {}{}",
                        l.xs,
                        range(ll.x + 4 * h, 4),
                        off(512 * u32::from(h))
                    ),
                    vec![],
                    vec![v(l.xs), vr(ll.x + 4 * h, 4)],
                )
                .memory(MemoryClass::DsStore),
            )?;
        }
        b.label(&sdone)?;
        if t == 0 {
            // The group's weights, reconstructed once (valid waves only) while the other waves publish.
            let udone = tag("udone");
            row_gate(b, &udone)?;
            unpack(b, spec, ll, net)?;
            b.label(&udone)?;
        }
        b.barrier(&[Transition::Ready(slot)])?;
        // Consume: the valid waves' per-slot chains on the shared weights, X read from the published tile.
        let cdone = tag("cdone");
        row_gate(b, &cdone)?;
        for i in 0..SLOT_TILE {
            let si = SLOT_TILE * t + i;
            if i > 0 {
                slot_index_gate(b, si, &cdone)?
            }
            for h in 0..nw / 4 {
                b.ds_load(
                    slot,
                    Instruction::new(
                        format!(
                            "ds_load_b128 {}, v{}{}",
                            range(ll.x + 4 * h, 4),
                            l.xlds,
                            off(i * pitch + 512 * u32::from(h))
                        ),
                        vec![vr(ll.x + 4 * h, 4)],
                        vec![v(l.xlds)],
                    )
                    .memory(MemoryClass::DsLoad),
                )?;
            }
            let dot = ll.dot + (si % 2) as u8;
            let x = ll.x;
            // Mq3I4: p = RN(w0*x0), then fma(w_j, x_j, p) for j = 1..; Mq4Sym: p = RN(w1*x1), then fma(w0, x0, p), then j = 2.. ascending.
            let first = if spec.codec == Codec::Mq4Sym { 1 } else { 0 };
            valu(
                b,
                &format!("v_mul_f32_e32 v{dot}, v{}, v{}", ll.w + first, x + first),
                &[dot],
                &[ll.w + first, x + first],
                &[],
            )?;
            if first == 1 {
                valu(
                    b,
                    &format!("v_fmac_f32_e32 v{dot}, v{}, v{x}", ll.w),
                    &[dot],
                    &[dot, ll.w, x],
                    &[],
                )?
            }
            for j in 2 - u8::from(first == 0)..nw {
                valu(
                    b,
                    &format!("v_fmac_f32_e32 v{dot}, v{}, v{}", ll.w + j, x + j),
                    &[dot],
                    &[dot, ll.w + j, x + j],
                    &[],
                )?;
            }
            let acc = ll.ac(si, chain);
            valu(
                b,
                &format!("v_add_f32_e32 v{acc}, v{acc}, v{dot}"),
                &[acc],
                &[acc, dot],
                &[],
            )?;
        }
        b.label(&cdone)?;
        b.barrier(&[Transition::Retire(slot)])?;
    }
    b.label(&gend)?;
    sop(
        b,
        format!("{} s{GI}, s{GI}, 1", s_add_i32(arch)),
        &[GI],
        &[GI],
    )?;
    b.wait_all()
}

/// Per slot fold (gate), then the snapshot shuffle-down tree interleaved over four slots at a time, then
/// the lane-0 stores. Valid waves only: a tail wave has run every barrier and has nothing to store.
fn epilogue(b: &mut Builder, spec: Spec, l: &WLay) -> R {
    let ll = &l.l;
    let exit = lbl(spec, "exit");
    row_gate(b, &exit)?;
    // Shuffle step k (offset 16 >> k): mask lane < 32 - off, address (lane + off or lane) * 4.
    let offsets = [16u32, 8, 4, 2, 1];
    for (k, off) in offsets.into_iter().enumerate() {
        let (m, sh) = (SHM + k as u8, ll.shuf + k as u8);
        op(
            b,
            format!("v_cmp_gt_u32_e64 s{m}, {}, v{}", 32 - off, l.lane),
            &[s(m)],
            &[v(l.lane)],
        )?;
        valu(
            b,
            &format!("v_cndmask_b32_e64 v{sh}, 0, {off}, s{m}"),
            &[sh],
            &[],
            &[m],
        )?;
        valu(
            b,
            &format!("v_add_lshl_u32 v{sh}, v{sh}, v{}, 2", l.lane),
            &[sh],
            &[sh, l.lane],
            &[],
        )?;
    }
    let red_done = lbl(spec, "red_done");
    for q in 0..l.cap / 4 {
        if q > 0 {
            slot_index_gate(b, 4 * q, &red_done)?
        }
        if spec.kind == Kind::GateUp {
            for j in 0..4u32 {
                let si = 4 * q + j;
                let (a0, a1, a2, a3) = (ll.ac(si, 0), ll.ac(si, 1), ll.ac(si, 2), ll.ac(si, 3));
                valu(
                    b,
                    &format!("v_add_f32_e32 v{a0}, v{a0}, v{a1}"),
                    &[a0],
                    &[a0, a1],
                    &[],
                )?;
                valu(
                    b,
                    &format!("v_add_f32_e32 v{a2}, v{a2}, v{a3}"),
                    &[a2],
                    &[a2, a3],
                    &[],
                )?;
                valu(
                    b,
                    &format!("v_add_f32_e32 v{a0}, v{a0}, v{a2}"),
                    &[a0],
                    &[a0, a2],
                    &[],
                )?;
            }
        }
        for k in 0..5u8 {
            let (m, sh) = (SHM + k, ll.shuf + k);
            for j in 0..4u32 {
                let (a0, ex) = (ll.ac(4 * q + j, 0), l.exch + j as u8);
                b.ds_crosslane(
                    Instruction::new(
                        format!("ds_bpermute_b32 v{ex}, v{sh}, v{a0}"),
                        vec![vreg(ex, 1)],
                        vec![vreg(sh, 1), vreg(a0, 1)],
                    )
                    .memory(MemoryClass::DsLoad),
                )?;
            }
            for j in 0..4u32 {
                let (a0, ex) = (ll.ac(4 * q + j, 0), l.exch + j as u8);
                // sum = acc + acc[lane + off]; lanes with lane + off > 31 (mask clear) keep acc unchanged.
                valu(
                    b,
                    &format!("v_add_f32_e32 v{ex}, v{a0}, v{ex}"),
                    &[ex],
                    &[a0, ex],
                    &[],
                )?;
                valu(
                    b,
                    &format!("v_cndmask_b32_e64 v{a0}, v{a0}, v{ex}, s{m}"),
                    &[a0],
                    &[a0, ex],
                    &[m],
                )?;
            }
        }
    }
    b.label(&red_done)?;
    // Lane 0 stores each live slot's value at its original slot's row.
    valu(
        b,
        &format!("v_cmpx_eq_u32_e32 0, v{}", l.lane),
        &[],
        &[l.lane],
        &[],
    )?;
    let st_done = lbl(spec, "st_done");
    for si in 0..l.cap {
        if si > 0 {
            slot_index_gate(b, si, &st_done)?
        }
        let tl = TL + (si % 4) as u8;
        let a0 = ll.ac(si, 0);
        op(
            b,
            format!("v_readlane_b32 s{tl}, v{}, {si}", ll.ya),
            &[s(tl)],
            &[v(ll.ya)],
        )?;
        valu(
            b,
            &format!("v_mov_b32_e32 v{}, s{tl}", ll.xa),
            &[ll.xa],
            &[],
            &[tl],
        )?;
        b.push(
            Instruction::new(
                format!("global_store_b32 v{}, v{a0}, s[{YSEL}:{}]", ll.xa, YSEL + 1),
                vec![],
                vec![vreg(ll.xa, 1), vreg(a0, 1), sreg(YSEL, 2)],
            )
            .memory(MemoryClass::VmemStore),
        )?;
    }
    b.label(&st_done)?;
    b.wait_all()?;
    b.label(&exit)?;
    branch(b, "s_endpgm")
}
