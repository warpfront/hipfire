// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Builder-emitted gfx1151 QSA block selector
//! (`indexed_attention_select_from_scores_pm_gfx1151`), with the kernarg ABI,
//! grid `[rows]`, block 256 and output bytes of the hipcc twin
//! `indexed_attention_select_from_scores` in `kernels/src/tensor_ops.hip`.
//!
//! Output per row (identical to hipcc): `visible = position_start + row + 1`,
//! `rbc = min(visible / compress, block_count)`, `chosen = min(budget, rbc)`;
//! the top `chosen` blocks by (score bits desc, block asc) fill slots
//! `k * compress + o` with `block_k * compress + o`, the causal tail
//! `[rbc * compress, visible)` follows at `chosen * compress`, every other
//! slot below `capacity` is `-1`, and `mirror` (row `rows - 1`, non-null) gets
//! the same bytes. Every slot is written exactly once, computed per slot (no
//! `-1` fill pass and no ordering between waves), through raw buffer
//! descriptors whose `num_records` drop the slots past `capacity` (and every
//! mirror store when the mirror is inactive).
//!
//! Scores are non-negative, non-NaN, never -0 floats, so their u32 bits order
//! like the values, and a key fits 31 bits.
//!
//! Selection (atomic-free, exact). A wave owns a contiguous run of 32-key
//! packets (`ceil(ceil(rbc / 32) / 8)` per wave) read with raw buffer loads
//! (lanes past `rbc` read 0 and are masked). Radix passes over 5-bit digit
//! windows `[31:27] .. [6:2]` and `[4:0]` (shifts 27, 22, 17, 12, 7, 2, 0),
//! each over the keys whose already-fixed high bits equal `prefix`:
//! - Every lane keeps a private bit-sliced counter per bin: the packet's
//!   digit becomes the one-hot word `1 << digit` and ripples through the
//!   plane words (`plane ^= carry; carry &= plane`), so no wave-private LDS
//!   read-modify-write (which the typed LDS core forbids), no ballots and no
//!   popcounts are needed per key. The plane count is picked per row by the
//!   packets a wave sees: 5 planes up to 31, 7 up to 127, 9 up to 511 (a
//!   lane counts at most one key per packet; 9 planes serve block_count up
//!   to 130816, the 65536 design point sees 256). Packets with no matching
//!   key skip their plane work.
//! - After the loop the 32 lanes' planes are summed by a five-level butterfly
//!   of ripple-carry adders (`v_mov_b32_dpp row_xmask` / `v_permlanex16_b32`
//!   exchange, carry = `bfi(a ^ b, carry, a)`), giving `planes + 5` identical plane
//!   words per lane, and lane `b` extracts bin `b`'s wave count
//!   (`bfe(plane_p, lane, 1) << p`). The 8 wave counts go through LDS (one
//!   `ds_store_b32` per lane, a barrier, four `ds_load_2addr_b32`), reversed
//!   so lane `L` holds bin `31 - L`.
//! - Every wave redundantly runs the descending inclusive scan (xor-butterfly
//!   prefix over `row_xmask` / `v_permlanex16_b32`), finds the lane where the
//!   cumulative count first reaches `need` (`v_cmp` ballot + `s_ctz_i32_b32`),
//!   reads that lane's bin count and cumulative count (`v_readlane_b32`), and
//!   updates `above += count above the digit`, `need -= count above`,
//!   `prefix |= digit << shift`. The workgroup-uniform state lives in SGPRs.
//!   A pass stops the refinement when `above + boundary <= 1024` (candidates
//!   are every key with processed high bits above `prefix` plus the whole
//!   boundary bucket) and, after the last window, takes the first `need`
//!   keys equal to `prefix` in ascending block order. An overflowing bucket
//!   is never dropped. `chosen == rbc` (or `chosen == 0`) skips the passes.
//!
//! Compaction (deterministic, ascending block order): each wave counts, with
//! `s_bcnt1_i32_b32` over its packets' class masks, the keys strictly above
//! the class prefix (`a`: `key & mask > prefix`) and in the boundary bucket
//! (`e`: `key & mask == prefix`), publishes `(a, e)` in 64 bytes of LDS, and
//! after one barrier every wave derives the exclusive scans and totals in
//! SGPRs. The fill pass places above keys at `tile[exA + rank]` and boundary
//! keys at `tile[A + exE + rank]` while `exE + rank < limit` (`limit` is
//! infinite after an early finish and for `chosen == rbc`, where mask and
//! prefix 0 make every key a boundary key; `need` after the last window; 0
//! when `chosen == 0`), ranks
//! from `v_mbcnt_lo_u32_b32` of the packet's class masks, as 64-bit entries
//! `(key << 32) | ~block` in the 8192-byte tile (at most 1024 entries).
//!
//! Sort: the tile entries are loaded into registers (entry `256 e + t` of
//! thread `t`, `e < 4`, zero past `n`, which sorts after every valid entry
//! because a valid entry's low word is `~block >= 0xffff0000`). A bitonic
//! network over the 64-bit values (signed 64-bit compare: the key sign bit is
//! 0) orders them descending, i.e. (key desc, block asc), with in-register
//! stages for strides 256 and 512, `v_mov_b32_dpp row_xmask` /
//! `v_permlanex16_b32` stages for strides below 32 and LDS exchanges (store,
//! barrier, load partner, barrier) for strides 32, 64 and 128. The stage list
//! for `n <= 256` stops after `k = 256`, for `n <= 512` after `k = 512`;
//! slices past `n` skip their compare-exchanges. Sorted entries `< 512` give
//! block ids `~lo` in a 2048-byte rank table.
//!
//! Output: slot `s = tid + 256 i` (four slots per thread per trip, one
//! `s_waitcnt_vscnt` per trip): ranked slots read `block = ranks[s / compress]`
//! and store `block * compress + s % compress`, the tail stores
//! `rbc * compress + (s - chosen * compress)` while below `visible`, else
//! `-1`. `s / compress` is the LLVM unsigned division (refined reciprocal,
//! `v_mul_hi_u32`, two remainder corrections), exact for every u32.
//!
//! Resources: wave32, workgroup 256, static LDS 10304 bytes = 8192 (tile:
//! wave counts, entries, sort exchange; phase-shared) + 2048 (rank table) +
//! 64 (per-wave `(a, e)` totals), no dynamic LDS, zero private segment, no
//! atomics. Every barrier is workgroup-uniform; the host guarantees gfx1151,
//! `budget_blocks <= 512`, buffer extents below `0x7fff_ff00` bytes and
//! `block_count <= 65536`.
use super::common::{lit, mem, op, s, smem, sop, sr, srd_tail, v, vr};
use crate::{
    insn::{Instruction, MemoryClass},
    ledger::Counter,
    reg::{Kind, Live, RegRef},
    Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan,
};
use peacemaker_author::{
    ready, retire, Carried, End, Free, Gfx1151, LdsRegion, Published, Scc, Uniform, Wave,
    WgUniform, Workgroup,
};
use std::cell::Cell;

type Wg<'b> = Workgroup<'b, Gfx1151, Builder>;
type W<'a> = Wave<'a, Gfx1151, Builder>;
type R = Result<(), String>;

pub fn symbol(arch: Arch) -> String {
    format!("indexed_attention_select_from_scores_pm_{}", arch.name())
}

// ---- LDS layout -------------------------------------------------------------
const H_BASE: u32 = 0;
const H_BYTES: u32 = 8192;
const R_BASE: u32 = 8192;
const R_BYTES: u32 = 2048;
const C_BASE: u32 = 10240;
const C_BYTES: u32 = 64;
pub const STATIC_LDS: u32 = 10304;

/// LDS region tags.
pub enum Tile {}
pub enum Ranks {}
pub enum Ctl {}

// ---- SGPRs ------------------------------------------------------------------
const KARG: u8 = 0;
const WGX: u8 = 2;
// kernarg blob: scores 8:9, stride 10, selected 12:13, rows 14, block_count 15,
// budget 16, compress 17, position_start 18, capacity 19, mirror 20:21.
const A0: u8 = 8;
const A1: u8 = 16;
const SC_PTR: u8 = 8;
const STRIDE: u8 = 10;
const SEL_PTR: u8 = 12;
const ROWS: u8 = 14;
const BCOUNT: u8 = 15;
const BUDGET: u8 = 16;
const COMP: u8 = 17;
const POS: u8 = 18;
const CAP: u8 = 19;
const MIR_PTR: u8 = 20;
const ROW: u8 = 24;
const VIS: u8 = 25;
const RBC: u8 = 26;
const CHOSEN: u8 = 27;
const TSTART: u8 = 28;
const TLEN: u8 = 29;
const NP: u8 = 30;
const PW: u8 = 31;
const WAVE: u8 = 32;
const P0: u8 = 33;
const P1: u8 = 34;
const RANKED: u8 = 35;
const TRIPS: u8 = 36;
const TRIP: u8 = 37;
const NCAND: u8 = 38;
const RMAG: u8 = 39;
const SRD_SC: u8 = 40;
const SRD_SEL: u8 = 44;
const SRD_MIR: u8 = 48;
const PREFIX: u8 = 52;
const NEED: u8 = 53;
const ABOVE: u8 = 54;
const MASKM: u8 = 55;
const SHIFT: u8 = 56;
const DONE: u8 = 57;
const LIMIT: u8 = 58;
const BD: u8 = 59;
const EXA: u8 = 60;
const EXE: u8 = 61;
const ATOT: u8 = 62;
const ETOT: u8 = 63;
const T0: u8 = 64;
const T1: u8 = 65;
const T2: u8 = 66;
const T3: u8 = 67;
const T4: u8 = 68;
const T5: u8 = 69;
const T6: u8 = 70;
const T7: u8 = 71;
/// Lane-mask pairs (the high word is never written in wave32; M7 reads VOP3 masks as pairs).
const MP0: u8 = 72;
const MP1: u8 = 74;
const MP2: u8 = 76;
const MP3: u8 = 78;
/// Lane masks with lane bit `k` set (the high word is zero).
const LB: [u8; 5] = [80, 82, 84, 86, 88];
const PSEL: u8 = 90;
const ARUN: u8 = 92;
const ERUN: u8 = 93;
const X1: u8 = 94;
const X2: u8 = 95;
/// Wave-bit masks of the sort (the totals' scratch, dead by then).
const WB: [u8; 3] = [60, 61, 62];
/// Low lane-select words of `v_permlane16_b32` for the in-row xor strides 1, 2, 4, 8.
const XSEL: [u8; 4] = [96, 97, 98, 99];
const S_END: u8 = 100;

// ---- VGPRs ------------------------------------------------------------------
const TID: u8 = 0;
const LANE: u8 = 1;
const LANE4: u8 = 2;
const TID8: u8 = 3;
const VWST: u8 = 4;
const VRD: u8 = 5;
const VCA: u8 = 6;
const VRA: u8 = 7;
const PL: u8 = 8; // 14 plane words 8..21
const KEY: u8 = 25; // packet u: lo 24 + 2u, key 25 + 2u
const VT: u8 = 32;
const VC0: u8 = 33;
const VC1: u8 = 34;
const VD: u8 = 35;
const VOH: u8 = 36;
const VB: u8 = 37;
const CW: u8 = 38;
const VCB: u8 = 39;
const CNT: u8 = 40;
const VP: u8 = 41;
const VTOT: u8 = 42;
const VX: u8 = 44; // 44..51 wave counts
const VPA: u8 = 52;
const VPE: u8 = 53;
const VPOS: u8 = 54;
const VAD: u8 = 55;
const CTL: u8 = 56; // 56..71
const ENT: u8 = 72; // slice e: 72 + 2e
const PART: u8 = 80; // slice e: 80 + 2e
const VI: u8 = 88;
const VPAD: u8 = 89;
const STMP: u8 = 90;
const VS: u8 = 96;
const VQ: u8 = 97;
const VRM: u8 = 98;
const VT3: u8 = 99;
const VRAD: u8 = 100;
const VBLK: u8 = 101;
const VV: u8 = 102;
const VJ: u8 = 103;
const VOFF: u8 = 104; // 104..107
const VVAL: u8 = 108; // 108..111
const V_END: u8 = 112;

/// Plane counts of the per-lane bit-sliced counters (`<= 2^planes - 1` packets per wave),
/// with the exclusive lower and inclusive upper bound of the per-wave packet count each serves.
/// Named `HIST_VARIANTS`, not `VARIANTS`: `const VARIANTS` is reserved for `.rip` variant discovery.
const HIST_VARIANTS: [(u8, Option<u32>, Option<u32>); 3] = [
    (5, None, Some(31)),
    (7, Some(31), Some(127)),
    (9, Some(127), None),
];
const SHIFT0: u32 = 27;
const EARLY: u32 = 1024;

fn lo(u: u8) -> u8 {
    24 + 2 * u
}
fn key(u: u8) -> u8 {
    KEY + 2 * u
}
fn ent(e: u32) -> u8 {
    ENT + 2 * e as u8
}
fn part(e: u32) -> u8 {
    PART + 2 * e as u8
}

#[derive(Default)]
struct Gen {
    n: Cell<u32>,
}
impl Gen {
    fn lbl(&self, base: &str) -> String {
        let n = self.n.get();
        self.n.set(n + 1);
        format!(".Lqsa_topk_{base}{n}")
    }
}

pub fn emit(arch: Arch) -> Result<Emitted, String> {
    if arch != Arch::Gfx1151 {
        return Err("qsa_topk: built for gfx1151 only".into());
    }
    let kernargs = KernargLayout::new(56)
        .pointer("scores", 0)
        .hidden("score_stride", 8, 4, "by_value")
        .pointer("selected", 16)
        .hidden("rows", 24, 4, "by_value")
        .hidden("block_count", 28, 4, "by_value")
        .hidden("budget_blocks", 32, 4, "by_value")
        .hidden("compress", 36, 4, "by_value")
        .hidden("position_start", 40, 4, "by_value")
        .hidden("capacity", 44, 4, "by_value")
        .pointer("mirror", 48);
    let kspec = KernelSpec {
        kernel_id: "qsa_select".into(),
        variant: "select".into(),
        arch,
        symbol: symbol(arch),
        kernargs,
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: 256,
        group_segment_fixed_size: STATIC_LDS,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan()?);
    b.enable_delay_alu();
    let g = Gen::default();
    kernel(&mut Workgroup::<Gfx1151, Builder>::new(&mut b)?, &g)?;
    b.finish()
}

fn plan() -> Result<RegPlan, String> {
    let mut p = RegPlan::new(128, 104)?;
    p.s::<2>("kernarg_ptr", KARG, Live::Whole)?;
    p.s::<1>("wg_x_in", WGX, Live::Whole)?;
    p.s::<8>("kernargs_0x00", A0, Live::Whole)?;
    p.s::<8>("kernargs_0x20", A1, Live::Whole)?;
    for base in (24u8..S_END).step_by(8) {
        p.add_range(&format!("s_blk{base}"), Kind::S, base, 8, Live::Whole)?;
    }
    for base in (0u8..V_END).step_by(8) {
        p.add_range(&format!("v_blk{base}"), Kind::V, base, 8, Live::Whole)?;
    }
    Ok(p)
}

// ---- small emission helpers -------------------------------------------------
fn mask(m: u8) -> RegRef {
    sr(m, 2)
}
fn vp(base: u8) -> String {
    format!("v[{}:{}]", base, base + 1)
}
fn cmp_wg(wg: &mut Wg, text: String, uses: &[RegRef]) -> Result<WgUniform<Scc>, String> {
    wg.scmp_wg_uniform(Instruction::new(text, vec![], uses.to_vec()))
}
fn cmp_w(w: &mut W, text: String, uses: &[RegRef]) -> Result<Uniform<Scc>, String> {
    w.scmp(Instruction::new(text, vec![], uses.to_vec()))
}
fn ds_ld(text: String, dst: RegRef, addr: u8) -> Instruction {
    Instruction::new(text, vec![dst], vec![v(addr)]).memory(MemoryClass::DsLoad)
}
fn ds_st(text: String, addr: u8, data: RegRef) -> Instruction {
    Instruction::new(text, vec![], vec![v(addr), data]).memory(MemoryClass::DsStore)
}
fn off(o: u32) -> String {
    if o == 0 {
        String::new()
    } else {
        format!(" offset:{o}")
    }
}
/// `v_permlane16_b32` lane-select word half (`hi`: lanes 8..15 of a row) for lane `l ^ m`.
fn xsel(m: u32, hi: bool) -> u32 {
    (0..8u32)
        .map(|i| ((i + 8 * u32::from(hi)) ^ m) << (4 * i))
        .sum()
}
/// `dst = src` of lane `l ^ m` for `m < 16` (inside a row of 16). A VOP3
/// VALU, unlike DPP, so every `s_delay_alu` window around it resolves.
fn xlane(b: &mut Builder, dst: u8, src: u8, m: u32) -> R {
    let k = m.trailing_zeros() as usize;
    op(
        b,
        format!(
            "v_permlane16_b32 v{dst}, v{src}, s{}, {}",
            XSEL[k],
            lit(xsel(m, true))
        ),
        &[v(dst)],
        &[v(src), s(XSEL[k])],
    )
}

/// `s[dst:dst+1] = s[src:src+1] + x * y` (64-bit).
fn add64_mul(b: &mut Builder, dst: u8, src: u8, x: u8, y: u8, lo: u8, hi: u8) -> R {
    sop(b, format!("s_mul_i32 s{lo}, s{x}, s{y}"), &[lo], &[x, y])?;
    sop(b, format!("s_mul_hi_u32 s{hi}, s{x}, s{y}"), &[hi], &[x, y])?;
    sop(
        b,
        format!("s_add_u32 s{dst}, s{src}, s{lo}"),
        &[dst],
        &[src, lo],
    )?;
    sop(
        b,
        format!("s_addc_u32 s{}, s{}, s{hi}", dst + 1, src + 1),
        &[dst + 1],
        &[src + 1, hi],
    )
}
/// The LLVM unsigned-division reciprocal over a uniform divisor `d`: `s{rmag}`
/// is the refined `2^32 / d` estimate (`vt` holds three scratch VGPRs).
fn magic(b: &mut Builder, rmag: u8, d: u8, vt: [u8; 3], neg: u8) -> R {
    let [f, r, h] = vt;
    op(b, format!("v_cvt_f32_u32_e32 v{f}, s{d}"), &[v(f)], &[s(d)])?;
    op(
        b,
        format!("v_rcp_iflag_f32_e32 v{f}, v{f}"),
        &[v(f)],
        &[v(f)],
    )?;
    op(
        b,
        format!("v_mul_f32_e32 v{f}, 0x4f7ffffe, v{f}"),
        &[v(f)],
        &[v(f)],
    )?;
    op(b, format!("v_cvt_u32_f32_e32 v{r}, v{f}"), &[v(r)], &[v(f)])?;
    sop(b, format!("s_sub_i32 s{neg}, 0, s{d}"), &[neg], &[d])?;
    op(
        b,
        format!("v_mul_lo_u32 v{h}, s{neg}, v{r}"),
        &[v(h)],
        &[s(neg), v(r)],
    )?;
    op(
        b,
        format!("v_mul_hi_u32 v{h}, v{r}, v{h}"),
        &[v(h)],
        &[v(r), v(h)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{r}, v{r}, v{h}"),
        &[v(r)],
        &[v(r), v(h)],
    )?;
    op(
        b,
        format!("v_readfirstlane_b32 s{rmag}, v{r}"),
        &[s(rmag)],
        &[v(r)],
    )
}
/// Exact `q = n / d` over unsigned 32-bit SGPRs (`d != 0`), as the gather's `udiv`.
fn udiv(b: &mut Builder, q: u8, n: u8, d: u8, vt: [u8; 3], t: [u8; 3]) -> R {
    let [neg, rem, alt] = t;
    magic(b, q, d, vt, neg)?;
    let h = vt[2];
    op(b, format!("v_mov_b32_e32 v{h}, s{q}"), &[v(h)], &[s(q)])?;
    op(
        b,
        format!("v_mul_hi_u32 v{h}, s{n}, v{h}"),
        &[v(h)],
        &[s(n), v(h)],
    )?;
    op(
        b,
        format!("v_readfirstlane_b32 s{q}, v{h}"),
        &[s(q)],
        &[v(h)],
    )?;
    for _ in 0..2 {
        sop(b, format!("s_mul_i32 s{rem}, s{q}, s{d}"), &[rem], &[q, d])?;
        sop(
            b,
            format!("s_sub_i32 s{rem}, s{n}, s{rem}"),
            &[rem],
            &[n, rem],
        )?;
        sop(b, format!("s_add_i32 s{alt}, s{q}, 1"), &[alt], &[q])?;
        sop(b, format!("s_cmp_ge_u32 s{rem}, s{d}"), &[], &[rem, d])?;
        sop(
            b,
            format!("s_cselect_b32 s{q}, s{alt}, s{q}"),
            &[q],
            &[alt, q],
        )?;
    }
    Ok(())
}

// ---- kernel -----------------------------------------------------------------
fn kernel(wg: &mut Wg, g: &Gen) -> R {
    let tile = wg.lds::<Tile>("tile", H_BASE, H_BYTES)?;
    let ctl = wg.lds::<Ctl>("ctl", C_BASE, C_BYTES)?;
    let ranks = wg.lds::<Ranks>("ranks", R_BASE, R_BYTES)?;
    let end = wg.exit(".Lqsa_topk_end")?;
    prologue(wg, &end)?;
    let tile = refine(wg, g, tile)?;
    let ctl = count_pass(wg, g, ctl)?;
    let (tile, ctl) = fill_pass(wg, g, tile, ctl)?;
    let tile = sort(wg, g, tile)?;
    let ranks = write_ranks(wg, ranks)?;
    output(wg, g, &ranks)?;
    let _ = (tile, ctl);
    wg.end(end)
}

fn prologue(wg: &mut Wg, end: &End) -> R {
    let b = wg.isa();
    smem(b, A0, 8, KARG, 0)?;
    smem(b, A1, 8, KARG, 0x20)?;
    sop(b, format!("s_mov_b32 s{ROW}, s{WGX}"), &[ROW], &[WGX])?;
    // Lane constants.
    op(
        b,
        format!("v_and_b32_e32 v{LANE}, 31, v{TID}"),
        &[v(LANE)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{LANE4}, 2, v{LANE}"),
        &[v(LANE4)],
        &[v(LANE)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{TID8}, 3, v{TID}"),
        &[v(TID8)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_lshrrev_b32_e32 v{VT}, 5, v{TID}"),
        &[v(VT)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_readfirstlane_b32 s{WAVE}, v{VT}"),
        &[s(WAVE)],
        &[v(VT)],
    )?;
    // Wave-count store address: wave * 128 + lane * 4.
    op(
        b,
        format!("v_lshl_add_u32 v{VWST}, s{WAVE}, 7, v{LANE4}"),
        &[v(VWST)],
        &[s(WAVE), v(LANE4)],
    )?;
    // Wave-count load address: bin 31 - lane.
    op(
        b,
        format!("v_sub_nc_u32_e32 v{VRD}, 31, v{LANE}"),
        &[v(VRD)],
        &[v(LANE)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{VRD}, 2, v{VRD}"),
        &[v(VRD)],
        &[v(VRD)],
    )?;
    // (a, e) store address, rank table store address and the broadcast address.
    op(
        b,
        format!("v_lshl_add_u32 v{VCA}, s{WAVE}, 3, {}", lit(C_BASE)),
        &[v(VCA)],
        &[s(WAVE)],
    )?;
    op(
        b,
        format!("v_lshl_add_u32 v{VRA}, v{TID}, 2, {}", lit(R_BASE)),
        &[v(VRA)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_mov_b32_e32 v{VCB}, {}", lit(C_BASE)),
        &[v(VCB)],
        &[],
    )?;
    // Lane-bit masks and the lane-select word of v_permlanex16_b32.
    for (k, m) in [
        0xaaaa_aaaau32,
        0xcccc_cccc,
        0xf0f0_f0f0,
        0xff00_ff00,
        0xffff_0000,
    ]
    .into_iter()
    .enumerate()
    {
        sop(
            b,
            format!("s_mov_b32 s{}, {}", LB[k], lit(m)),
            &[LB[k]],
            &[],
        )?;
        sop(b, format!("s_mov_b32 s{}, 0", LB[k] + 1), &[LB[k] + 1], &[])?;
    }
    sop(b, format!("s_mov_b32 s{PSEL}, 0x76543210"), &[PSEL], &[])?;
    for (k, m) in [1u32, 2, 4, 8].into_iter().enumerate() {
        sop(
            b,
            format!("s_mov_b32 s{}, {}", XSEL[k], lit(xsel(m, false))),
            &[XSEL[k]],
            &[],
        )?;
    }
    for m in [MP0, MP1, MP2, MP3] {
        sop(b, format!("s_mov_b32 s{}, 0", m + 1), &[m + 1], &[])?;
    }
    let outside = cmp_wg(
        wg,
        format!("s_cmp_ge_i32 s{ROW}, s{ROWS}"),
        &[s(ROW), s(ROWS)],
    )?;
    wg.exit_if(outside, end)?;
    let b = wg.isa();
    // Selection length as the hipcc kernel derives it.
    sop(
        b,
        format!("s_add_i32 s{VIS}, s{POS}, s{ROW}"),
        &[VIS],
        &[POS, ROW],
    )?;
    sop(b, format!("s_add_i32 s{VIS}, s{VIS}, 1"), &[VIS], &[VIS])?;
    udiv(b, T0, VIS, COMP, [VT, VC0, VC1], [T1, T2, T3])?;
    sop(
        b,
        format!("s_min_i32 s{RBC}, s{T0}, s{BCOUNT}"),
        &[RBC],
        &[T0, BCOUNT],
    )?;
    sop(
        b,
        format!("s_min_i32 s{CHOSEN}, s{BUDGET}, s{RBC}"),
        &[CHOSEN],
        &[BUDGET, RBC],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{TSTART}, s{RBC}, s{COMP}"),
        &[TSTART],
        &[RBC, COMP],
    )?;
    sop(
        b,
        format!("s_sub_i32 s{TLEN}, s{VIS}, s{TSTART}"),
        &[TLEN],
        &[VIS, TSTART],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{RANKED}, s{CHOSEN}, s{COMP}"),
        &[RANKED],
        &[CHOSEN, COMP],
    )?;
    // Packets: np = ceil(rbc / 32), per wave ceil(np / 8), a wave's run [p0, p1).
    sop(b, format!("s_add_i32 s{NP}, s{RBC}, 31"), &[NP], &[RBC])?;
    sop(b, format!("s_lshr_b32 s{NP}, s{NP}, 5"), &[NP], &[NP])?;
    sop(b, format!("s_add_i32 s{PW}, s{NP}, 7"), &[PW], &[NP])?;
    sop(b, format!("s_lshr_b32 s{PW}, s{PW}, 3"), &[PW], &[PW])?;
    sop(
        b,
        format!("s_mul_i32 s{P0}, s{WAVE}, s{PW}"),
        &[P0],
        &[WAVE, PW],
    )?;
    sop(
        b,
        format!("s_add_i32 s{P1}, s{P0}, s{PW}"),
        &[P1],
        &[P0, PW],
    )?;
    sop(
        b,
        format!("s_min_u32 s{P1}, s{P1}, s{NP}"),
        &[P1],
        &[P1, NP],
    )?;
    sop(b, format!("s_add_i32 s{TRIPS}, s{PW}, 3"), &[TRIPS], &[PW])?;
    sop(
        b,
        format!("s_lshr_b32 s{TRIPS}, s{TRIPS}, 2"),
        &[TRIPS],
        &[TRIPS],
    )?;
    // Descriptors: scores row, selected row, mirror (records 0 when inactive).
    sop(
        b,
        format!("s_lshl_b32 s{T0}, s{STRIDE}, 2"),
        &[T0],
        &[STRIDE],
    )?;
    add64_mul(b, SRD_SC, SC_PTR, ROW, T0, T1, T2)?;
    sop(b, format!("s_lshl_b32 s{T0}, s{RBC}, 2"), &[T0], &[RBC])?;
    srd_tail(b, SRD_SC, Some(T0))?;
    sop(b, format!("s_lshl_b32 s{T0}, s{CAP}, 2"), &[T0], &[CAP])?;
    add64_mul(b, SRD_SEL, SEL_PTR, ROW, T0, T1, T2)?;
    srd_tail(b, SRD_SEL, Some(T0))?;
    sop(b, format!("s_sub_i32 s{T1}, s{ROWS}, 1"), &[T1], &[ROWS])?;
    sop(b, format!("s_cmp_eq_u32 s{ROW}, s{T1}"), &[], &[ROW, T1])?;
    sop(b, format!("s_cselect_b32 s{T1}, s{T0}, 0"), &[T1], &[T0])?;
    op(
        b,
        format!("s_cmp_lg_u64 s[{MIR_PTR}:{}], 0", MIR_PTR + 1),
        &[],
        &[sr(MIR_PTR, 2)],
    )?;
    sop(b, format!("s_cselect_b32 s{T1}, s{T1}, 0"), &[T1], &[T1])?;
    sop(
        b,
        format!("s_mov_b32 s{SRD_MIR}, s{MIR_PTR}"),
        &[SRD_MIR],
        &[MIR_PTR],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{}", SRD_MIR + 1, MIR_PTR + 1),
        &[SRD_MIR + 1],
        &[MIR_PTR + 1],
    )?;
    srd_tail(b, SRD_MIR, Some(T1))?;
    // Refinement state: no digit fixed; every key is a candidate class member
    // unless chosen == 0 (limit 0 keeps the tile empty).
    sop(b, format!("s_mov_b32 s{PREFIX}, 0"), &[PREFIX], &[])?;
    sop(
        b,
        format!("s_mov_b32 s{NEED}, s{CHOSEN}"),
        &[NEED],
        &[CHOSEN],
    )?;
    sop(b, format!("s_mov_b32 s{ABOVE}, 0"), &[ABOVE], &[])?;
    sop(b, format!("s_mov_b32 s{MASKM}, 0"), &[MASKM], &[])?;
    sop(b, format!("s_mov_b32 s{SHIFT}, {SHIFT0}"), &[SHIFT], &[])?;
    sop(b, format!("s_mov_b32 s{DONE}, 0"), &[DONE], &[])?;
    sop(b, format!("s_cmp_eq_u32 s{CHOSEN}, 0"), &[], &[CHOSEN])?;
    sop(b, format!("s_cselect_b32 s{LIMIT}, 0, -1"), &[LIMIT], &[])
}

// ---- packet loops -----------------------------------------------------------
/// The four key loads of one trip: packets `q .. q + 3` with `q = p0 + 4 * trip`
/// (lanes past `rbc`, and packets of other waves, are masked by the callers).
fn issue_loads(b: &mut Builder) -> R {
    sop(b, format!("s_lshl_b32 s{T0}, s{TRIP}, 2"), &[T0], &[TRIP])?;
    sop(
        b,
        format!("s_add_i32 s{T0}, s{T0}, s{P0}"),
        &[T0],
        &[T0, P0],
    )?;
    for u in 0..4u8 {
        sop(b, format!("s_add_i32 s{T1}, s{T0}, {u}"), &[T1], &[T0])?;
        sop(b, format!("s_lshl_b32 s{T1}, s{T1}, 7"), &[T1], &[T1])?;
        mem(
            b,
            format!(
                "buffer_load_b32 v{}, v{LANE4}, s[{SRD_SC}:{}], s{T1} offen",
                key(u),
                SRD_SC + 3
            ),
            &[v(key(u))],
            &[v(LANE4), sr(SRD_SC, 4), s(T1)],
            MemoryClass::VmemLoad,
        )?;
    }
    b.wait(Counter::Vm, 0)
}

/// A workgroup-uniform loop over this wave's packets, four per trip: every
/// wave runs `ceil(per-wave / 4)` trips and skips the packets past its run.
/// `per_packet` sees `T0 = q`, `T1 = q + u` and the loaded keys.
fn packet_loop<S: Carried>(
    wg: &mut Wg,
    g: &Gen,
    head: &str,
    st: S,
    drain_ds: bool,
    per_packet: impl Fn(&mut W, S, u8) -> Result<S, String>,
) -> Result<S, String> {
    sop(wg.isa(), format!("s_mov_b32 s{TRIP}, 0"), &[TRIP], &[])?;
    wg.loop_carried(&g.lbl(head), st, |wg, mut st| {
        issue_loads(wg.isa())?;
        for u in 0..4u8 {
            sop(
                wg.isa(),
                format!("s_add_i32 s{T1}, s{T0}, {u}"),
                &[T1],
                &[T0],
            )?;
            let past = cmp_w(wg, format!("s_cmp_ge_u32 s{T1}, s{P1}"), &[s(T1), s(P1)])?;
            st = wg.skip_if(past, &g.lbl("pk"), st, |w, st| per_packet(w, st, u))?;
        }
        let b = wg.isa();
        if drain_ds {
            b.wait(Counter::Lgkm, 0)?;
        }
        sop(
            b,
            format!("s_add_i32 s{TRIP}, s{TRIP}, 1"),
            &[TRIP],
            &[TRIP],
        )?;
        let more = cmp_wg(
            wg,
            format!("s_cmp_lt_u32 s{TRIP}, s{TRIPS}"),
            &[s(TRIP), s(TRIPS)],
        )?;
        Ok((st, more))
    })
}

/// Lane masks of packet `q + u` (`T1`): MP0 = lanes inside the row, MP1 = keys
/// strictly above the class prefix, MP2 = keys in the boundary bucket (both
/// restricted to MP0). Leaves `T2 = first key index of the packet`.
fn class_masks(b: &mut Builder, u: u8) -> R {
    sop(b, format!("s_lshl_b32 s{T2}, s{T1}, 5"), &[T2], &[T1])?;
    sop(
        b,
        format!("s_sub_u32 s{T3}, s{RBC}, s{T2}"),
        &[T3],
        &[RBC, T2],
    )?;
    op(
        b,
        format!("v_cmp_gt_u32_e64 s{MP0}, s{T3}, v{LANE}"),
        &[mask(MP0)],
        &[s(T3), v(LANE)],
    )?;
    op(
        b,
        format!("v_and_b32_e32 v{VT}, s{MASKM}, v{}", key(u)),
        &[v(VT)],
        &[s(MASKM), v(key(u))],
    )?;
    op(
        b,
        format!("v_cmp_gt_u32_e64 s{MP1}, v{VT}, s{PREFIX}"),
        &[mask(MP1)],
        &[v(VT), s(PREFIX)],
    )?;
    op(
        b,
        format!("v_cmp_eq_u32_e64 s{MP2}, s{PREFIX}, v{VT}"),
        &[mask(MP2)],
        &[s(PREFIX), v(VT)],
    )?;
    op(
        b,
        format!("s_and_b32 s{MP1}, s{MP1}, s{MP0}"),
        &[mask(MP1)],
        &[mask(MP1), mask(MP0)],
    )?;
    op(
        b,
        format!("s_and_b32 s{MP2}, s{MP2}, s{MP0}"),
        &[mask(MP2)],
        &[mask(MP2), mask(MP0)],
    )
}

// ---- refinement -------------------------------------------------------------
fn refine(
    wg: &mut Wg,
    g: &Gen,
    tile: LdsRegion<Tile, Free>,
) -> Result<LdsRegion<Tile, Free>, String> {
    let b = wg.isa();
    // Skip when chosen * (rbc - chosen) == 0: chosen == rbc keeps every key,
    // chosen == 0 keeps none.
    sop(
        b,
        format!("s_sub_i32 s{T0}, s{RBC}, s{CHOSEN}"),
        &[T0],
        &[RBC, CHOSEN],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{T0}, s{T0}, s{CHOSEN}"),
        &[T0],
        &[T0, CHOSEN],
    )?;
    let skip = cmp_wg(wg, format!("s_cmp_eq_u32 s{T0}, 0"), &[s(T0)])?;
    wg.wg_skip_if(skip, &g.lbl("refine_done"), tile, |wg, tile| {
        let tile = wg.loop_carried(&g.lbl("refine"), tile, |wg, tile| pass(wg, g, tile))?;
        // Limit of the equal-key class: unbounded after an early finish, the
        // remaining `need` after the last window.
        let b = wg.isa();
        sop(b, format!("s_cmp_eq_u32 s{DONE}, 0"), &[], &[DONE])?;
        sop(
            b,
            format!("s_cselect_b32 s{LIMIT}, s{NEED}, -1"),
            &[LIMIT],
            &[NEED],
        )?;
        Ok(tile)
    })
}

/// Run `body` only when `lower < per-wave packets <= upper` (a missing bound is open).
fn gated(
    wg: &mut Wg,
    g: &Gen,
    lower: Option<u32>,
    upper: Option<u32>,
    body: impl FnOnce(&mut Wg) -> R,
) -> R {
    match (lower, upper) {
        (Some(lo), _) => {
            let skip = cmp_wg(wg, format!("s_cmp_le_u32 s{PW}, {}", lit(lo)), &[s(PW)])?;
            wg.wg_skip_if(skip, &g.lbl("pv"), (), |wg, ()| {
                gated(wg, g, None, upper, body)
            })
        }
        (None, Some(hi)) => {
            let skip = cmp_wg(wg, format!("s_cmp_gt_u32 s{PW}, {}", lit(hi)), &[s(PW)])?;
            wg.wg_skip_if(skip, &g.lbl("pv"), (), |wg, ()| body(wg))
        }
        (None, None) => body(wg),
    }
}

/// The wave's count of bin `lane` in `CW`: zero the planes, count the packets,
/// sum the lanes and extract.
fn hist_variant(wg: &mut Wg, g: &Gen, planes: u8) -> R {
    let b = wg.isa();
    for p in 0..planes {
        op(
            b,
            format!("v_mov_b32_e32 v{}, 0", PL + p),
            &[v(PL + p)],
            &[],
        )?;
    }
    packet_loop(wg, g, "hist", (), false, |w, (), u| {
        hist_packet(w, g, u, planes)
    })?;
    let b = wg.isa();
    reduce_planes(b, planes)?;
    // Bin `lane` of this wave: sum over planes of ((plane >> lane) & 1) << p.
    op(
        b,
        format!("v_bfe_u32 v{CW}, v{PL}, v{LANE}, 1"),
        &[v(CW)],
        &[v(PL), v(LANE)],
    )?;
    for p in 1..planes + 5 {
        op(
            b,
            format!("v_bfe_u32 v{VT}, v{}, v{LANE}, 1", PL + p),
            &[v(VT)],
            &[v(PL + p), v(LANE)],
        )?;
        op(
            b,
            format!("v_lshl_add_u32 v{CW}, v{VT}, {p}, v{CW}"),
            &[v(CW)],
            &[v(VT), v(CW)],
        )?;
    }
    Ok(())
}

/// One radix pass over the 5-bit window at `SHIFT`.
fn pass(
    wg: &mut Wg,
    g: &Gen,
    tile: LdsRegion<Tile, Free>,
) -> Result<(LdsRegion<Tile, Free>, WgUniform<Scc>), String> {
    op(wg.isa(), format!("v_mov_b32_e32 v{CW}, 0"), &[v(CW)], &[])?;
    // The plane count is picked by the wave's packet count: fewer planes, shorter ripples.
    for (planes, lower, upper) in HIST_VARIANTS {
        gated(wg, g, lower, upper, |wg| hist_variant(wg, g, planes))?;
    }
    // Publish the eight wave counts, then read bin `31 - lane` of every wave.
    let st = wg.begin_write(tile);
    let (tw, pend) = wg.ds_store(
        st,
        ds_st(format!("ds_store_b32 v{VWST}, v{CW}"), VWST, v(CW)),
    )?;
    let d = wg.wait(pend)?;
    let (tp,) = wg.barrier((ready(tw, d),))?;
    for k in 0..4u32 {
        let o0 = 64 * k;
        let offs = if o0 == 0 {
            format!(" offset1:{}", o0 + 32)
        } else {
            format!(" offset0:{o0} offset1:{}", o0 + 32)
        };
        let dst = VX + 2 * k as u8;
        wg.ds_load(
            &tp,
            ds_ld(
                format!("ds_load_2addr_b32 {}, v{VRD}{offs}", vp(dst)),
                vr(dst, 2),
                VRD,
            ),
        )?;
    }
    let b = wg.isa();
    op(
        b,
        format!("v_add_nc_u32_e32 v{CNT}, v{VX}, v{}", VX + 1),
        &[v(CNT)],
        &[v(VX), v(VX + 1)],
    )?;
    for i in 2..8u8 {
        op(
            b,
            format!("v_add_nc_u32_e32 v{CNT}, v{CNT}, v{}", VX + i),
            &[v(CNT)],
            &[v(CNT), v(VX + i)],
        )?;
    }
    let (tile,) = wg.barrier((retire(tp),))?;
    let b = wg.isa();
    // Inclusive prefix over lanes (lane L = digit 31 - L): xor-butterfly with
    // the lower partner's total added on lanes whose partner bit is set.
    op(
        b,
        format!("v_mov_b32_e32 v{VP}, v{CNT}"),
        &[v(VP)],
        &[v(CNT)],
    )?;
    op(
        b,
        format!("v_mov_b32_e32 v{VTOT}, v{CNT}"),
        &[v(VTOT)],
        &[v(CNT)],
    )?;
    for (k, m) in [1u32, 2, 4, 8].into_iter().enumerate() {
        xlane(b, VB, VTOT, m)?;
        op(
            b,
            format!("v_cndmask_b32_e64 v{VT}, 0, v{VB}, s{}", LB[k]),
            &[v(VT)],
            &[v(VB), mask(LB[k])],
        )?;
        op(
            b,
            format!("v_add_nc_u32_e32 v{VP}, v{VP}, v{VT}"),
            &[v(VP)],
            &[v(VP), v(VT)],
        )?;
        op(
            b,
            format!("v_add_nc_u32_e32 v{VTOT}, v{VTOT}, v{VB}"),
            &[v(VTOT)],
            &[v(VTOT), v(VB)],
        )?;
    }
    op(
        b,
        format!("v_permlanex16_b32 v{VB}, v{VTOT}, s{PSEL}, 0xfedcba98"),
        &[v(VB)],
        &[v(VTOT), s(PSEL)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{VT}, 0, v{VB}, s{}", LB[4]),
        &[v(VT)],
        &[v(VB), mask(LB[4])],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{VP}, v{VP}, v{VT}"),
        &[v(VP)],
        &[v(VP), v(VT)],
    )?;
    // Boundary lane: first lane whose cumulative count reaches need.
    op(
        b,
        format!("v_cmp_le_u32_e64 s{MP0}, s{NEED}, v{VP}"),
        &[mask(MP0)],
        &[s(NEED), v(VP)],
    )?;
    op(
        b,
        format!("s_ctz_i32_b32 s{T4}, s{MP0}"),
        &[s(T4)],
        &[mask(MP0)],
    )?;
    op(
        b,
        format!("v_readlane_b32 s{T5}, v{CNT}, s{T4}"),
        &[s(T5)],
        &[v(CNT), s(T4)],
    )?;
    op(
        b,
        format!("v_readlane_b32 s{T6}, v{VP}, s{T4}"),
        &[s(T6)],
        &[v(VP), s(T4)],
    )?;
    sop(
        b,
        format!("s_sub_u32 s{T7}, s{T6}, s{T5}"),
        &[T7],
        &[T6, T5],
    )?;
    sop(b, format!("s_sub_u32 s{T4}, 31, s{T4}"), &[T4], &[T4])?;
    // above += count above the digit; need -= it; prefix |= digit << shift.
    sop(b, format!("s_mov_b32 s{BD}, s{T5}"), &[BD], &[T5])?;
    sop(
        b,
        format!("s_sub_u32 s{NEED}, s{NEED}, s{T7}"),
        &[NEED],
        &[NEED, T7],
    )?;
    sop(
        b,
        format!("s_add_u32 s{ABOVE}, s{ABOVE}, s{T7}"),
        &[ABOVE],
        &[ABOVE, T7],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{T4}, s{T4}, s{SHIFT}"),
        &[T4],
        &[T4, SHIFT],
    )?;
    sop(
        b,
        format!("s_or_b32 s{PREFIX}, s{PREFIX}, s{T4}"),
        &[PREFIX],
        &[PREFIX, T4],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{MASKM}, -1, s{SHIFT}"),
        &[MASKM],
        &[SHIFT],
    )?;
    // Early finish when the candidates fit the tile.
    sop(
        b,
        format!("s_add_u32 s{T6}, s{ABOVE}, s{BD}"),
        &[T6],
        &[ABOVE, BD],
    )?;
    sop(b, format!("s_cmp_le_u32 s{T6}, {}", lit(EARLY)), &[], &[T6])?;
    sop(b, format!("s_cselect_b32 s{DONE}, 1, 0"), &[DONE], &[])?;
    sop(b, format!("s_cmp_eq_u32 s{SHIFT}, 0"), &[], &[SHIFT])?;
    sop(b, format!("s_cselect_b32 s{T6}, 1, 0"), &[T6], &[])?;
    sop(
        b,
        format!("s_or_b32 s{T6}, s{T6}, s{DONE}"),
        &[T6],
        &[T6, DONE],
    )?;
    sop(
        b,
        format!("s_sub_i32 s{SHIFT}, s{SHIFT}, 5"),
        &[SHIFT],
        &[SHIFT],
    )?;
    sop(
        b,
        format!("s_max_i32 s{SHIFT}, s{SHIFT}, 0"),
        &[SHIFT],
        &[SHIFT],
    )?;
    let more = cmp_wg(wg, format!("s_cmp_eq_u32 s{T6}, 0"), &[s(T6)])?;
    Ok((tile, more))
}

/// Digit counting of packet `q + u`: the one-hot word of each matching key
/// ripples through the lane's plane counters.
fn hist_packet(w: &mut W, g: &Gen, u: u8, planes: u8) -> R {
    let b = w.isa();
    sop(b, format!("s_lshl_b32 s{T2}, s{T1}, 5"), &[T2], &[T1])?;
    sop(
        b,
        format!("s_sub_u32 s{T3}, s{RBC}, s{T2}"),
        &[T3],
        &[RBC, T2],
    )?;
    op(
        b,
        format!("v_cmp_gt_u32_e64 s{MP0}, s{T3}, v{LANE}"),
        &[mask(MP0)],
        &[s(T3), v(LANE)],
    )?;
    op(
        b,
        format!("v_and_b32_e32 v{VT}, s{MASKM}, v{}", key(u)),
        &[v(VT)],
        &[s(MASKM), v(key(u))],
    )?;
    op(
        b,
        format!("v_cmp_eq_u32_e64 s{MP1}, s{PREFIX}, v{VT}"),
        &[mask(MP1)],
        &[s(PREFIX), v(VT)],
    )?;
    op(
        b,
        format!("s_and_b32 s{MP0}, s{MP0}, s{MP1}"),
        &[mask(MP0)],
        &[mask(MP0), mask(MP1)],
    )?;
    let none = cmp_w(w, format!("s_cmp_eq_u32 s{MP0}, 0"), &[mask(MP0)])?;
    w.skip_if(none, &g.lbl("hs"), (), |w, ()| {
        let b = w.isa();
        op(
            b,
            format!("v_bfe_u32 v{VD}, v{}, s{SHIFT}, 5", key(u)),
            &[v(VD)],
            &[v(key(u)), s(SHIFT)],
        )?;
        op(
            b,
            format!("v_lshlrev_b32_e64 v{VOH}, v{VD}, 1"),
            &[v(VOH)],
            &[v(VD)],
        )?;
        op(
            b,
            format!("v_cndmask_b32_e64 v{VC0}, 0, v{VOH}, s{MP0}"),
            &[v(VC0)],
            &[v(VOH), mask(MP0)],
        )?;
        let (mut c, mut cn) = (VC0, VC1);
        for p in 0..planes {
            let pl = PL + p;
            if p + 1 < planes {
                op(
                    b,
                    format!("v_and_b32_e32 v{cn}, v{pl}, v{c}"),
                    &[v(cn)],
                    &[v(pl), v(c)],
                )?;
            }
            op(
                b,
                format!("v_xor_b32_e32 v{pl}, v{pl}, v{c}"),
                &[v(pl)],
                &[v(pl), v(c)],
            )?;
            std::mem::swap(&mut c, &mut cn);
        }
        Ok(())
    })
}

/// Sum the 32 lanes' plane counters with a five-level butterfly of
/// ripple-carry adders: afterwards every lane holds the wave's `planes + 5` plane words.
fn reduce_planes(b: &mut Builder, planes: u8) -> R {
    let mut n = planes;
    for m in [1u32, 2, 4, 8, 16] {
        let (mut c, mut cn) = (VC0, VC1);
        for p in 0..n {
            let a = PL + p;
            if m < 16 {
                xlane(b, VB, a, m)?;
            } else {
                op(
                    b,
                    format!("v_permlanex16_b32 v{VB}, v{a}, s{PSEL}, 0xfedcba98"),
                    &[v(VB)],
                    &[v(a), s(PSEL)],
                )?;
            }
            if p == 0 {
                op(
                    b,
                    format!("v_and_b32_e32 v{c}, v{a}, v{VB}"),
                    &[v(c)],
                    &[v(a), v(VB)],
                )?;
                op(
                    b,
                    format!("v_xor_b32_e32 v{a}, v{VB}, v{a}"),
                    &[v(a)],
                    &[v(VB), v(a)],
                )?;
            } else {
                op(
                    b,
                    format!("v_xor_b32_e32 v{VT}, v{VB}, v{a}"),
                    &[v(VT)],
                    &[v(VB), v(a)],
                )?;
                op(
                    b,
                    format!("v_bfi_b32 v{cn}, v{VT}, v{c}, v{a}"),
                    &[v(cn)],
                    &[v(VT), v(c), v(a)],
                )?;
                op(
                    b,
                    format!("v_xor_b32_e32 v{a}, v{VT}, v{c}"),
                    &[v(a)],
                    &[v(VT), v(c)],
                )?;
                std::mem::swap(&mut c, &mut cn);
            }
        }
        op(
            b,
            format!("v_mov_b32_e32 v{}, v{c}", PL + n),
            &[v(PL + n)],
            &[v(c)],
        )?;
        n += 1;
    }
    Ok(())
}

// ---- compaction -------------------------------------------------------------
fn count_pass(
    wg: &mut Wg,
    g: &Gen,
    ctl: LdsRegion<Ctl, Free>,
) -> Result<LdsRegion<Ctl, Published>, String> {
    let b = wg.isa();
    sop(b, format!("s_mov_b32 s{ARUN}, 0"), &[ARUN], &[])?;
    sop(b, format!("s_mov_b32 s{ERUN}, 0"), &[ERUN], &[])?;
    packet_loop(wg, g, "count", (), false, |w, (), u| {
        let b = w.isa();
        class_masks(b, u)?;
        op(
            b,
            format!("s_bcnt1_i32_b32 s{T4}, s{MP1}"),
            &[s(T4)],
            &[mask(MP1)],
        )?;
        sop(
            b,
            format!("s_add_u32 s{ARUN}, s{ARUN}, s{T4}"),
            &[ARUN],
            &[ARUN, T4],
        )?;
        op(
            b,
            format!("s_bcnt1_i32_b32 s{T4}, s{MP2}"),
            &[s(T4)],
            &[mask(MP2)],
        )?;
        sop(
            b,
            format!("s_add_u32 s{ERUN}, s{ERUN}, s{T4}"),
            &[ERUN],
            &[ERUN, T4],
        )
    })?;
    // This wave's (a, e) pair, then every wave's pairs.
    let b = wg.isa();
    op(
        b,
        format!("v_mov_b32_e32 v{VT}, s{ARUN}"),
        &[v(VT)],
        &[s(ARUN)],
    )?;
    op(
        b,
        format!("v_mov_b32_e32 v{VC0}, s{ERUN}"),
        &[v(VC0)],
        &[s(ERUN)],
    )?;
    let (cw, pend) = wg.ds_store(
        ctl,
        ds_st(format!("ds_store_b64 v{VCA}, {}", vp(VT)), VCA, vr(VT, 2)),
    )?;
    let d = wg.wait(pend)?;
    let (cp,) = wg.barrier((ready(cw, d),))?;
    for k in 0..4u8 {
        let dst = CTL + 4 * k;
        wg.ds_load(
            &cp,
            ds_ld(
                format!(
                    "ds_load_b128 v[{}:{}], v{VCB}{}",
                    dst,
                    dst + 3,
                    off(16 * u32::from(k))
                ),
                vr(dst, 4),
                VCB,
            ),
        )?;
    }
    let b = wg.isa();
    // (a_k, e_k) of wave k land in T0 + 2k, T0 + 2k + 1 (s64 .. s79).
    for i in 0..16u8 {
        op(
            b,
            format!("v_readfirstlane_b32 s{}, v{}", T0 + i, CTL + i),
            &[s(T0 + i)],
            &[v(CTL + i)],
        )?;
    }
    for r in [EXA, EXE, ATOT, ETOT] {
        sop(b, format!("s_mov_b32 s{r}, 0"), &[r], &[])?;
    }
    for k in 0..8u8 {
        let (a, e) = (T0 + 2 * k, T0 + 2 * k + 1);
        sop(b, format!("s_cmp_gt_u32 s{WAVE}, {k}"), &[], &[WAVE])?;
        sop(b, format!("s_cselect_b32 s{X1}, s{a}, 0"), &[X1], &[a])?;
        sop(b, format!("s_cselect_b32 s{X2}, s{e}, 0"), &[X2], &[e])?;
        sop(
            b,
            format!("s_add_u32 s{EXA}, s{EXA}, s{X1}"),
            &[EXA],
            &[EXA, X1],
        )?;
        sop(
            b,
            format!("s_add_u32 s{EXE}, s{EXE}, s{X2}"),
            &[EXE],
            &[EXE, X2],
        )?;
        sop(
            b,
            format!("s_add_u32 s{ATOT}, s{ATOT}, s{a}"),
            &[ATOT],
            &[ATOT, a],
        )?;
        sop(
            b,
            format!("s_add_u32 s{ETOT}, s{ETOT}, s{e}"),
            &[ETOT],
            &[ETOT, e],
        )?;
    }
    // n = A + min(E, limit) candidates; the fill pass counts from the wave's prefixes.
    sop(
        b,
        format!("s_min_u32 s{X1}, s{ETOT}, s{LIMIT}"),
        &[X1],
        &[ETOT, LIMIT],
    )?;
    sop(
        b,
        format!("s_add_u32 s{NCAND}, s{ATOT}, s{X1}"),
        &[NCAND],
        &[ATOT, X1],
    )?;
    sop(b, format!("s_mov_b32 s{ARUN}, s{EXA}"), &[ARUN], &[EXA])?;
    sop(b, format!("s_mov_b32 s{ERUN}, s{EXE}"), &[ERUN], &[EXE])?;
    // The readbacks used the lane-mask pairs as scalars: their high words are zero again.
    for m in [MP0, MP1, MP2, MP3] {
        sop(b, format!("s_mov_b32 s{}, 0", m + 1), &[m + 1], &[])?;
    }
    Ok(cp)
}

fn fill_pass(
    wg: &mut Wg,
    g: &Gen,
    tile: LdsRegion<Tile, Free>,
    ctl: LdsRegion<Ctl, Published>,
) -> Result<(LdsRegion<Tile, Published>, LdsRegion<Ctl, Free>), String> {
    let st = wg.begin_write(tile);
    let st = packet_loop(wg, g, "fill", st, true, |w, st, u| {
        let b = w.isa();
        class_masks(b, u)?;
        // Ranks inside the packet's two classes, offset by the wave's running counts.
        op(
            b,
            format!("v_mbcnt_lo_u32_b32 v{VPA}, s{MP1}, s{ARUN}"),
            &[v(VPA)],
            &[mask(MP1), s(ARUN)],
        )?;
        op(
            b,
            format!("v_mbcnt_lo_u32_b32 v{VPE}, s{MP2}, s{ERUN}"),
            &[v(VPE)],
            &[mask(MP2), s(ERUN)],
        )?;
        // Boundary keys are stored while their rank is below the limit.
        op(
            b,
            format!("v_cmp_gt_u32_e64 s{MP3}, s{LIMIT}, v{VPE}"),
            &[mask(MP3)],
            &[s(LIMIT), v(VPE)],
        )?;
        op(
            b,
            format!("s_and_b32 s{MP3}, s{MP3}, s{MP2}"),
            &[mask(MP3)],
            &[mask(MP3), mask(MP2)],
        )?;
        op(
            b,
            format!("s_or_b32 s{MP3}, s{MP3}, s{MP1}"),
            &[mask(MP3)],
            &[mask(MP3), mask(MP1)],
        )?;
        op(
            b,
            format!("v_add_nc_u32_e32 v{VPE}, s{ATOT}, v{VPE}"),
            &[v(VPE)],
            &[s(ATOT), v(VPE)],
        )?;
        op(
            b,
            format!("v_cndmask_b32_e64 v{VPOS}, v{VPE}, v{VPA}, s{MP1}"),
            &[v(VPOS)],
            &[v(VPE), v(VPA), mask(MP1)],
        )?;
        op(
            b,
            format!("v_lshlrev_b32_e32 v{VAD}, 3, v{VPOS}"),
            &[v(VAD)],
            &[v(VPOS)],
        )?;
        // The position is below 1024: the mask keeps the address provably in the tile.
        op(
            b,
            format!("v_and_b32_e32 v{VAD}, {}, v{VAD}", lit(H_BYTES - 8)),
            &[v(VAD)],
            &[v(VAD)],
        )?;
        // Entry (key << 32) | ~block as the pair (lo, key).
        op(
            b,
            format!("v_add_nc_u32_e32 v{}, s{T2}, v{LANE}", lo(u)),
            &[v(lo(u))],
            &[s(T2), v(LANE)],
        )?;
        op(
            b,
            format!("v_xor_b32_e32 v{0}, -1, v{0}", lo(u)),
            &[v(lo(u))],
            &[v(lo(u))],
        )?;
        op(
            b,
            format!("s_bcnt1_i32_b32 s{T4}, s{MP1}"),
            &[s(T4)],
            &[mask(MP1)],
        )?;
        sop(
            b,
            format!("s_add_u32 s{ARUN}, s{ARUN}, s{T4}"),
            &[ARUN],
            &[ARUN, T4],
        )?;
        op(
            b,
            format!("s_bcnt1_i32_b32 s{T4}, s{MP2}"),
            &[s(T4)],
            &[mask(MP2)],
        )?;
        sop(
            b,
            format!("s_add_u32 s{ERUN}, s{ERUN}, s{T4}"),
            &[ERUN],
            &[ERUN, T4],
        )?;
        op(b, format!("s_mov_b32 exec_lo, s{MP3}"), &[], &[mask(MP3)])?;
        let st = w.ds_store(
            st,
            ds_st(
                format!("ds_store_b64 v{VAD}, {}", vp(lo(u))),
                VAD,
                vr(lo(u), 2),
            ),
        )?;
        op(w.isa(), "s_mov_b32 exec_lo, -1", &[], &[])?;
        Ok(st)
    })?;
    let (tw, pend) = st;
    let d = wg.wait(pend)?;
    wg.barrier((ready(tw, d), retire(ctl)))
}

// ---- sort -------------------------------------------------------------------
#[derive(Clone, Copy)]
enum Msk {
    Reg(u8),
    Zero,
    Ones,
}
/// Bit `q` of a sort index `256 e + 32 wave + lane`, as a lane mask.
fn index_bit(q: u32, e: u32) -> Msk {
    match q {
        0..=4 => Msk::Reg(LB[q as usize]),
        5..=7 => Msk::Reg(WB[(q - 5) as usize]),
        _ => {
            if (e >> (q - 8)) & 1 == 1 {
                Msk::Ones
            } else {
                Msk::Zero
            }
        }
    }
}
/// `T0 = mask(bit j) ^ mask(bit k)` of slice `e`: the complement of the lanes
/// that want the better entry of the pair.
fn wbar(b: &mut Builder, j: u32, k: u32, e: u32) -> R {
    let (mj, mk) = (
        index_bit(j.trailing_zeros(), e),
        index_bit(k.trailing_zeros(), e),
    );
    match (mj, mk) {
        (Msk::Reg(x), Msk::Reg(y)) => {
            sop(b, format!("s_xor_b32 s{T0}, s{x}, s{y}"), &[T0], &[x, y])
        }
        (Msk::Reg(x), Msk::Zero) | (Msk::Zero, Msk::Reg(x)) => {
            sop(b, format!("s_mov_b32 s{T0}, s{x}"), &[T0], &[x])
        }
        (Msk::Reg(x), Msk::Ones) | (Msk::Ones, Msk::Reg(x)) => {
            sop(b, format!("s_not_b32 s{T0}, s{x}"), &[T0], &[x])
        }
        (Msk::Zero, Msk::Zero) | (Msk::Ones, Msk::Ones) => {
            sop(b, format!("s_mov_b32 s{T0}, 0"), &[T0], &[])
        }
        _ => sop(b, format!("s_mov_b32 s{T0}, -1"), &[T0], &[]),
    }
}
/// Run `body` only when slice `e` can hold valid entries (`n > 256 e`).
fn gate(w: &mut W, g: &Gen, e: u32, body: impl FnOnce(&mut W) -> R) -> R {
    if e == 0 {
        return body(w);
    }
    let limit = if e == 1 { 256 } else { 512 };
    let skip = cmp_w(
        w,
        format!("s_cmp_le_u32 s{NCAND}, {}", lit(limit)),
        &[s(NCAND)],
    )?;
    w.skip_if(skip, &g.lbl("sl"), (), |w, ()| body(w))
}
/// Keep the better or worse entry of slice `e` against its partner `part(e)`.
fn compare_exchange(b: &mut Builder, j: u32, k: u32, e: u32) -> R {
    let (x, p) = (ent(e), part(e));
    op(
        b,
        format!("v_cmp_gt_i64_e64 s{MP0}, {}, {}", vp(x), vp(p)),
        &[mask(MP0)],
        &[vr(x, 2), vr(p, 2)],
    )?;
    wbar(b, j, k, e)?;
    op(
        b,
        format!("s_xor_b32 s{MP0}, s{MP0}, s{T0}"),
        &[mask(MP0)],
        &[mask(MP0), s(T0)],
    )?;
    for d in 0..2u8 {
        op(
            b,
            format!("v_cndmask_b32_e64 v{0}, v{1}, v{0}, s{MP0}", x + d, p + d),
            &[v(x + d)],
            &[v(p + d), v(x + d), mask(MP0)],
        )?;
    }
    Ok(())
}
/// Strides below 32: the partner is lane `l ^ j` of the same wave.
fn lane_stage(wg: &mut Wg, g: &Gen, k: u32, j: u32) -> R {
    for e in 0..4u32 {
        gate(wg, g, e, |w| {
            let b = w.isa();
            let (x, p) = (ent(e), part(e));
            for d in 0..2u8 {
                if j < 16 {
                    xlane(b, p + d, x + d, j)?;
                } else {
                    op(
                        b,
                        format!(
                            "v_permlanex16_b32 v{}, v{}, s{PSEL}, 0xfedcba98",
                            p + d,
                            x + d
                        ),
                        &[v(p + d)],
                        &[v(x + d), s(PSEL)],
                    )?;
                }
            }
            compare_exchange(b, j, k, e)
        })?;
    }
    Ok(())
}
/// Strides 256 and 512: the partner is another slice of the same thread.
fn thread_stage(wg: &mut Wg, g: &Gen, k: u32, j: u32) -> R {
    let pairs: [(u32, u32); 2] = if j == 256 {
        [(0, 1), (2, 3)]
    } else {
        [(0, 2), (1, 3)]
    };
    for (ea, eb) in pairs {
        // Slice `ea`'s sort direction (bit k of its index): the merge of
        // k = 512 runs descending on the upper half.
        let desc = k == 512 && (ea >> 1) & 1 == 1;
        gate(wg, g, eb, |w| {
            let b = w.isa();
            let (a, c) = (ent(ea), ent(eb));
            op(
                b,
                format!("v_cmp_gt_i64_e64 s{MP0}, {}, {}", vp(a), vp(c)),
                &[mask(MP0)],
                &[vr(a, 2), vr(c, 2)],
            )?;
            for d in 0..2u8 {
                let (ad, cd, t) = (a + d, c + d, STMP + d);
                // asc: a' = gt ? a : c, c' = gt ? c : a; desc swaps the roles.
                let (first, second) = if desc { (ad, cd) } else { (cd, ad) };
                op(
                    b,
                    format!("v_cndmask_b32_e64 v{t}, v{first}, v{second}, s{MP0}"),
                    &[v(t)],
                    &[v(first), v(second), mask(MP0)],
                )?;
                op(
                    b,
                    format!("v_cndmask_b32_e64 v{cd}, v{second}, v{first}, s{MP0}"),
                    &[v(cd)],
                    &[v(second), v(first), mask(MP0)],
                )?;
                op(b, format!("v_mov_b32_e32 v{ad}, v{t}"), &[v(ad)], &[v(t)])?;
            }
            Ok(())
        })?;
    }
    Ok(())
}
/// Strides 32, 64, 128: the partner is the same lane of wave `wave ^ j/32`,
/// exchanged through the tile.
fn wave_stage(
    wg: &mut Wg,
    g: &Gen,
    k: u32,
    j: u32,
    tile: LdsRegion<Tile, Free>,
) -> Result<LdsRegion<Tile, Free>, String> {
    let mut st = wg.begin_write(tile);
    for e in 0..4u32 {
        st = wg.ds_store(
            st,
            ds_st(
                format!("ds_store_b64 v{TID8}, {}{}", vp(ent(e)), off(2048 * e)),
                TID8,
                vr(ent(e), 2),
            ),
        )?;
    }
    let (tw, pend) = st;
    let d = wg.wait(pend)?;
    let (tp,) = wg.barrier((ready(tw, d),))?;
    op(
        wg.isa(),
        format!("v_xor_b32_e32 v{VPAD}, {}, v{TID8}", lit(8 * j)),
        &[v(VPAD)],
        &[v(TID8)],
    )?;
    // tid ^ j < 256: the mask keeps the partner address provably in its slice.
    op(
        wg.isa(),
        format!("v_and_b32_e32 v{VPAD}, {}, v{VPAD}", lit(2048 - 8)),
        &[v(VPAD)],
        &[v(VPAD)],
    )?;
    for e in 0..4u32 {
        wg.ds_load(
            &tp,
            ds_ld(
                format!("ds_load_b64 {}, v{VPAD}{}", vp(part(e)), off(2048 * e)),
                vr(part(e), 2),
                VPAD,
            ),
        )?;
    }
    for e in 0..4u32 {
        gate(wg, g, e, |w| compare_exchange(w.isa(), j, k, e))?;
    }
    let (tile,) = wg.barrier((retire(tp),))?;
    Ok(tile)
}
fn stage(
    wg: &mut Wg,
    g: &Gen,
    k: u32,
    j: u32,
    tile: LdsRegion<Tile, Free>,
) -> Result<LdsRegion<Tile, Free>, String> {
    if j >= 256 {
        thread_stage(wg, g, k, j)?;
        Ok(tile)
    } else if j < 32 {
        lane_stage(wg, g, k, j)?;
        Ok(tile)
    } else {
        wave_stage(wg, g, k, j, tile)
    }
}
/// The stages of merge size `k`.
fn merge(
    wg: &mut Wg,
    g: &Gen,
    k: u32,
    mut tile: LdsRegion<Tile, Free>,
) -> Result<LdsRegion<Tile, Free>, String> {
    let mut j = k / 2;
    while j >= 1 {
        tile = stage(wg, g, k, j, tile)?;
        j /= 2;
    }
    Ok(tile)
}

fn sort(
    wg: &mut Wg,
    g: &Gen,
    tile: LdsRegion<Tile, Published>,
) -> Result<LdsRegion<Tile, Free>, String> {
    let b = wg.isa();
    for k in 0..3u8 {
        sop(b, format!("s_bitcmp1_b32 s{WAVE}, {k}"), &[], &[WAVE])?;
        sop(
            b,
            format!("s_cselect_b32 s{}, -1, 0", WB[usize::from(k)]),
            &[WB[usize::from(k)]],
            &[],
        )?;
    }
    // Entry 256 e + t of thread t, zero (the minimum) past n.
    for e in 0..4u32 {
        wg.ds_load(
            &tile,
            ds_ld(
                format!("ds_load_b64 {}, v{TID8}{}", vp(ent(e)), off(2048 * e)),
                vr(ent(e), 2),
                TID8,
            ),
        )?;
    }
    let b = wg.isa();
    for e in 0..4u32 {
        op(
            b,
            format!("v_add_nc_u32_e32 v{VI}, {}, v{TID}", lit(256 * e)),
            &[v(VI)],
            &[v(TID)],
        )?;
        op(
            b,
            format!("v_cmp_gt_u32_e64 s{MP0}, s{NCAND}, v{VI}"),
            &[mask(MP0)],
            &[s(NCAND), v(VI)],
        )?;
        for d in 0..2u8 {
            op(
                b,
                format!("v_cndmask_b32_e64 v{0}, 0, v{0}, s{MP0}", ent(e) + d),
                &[v(ent(e) + d)],
                &[v(ent(e) + d), mask(MP0)],
            )?;
        }
    }
    let (mut tile,) = wg.barrier((retire(tile),))?;
    let mut k = 2;
    while k <= 256 {
        tile = merge(wg, g, k, tile)?;
        k *= 2;
    }
    // k = 512 for n > 256, k = 1024 for n > 512.
    let small = cmp_wg(
        wg,
        format!("s_cmp_le_u32 s{NCAND}, {}", lit(256)),
        &[s(NCAND)],
    )?;
    let tile = wg.wg_skip_if(small, &g.lbl("sort_b"), tile, |wg, tile| {
        merge(wg, g, 512, tile)
    })?;
    let medium = cmp_wg(
        wg,
        format!("s_cmp_le_u32 s{NCAND}, {}", lit(512)),
        &[s(NCAND)],
    )?;
    wg.wg_skip_if(medium, &g.lbl("sort_c"), tile, |wg, tile| {
        merge(wg, g, 1024, tile)
    })
}

/// Block ids `~lo` of the sorted entries `< 512`.
fn write_ranks(
    wg: &mut Wg,
    ranks: LdsRegion<Ranks, Free>,
) -> Result<LdsRegion<Ranks, Published>, String> {
    let mut st = wg.begin_write(ranks);
    for e in 0..2u32 {
        let t = VI + e as u8;
        op(
            wg.isa(),
            format!("v_xor_b32_e32 v{t}, -1, v{}", ent(e)),
            &[v(t)],
            &[v(ent(e))],
        )?;
        st = wg.ds_store(
            st,
            ds_st(
                format!("ds_store_b32 v{VRA}, v{t}{}", off(1024 * e)),
                VRA,
                v(t),
            ),
        )?;
    }
    let (rw, pend) = st;
    let d = wg.wait(pend)?;
    let (rp,) = wg.barrier((ready(rw, d),))?;
    Ok(rp)
}

// ---- output -----------------------------------------------------------------
fn output(wg: &mut Wg, g: &Gen, ranks: &LdsRegion<Ranks, Published>) -> R {
    let b = wg.isa();
    magic(b, RMAG, COMP, [VT, VC0, VC1], T0)?;
    sop(
        b,
        format!("s_add_i32 s{T0}, s{CAP}, {}", lit(1023)),
        &[T0],
        &[CAP],
    )?;
    sop(
        b,
        format!("s_lshr_b32 s{TRIPS}, s{T0}, 10"),
        &[TRIPS],
        &[T0],
    )?;
    sop(b, format!("s_mov_b32 s{TRIP}, 0"), &[TRIP], &[])?;
    wg.loop_carried(&g.lbl("out"), (), |wg, ()| {
        for u in 0..4u8 {
            let b = wg.isa();
            sop(b, format!("s_lshl_b32 s{T0}, s{TRIP}, 10"), &[T0], &[TRIP])?;
            sop(
                b,
                format!("s_add_i32 s{T0}, s{T0}, {}", lit(256 * u32::from(u))),
                &[T0],
                &[T0],
            )?;
            op(
                b,
                format!("v_add_nc_u32_e32 v{VS}, s{T0}, v{TID}"),
                &[v(VS)],
                &[s(T0), v(TID)],
            )?;
            // q = s / compress, rm = s % compress.
            op(
                b,
                format!("v_mul_hi_u32 v{VQ}, v{VS}, s{RMAG}"),
                &[v(VQ)],
                &[v(VS), s(RMAG)],
            )?;
            for _ in 0..2 {
                op(
                    b,
                    format!("v_mul_lo_u32 v{VT3}, v{VQ}, s{COMP}"),
                    &[v(VT3)],
                    &[v(VQ), s(COMP)],
                )?;
                op(
                    b,
                    format!("v_sub_nc_u32_e32 v{VRM}, v{VS}, v{VT3}"),
                    &[v(VRM)],
                    &[v(VS), v(VT3)],
                )?;
                op(
                    b,
                    format!("v_cmp_le_u32_e64 s{MP1}, s{COMP}, v{VRM}"),
                    &[mask(MP1)],
                    &[s(COMP), v(VRM)],
                )?;
                op(
                    b,
                    format!("v_add_nc_u32_e32 v{VT3}, 1, v{VQ}"),
                    &[v(VT3)],
                    &[v(VQ)],
                )?;
                op(
                    b,
                    format!("v_cndmask_b32_e64 v{VQ}, v{VQ}, v{VT3}, s{MP1}"),
                    &[v(VQ)],
                    &[v(VQ), v(VT3), mask(MP1)],
                )?;
            }
            op(
                b,
                format!("v_mul_lo_u32 v{VT3}, v{VQ}, s{COMP}"),
                &[v(VT3)],
                &[v(VQ), s(COMP)],
            )?;
            op(
                b,
                format!("v_sub_nc_u32_e32 v{VRM}, v{VS}, v{VT3}"),
                &[v(VRM)],
                &[v(VS), v(VT3)],
            )?;
            op(
                b,
                format!("v_min_u32_e32 v{VRAD}, {}, v{VQ}", lit(511)),
                &[v(VRAD)],
                &[v(VQ)],
            )?;
            op(
                b,
                format!("v_lshlrev_b32_e32 v{VRAD}, 2, v{VRAD}"),
                &[v(VRAD)],
                &[v(VRAD)],
            )?;
            // min(q, 511) * 4: the mask keeps the rank address provably in the table.
            op(
                b,
                format!("v_and_b32_e32 v{VRAD}, {}, v{VRAD}", lit(R_BYTES - 4)),
                &[v(VRAD)],
                &[v(VRAD)],
            )?;
            wg.ds_load(
                ranks,
                ds_ld(
                    format!("ds_load_b32 v{VBLK}, v{VRAD}{}", off(R_BASE)),
                    v(VBLK),
                    VRAD,
                ),
            )?;
            let b = wg.isa();
            op(
                b,
                format!("v_cmp_gt_u32_e64 s{MP0}, s{RANKED}, v{VS}"),
                &[mask(MP0)],
                &[s(RANKED), v(VS)],
            )?;
            op(
                b,
                format!("v_mul_lo_u32 v{VV}, v{VBLK}, s{COMP}"),
                &[v(VV)],
                &[v(VBLK), s(COMP)],
            )?;
            op(
                b,
                format!("v_add_nc_u32_e32 v{VV}, v{VV}, v{VRM}"),
                &[v(VV)],
                &[v(VV), v(VRM)],
            )?;
            op(
                b,
                format!("v_subrev_nc_u32_e32 v{VJ}, s{RANKED}, v{VS}"),
                &[v(VJ)],
                &[s(RANKED), v(VS)],
            )?;
            op(
                b,
                format!("v_cmp_gt_u32_e64 s{MP1}, s{TLEN}, v{VJ}"),
                &[mask(MP1)],
                &[s(TLEN), v(VJ)],
            )?;
            op(
                b,
                format!("v_add_nc_u32_e32 v{VT3}, s{TSTART}, v{VJ}"),
                &[v(VT3)],
                &[s(TSTART), v(VJ)],
            )?;
            op(
                b,
                format!("v_cndmask_b32_e64 v{VT3}, -1, v{VT3}, s{MP1}"),
                &[v(VT3)],
                &[v(VT3), mask(MP1)],
            )?;
            let (val, offv) = (VVAL + u, VOFF + u);
            op(
                b,
                format!("v_cndmask_b32_e64 v{val}, v{VT3}, v{VV}, s{MP0}"),
                &[v(val)],
                &[v(VT3), v(VV), mask(MP0)],
            )?;
            op(
                b,
                format!("v_lshlrev_b32_e32 v{offv}, 2, v{VS}"),
                &[v(offv)],
                &[v(VS)],
            )?;
            for srd in [SRD_SEL, SRD_MIR] {
                mem(
                    b,
                    format!(
                        "buffer_store_b32 v{val}, v{offv}, s[{srd}:{}], 0 offen",
                        srd + 3
                    ),
                    &[],
                    &[v(offv), v(val), sr(srd, 4)],
                    MemoryClass::VmemStore,
                )?;
            }
        }
        let b = wg.isa();
        b.wait(Counter::Vs, 0)?;
        sop(
            b,
            format!("s_add_i32 s{TRIP}, s{TRIP}, 1"),
            &[TRIP],
            &[TRIP],
        )?;
        let more = cmp_wg(
            wg,
            format!("s_cmp_lt_u32 s{TRIP}, s{TRIPS}"),
            &[s(TRIP), s(TRIPS)],
        )?;
        Ok(((), more))
    })
}
