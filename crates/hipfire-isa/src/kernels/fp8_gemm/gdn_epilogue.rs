//! GDN preparation fused into the F2 QKVZA epilogue (`Epi::QkvzaGdn`).
//!
//! A q/k/v row tile (256 channels = two 128-channel heads) x 128 tokens is
//! turned into `gdn_chunk_prep`'s FP16 q/k/v directly: width-4 causal conv1d
//! over tokens, SiLU, and for q/k the per-token head RMS norm (and q_scale).
//! The f32 projection itself is never stored except the raw rows the
//! completion pass `gdn_chunk_prep_fixup` needs: tile positions 0..2 and
//! 125..127 (the 3-token conv halo across tiles) and the last three tokens
//! (the persistent conv ring). Tile heads 128j..128j+2 (j >= 1) are left to
//! that pass; the first tile reads the conv ring as its halo.
//!
//! LDS is one 19-row ring of 256 f32 channels with rows 1,040 bytes apart
//! (1,024 data bytes plus a 16-byte bank skew): 19,760 bytes, the 19,456-byte
//! F2 launch allocation plus `FIXED_LDS` in the kernel descriptor. Token t of
//! the tile lives in row (t + 3) mod 19, so the 16 tokens of block b and the 3
//! previous ones are always resident. Block b
//! (tokens 16b..16b+15) is drained from the accumulators of token half
//! b / 4, column block b % 4, by that half's four waves, which then run the
//! per-token regions in `gdn_chunk_prep`'s lane layout: a wave owns
//! one head (lane = 4 channels) and 8 consecutive tokens. The conv+SiLU is
//! the exhaustively proven lean region (`Region::conv_silu_lean`); the norms
//! are the imported goldens.
use super::{Builder, vo, vload};
use crate::kernels::common::{mem, op, s, sop, sr, v, vr};
use super::gdn_region::{self, Binding, Half, Region};
use crate::{insn::{Instruction, MemoryClass}, lds::Transition, ledger::Counter};

pub(super) const ENTRY: &str = ".Lfp8_gdn";
const RING_ROWS: u32 = 19;
/// Ring row pitch: 1,024 bytes of data plus 16 (bank skew; keeps b128 alignment).
pub const ROW_BYTES: u32 = 1040;
pub const RING_BYTES: u32 = RING_ROWS * ROW_BYTES;
/// Fixed LDS of the GDN-fused symbol: the padded ring beyond the launch's dynamic bytes.
pub const FIXED_LDS: u32 = RING_BYTES - super::spec::LDS_BYTES;

// VGPRs of the GDN path (the accumulators v0..v127 drain block by block).
const W: u8 = 128; // conv taps: w[tap][c] at W + 4*tap + c
// window: four slots of four channels at ROWS + 4*slot + c. The n-th token of
// a P run (j = n mod 4) reads rows t-3..t from slots j, j+1, j+2, j+3 (mod 4)
// and loads only row t, into slot (j + 3) % 4, over the slot of row t-4.
const ROWS: u8 = 144;
const CONV: u8 = 160; // two conv+SiLU instances x 8 temps; the norm reuses CONV..CONV+7
const OUT: u8 = 176; // o0..o3 (SiLU outputs of channels 0..3)
// FP16 halves h0..h3 in v0.l v0.h v1.l v1.h. A true16 VOP1 destination must
// be below v128; accumulator column block 0 is drained before any wave runs
// its first token (W(4h) precedes P(4h) for both token halves).
const PACK: u8 = 0;
const VA: u8 = 182; // W: LDS address of this lane's ring row
const VC: u8 = 183; // W: channel byte offset (rg*64 + hi*8)*4
const VL: u8 = 184; // lane & 15
const VP: u8 = 185; // P: hd*512 + lane*16 (head channels in a ring row)
const VPA: u8 = 186; // P: row address / scratch
const TID: u8 = 187;
const VOUT: u8 = 188; // out: hd*256 + lane*8
const VRAW: u8 = 189; // raw rows / weights: 4*ch, ch = rt*256 + hd*128 + lane*4
const VWOFF: u8 = 190; // weights: 16*ch (only before the block loop)
const VOUTA: u8 = 190; // out store address (reuses VWOFF once the taps landed)
const VRAWA: u8 = 191; // raw store address

// SGPRs (K-loop descriptors and the Ew/D descriptors are dead here).
const KA: u8 = 32; // s[32:39] = ConvW ConvState Q K ; s[40:43] = V QScale Eps
const CLAMP: u8 = 36; // 128.0, the lean SiLU's exp clamp (Q's pointer is dead once OUTD is built)
const QSCALE: u8 = 42;
const EPS: u8 = 43;
const NORM_MASK: u8 = 48;
const RAWD: u8 = 52; // Y0 rebuilt with num_records = N * 10240 * 4
const OUTD: u8 = 56; // selected q/k/v, num_records = N * heads * 256
const CWD: u8 = 60;
const CSD: u8 = 64;
const TOKSTRIDE: u8 = 68; // out bytes per token: heads * 256
const LANESEL: u8 = 69; // 0x76543210
const HEAD0: u8 = 70; // out byte offset of the tile's first head
const CLASS: u8 = 71; // 0 q, 1 k, 2 v
const HD: u8 = 72; // rg >> 1: the head a P wave owns
const SUB: u8 = 73; // rg & 1: which 8 tokens of the block
const BLK: u8 = 90;
const TT: u8 = 91;
const HALF: u8 = 92;
const RB: u8 = 93; // ring row of token 16b
const R: u8 = 94; // ring row of token t-3
const T: u8 = 95; // tile-local token
const TG: u8 = 96; // global token
const S0: u8 = 97;
const S1: u8 = 98;
const S2: u8 = 99;
const J: u8 = 100;
const OUTOFF: u8 = 101;
const RAWOFF: u8 = 102;

fn descriptor(b: &mut Builder, dst: u8, lo: u8, records: Option<&str>) -> Result<(), String> {
    sop(b, format!("s_mov_b32 s{dst}, s{lo}"), &[dst], &[lo])?;
    sop(b, format!("s_and_b32 s{}, s{}, 0xffff", dst + 1, lo + 1), &[dst + 1], &[lo + 1])?;
    match records {
        Some(r) => op(b, format!("s_mov_b32 s{}, {r}", dst + 2), &[s(dst + 2)], &[])?,
        None => sop(b, format!("s_mov_b32 s{}, -1", dst + 2), &[dst + 2], &[])?,
    }
    sop(b, format!("s_mov_b32 s{}, 0x31004000", dst + 3), &[dst + 3], &[])
}
/// `dst = (x + k) mod 19` for x < 19, k <= 19 (one conditional subtraction).
fn mod19(b: &mut Builder, dst: u8, x: u8, k: u32) -> Result<(), String> {
    if k == 0 { sop(b, format!("s_mov_b32 s{dst}, s{x}"), &[dst], &[x])?; } else { sop(b, format!("s_add_co_i32 s{dst}, s{x}, {k}"), &[dst], &[x])?; }
    sop(b, format!("s_sub_co_i32 s{S2}, s{dst}, {RING_ROWS}"), &[S2], &[dst])?;
    sop(b, format!("s_cmp_ge_u32 s{dst}, {RING_ROWS}"), &[], &[dst])?;
    sop(b, format!("s_cselect_b32 s{dst}, s{S2}, s{dst}"), &[dst], &[S2, dst])
}
fn ds_store_b128(b: &mut Builder, slot: usize, addr: u8, data: u8, offset: u32) -> Result<(), String> {
    let off = if offset == 0 { String::new() } else { format!(" offset:{offset}") };
    b.ds_store(slot, Instruction::new(format!("ds_store_b128 v{addr}, {}{off}", vr(data, 4)), vec![], vec![v(addr), vr(data, 4)]).memory(MemoryClass::DsStore))
}
fn ds_load_b128(b: &mut Builder, slot: usize, data: u8, addr: u8) -> Result<(), String> {
    b.ds_load(slot, Instruction::new(format!("ds_load_b128 {}, v{addr}", vr(data, 4)), vec![vr(data, 4)], vec![v(addr)]).memory(MemoryClass::DsLoad))
}

struct Regions { conv: Region, norm_q: Region, norm_k: Region, cvt_v: Region }

/// Instance `k` (0/1) of a conv+SiLU pair for channel `c` of token `j`
/// (window rotation): temps rotated by `k` so VOPD partners differ in bank
/// and destination parity.
fn conv_bind(r: &Region, c: u8, k: u8, j: u8) -> Result<Binding, String> {
    let mut inputs = vec![0u8; r.inputs.len()];
    let slot = |d: u8| ROWS + 4 * ((j + d) % 4) + c;
    for (name, reg) in [("w0", W + c), ("w1", W + 4 + c), ("w2", W + 8 + c), ("w3", W + 12 + c),
        ("win0", slot(0)), ("win1", slot(1)), ("win2", slot(2)), ("cur", slot(3))] {
        inputs[r.input(name)?] = reg;
    }
    if r.temps > 8 || r.masks > 0 { return Err("conv+SiLU region exceeds its GDN registers".into()) }
    let temps = (0..r.temps as u8).map(|t| CONV + 8 * k + (t + k) % 8).collect();
    let mut sinputs = vec![0u8; r.sinputs.len()];
    sinputs[r.sinput("clamp")?] = CLAMP;
    Ok(Binding { inputs, sinputs, temps, masks: vec![], outputs: vec![(OUT + c, Half::Full)], lane_select: None })
}
fn norm_bind(r: &Region) -> Result<Binding, String> {
    let mut inputs = vec![0u8; r.inputs.len()];
    for c in 0..4u8 { inputs[r.input(&format!("o{c}"))?] = OUT + c; }
    let mut sinputs = vec![0u8; r.sinputs.len()];
    if !r.sinputs.is_empty() { sinputs[r.sinput("eps")?] = EPS; }
    if r.sinputs.len() > 1 { sinputs[r.sinput("qscale")?] = QSCALE; }
    if r.temps > 8 || r.masks > 1 { return Err("norm region exceeds its GDN registers".into()) }
    let mut outputs = vec![(0u8, Half::Full); 4];
    for (k, (reg, half)) in [(PACK, Half::Lo), (PACK, Half::Hi), (PACK + 1, Half::Lo), (PACK + 1, Half::Hi)].into_iter().enumerate() {
        outputs[r.output(&format!("h{k}"))?] = (reg, half);
    }
    Ok(Binding { inputs, sinputs, temps: (0..r.temps as u8).map(|t| CONV + t).collect(), masks: vec![NORM_MASK], outputs, lane_select: Some(LANESEL) })
}

/// Load ring row `(R + k) mod 19` of this head into window slot `slot`.
fn load_row(b: &mut Builder, g: usize, k: u8, slot: u8) -> Result<(), String> {
    mod19(b, S0, R, u32::from(k))?;
    sop(b, format!("s_mul_i32 s{S0}, s{S0}, {ROW_BYTES:#x}"), &[S0], &[S0])?;
    vo(b, format!("v_add_nc_u32_e32 v{VPA}, s{S0}, v{VP}"), &[VPA], &[VP], &[S0])?;
    ds_load_b128(b, g, ROWS + 4 * slot, VPA)
}

/// Token `j` (window rotation, 0..3) of a P wave: tile-local token `T`,
/// global `TG`, window rows starting at ring row `R`; rows t-3..t-1 are
/// already in the window. `tag` keeps labels unique per emission site.
fn token(b: &mut Builder, g: usize, re: &Regions, tag: &str, j: u8) -> Result<(), String> {
    // The previous token's stores have read their sources (rows, halves, addresses).
    b.release_store_sources()?;
    // Row t of this head from the ring, over row t-4.
    let cur = ROWS + 4 * ((j + 3) % 4);
    load_row(b, g, 3, (j + 3) % 4)?;
    // Raw row for the completion pass: tile positions 0..2 / 125..127 and the
    // last three tokens. Others (and tokens >= N) go past num_records.
    sop(b, format!("s_cmp_lt_u32 s{T}, 3"), &[], &[T])?;
    sop(b, format!("s_cselect_b32 s{S0}, 1, 0"), &[S0], &[])?;
    sop(b, format!("s_cmp_ge_u32 s{T}, 0x7d"), &[], &[T])?;
    sop(b, format!("s_cselect_b32 s{S1}, 1, 0"), &[S1], &[])?;
    sop(b, format!("s_or_b32 s{S0}, s{S0}, s{S1}"), &[S0], &[S0, S1])?;
    sop(b, format!("s_add_co_i32 s{S1}, s{TG}, 3"), &[S1], &[TG])?;
    sop(b, format!("s_cmp_ge_u32 s{S1}, s31"), &[], &[S1, 31])?;
    sop(b, format!("s_cselect_b32 s{S1}, 1, 0"), &[S1], &[])?;
    sop(b, format!("s_or_b32 s{S0}, s{S0}, s{S1}"), &[S0], &[S0, S1])?;
    sop(b, format!("s_mul_i32 s{RAWOFF}, s{TG}, 0xa000"), &[RAWOFF], &[TG])?;
    sop(b, format!("s_cmp_eq_u32 s{S0}, 0"), &[], &[S0])?;
    sop(b, format!("s_cselect_b32 s{RAWOFF}, s{}, s{RAWOFF}", RAWD + 2), &[RAWOFF], &[RAWD + 2, RAWOFF])?;
    // The token offset rides in VOFFSET: raw-buffer range checks cover it.
    vo(b, format!("v_add_nc_u32_e32 v{VRAWA}, s{RAWOFF}, v{VRAW}"), &[VRAWA], &[VRAW], &[RAWOFF])?;
    mem(b, format!("buffer_store_b128 {}, v{VRAWA}, s[{RAWD}:{}], null offen", vr(cur, 4), RAWD + 3),
        &[], &[vr(cur, 4), v(VRAWA), sr(RAWD, 4)], MemoryClass::VmemStore)?;
    // conv1d + SiLU of the four channels, two interleaved instances at a time.
    for pair in [0u8, 2] {
        let binds = vec![conv_bind(&re.conv, pair, 0, j)?, conv_bind(&re.conv, pair + 1, 1, j)?];
        gdn_region::emit_interleaved(b, &re.conv, &binds)?;
    }
    // Head norm (q, k) or plain conversion (v), selected by the tile's class.
    sop(b, format!("s_cmp_eq_u32 s{CLASS}, 2"), &[], &[CLASS])?;
    op(b, format!("s_cbranch_scc1 {ENTRY}_v_{tag}"), &[], &[])?;
    sop(b, format!("s_cmp_eq_u32 s{CLASS}, 1"), &[], &[CLASS])?;
    op(b, format!("s_cbranch_scc1 {ENTRY}_k_{tag}"), &[], &[])?;
    gdn_region::emit_interleaved(b, &re.norm_q, &[norm_bind(&re.norm_q)?])?;
    op(b, format!("s_branch {ENTRY}_norm_{tag}"), &[], &[])?;
    b.label(&format!("{ENTRY}_k_{tag}"))?;
    gdn_region::emit_interleaved(b, &re.norm_k, &[norm_bind(&re.norm_k)?])?;
    op(b, format!("s_branch {ENTRY}_norm_{tag}"), &[], &[])?;
    b.label(&format!("{ENTRY}_v_{tag}"))?;
    gdn_region::emit_interleaved(b, &re.cvt_v, &[norm_bind(&re.cvt_v)?])?;
    b.label(&format!("{ENTRY}_norm_{tag}"))?;
    // FP16 q/k/v: tile heads 0..2 of tiles after the first belong to the completion pass.
    sop(b, format!("s_mul_i32 s{OUTOFF}, s{TG}, s{TOKSTRIDE}"), &[OUTOFF], &[TG, TOKSTRIDE])?;
    sop(b, format!("s_add_co_i32 s{OUTOFF}, s{OUTOFF}, s{HEAD0}"), &[OUTOFF], &[OUTOFF, HEAD0])?;
    sop(b, "s_cmp_lg_u32 s82, 0", &[], &[82])?;
    sop(b, format!("s_cselect_b32 s{S0}, 1, 0"), &[S0], &[])?;
    sop(b, format!("s_cmp_lt_u32 s{T}, 3"), &[], &[T])?;
    sop(b, format!("s_cselect_b32 s{S1}, 1, 0"), &[S1], &[])?;
    sop(b, format!("s_and_b32 s{S0}, s{S0}, s{S1}"), &[S0], &[S0, S1])?;
    sop(b, format!("s_cmp_eq_u32 s{S0}, 0"), &[], &[S0])?;
    sop(b, format!("s_cselect_b32 s{OUTOFF}, s{OUTOFF}, s{}", OUTD + 2), &[OUTOFF], &[OUTOFF, OUTD + 2])?;
    vo(b, format!("v_add_nc_u32_e32 v{VOUTA}, s{OUTOFF}, v{VOUT}"), &[VOUTA], &[VOUT], &[OUTOFF])?;
    mem(b, format!("buffer_store_b64 {}, v{VOUTA}, s[{OUTD}:{}], null offen", vr(PACK, 2), OUTD + 3),
        &[], &[vr(PACK, 2), v(VOUTA), sr(OUTD, 4)], MemoryClass::VmemStore)?;
    // Next token.
    sop(b, format!("s_add_co_i32 s{T}, s{T}, 1"), &[T], &[T])?;
    sop(b, format!("s_add_co_i32 s{TG}, s{TG}, 1"), &[TG], &[TG])?;
    mod19(b, R, R, 1)
}

pub(super) fn emit(b: &mut Builder) -> Result<(), String> {
    let re = Regions { conv: Region::conv_silu_lean()?, norm_q: Region::norm_q()?, norm_k: Region::norm_k()?, cvt_v: Region::cvt_v()? };
    b.label(ENTRY)?;
    mem(b, "s_load_b256 s[32:39], s[0:1], 0x60", &[sr(KA, 4), sr(KA + 4, 4)], &[sr(0, 2)], MemoryClass::SmemLoad)?;
    mem(b, "s_load_b128 s[40:43], s[0:1], 0x80", &[sr(KA + 8, 4)], &[sr(0, 2)], MemoryClass::SmemLoad)?;
    // Class of this row tile (M0 = 10240 = 2048 q + 2048 k + 6144 v rows).
    sop(b, "s_cmp_ge_u32 s83, 8", &[], &[83])?;
    sop(b, format!("s_cselect_b32 s{CLASS}, 1, 0"), &[CLASS], &[])?;
    sop(b, "s_cmp_ge_u32 s83, 16", &[], &[83])?;
    sop(b, format!("s_cselect_b32 s{S0}, 1, 0"), &[S0], &[])?;
    sop(b, format!("s_add_co_i32 s{CLASS}, s{CLASS}, s{S0}"), &[CLASS], &[CLASS, S0])?;
    // First head of the tile within its q/k/v tensor, as an out byte offset.
    sop(b, format!("s_lshl_b32 s{S0}, s{CLASS}, 4"), &[S0], &[CLASS])?;
    sop(b, format!("s_lshl_b32 s{HEAD0}, s83, 1"), &[HEAD0], &[83])?;
    sop(b, format!("s_sub_co_i32 s{HEAD0}, s{HEAD0}, s{S0}"), &[HEAD0], &[HEAD0, S0])?;
    sop(b, format!("s_lshl_b32 s{HEAD0}, s{HEAD0}, 8"), &[HEAD0], &[HEAD0])?;
    sop(b, format!("s_movk_i32 s{TOKSTRIDE}, 0x1000"), &[TOKSTRIDE], &[])?;
    sop(b, format!("s_cmp_eq_u32 s{CLASS}, 2"), &[], &[CLASS])?;
    sop(b, format!("s_cselect_b32 s{TOKSTRIDE}, 0x3000, s{TOKSTRIDE}"), &[TOKSTRIDE], &[TOKSTRIDE])?;
    // Output base: Q, K or V.
    sop(b, format!("s_cmp_eq_u32 s{CLASS}, 0"), &[], &[CLASS])?;
    op(b, format!("s_cselect_b64 s[{S1}:{S2}], s[{}:{}], s[{}:{}]", KA + 4, KA + 5, KA + 6, KA + 7), &[s(S1), s(S2)], &[sr(KA + 4, 4)])?;
    sop(b, format!("s_cmp_eq_u32 s{CLASS}, 2"), &[], &[CLASS])?;
    op(b, format!("s_cselect_b64 s[{S1}:{S2}], s[{}:{}], s[{S1}:{S2}]", KA + 8, KA + 9), &[s(S1), s(S2)], &[sr(KA + 8, 2), s(S1), s(S2)])?;
    descriptor(b, OUTD, S1, None)?;
    sop(b, format!("s_mov_b32 s{CLAMP}, 0x43000000"), &[CLAMP], &[])?;
    sop(b, format!("s_mul_i32 s{}, s31, s{TOKSTRIDE}", OUTD + 2), &[OUTD + 2], &[31, TOKSTRIDE])?;
    descriptor(b, CWD, KA, None)?;
    descriptor(b, CSD, KA + 2, None)?;
    descriptor(b, RAWD, 18, None)?;
    sop(b, format!("s_mul_i32 s{}, s31, 0xa000", RAWD + 2), &[RAWD + 2], &[31])?;
    sop(b, format!("s_mov_b32 s{LANESEL}, 0x76543210"), &[LANESEL], &[])?;
    // Per-lane constants.
    sop(b, format!("s_lshr_b32 s{HD}, s88, 1"), &[HD], &[88])?;
    sop(b, format!("s_and_b32 s{SUB}, s88, 1"), &[SUB], &[88])?;
    vo(b, format!("v_and_b32_e32 v{VL}, 15, v{TID}"), &[VL], &[TID], &[])?;
    vo(b, format!("v_lshrrev_b32_e32 v{VC}, 4, v{TID}"), &[VC], &[TID], &[])?;
    vo(b, format!("v_and_b32_e32 v{VC}, 1, v{VC}"), &[VC], &[VC], &[])?;
    vo(b, format!("v_lshlrev_b32_e32 v{VC}, 5, v{VC}"), &[VC], &[VC], &[])?;
    sop(b, format!("s_lshl_b32 s{S0}, s88, 8"), &[S0], &[88])?;
    vo(b, format!("v_add_nc_u32_e32 v{VC}, s{S0}, v{VC}"), &[VC], &[VC], &[S0])?;
    vo(b, format!("v_and_b32_e32 v{VP}, 31, v{TID}"), &[VP], &[TID], &[])?;
    vo(b, format!("v_lshlrev_b32_e32 v{VP}, 4, v{VP}"), &[VP], &[VP], &[])?;
    sop(b, format!("s_lshl_b32 s{S0}, s{HD}, 9"), &[S0], &[HD])?;
    vo(b, format!("v_add_nc_u32_e32 v{VP}, s{S0}, v{VP}"), &[VP], &[VP], &[S0])?;
    vo(b, format!("v_and_b32_e32 v{VOUT}, 31, v{TID}"), &[VOUT], &[TID], &[])?;
    vo(b, format!("v_lshlrev_b32_e32 v{VOUT}, 3, v{VOUT}"), &[VOUT], &[VOUT], &[])?;
    sop(b, format!("s_lshl_b32 s{S0}, s{HD}, 8"), &[S0], &[HD])?;
    vo(b, format!("v_add_nc_u32_e32 v{VOUT}, s{S0}, v{VOUT}"), &[VOUT], &[VOUT], &[S0])?;
    sop(b, format!("s_lshl_b32 s{S0}, s83, 10"), &[S0], &[83])?;
    vo(b, format!("v_add_nc_u32_e32 v{VRAW}, s{S0}, v{VP}"), &[VRAW], &[VP], &[S0])?;
    vo(b, format!("v_lshlrev_b32_e32 v{VWOFF}, 2, v{VRAW}"), &[VWOFF], &[VRAW], &[])?;
    // conv taps of this lane's four channels: w[tap][c] = conv_w[(ch + c) * 4 + tap].
    for c in 0..4u8 { for tap in 0..4u8 { vload(b, W + 4 * tap + c, 1, VWOFF, CWD, None, u32::from(c) * 16 + u32::from(tap) * 4)?; } }
    // The K-loop's last barrier retired every staging slot.
    b.lds_relayout()?;
    let g = b.lds_slot("gdn_ring", 0, RING_BYTES)?;
    sop(b, format!("s_mov_b32 s{BLK}, 0"), &[BLK], &[])?;
    sop(b, format!("s_mov_b32 s{RB}, 3"), &[RB], &[])?;
    b.wait_all()?;
    b.loop_(&format!("{ENTRY}_block"), |b| {
        sop(b, format!("s_and_b32 s{TT}, s{BLK}, 3"), &[TT], &[BLK])?;
        sop(b, format!("s_lshr_b32 s{HALF}, s{BLK}, 2"), &[HALF], &[BLK])?;
        // W: the token half holding tokens 16b..16b+15 stores column block b % 4.
        sop(b, format!("s_cmp_lg_u32 s89, s{HALF}"), &[], &[89, HALF])?;
        op(b, format!("s_cbranch_scc1 {ENTRY}_w_skip"), &[], &[])?;
        vo(b, format!("v_add_nc_u32_e32 v{VA}, s{RB}, v{VL}"), &[VA], &[VL], &[RB])?;
        vo(b, format!("v_subrev_nc_u32_e32 v{VPA}, {RING_ROWS}, v{VA}"), &[VPA], &[VA], &[])?;
        vo(b, format!("v_min_u32_e32 v{VA}, v{VA}, v{VPA}"), &[VA], &[VA, VPA], &[])?;
        vo(b, format!("v_mad_u32_u24 v{VA}, v{VA}, {ROW_BYTES:#x}, v{VC}"), &[VA], &[VA, VC], &[])?;
        for tt in 0..4u8 {
            sop(b, format!("s_cmp_eq_u32 s{TT}, {tt}"), &[], &[TT])?;
            op(b, format!("s_cbranch_scc1 {ENTRY}_w{tt}"), &[], &[])?;
        }
        for tt in 0..4u8 {
            b.label(&format!("{ENTRY}_w{tt}"))?;
            for i in 0..4u8 { for h in 0..2u8 {
                ds_store_b128(b, g, VA, 32 * i + 8 * tt + 4 * h, u32::from(i) * 64 + u32::from(h) * 16)?;
            }}
            op(b, format!("s_branch {ENTRY}_w_done"), &[], &[])?;
        }
        b.label(&format!("{ENTRY}_w_done"))?;
        // Halo rows 0..2 (tokens -3..-1) before block 0: the conv ring for the
        // first tile, zeros elsewhere (those three outputs are the completion pass's).
        sop(b, format!("s_cmp_lg_u32 s{BLK}, 0"), &[], &[BLK])?;
        op(b, format!("s_cbranch_scc1 {ENTRY}_w_skip"), &[], &[])?;
        sop(b, format!("s_cmp_lg_u32 s{SUB}, 0"), &[], &[SUB])?;
        op(b, format!("s_cbranch_scc1 {ENTRY}_w_skip"), &[], &[])?;
        sop(b, "s_cmp_lg_u32 s82, 0", &[], &[82])?;
        op(b, format!("s_cbranch_scc1 {ENTRY}_halo_zero"), &[], &[])?;
        vo(b, format!("v_mul_u32_u24_e32 v{VPA}, 3, v{VRAW}"), &[VPA], &[VRAW], &[])?;
        for c in 0..4u8 { for k in 0..3u8 {
            // conv_state[ch*3 + k] is x[-1-k]: ring row 2-k.
            vload(b, ROWS + 4 * (2 - k) + c, 1, VPA, CSD, None, u32::from(c) * 12 + u32::from(k) * 4)?;
        }}
        // Both paths reach the stores with nothing pending (the ledger is path-insensitive).
        b.wait(Counter::Load, 0)?;
        op(b, format!("s_branch {ENTRY}_halo_store"), &[], &[])?;
        b.label(&format!("{ENTRY}_halo_zero"))?;
        for r in 0..12u8 { vo(b, format!("v_mov_b32_e32 v{}, 0", ROWS + r), &[ROWS + r], &[], &[])?; }
        b.label(&format!("{ENTRY}_halo_store"))?;
        for r in 0..3u8 { ds_store_b128(b, g, VP, ROWS + 4 * r, u32::from(r) * ROW_BYTES)?; }
        b.label(&format!("{ENTRY}_w_skip"))?;
        b.barrier(&[Transition::Ready(g)])?;
        // P: wave rg owns head rg >> 1 and tokens 16b + 8(rg & 1) .. +7.
        sop(b, format!("s_cmp_lg_u32 s89, s{HALF}"), &[], &[89, HALF])?;
        op(b, format!("s_cbranch_scc1 {ENTRY}_p_skip"), &[], &[])?;
        sop(b, format!("s_lshl_b32 s{T}, s{BLK}, 4"), &[T], &[BLK])?;
        sop(b, format!("s_lshl_b32 s{S0}, s{SUB}, 3"), &[S0], &[SUB])?;
        sop(b, format!("s_add_co_i32 s{T}, s{T}, s{S0}"), &[T], &[T, S0])?;
        sop(b, format!("s_add_co_i32 s{TG}, s82, s{T}"), &[TG], &[82, T])?;
        // Row of token t0-3 = t0 mod 19 = (RB + 16 + 8*sub) mod 19.
        mod19(b, R, RB, 16)?;
        sop(b, format!("s_add_co_i32 s{R}, s{R}, s{S0}"), &[R], &[R, S0])?;
        mod19(b, R, R, 0)?;
        // Window rows t0-3..t0-1 into slots 0..2; the loop body starts empty.
        for k in 0..3u8 { load_row(b, g, k, k)?; }
        b.wait(Counter::Ds, 0)?;
        sop(b, format!("s_mov_b32 s{J}, 2"), &[J], &[])?;
        b.loop_(&format!("{ENTRY}_token"), |b| {
            for j in 0..4u8 { token(b, g, &re, &format!("l{j}"), j)?; }
            b.release_store_sources()?;
            sop(b, format!("s_add_co_i32 s{J}, s{J}, -1"), &[J], &[J])?;
            sop(b, format!("s_cmp_lg_u32 s{J}, 0"), &[], &[J])?;
            op(b, format!("s_cbranch_scc1 {ENTRY}_token"), &[], &[])
        })?;
        b.label(&format!("{ENTRY}_p_skip"))?;
        b.barrier(&[Transition::Retire(g)])?;
        // Every block starts from an empty ledger (loop fixpoint).
        b.release_store_sources()?;
        sop(b, format!("s_add_co_i32 s{BLK}, s{BLK}, 1"), &[BLK], &[BLK])?;
        mod19(b, RB, RB, 16)?;
        sop(b, format!("s_cmp_lg_u32 s{BLK}, 8"), &[], &[BLK])?;
        op(b, format!("s_cbranch_scc1 {ENTRY}_block"), &[], &[])
    })?;
    b.wait_all()
}
