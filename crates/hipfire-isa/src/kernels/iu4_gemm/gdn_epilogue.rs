//! GDN preparation fused into the fused A4 input projection
//! (`Epi::QkvzaGdn`), the `_b1` port of F2's `fp8_gemm::gdn_epilogue`.
//!
//! A QKV row tile (128 channels = one 128-channel q, k or v head) x 128
//! tokens is turned into `gdn_chunk_prep`'s FP16 q/k/v directly: width-4
//! causal conv1d over tokens, SiLU, and for q/k the per-token head RMS norm
//! (and q_scale). The f32 projection is never stored except the raw rows the
//! completion pass `gdn_chunk_prep_fixup` needs: tile positions 0..2 and
//! 125..127 and the last three tokens, into `X` (stride 10240). Tile heads
//! 128j..128j+2 (j >= 1) are left to that pass; the first token tile reads
//! the persistent conv ring as its halo.
//!
//! LDS is one 19-row ring of 128 f32 channels (rows 528 bytes apart, 10,032
//! bytes of the 20,480-byte `_b1` allocation): token t of the tile lives in
//! row (t + 3) mod 19. Block b (tokens 16b..16b+15) is drained from column
//! block b % 4 of token half b / 4 by that half's four waves (lane (m, k) of
//! wave pair p stores rows 32p + 16rg + 8k + 0..7 of token m); then each of
//! those waves runs four consecutive tokens 16b + 4p .. +3 in
//! `gdn_chunk_prep`'s lane layout (lane = 4 channels). Region instances (lean
//! conv+SiLU, the imported norm goldens) use F2's register numbers, so their
//! VOPD packing is F2's.
//!
//! Two choices only schedule the same instructions:
//! - The 16 bytes of row padding put the 16 token rows of a W store four
//!   dword banks apart. With 512-byte rows every lane of a store hit the same
//!   four banks (16-way conflicts on the LDS the co-resident K-loops need).
//! - The GDN path runs at wave priority 0 under the kernel's priority 1
//!   (`mod.rs`), so co-resident K-loop waves win instruction arbitration and
//!   the epilogue's VALU work fills their stall cycles.
use super::{GDN, Gen, Lds, Wg, Wv, publish::vload};
use crate::kernels::common::{mem, op, s, sop, sr, v, vr};
use crate::{RegPlan, insn::{Instruction, MemoryClass}, kernels::fp8_gemm::gdn_region::{self, Binding, Half, Region}, ledger::Counter, reg::Live};
use peacemaker_author::{LdsRegion, Published, Scc, Uniform, ready, retire, retire_cur};

/// LDS tag of the token ring (19 rows of 528 bytes) after the K loop's re-layout.
pub enum GdnRing {}

const RING_ROWS: u32 = 19;
/// Ring row pitch: 512 bytes of data plus 16 (bank skew; keeps b128 alignment).
const ROW_BYTES: u32 = 528;
pub const RING_BYTES: u32 = RING_ROWS * ROW_BYTES;

// VGPRs (accumulators v64..v127 drain block by block; v185 = sc_addr is the
// lane's ring channel offset (32p + 8k) * 4).
const PACK: u8 = 0; // FP16 halves h0..h3 in v0.l v0.h v1.l v1.h (true16: below v128)
const W: u8 = 128; // conv taps: w[tap][c] at W + 4*tap + c
// window: four slots of four channels at ROWS + 4*slot + c. Token j of a run
// reads rows t-3..t from slots j, j+1, j+2, j+3 (mod 4) and loads only row t,
// into slot (j + 3) % 4, over the slot of row t-4.
const ROWS: u8 = 144;
const CONV: u8 = 160; // two conv+SiLU instances x 8 temps; the norm reuses CONV..CONV+7
const OUT: u8 = 176; // o0..o3 (SiLU outputs of channels 0..3)
const VA: u8 = 180; // W: LDS address of this lane's ring row
const VL: u8 = 182; // lane & 15
const VP: u8 = 183; // P: lane*16 (head channels in a ring row)
const VPA: u8 = 184; // P: row address / scratch
const VC: u8 = 185; // sc_addr
const LANE: u8 = 187;
const VOUT: u8 = 188; // out: lane*8
const VRAW: u8 = 189; // raw rows / weights: 4*ch, ch = rs + lane*4
const VWOFF: u8 = 190; // weights: 16*ch (only before the block loop)
const VOUTA: u8 = 190; // out store address (reuses VWOFF once the taps landed)
const VRAWA: u8 = 191; // raw store address

// SGPRs (K-loop descriptors, the store-path scalars and temporaries are dead).
const N: u8 = 23;
const RS: u8 = 48; // segment-relative first row: the tile's head is RS / 128
const BS: u8 = 49; // first token of the tile
const WAVE: u8 = 50;
const KA: u8 = 32; // s[32:39] = ConvW ConvState Q K16 ; s[40:43] = V X ; s[44:45] = QScale Eps
const CLAMP: u8 = 36; // 128.0, the lean SiLU's exp clamp (Q's pointer is dead once OUTD is built)
const QSCALE: u8 = 44;
const EPS: u8 = 45;
const NORM_MASK: u8 = 46;
const RAWD: u8 = 52; // X with num_records = N * 10240 * 4
const OUTD: u8 = 56; // selected q/k/v, num_records = N * heads * 256
const CWD: u8 = 60;
const CSD: u8 = 64;
const TOKSTRIDE: u8 = 68; // out bytes per token: heads * 256
const LANESEL: u8 = 69; // 0x76543210
const HEAD0: u8 = 70; // out byte offset of the tile's head
const CLASS: u8 = 71; // 0 q, 1 k, 2 v
const TH: u8 = 72; // wave & 1: the wave's token half
const SUB: u8 = 73; // wave >> 1: which 4 tokens of a block the wave runs
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
const OUTOFF: u8 = 101;
const RAWOFF: u8 = 102;

/// GDN-phase register ranges (the K-loop/store ranges end at `GDN`).
pub(crate) fn plan(p: &mut RegPlan) -> Result<(), String> {
    let g = || Live::Between(GDN.into(), super::END.into());
    p.v::<2>("gdn_pack", PACK, g())?;
    for i in 0..6u8 { p.v::<8>("gdn_taps_window_conv", W + 8 * i, g())?; }
    p.v::<4>("gdn_out", OUT, g())?;
    for r in [VA, 181, VL, VP, VPA, LANE, VOUT, VRAW, VWOFF, VRAWA] { p.v::<1>("gdn_lane", r, g())?; }
    p.s::<8>("gdn_kernargs", KA, g())?;
    p.s::<4>("gdn_kernargs_v_x", KA + 8, g())?;
    p.s::<2>("gdn_scale_eps", QSCALE, g())?;
    p.s::<1>("gdn_norm_mask", NORM_MASK, g())?;
    p.s::<1>("gdn_s47", 47, g())?;
    p.s::<1>("gdn_s51", 51, g())?;
    for d in [RAWD, OUTD, CWD, CSD] { p.s::<4>("gdn_descriptor", d, g())?; }
    for r in 68..104u8 { p.s::<1>("gdn_scalar", r, g())?; }
    Ok(())
}

fn vo(b: &mut crate::Builder, text: impl Into<String>, defs: &[u8], uses: &[u8], su: &[u8]) -> Result<(), String> {
    op(b, text, &defs.iter().map(|&n| v(n)).collect::<Vec<_>>(), &uses.iter().map(|&n| v(n)).chain(su.iter().map(|&n| s(n))).collect::<Vec<_>>())
}
fn descriptor(b: &mut crate::Builder, dst: u8, lo: u8) -> Result<(), String> {
    sop(b, format!("s_mov_b32 s{dst}, s{lo}"), &[dst], &[lo])?;
    sop(b, format!("s_and_b32 s{}, s{}, 0xffff", dst + 1, lo + 1), &[dst + 1], &[lo + 1])?;
    sop(b, format!("s_mov_b32 s{}, -1", dst + 2), &[dst + 2], &[])?;
    sop(b, format!("s_mov_b32 s{}, 0x31004000", dst + 3), &[dst + 3], &[])
}
/// `dst = (x + k) mod 19` for x < 19, k <= 19 (one conditional subtraction).
fn mod19(b: &mut crate::Builder, dst: u8, x: u8, k: u32) -> Result<(), String> {
    if k == 0 { sop(b, format!("s_mov_b32 s{dst}, s{x}"), &[dst], &[x])?; } else { sop(b, format!("s_add_co_i32 s{dst}, s{x}, {k}"), &[dst], &[x])?; }
    sop(b, format!("s_sub_co_i32 s{S2}, s{dst}, {RING_ROWS}"), &[S2], &[dst])?;
    sop(b, format!("s_cmp_ge_u32 s{dst}, {RING_ROWS}"), &[], &[dst])?;
    sop(b, format!("s_cselect_b32 s{dst}, s{S2}, s{dst}"), &[dst], &[S2, dst])
}
fn ds_store_b128(addr: u8, data: u8, offset: u32) -> Instruction {
    let off = if offset == 0 { String::new() } else { format!(" offset:{offset}") };
    Instruction::new(format!("ds_store_b128 v{addr}, {}{off}", vr(data, 4)), vec![], vec![v(addr), vr(data, 4)]).memory(MemoryClass::DsStore)
}
fn ds_load_b128(data: u8, addr: u8) -> Instruction {
    Instruction::new(format!("ds_load_b128 {}, v{addr}", vr(data, 4)), vec![vr(data, 4)], vec![v(addr)]).memory(MemoryClass::DsLoad)
}
/// `s_cmp_lg_u32 TH, HALF`: this wave's token half is not block `BLK`'s.
fn other_half(w: &mut Wv) -> Result<Uniform<Scc>, String> {
    cmp(w, format!("s_cmp_lg_u32 s{TH}, s{HALF}"), &[TH, HALF])
}
/// A scalar compare over single SGPRs (wave-uniform SCC).
fn cmp(w: &mut Wv, text: String, uses: &[u8]) -> Result<Uniform<Scc>, String> {
    w.scmp(Instruction::new(text, vec![], uses.iter().map(|&n| s(n)).collect()))
}

struct Regions { conv: Region, norm_q: Region, norm_k: Region, cvt_v: Region }

/// Instance `k` (0/1) of a conv+SiLU pair for channel `c` of token `j`
/// (window rotation): temps rotated by `k` so VOPD partners differ in bank
/// and destination parity (F2's binding).
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

/// Load ring row `(R + k) mod 19` into window slot `slot`.
fn load_row(w: &mut Wv, ring: &LdsRegion<GdnRing, Published>, k: u8, slot: u8) -> Result<(), String> {
    let b = w.isa();
    mod19(b, S0, R, u32::from(k))?;
    sop(b, format!("s_mul_i32 s{S0}, s{S0}, {ROW_BYTES:#x}"), &[S0], &[S0])?;
    vo(b, format!("v_add_nc_u32_e32 v{VPA}, s{S0}, v{VP}"), &[VPA], &[VP], &[S0])?;
    w.ds_load(ring, ds_load_b128(ROWS + 4 * slot, VPA))
}

/// Token `j` (window rotation, 0..3) of a P wave: tile-local token `T`,
/// global `TG`, window rows starting at ring row `R`; rows t-3..t-1 are
/// already in the window. `tag` keeps labels unique per emission site.
fn token(w: &mut Wv, ring: &LdsRegion<GdnRing, Published>, re: &Regions, tag: &str, j: u8) -> Result<(), String> {
    // The previous token's stores have read their sources (rows, halves, addresses).
    w.isa().release_store_sources()?;
    // Row t from the ring, over row t-4.
    let cur = ROWS + 4 * ((j + 3) % 4);
    load_row(w, ring, 3, (j + 3) % 4)?;
    let b = w.isa();
    // Raw row for the completion pass: tile positions 0..2 / 125..127 and the
    // last three tokens. Others (and tokens >= N) go past num_records.
    sop(b, format!("s_cmp_lt_u32 s{T}, 3"), &[], &[T])?;
    sop(b, format!("s_cselect_b32 s{S0}, 1, 0"), &[S0], &[])?;
    sop(b, format!("s_cmp_ge_u32 s{T}, 0x7d"), &[], &[T])?;
    sop(b, format!("s_cselect_b32 s{S1}, 1, 0"), &[S1], &[])?;
    sop(b, format!("s_or_b32 s{S0}, s{S0}, s{S1}"), &[S0], &[S0, S1])?;
    sop(b, format!("s_add_co_i32 s{S1}, s{TG}, 3"), &[S1], &[TG])?;
    sop(b, format!("s_cmp_ge_u32 s{S1}, s{N}"), &[], &[S1, N])?;
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
    let (v_arm, k_arm, norm) = (format!("{GDN}_v_{tag}"), format!("{GDN}_k_{tag}"), format!("{GDN}_norm_{tag}"));
    w.forward(|w, f| {
        let v_class = cmp(w, format!("s_cmp_eq_u32 s{CLASS}, 2"), &[CLASS])?;
        f.branch_if(w, v_class, &v_arm)?;
        let k_class = cmp(w, format!("s_cmp_eq_u32 s{CLASS}, 1"), &[CLASS])?;
        f.branch_if(w, k_class, &k_arm)?;
        gdn_region::emit_interleaved(w.isa(), &re.norm_q, &[norm_bind(&re.norm_q)?])?;
        f.goto(w, &norm)?;
        f.place(w, &k_arm)?;
        gdn_region::emit_interleaved(w.isa(), &re.norm_k, &[norm_bind(&re.norm_k)?])?;
        f.goto(w, &norm)?;
        f.place(w, &v_arm)?;
        gdn_region::emit_interleaved(w.isa(), &re.cvt_v, &[norm_bind(&re.cvt_v)?])?;
        f.place(w, &norm)
    })?;
    let b = w.isa();
    // FP16 q/k/v: tile heads 0..2 of token tiles after the first belong to the completion pass.
    sop(b, format!("s_mul_i32 s{OUTOFF}, s{TG}, s{TOKSTRIDE}"), &[OUTOFF], &[TG, TOKSTRIDE])?;
    sop(b, format!("s_add_co_i32 s{OUTOFF}, s{OUTOFF}, s{HEAD0}"), &[OUTOFF], &[OUTOFF, HEAD0])?;
    sop(b, format!("s_cmp_lg_u32 s{BS}, 0"), &[], &[BS])?;
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

pub(crate) fn emit(wg: &mut Wg, g: &Gen, (ra, rw, rd, rs): Lds) -> Result<(), String> {
    let b = wg.isa();
    let re = Regions { conv: Region::conv_silu_lean()?, norm_q: Region::norm_q()?, norm_k: Region::norm_k()?, cvt_v: Region::cvt_v()? };
    // The epilogue's exit branch placed GDN (`epilogue::fused_stores`).
    // Below the K-loop's priority 1: co-resident K-loops issue first.
    op(b, "s_setprio 0", &[], &[])?;
    // The GDN arithmetic is VALU-issue bound: give it F2's issue hints.
    b.enable_delay_alu();
    mem(b, format!("s_load_b256 s[{KA}:{}], s[0:1], 0x40", KA + 7), &[sr(KA, 8)], &[sr(0, 2)], MemoryClass::SmemLoad)?;
    mem(b, format!("s_load_b128 s[{}:{}], s[0:1], 0x60", KA + 8, KA + 11), &[sr(KA + 8, 4)], &[sr(0, 2)], MemoryClass::SmemLoad)?;
    mem(b, format!("s_load_b64 s[{QSCALE}:{EPS}], s[0:1], 0x70"), &[sr(QSCALE, 2)], &[sr(0, 2)], MemoryClass::SmemLoad)?;
    // Class of this head tile (M0 = 10240 = 2048 q + 2048 k + 6144 v rows).
    sop(b, format!("s_lshr_b32 s{S1}, s{RS}, 7"), &[S1], &[RS])?;
    sop(b, format!("s_cmp_ge_u32 s{S1}, 16"), &[], &[S1])?;
    sop(b, format!("s_cselect_b32 s{CLASS}, 1, 0"), &[CLASS], &[])?;
    sop(b, format!("s_cmp_ge_u32 s{S1}, 32"), &[], &[S1])?;
    sop(b, format!("s_cselect_b32 s{S0}, 1, 0"), &[S0], &[])?;
    sop(b, format!("s_add_co_i32 s{CLASS}, s{CLASS}, s{S0}"), &[CLASS], &[CLASS, S0])?;
    // The tile's head within its q/k/v tensor, as an out byte offset.
    sop(b, format!("s_lshl_b32 s{S0}, s{CLASS}, 4"), &[S0], &[CLASS])?;
    sop(b, format!("s_sub_co_i32 s{HEAD0}, s{S1}, s{S0}"), &[HEAD0], &[S1, S0])?;
    sop(b, format!("s_lshl_b32 s{HEAD0}, s{HEAD0}, 8"), &[HEAD0], &[HEAD0])?;
    sop(b, format!("s_movk_i32 s{TOKSTRIDE}, 0x1000"), &[TOKSTRIDE], &[])?;
    sop(b, format!("s_cmp_eq_u32 s{CLASS}, 2"), &[], &[CLASS])?;
    sop(b, format!("s_cselect_b32 s{TOKSTRIDE}, 0x3000, s{TOKSTRIDE}"), &[TOKSTRIDE], &[TOKSTRIDE])?;
    // Output base: Q, K or V.
    sop(b, format!("s_cmp_eq_u32 s{CLASS}, 0"), &[], &[CLASS])?;
    op(b, format!("s_cselect_b64 s[{S1}:{S2}], s[{}:{}], s[{}:{}]", KA + 4, KA + 5, KA + 6, KA + 7), &[s(S1), s(S2)], &[sr(KA + 4, 4)])?;
    sop(b, format!("s_cmp_eq_u32 s{CLASS}, 2"), &[], &[CLASS])?;
    op(b, format!("s_cselect_b64 s[{S1}:{S2}], s[{}:{}], s[{S1}:{S2}]", KA + 8, KA + 9), &[s(S1), s(S2)], &[sr(KA + 8, 2), s(S1), s(S2)])?;
    descriptor(b, OUTD, S1)?;
    sop(b, format!("s_mov_b32 s{CLAMP}, 0x43000000"), &[CLAMP], &[])?;
    sop(b, format!("s_mul_i32 s{}, s{N}, s{TOKSTRIDE}", OUTD + 2), &[OUTD + 2], &[N, TOKSTRIDE])?;
    descriptor(b, CWD, KA)?;
    descriptor(b, CSD, KA + 2)?;
    descriptor(b, RAWD, KA + 10)?;
    sop(b, format!("s_mul_i32 s{}, s{N}, 0xa000", RAWD + 2), &[RAWD + 2], &[N])?;
    sop(b, format!("s_mov_b32 s{LANESEL}, 0x76543210"), &[LANESEL], &[])?;
    // Per-lane constants.
    sop(b, format!("s_and_b32 s{TH}, s{WAVE}, 1"), &[TH], &[WAVE])?;
    sop(b, format!("s_lshr_b32 s{SUB}, s{WAVE}, 1"), &[SUB], &[WAVE])?;
    vo(b, format!("v_mbcnt_lo_u32_b32 v{LANE}, -1, 0"), &[LANE], &[], &[])?;
    vo(b, format!("v_and_b32_e32 v{VL}, 15, v{LANE}"), &[VL], &[LANE], &[])?;
    vo(b, format!("v_lshlrev_b32_e32 v{VP}, 4, v{LANE}"), &[VP], &[LANE], &[])?;
    vo(b, format!("v_lshlrev_b32_e32 v{VOUT}, 3, v{LANE}"), &[VOUT], &[LANE], &[])?;
    sop(b, format!("s_lshl_b32 s{S0}, s{RS}, 2"), &[S0], &[RS])?;
    vo(b, format!("v_add_nc_u32_e32 v{VRAW}, s{S0}, v{VP}"), &[VRAW], &[VP], &[S0])?;
    vo(b, format!("v_lshlrev_b32_e32 v{VWOFF}, 2, v{VRAW}"), &[VWOFF], &[VRAW], &[])?;
    // conv taps of this lane's four channels: w[tap][c] = conv_w[(ch + c) * 4 + tap].
    for c in 0..4u8 { for tap in 0..4u8 { vload(b, W + 4 * tap + c, 1, VWOFF, CWD, None, u32::from(c) * 16 + u32::from(tap) * 4)?; } }
    // Every wave has finished its last block's fragment and plane reads.
    let lds = wg.barrier((retire_cur(ra), retire_cur(rw), retire_cur(rd), retire_cur(rs)))?;
    wg.relayout(lds)?;
    let ring = wg.lds::<GdnRing>("gdn_ring", 0, RING_BYTES)?;
    if RING_BYTES > g.layout.launch { return Err("GDN ring exceeds the _b1 LDS allocation".into()) }
    let b = wg.isa();
    sop(b, format!("s_mov_b32 s{BLK}, 0"), &[BLK], &[])?;
    sop(b, format!("s_mov_b32 s{RB}, 3"), &[RB], &[])?;
    b.wait_all()?;
    let _free = wg.loop_carried(&format!("{GDN}_block"), ring, |wg, ring| {
        let b = wg.isa();
        sop(b, format!("s_and_b32 s{TT}, s{BLK}, 3"), &[TT], &[BLK])?;
        sop(b, format!("s_lshr_b32 s{HALF}, s{BLK}, 2"), &[HALF], &[BLK])?;
        // W: the token half holding tokens 16b..16b+15 stores column block b % 4.
        let other = other_half(wg)?;
        let staged = wg.begin_write(ring);
        let (skip, done, halo_zero, halo_store) =
            (format!("{GDN}_w_skip"), format!("{GDN}_w_done"), format!("{GDN}_halo_zero"), format!("{GDN}_halo_store"));
        let (ring, stores) = wg.forward(|w, f| {
            let mut st = staged;
            f.branch_if(w, other, &skip)?;
            let b = w.isa();
            vo(b, format!("v_add_nc_u32_e32 v{VA}, s{RB}, v{VL}"), &[VA], &[VL], &[RB])?;
            vo(b, format!("v_subrev_nc_u32_e32 v{VPA}, {RING_ROWS}, v{VA}"), &[VPA], &[VA], &[])?;
            vo(b, format!("v_min_u32_e32 v{VA}, v{VA}, v{VPA}"), &[VA], &[VA, VPA], &[])?;
            vo(b, format!("v_mad_u32_u24 v{VA}, v{VA}, {ROW_BYTES:#x}, v{VC}"), &[VA], &[VA, VC], &[])?;
            for tt in 0..4u8 {
                let c = cmp(w, format!("s_cmp_eq_u32 s{TT}, {tt}"), &[TT])?;
                f.branch_if(w, c, &format!("{GDN}_w{tt}"))?;
            }
            for tt in 0..4u8 {
                f.place(w, &format!("{GDN}_w{tt}"))?;
                for rg in 0..2u8 { for c in 0..2u8 {
                    st = w.ds_store(st, ds_store_b128(VA, g.acc + 8 * (2 * tt + rg) + 4 * c, u32::from(rg) * 64 + u32::from(c) * 16))?;
                }}
                f.goto(w, &done)?;
            }
            f.place(w, &done)?;
            // Halo rows 0..2 (tokens -3..-1) before block 0: the conv ring for the
            // first token tile, zeros elsewhere (those three outputs are the
            // completion pass's). One wave (token half 0, pair 0) writes all 128 channels.
            let c = cmp(w, format!("s_cmp_lg_u32 s{BLK}, 0"), &[BLK])?;
            f.branch_if(w, c, &skip)?;
            let c = cmp(w, format!("s_cmp_lg_u32 s{SUB}, 0"), &[SUB])?;
            f.branch_if(w, c, &skip)?;
            let c = cmp(w, format!("s_cmp_lg_u32 s{BS}, 0"), &[BS])?;
            f.branch_if(w, c, &halo_zero)?;
            let b = w.isa();
            vo(b, format!("v_mul_u32_u24_e32 v{VPA}, 3, v{VRAW}"), &[VPA], &[VRAW], &[])?;
            for c in 0..4u8 { for k in 0..3u8 {
                // conv_state[ch*3 + k] is x[-1-k]: ring row 2-k.
                vload(b, ROWS + 4 * (2 - k) + c, 1, VPA, CSD, None, u32::from(c) * 12 + u32::from(k) * 4)?;
            }}
            // The conv-state rows are complete before the zero path joins.
            b.wait(Counter::Load, 0)?;
            f.goto(w, &halo_store)?;
            f.place(w, &halo_zero)?;
            let b = w.isa();
            for r in 0..12u8 { vo(b, format!("v_mov_b32_e32 v{}, 0", ROWS + r), &[ROWS + r], &[], &[])?; }
            f.place(w, &halo_store)?;
            for r in 0..3u8 { st = w.ds_store(st, ds_store_b128(VP, ROWS + 4 * r, u32::from(r) * ROW_BYTES))?; }
            f.place(w, &skip)?;
            Ok(st)
        })?;
        let drained = wg.wait(stores)?;
        let (ring,) = wg.barrier((ready(ring, drained),))?;
        // P: wave pair SUB runs tokens 16b + 4*SUB .. +3.
        let other = other_half(wg)?;
        wg.skip_if(other, &format!("{GDN}_p_skip"), (), |w, ()| {
            let b = w.isa();
            sop(b, format!("s_lshl_b32 s{T}, s{BLK}, 4"), &[T], &[BLK])?;
            sop(b, format!("s_lshl_b32 s{S0}, s{SUB}, 2"), &[S0], &[SUB])?;
            sop(b, format!("s_add_co_i32 s{T}, s{T}, s{S0}"), &[T], &[T, S0])?;
            sop(b, format!("s_add_co_i32 s{TG}, s{BS}, s{T}"), &[TG], &[BS, T])?;
            // Row of token t0-3 = t0 mod 19 = (RB + 16 + 4*sub) mod 19.
            mod19(b, R, RB, 16)?;
            sop(b, format!("s_add_co_i32 s{R}, s{R}, s{S0}"), &[R], &[R, S0])?;
            mod19(b, R, R, 0)?;
            // Window rows t0-3..t0-1 into slots 0..2.
            for k in 0..3u8 { load_row(w, &ring, k, k)?; }
            w.isa().wait(Counter::Ds, 0)?;
            for j in 0..4u8 { token(w, &ring, &re, &format!("t{j}"), j)?; }
            w.isa().release_store_sources()
        })?;
        let (ring,) = wg.barrier((retire(ring),))?;
        let b = wg.isa();
        // Every block starts from an empty ledger (loop fixpoint).
        b.release_store_sources()?;
        sop(b, format!("s_add_co_i32 s{BLK}, s{BLK}, 1"), &[BLK], &[BLK])?;
        mod19(b, RB, RB, 16)?;
        // BLK counts blocks from the constant 0: workgroup-uniform.
        let more = wg.scmp_wg_uniform(Instruction::new(format!("s_cmp_lg_u32 s{BLK}, 8"), vec![], vec![s(BLK)]))?;
        Ok((ring, more))
    })?;
    wg.isa().wait_all()
}

/// Certify the fused projection's LDS use beyond the builder's slot model:
/// the store paths touch no LDS, and every GDN ring access uses one of three
/// address registers whose derivations are matched exactly, with the ring-row
/// SGPRs only changing through `mod 19` idioms that keep them in [0, 19).
/// Lines exclude labels, waits and issue hints; adjacency is over the rest.
/// Returns the largest byte end of a GDN access (the K-loop's slots end at
/// `Layout::end`, proven by the slot model).
pub fn check_lds_access(source: &str, symbol: &str, launch_dynamic: u32) -> Result<u32, String> {
    let fail = |what: String| Err(format!("{symbol}: GDN LDS certificate: {what}"));
    let launch = super::Tile::T128x128x8.layout().launch;
    if launch_dynamic != launch { return fail(format!("expected {launch} launch LDS bytes, got {launch_dynamic}")) }
    let marker = format!("\n{symbol}:\n");
    let (_, rest) = source.split_once(&marker).ok_or("iu4 GDN LDS certificate missing symbol")?;
    let (kernel, _) = rest.split_once("\ts_endpgm").ok_or("iu4 GDN LDS certificate missing kernel end")?;
    let (body, epilogue) = kernel.split_once("_epilogue:\n").ok_or("missing epilogue")?;
    let (plain, gdn) = epilogue.split_once("_gdn:\n").ok_or("missing GDN label")?;
    if plain.lines().any(|l| l.trim().starts_with("ds_")) { return fail("LDS access in the store paths".into()) }
    let clean = |t: &str| -> Vec<String> { t.lines().map(str::trim).filter(|l| !l.is_empty() && !l.ends_with(':') && !l.starts_with("s_delay_alu") && !l.starts_with("s_wait")).map(String::from).collect() };
    fn dst(l: &str) -> &str { l.split_whitespace().nth(1).map(|t| t.trim_end_matches(',')).unwrap_or("") }
    let writes = |l: &str| !["ds_", "buffer_store", "s_cmp", "s_bitcmp", "s_cbranch", "s_branch", "s_barrier"].iter().any(|p| l.starts_with(p));
    // sc_addr v185 = (wave>>1)*128 + (tid&16)*2 <= 416, defined once in the prologue.
    let pro = clean(body);
    let at_p = |i: usize| pro.get(i).map(String::as_str).unwrap_or("");
    let before_p = |reg: &str, i: usize| (0..i).rev().find(|&j| writes(at_p(j)) && dst(at_p(j)) == reg).map(at_p);
    let sc: Vec<usize> = pro.iter().enumerate().filter(|(_, l)| writes(l) && dst(l) == "v185").map(|(i, _)| i).collect();
    let [i] = sc[..] else { return fail("v185 must have one prologue definition".into()) };
    let last_def = |reg: &str, i: usize| (0..i).rev().find(|&j| writes(at_p(j)) && dst(at_p(j)) == reg);
    let (j13, j58) = (last_def("v13", i).unwrap_or(0), last_def("s58", i).unwrap_or(0));
    let chain = at_p(i) == "v_add_nc_u32_e32 v185, s58, v13"
        && at_p(j13) == "v_lshlrev_b32_e32 v13, 1, v2"
        && before_p("v2", j13) == Some("v_and_b32_e32 v2, 16, v0")
        && at_p(j58) == "s_lshl_b32 s58, s57, 7"
        && before_p("s57", j58) == Some("s_lshr_b32 s57, s50, 1")
        && pro.iter().filter(|l| writes(l) && dst(l) == "s50").map(String::as_str).collect::<Vec<_>>() == ["v_readfirstlane_b32 s50, v0", "s_lshr_b32 s50, s50, 5"]
        && !pro[..i].iter().any(|l| writes(l) && dst(l) == "v0");
    if !chain { return fail("sc_addr v185 derivation changed".into()) }
    let lines = clean(gdn);
    let at = |i: usize| lines.get(i).map(String::as_str).unwrap_or("");
    let defs = |reg: &str| -> Vec<usize> { lines.iter().enumerate().filter(|(_, l)| writes(l) && dst(l) == reg).map(|(i, _)| i).collect() };
    if !defs("v185").is_empty() { return fail("v185 is redefined in the GDN path".into()) }
    // Lane terms: lane <= 31, lane15 <= 15, lane*16 <= 496.
    for (reg, expected) in [
        ("v187", &["v_mbcnt_lo_u32_b32 v187, -1, 0"][..]),
        ("v182", &["v_and_b32_e32 v182, 15, v187"][..]),
        ("v183", &["v_lshlrev_b32_e32 v183, 4, v187"][..]),
    ] {
        if defs(reg).iter().map(|&i| at(i)).collect::<Vec<_>>() != expected { return fail(format!("{reg} derivation changed")) }
    }
    let before = |reg: &str, i: usize| (0..i).rev().find(|&j| writes(at(j)) && dst(at(j)) == reg).map(at);
    // `x = (x + k) mod 19`: the add, then subtract-and-select once (valid for x + k < 38).
    let mod19 = |i: usize, reg: &str| at(i + 1) == format!("s_sub_co_i32 s99, {reg}, {RING_ROWS}") && at(i + 2) == format!("s_cmp_ge_u32 {reg}, {RING_ROWS}") && at(i + 3) == format!("s_cselect_b32 {reg}, s99, {reg}");
    let in_idiom = |i: usize, reg: &str| at(i) == format!("s_cselect_b32 {reg}, s99, {reg}") && i >= 3 && mod19(i - 3, reg);
    for i in defs("s93") {
        let l = at(i);
        if l == "s_mov_b32 s93, 3" || in_idiom(i, "s93") || (l == "s_add_co_i32 s93, s93, 16" && mod19(i, "s93")) { continue }
        return fail(format!("unproven ring row update `{l}`"))
    }
    for i in defs("s94") {
        let l = at(i);
        let ok = in_idiom(i, "s94")
            || (l == "s_add_co_i32 s94, s93, 16" && mod19(i, "s94"))
            || (l == "s_add_co_i32 s94, s94, 1" && mod19(i, "s94"))
            || (l == "s_mov_b32 s94, s94" && mod19(i, "s94"))
            // + 4*(wave >> 1) <= 12, reduced by the idiom that follows.
            || (l == "s_add_co_i32 s94, s94, s97" && before("s97", i) == Some("s_lshl_b32 s97, s73, 2")
                && before("s73", i) == Some("s_lshr_b32 s73, s50, 1") && at(i + 1) == "s_mov_b32 s94, s94" && mod19(i + 1, "s94"));
        if !ok { return fail(format!("unproven ring row update `{l}`")) }
    }
    let (mut max_end, mut accesses) = (0u32, 0usize);
    for (i, l) in lines.iter().enumerate() {
        let Some((opcode, operands)) = l.split_once(' ') else { continue };
        if !opcode.starts_with("ds_") { continue }
        let (args, offset) = match operands.split_once(" offset:") { Some((a, o)) => (a, o.parse::<u32>().map_err(|_| "invalid GDN LDS offset")?), None => (operands, 0) };
        let addr = if opcode == "ds_store_b128" { args.split(',').next() } else if opcode == "ds_load_b128" { args.split(',').nth(1) } else { return fail(format!("unexpected {opcode}")) }.unwrap_or("").trim();
        let base_max = match (opcode, addr) {
            // Ring row min(r, r - 19) of r = s93 + lane15 <= 33, times ROW_BYTES, plus sc_addr <= 416.
            ("ds_store_b128", "v180") => {
                let d = defs("v180");
                let j = (0..i).rev().find(|j| d.contains(j)).ok_or("W address undefined")?;
                if !(j >= 3 && at(j) == format!("v_mad_u32_u24 v180, v180, {ROW_BYTES:#x}, v185") && at(j - 1) == "v_min_u32_e32 v180, v180, v184"
                    && at(j - 2) == format!("v_subrev_nc_u32_e32 v184, {RING_ROWS}, v180") && at(j - 3) == "v_add_nc_u32_e32 v180, s93, v182") {
                    return fail("W ring address derivation changed".into())
                }
                (RING_ROWS - 1) * ROW_BYTES + 416
            }
            // Halo rows 0..2 through the lane's channel base.
            ("ds_store_b128", "v183") if offset <= 2 * ROW_BYTES => 496,
            // P rows: ((s94 + k) mod 19) * ROW_BYTES over the lane's channel base.
            ("ds_load_b128", "v184") => {
                let row_ok = i >= 6 && at(i - 1) == "v_add_nc_u32_e32 v184, s97, v183" && at(i - 2) == format!("s_mul_i32 s97, s97, {ROW_BYTES:#x}") && mod19(i - 6, "s97")
                    && (at(i - 6) == "s_mov_b32 s97, s94" || (1..=3).any(|k| at(i - 6) == format!("s_add_co_i32 s97, s94, {k}")));
                if !row_ok { return fail(format!("P row address before line `{l}` changed")) }
                (RING_ROWS - 1) * ROW_BYTES + 496
            }
            _ => return fail(format!("unproven LDS base {addr} in `{l}`")),
        };
        let end = offset + base_max + 16;
        if end > RING_BYTES || end > launch { return fail(format!("`{l}` reaches LDS byte {end}")) }
        max_end = max_end.max(end); accesses += 1;
    }
    if accesses == 0 { return fail("no accesses".into()) }
    Ok(max_end)
}

#[cfg(test)]
mod tests {
    use super::{RING_ROWS, ROW_BYTES, check_lds_access};
    use crate::kernels::iu4_gemm::{Epi, Spec, emit};

    #[test]
    fn gdn_lds_certificate_bounds_the_ring_and_rejects_changes() {
        let spec = Spec::control(Epi::QkvzaGdn);
        let source = emit(spec).unwrap().s_text;
        // The last ring row's 512 data bytes end the GDN accesses (its padding is never touched).
        assert_eq!(check_lds_access(&source, &spec.symbol(), 20480).unwrap(), (RING_ROWS - 1) * ROW_BYTES + 512);
        assert!(check_lds_access(&source, &spec.symbol(), 20479).is_err());
        for (from, to) in [
            ("v_mad_u32_u24 v180, v180, 0x210, v185", "v_mad_u32_u24 v180, v180, 0x220, v185"),
            ("s_mul_i32 s97, s97, 0x210", "s_mul_i32 s97, s97, 0x200"),
            ("ds_store_b128 v183, v[152:155] offset:1056", "ds_store_b128 v183, v[152:155] offset:10032"),
            ("s_lshl_b32 s97, s73, 2", "s_lshl_b32 s97, s73, 3"),
            ("s_add_co_i32 s93, s93, 16", "s_add_co_i32 s93, s93, 17"),
        ] {
            let mutant = source.replacen(from, to, 1);
            assert_ne!(mutant, source, "{from}");
            assert!(check_lds_access(&mutant, &spec.symbol(), 20480).is_err(), "{from}");
        }
    }
}
