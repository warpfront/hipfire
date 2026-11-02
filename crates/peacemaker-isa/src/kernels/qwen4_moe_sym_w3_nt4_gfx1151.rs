// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1151 `WeightCodec::Mq3I4` NT4 expert-run entries: gate/up
//! `qwen4_moe_gate_up_silu_iu4_sym_w3_pm_gfx1151_nt4`, down
//! `qwen4_moe_down_iu4_sym_w3_pm_gfx1151_nt4`, the down row repeats `..._nt4r2` and
//! `..._nt4r4`, and the F32 `_acc` qualification twins of the two `rr == 1` entries, as the
//! gfx1151 members of the module `qwen4_moe_iu4_sym_w3_nt4_pm_gfx1151`. Sibling of the gfx1201
//! body in `qwen4_moe_sym_w3_nt4.rs`, which is untouched; mounted and dispatched from there.
//!
//! The pieces, each taken whole from its source so no new tile design is involved:
//!  * grouping, leader walk (up to four consecutive 16-slot tiles of one expert run per CTA),
//!    dead-tile handling, K loop, tail, BF16 epilogue, gate/up LDS handoff and the down row
//!    repeats (`rr` 64-row blocks per CTA, block `b` reads logical weight set `p` from physical
//!    set `p ^ (b & 1)`): the MQ4 gfx1151 NT4 entry's (`qwen4_moe_sym::nt`), register for
//!    register (the gfx11 operand layout: sums, weight sets, headers, `d`, X, epilogue
//!    aliases are at the same VGPRs);
//!  * the weight side: the gfx1151 W3 NT1 entry's (`qwen4_moe_sym_w3_gfx1151.rs`). Planar
//!    qt131/qt132 `[codes: M*K*3/8 B][scales: M*(K/128) f16]`, weight `d * (2c - 7)`. Lanes `r`
//!    and `r + 16` carry the whole row, so every lane loads all sixteen 8-code chunks of its
//!    row's 48-byte epoch group, chunk `c` as one unaligned dword at byte `3c` (low 24 bits are
//!    the codes), spread in place to signed nibbles (eight VALU per chunk: three bit-field inserts
//!    finished to the clean nibble `2c + 1`, then the rebias XOR); K16 step `i` consumes dwords
//!    `2i`, `2i+1`.
//!    The odd epoch's half of the trip's scale pair is moved down first. Each output keeps the
//!    NT1 entry's ascending per-epoch fold `sum = fma(RN(d * sc), C - magic, sum)`, so every
//!    output byte equals the gfx1151 NT1 W3 entry's.
//!
//! Resources (exact): 208 VGPRs, the MQ4 gfx1151 NT4 layout's (weight nibble sets are 16 dwords
//! per epoch exactly like the raw code sets; the two unpack temporaries sit at v142..v143, the
//! pair its `d` and X blocks leave free), 80 SGPRs (highest s79; the
//! permlane selector sits on the free s73), no scratch, 8192 B LDS (gate/up) / 0 (down, acc).
//! Every `rr` has the same registers (the row repeat reuses the dead fold registers for its
//! store temporaries, as the MQ4 entry).
//!
//! `_acc` entries (`rr == 1` only) store each accumulator `j` of tile `t` as one dword at row
//! `rbase + 2j + hi` of slot `lr`: `Y[P][M]` f32, tile `t` at `+16 * t * M * 4` bytes, the up
//! waves M/2 columns further, padding slots `+0`: the same bytes as the gfx1201 `_acc` twin.
use super::*;

/// The gate/up handoff region: RNE(gate) of every tile.
enum Gate {}

/// 16-slot tiles per weight stream.
const NT: u8 = 4;

// VGPRs: the MQ4 gfx1151 NT4 layout (`qwen4_moe_sym::nt`, gfx11) plus the unpack registers.
const XOFF: u8 = 16; // + j: activation row offset of tile j (or X_OOB)
const WOFF: u8 = 32; // lr * row_bytes
const HOFF: u8 = 33; // scale row (C rows 2j + hi: (lr & 14) | hi) * scale_row_bytes
const YOFF: u8 = 34;
const LDSA: u8 = 35; // gate/up: (32*(w&1) + lane) * 32
const HF: u8 = 36;
const MAGIC8: u8 = 40;
const CACC: u8 = 48;
const TPROD: u8 = 56;
const SCF: u8 = 64;
const SUM0: u8 = 72; // + 8j
const SETS: u8 = 104; // physical weight set s at SETS + 16 s: 16 code dwords (unpacked in place)
const WH0: u8 = 136; // + s: scale pair
const D0: u8 = 138; // + j: activation scale of tile j
const X0: u8 = 144; // + 16 j: activation nibbles of tile j
const TMPN: u8 = 142; // 2: unpack temporaries (the two registers between `d` and X)
                      // Epilogue aliases (dead K-loop ranges).
const SILU_TMP: u8 = 40; // 24: magic, cacc, fold products
const RT_TMP: u8 = 64; // 8: row scales

// SGPRs (the gfx1201 NT4 entry's, plus the gfx11 workgroup-id inputs and the permlane selector).
const SRD_W: u8 = 24;
const WPTR: u8 = 28;
const BACK: u8 = 30;
const POS: u8 = 31;
const SRD_X: [u8; 2] = [32, 36];
const SRD_S: u8 = 40;
const TRIPS: u8 = 44;
const XS2: u8 = 45;
const ROWB: u8 = 46;
const TILE16: u8 = 47;
const WAVE: u8 = 48;
const RBASE: u8 = 49;
const WGX: u8 = 50;
const WGY: u8 = 51;
const CNT: u8 = 52;
const YSTEP: u8 = 53;
const MASKA: u8 = 54; // prologue lane mask; epilogue `hi_mask`
const MASKT: [u8; 2] = [56, 58];
const LIVE0: u8 = 60; // + 2j
const SRD_WS: u8 = 68; // scale plane descriptor
const SROW: u8 = 72; // scale row bytes (K/64)
const PSEL: u8 = 73;
const SRD_Y: u8 = 76;
const SILU_MASK: u8 = 24;

fn arch() -> Arch {
    Arch::Gfx1151
}
fn sub_i32() -> &'static str {
    "s_sub_i32"
}

/// The subset built here: gfx1151, MQ3-I4, `nt == 4`; `rr == 1` for both kinds, `rr` 2 or 4 for the
/// down entry. (The parent `Spec::validate` admits the MQ3-I4 codec at NT1 only and is not edited.)
pub(super) fn validate(spec: Spec) -> R {
    if spec.arch != Arch::Gfx1151 {
        return Err("qwen4_moe_sym w3 nt4 gfx1151: gfx1151 only".into());
    }
    if spec.nt != NT {
        return Err(format!(
            "qwen4_moe_sym w3 nt4 gfx1151: tile width {} (built for nt 4)",
            spec.nt
        ));
    }
    if spec.codec != WeightCodec::Mq3I4 {
        return Err("qwen4_moe_sym w3 nt4 gfx1151: built for the MQ3-I4 codec only".into());
    }
    if !matches!(spec.rr, 1 | 2 | 4) {
        return Err(format!(
            "qwen4_moe_sym w3 nt4 gfx1151: row repeat {} (1, 2 or 4)",
            spec.rr
        ));
    }
    if spec.rr > 1 && spec.kind != Kind::Down {
        return Err(
            "qwen4_moe_sym w3 nt4 gfx1151: row repeats are built for the down entry only".into(),
        );
    }
    Ok(())
}

/// Every entry in module order: gate/up, down, then the down row repeats.
pub(super) fn module_specs() -> Vec<Spec> {
    let spec = |kind, rr| Spec {
        arch: Arch::Gfx1151,
        kind,
        nt: NT,
        rr,
        codec: WeightCodec::Mq3I4,
    };
    vec![
        spec(Kind::GateUp, 1),
        spec(Kind::Down, 1),
        spec(Kind::Down, 2),
        spec(Kind::Down, 4),
    ]
}

#[derive(Clone, Copy)]
struct G {
    spec: Spec,
    blk: u8,
    acc: bool,
}
impl G {
    fn rr(&self) -> u8 {
        self.spec.rr
    }
    fn at(&self, blk: u8) -> G {
        G { blk, ..*self }
    }
    fn gate_up(&self) -> bool {
        self.spec.kind == Kind::GateUp
    }
    /// Gate/up BF16 entry: RNE(gate) goes through LDS to the up waves.
    fn handoff(&self) -> bool {
        self.gate_up() && !self.acc
    }
    fn sfx(&self) -> &'static str {
        if self.acc {
            "acc"
        } else {
            ""
        }
    }
    fn label(&self, name: &str) -> String {
        if self.rr() == 1 {
            format!(".Lq4s_{}_w3nt4{}_{name}", self.spec.kind.tag(), self.sfx())
        } else {
            format!(
                ".Lq4s_{}_w3nt4r{}b{}_{name}",
                self.spec.kind.tag(),
                self.rr(),
                self.blk
            )
        }
    }
    fn exit_label(&self) -> String {
        if self.rr() == 1 {
            self.label("end")
        } else {
            format!(".Lq4s_{}_w3nt4r{}_end", self.spec.kind.tag(), self.rr())
        }
    }
    /// Parity of the physical weight sets: odd row blocks swap them.
    fn flip(&self) -> usize {
        usize::from(self.blk & 1)
    }
    fn set_at(&self, set: usize) -> u8 {
        SETS + 16 * set as u8
    }
    /// Code dword `c` (chunk `c`) of epoch parity `p` in this block.
    fn raw(&self, p: usize, c: u8) -> u8 {
        self.set_at(p ^ self.flip()) + c
    }
    fn wh(&self, p: usize) -> u8 {
        WH0 + (p ^ self.flip()) as u8
    }
    fn sum(&self, j: u8) -> u8 {
        SUM0 + 8 * j
    }
    fn d(&self, j: u8) -> u8 {
        D0 + j
    }
    fn x(&self, j: u8) -> u8 {
        X0 + 16 * j
    }
    fn live(&self, j: u8) -> u8 {
        LIVE0 + 2 * j
    }
    fn gval(&self) -> u8 {
        self.x(0)
    }
    // Store temporaries of the gfx11 lane exchange: over the dead weight sets, or (row repeats keep
    // every set live across a transition) over the dead fold registers and tile 0's activations.
    fn own(&self) -> u8 {
        if self.rr() > 1 {
            CACC
        } else {
            self.set_at(self.flip())
        }
    }
    fn send(&self) -> u8 {
        if self.rr() > 1 {
            CACC + 4
        } else {
            self.own() + 4
        }
    }
    fn recv(&self) -> u8 {
        if self.rr() > 1 {
            CACC + 6
        } else {
            self.own() + 6
        }
    }
    fn sel(&self) -> u8 {
        if self.rr() > 1 {
            TPROD
        } else {
            self.own() + 8
        }
    }
    fn out(&self) -> u8 {
        if self.rr() > 1 {
            TPROD + 4
        } else {
            self.own() + 12
        }
    }
    fn pack(&self) -> u8 {
        if self.rr() > 1 {
            self.x(0)
        } else {
            self.own() + 16
        }
    }

    fn plan(&self) -> Result<RegPlan, String> {
        let mut p = RegPlan::new(VGPR_CEILING, 104)?;
        let rr = self.rr();
        let blk = |b: u8| self.at(b);
        let l = |n: &str| blk(0).label(n);
        let exit = self.exit_label();
        // Whole-run registers: live to the last block's epilogue.
        let kernel = || Live::Between("entry".into(), blk(rr - 1).label("epilogue"));
        let pro = || Live::Between("entry".into(), l("k_begin"));
        // Registers the store temporaries alias: block `b`'s body runs from the previous block's
        // reload (entry for block 0) to its epilogue, its temporaries from there to its own reload
        // (the exit for the last block). `rr = 1` is `kernel`/`epi`.
        let body = |b: u8| {
            Live::Between(
                if b == 0 {
                    "entry".to_string()
                } else {
                    blk(b - 1).label("xreload")
                },
                blk(b).label("epilogue"),
            )
        };
        let epi = |b: u8| {
            Live::Between(
                blk(b).label("epilogue"),
                if b + 1 == rr {
                    exit.clone()
                } else {
                    blk(b).label("xreload")
                },
            )
        };
        p.v::<1>("tid", 0, pro())?;
        for i in 1..16u8 {
            p.v::<1>("prologue_tmp", i, pro())?;
        }
        for j in 0..NT {
            p.v::<1>("x_off", XOFF + j, kernel())?;
            p.v::<8>("sum", self.sum(j), Live::Whole)?;
            p.v::<1>("d", self.d(j), kernel())?;
            for c in 0..4u8 {
                if j == 0 && c == 0 {
                    for b in 0..rr {
                        p.v::<4>("x", self.x(0), body(b))?;
                    }
                } else {
                    p.v::<4>("x", self.x(j) + 4 * c, kernel())?;
                }
            }
            p.s::<2>("live", self.live(j), Live::Whole)?;
        }
        for (name, r) in [("w_off", WOFF), ("h_off", HOFF), ("scale_f32", HF)] {
            p.v::<1>(name, r, kernel())?;
        }
        p.v::<1>("y_off", YOFF, Live::Whole)?;
        if self.handoff() {
            p.v::<1>("lds_addr", LDSA, Live::Whole)?;
        }
        p.v::<8>("magic8", MAGIC8, kernel())?;
        for b in 0..rr {
            p.v::<8>("cacc", CACC, body(b))?;
        }
        for b in 0..rr {
            p.v::<8>("fold_t", TPROD, body(b))?;
        }
        for b in 0..rr {
            p.v::<8>("scale_rows", SCF, body(b))?;
        }
        // Operands at their WMMA/load granularity: code-dword pairs (the K16 A operand).
        for set in 0..2usize {
            for k in 0..8u8 {
                p.v::<2>("operand", self.set_at(set) + 2 * k, kernel())?;
            }
            p.v::<1>("scale_pair", WH0 + set as u8, kernel())?;
        }
        for i in 0..2u8 {
            p.v::<1>("unpack_tmp", TMPN + i, kernel())?;
        }
        for b in 0..rr {
            for i in 0..8u8 {
                p.v::<1>("rt_tmp", RT_TMP + i, epi(b))?;
            }
            p.v::<4>("out", self.out(), epi(b))?;
            if self.handoff() {
                p.v::<8>("gate", self.gval(), epi(b))?;
                for i in 0..(6 * SILU_N) {
                    p.v::<1>("silu_tmp", SILU_TMP + i, epi(b))?;
                }
                for i in 0..(3 * SILU_N) {
                    p.s::<2>("silu_mask", SILU_MASK + 2 * i, epi(b))?;
                }
            }
            for i in 0..4u8 {
                p.v::<1>("own", self.own() + i, epi(b))?;
                p.v::<1>("pack", self.pack() + i, epi(b))?;
            }
            for i in 0..2u8 {
                p.v::<1>("send", self.send() + i, epi(b))?;
                p.v::<1>("recv", self.recv() + i, epi(b))?;
                p.v::<1>("sel", self.sel() + i, epi(b))?;
            }
            p.s::<2>("hi_mask", MASKA, epi(b))?;
            p.s::<1>("permlane_sel", PSEL, epi(b))?;
        }
        p.s::<1>("wg_x_in", 2, Live::Whole)?;
        p.s::<1>("wg_y_in", 3, Live::Whole)?;
        p.s::<2>("kernarg_ptr", 0, Live::Whole)?;
        for b in 0..rr {
            p.s::<4>("srd_y", SRD_Y, epi(b))?;
        }
        p.s::<1>("s_tmp0", 4, kernel())?;
        p.s::<1>("s_tmp1", 5, kernel())?;
        p.s::<2>("s_tmp64", 6, kernel())?;
        p.s::<8>("kernargs_0x00", 8, kernel())?;
        p.s::<2>("y_base", 16, Live::Whole)?;
        p.s::<2>("kernargs_m_k", 18, kernel())?;
        p.s::<2>("kernargs_0x30", 20, kernel())?;
        p.s::<1>("x_src_rows", 22, kernel())?;
        p.s::<1>("expert", 23, kernel())?;
        p.s::<4>("srd_codes", SRD_W, kernel())?;
        p.s::<4>("srd_scales", SRD_WS, kernel())?;
        p.s::<2>("w_ptr", WPTR, kernel())?;
        p.s::<1>("run_back", BACK, pro())?;
        p.s::<1>("run_pos", POS, pro())?;
        for r in SRD_X {
            p.s::<4>("srd_x", r, kernel())?;
        }
        p.s::<4>("srd_tiles_sorted", SRD_S, kernel())?;
        for (name, r) in [
            ("trips", TRIPS),
            ("x_stride2", XS2),
            ("row_bytes", ROWB),
            ("scale_row_bytes", SROW),
            ("tile16", TILE16),
            ("row_base", RBASE),
            ("wg_x", WGX),
            ("wg_y", WGY),
        ] {
            p.s::<1>(name, r, kernel())?;
        }
        p.s::<1>("wave", WAVE, Live::Whole)?;
        p.s::<1>("tile_count", CNT, Live::Whole)?;
        p.s::<1>("y_step", YSTEP, Live::Whole)?;
        p.s::<2>("lane_mask", MASKA, pro())?;
        for m in MASKT {
            p.s::<2>("cmp_mask", m, Live::Whole)?;
        }
        let top = p.next_free_vgpr();
        if top % 8 != 0 {
            p.v::<1>("granule_pad", (top.div_ceil(8) * 8 - 1) as u8, Live::Whole)?;
        }
        Ok(p)
    }
}

/// Lane mask `m` (a pair; the high word is unused in wave32) |= `n`.
fn or_mask(b: &mut Builder, m: u8, n: u8) -> R {
    sop(b, format!("s_or_b32 s{m}, s{m}, s{n}"), &[m], &[m, n])
}

/// Words 0..3 of both activation descriptors (epoch 0, then epoch 1)
/// from the kernel arguments; leaves `s4 = x_src_rows * 72`.
fn x_descriptors(b: &mut Builder) -> R {
    sop(b, "s_mul_i32 s4, s22, 0x48", &[4], &[22])?;
    sop(
        b,
        format!("s_mov_b32 s{}, s14", SRD_X[0]),
        &[SRD_X[0]],
        &[14],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, s15", SRD_X[0] + 1),
        &[SRD_X[0] + 1],
        &[15],
    )?;
    srd_tail(b, SRD_X[0], Some(4))?;
    add64(b, SRD_X[1], 14, 4)?;
    srd_tail(b, SRD_X[1], Some(4))
}

/// Code descriptor at rows `row_base..` of the expert, and the scale
/// descriptor (the scale plane starts `M * row_bytes` after the tensor base).
fn w_descriptors(b: &mut Builder) -> R {
    sop(
        b,
        format!("s_mul_i32 s4, s{RBASE}, s{ROWB}"),
        &[4],
        &[RBASE, ROWB],
    )?;
    add64(b, SRD_W, WPTR, 4)?;
    srd_tail(b, SRD_W, None)?;
    sop(b, format!("s_mul_i32 s4, s18, s{ROWB}"), &[4], &[18, ROWB])?;
    add64(b, SRD_WS, WPTR, 4)?;
    sop(
        b,
        format!("s_mul_i32 s5, s{RBASE}, s{SROW}"),
        &[5],
        &[RBASE, SROW],
    )?;
    add64(b, SRD_WS, SRD_WS, 5)?;
    srd_tail(b, SRD_WS, None)
}

/// Two-epoch trips before the tail: (K/128 - 1) / 2.
fn set_trips(b: &mut Builder) -> R {
    let k = 19;
    sop(b, format!("s_lshr_b32 s{TRIPS}, s{k}, 7"), &[TRIPS], &[k])?;
    sop(
        b,
        format!("{} s{TRIPS}, s{TRIPS}, -1", s_add_i32(arch())),
        &[TRIPS],
        &[TRIPS],
    )?;
    sop(
        b,
        format!("s_lshr_b32 s{TRIPS}, s{TRIPS}, 1"),
        &[TRIPS],
        &[TRIPS],
    )
}

fn prologue(wg: &mut Wg<Gfx1151>, g: &G, end: &End) -> R {
    let a = arch();
    let b = wg.isa();
    let gate_up = g.gate_up();
    // The LDS address comes first: certification derives it from the
    // work-item id in the entry block. (32*(w&1) + lane) * 32 bytes.
    if g.handoff() {
        op(
            b,
            format!("v_and_b32_e32 v{LDSA}, 63, v0"),
            &[v(LDSA)],
            &[v(0)],
        )?;
        op(
            b,
            format!("v_lshlrev_b32_e32 v{LDSA}, 5, v{LDSA}"),
            &[v(LDSA)],
            &[v(LDSA)],
        )?;
    }
    smem(b, 8, 8, 0, 0)?;
    smem(b, 16, 2, 0, 0x20)?;
    smem(b, 18, 2, 0, 0x28)?;
    smem(b, 20, 2, 0, 0x30)?;
    smem(b, 22, 1, 0, 0x38)?;
    // Workgroup ids arrive in s2 (x) and s3 (y).
    sop(b, format!("s_mov_b32 s{WGX}, s2"), &[WGX], &[2])?;
    sop(b, format!("s_mov_b32 s{WGY}, s3"), &[WGY], &[3])?;
    // v1 = wave, v2 = lane, v3 = lr, v4 = hi.
    op(b, "v_lshrrev_b32_e32 v1, 5, v0", &[v(1)], &[v(0)])?;
    op(b, "v_and_b32_e32 v2, 31, v0", &[v(2)], &[v(0)])?;
    op(
        b,
        format!("v_readfirstlane_b32 s{WAVE}, v1"),
        &[s(WAVE)],
        &[v(1)],
    )?;
    op(b, "v_and_b32_e32 v3, 15, v2", &[v(3)], &[v(2)])?;
    op(b, "v_lshrrev_b32_e32 v4, 4, v2", &[v(4)], &[v(2)])?;
    // A tile at or past m_total, or a sentinel tile, returns.
    sop(
        b,
        format!("s_lshl_b32 s{TILE16}, s{WGY}, 4"),
        &[TILE16],
        &[WGY],
    )?;
    let outside = wg.scmp_wg_uniform(Instruction::new(
        format!("s_cmp_ge_i32 s{TILE16}, s21"),
        vec![],
        vec![s(TILE16), s(21)],
    ))?;
    wg.exit_if(outside, end)?;
    let b = wg.isa();
    sop(b, format!("s_lshl_b32 s4, s{WGY}, 2"), &[4], &[WGY])?;
    add64(b, 6, 10, 4)?;
    smem(b, 23, 1, 6, 0)?;
    // Fail closed: an expert id outside [0, n_experts) exits before any `expert_weight_ptrs[id]` load (one unsigned compare).
    smem(b, 5, 1, 0, 0x3c)?;
    let sentinel = wg.scmp_wg_uniform(Instruction::new(
        "s_cmp_ge_u32 s23, s5",
        vec![],
        vec![s(23), s(5)],
    ))?;
    wg.exit_if(sentinel, end)?;
    let b = wg.isa();
    sop(b, "s_lshl_b32 s4, s23, 3", &[4], &[23])?;
    add64(b, 6, 8, 4)?;
    smem(b, WPTR, 2, 6, 0)?;
    // Tile ids through a descriptor of m_total/16 records: an id read at
    // or past the end (or before the start) reads 0, so both bounds are
    // tested explicitly.
    sop(b, format!("s_mov_b32 s{SRD_S}, s10"), &[SRD_S], &[10])?;
    sop(
        b,
        format!("s_mov_b32 s{}, s11", SRD_S + 1),
        &[SRD_S + 1],
        &[11],
    )?;
    sop(b, "s_lshr_b32 s5, s21, 2", &[5], &[21])?;
    srd_tail(b, SRD_S, Some(5))?;
    // Run position: lanes test tiles tile-1-back-lane; the first that is
    // before the start or another expert ends the run.
    sop(b, format!("s_mov_b32 s{BACK}, 0"), &[BACK], &[])?;
    let (walk, found) = (g.label("run_walk"), g.label("run_found"));
    wg.loop_until(&walk, &found, |w, exit| {
        let b = w.isa();
        sop(b, format!("{} s4, s{WGY}, -1", s_add_i32(a)), &[4], &[WGY])?;
        sop(
            b,
            format!("{} s4, s4, s{BACK}", sub_i32()),
            &[4],
            &[4, BACK],
        )?;
        op(b, "v_sub_nc_u32_e32 v5, s4, v2", &[v(5)], &[s(4), v(2)])?;
        op(b, "v_lshlrev_b32_e32 v6, 2, v5", &[v(6)], &[v(5)])?;
        bload(b, 1, 7, 6, SRD_S, 0)?;
        op(
            b,
            format!("v_cmp_ne_u32_e64 s{MASKA}, s23, v7"),
            &[s(MASKA)],
            &[s(23), v(7)],
        )?;
        op(
            b,
            format!("v_cmp_gt_i32_e64 s{}, 0, v5", MASKT[0]),
            &[s(MASKT[0])],
            &[v(5)],
        )?;
        or_mask(b, MASKA, MASKT[0])?;
        let hit = w.scmp(Instruction::new(
            format!("s_cmp_lg_u32 s{MASKA}, 0"),
            vec![],
            vec![s(MASKA)],
        ))?;
        w.break_if(hit, exit)?;
        sop(
            w.isa(),
            format!("{} s{BACK}, s{BACK}, 32", s_add_i32(a)),
            &[BACK],
            &[BACK],
        )
    })?;
    let b = wg.isa();
    sop(
        b,
        format!("s_ctz_i32_b32 s{POS}, s{MASKA}"),
        &[POS],
        &[MASKA],
    )?;
    sop(
        b,
        format!("{} s{POS}, s{POS}, s{BACK}", s_add_i32(a)),
        &[POS],
        &[POS, BACK],
    )?;
    // Not a leader: another CTA folds this tile.
    sop(
        b,
        format!("s_and_b32 s{POS}, s{POS}, {}", NT - 1),
        &[POS],
        &[POS],
    )?;
    let follower = wg.scmp_wg_uniform(Instruction::new(
        format!("s_cmp_lg_u32 s{POS}, 0"),
        vec![],
        vec![s(POS)],
    ))?;
    wg.exit_if(follower, end)?;
    let b = wg.isa();
    // cnt: the first lane l < nt whose tile tile+l is past the end or of
    // another expert (lanes >= nt always stop).
    op(
        b,
        format!("v_add_nc_u32_e32 v5, s{WGY}, v2"),
        &[v(5)],
        &[s(WGY), v(2)],
    )?;
    op(b, "v_lshlrev_b32_e32 v6, 2, v5", &[v(6)], &[v(5)])?;
    bload(b, 1, 7, 6, SRD_S, 0)?;
    sop(b, "s_lshr_b32 s5, s21, 4", &[5], &[21])?;
    op(
        b,
        format!("v_cmp_ne_u32_e64 s{MASKA}, s23, v7"),
        &[s(MASKA)],
        &[s(23), v(7)],
    )?;
    op(
        b,
        format!("v_cmp_le_u32_e64 s{}, s5, v5", MASKT[0]),
        &[s(MASKT[0])],
        &[s(5), v(5)],
    )?;
    or_mask(b, MASKA, MASKT[0])?;
    op(
        b,
        format!("v_cmp_le_u32_e64 s{}, {NT}, v2", MASKT[0]),
        &[s(MASKT[0])],
        &[v(2)],
    )?;
    or_mask(b, MASKA, MASKT[0])?;
    sop(
        b,
        format!("s_ctz_i32_b32 s{CNT}, s{MASKA}"),
        &[CNT],
        &[MASKA],
    )?;
    // Code row bytes K*3/8; scale row bytes (K/128)*2.
    let (m, k) = (18u8, 19u8);
    sop(b, format!("s_mul_i32 s{ROWB}, s{k}, 3"), &[ROWB], &[k])?;
    sop(
        b,
        format!("s_lshr_b32 s{ROWB}, s{ROWB}, 3"),
        &[ROWB],
        &[ROWB],
    )?;
    sop(b, format!("s_lshr_b32 s{SROW}, s{k}, 6"), &[SROW], &[k])?;
    set_trips(b)?;
    // Output column base: gate/up 32x + 16*(w&1), down 64*rr*x + 16w.
    if gate_up {
        sop(
            b,
            format!("s_lshl_b32 s{RBASE}, s{WGX}, 5"),
            &[RBASE],
            &[WGX],
        )?;
        sop(b, format!("s_and_b32 s4, s{WAVE}, 1"), &[4], &[WAVE])?;
    } else {
        sop(
            b,
            format!(
                "s_lshl_b32 s{RBASE}, s{WGX}, {}",
                6 + g.rr().trailing_zeros()
            ),
            &[RBASE],
            &[WGX],
        )?;
        sop(b, format!("s_mov_b32 s4, s{WAVE}"), &[4], &[WAVE])?;
    }
    sop(b, "s_lshl_b32 s4, s4, 4", &[4], &[4])?;
    sop(
        b,
        format!("{} s{RBASE}, s{RBASE}, s4", s_add_i32(a)),
        &[RBASE],
        &[RBASE, 4],
    )?;
    if g.acc && gate_up {
        // F32 rows: an up wave's weight rows and its stored columns are both M/2 further.
        sop(b, format!("s_lshr_b32 s4, s{WAVE}, 1"), &[4], &[WAVE])?;
        sop(b, format!("s_lshr_b32 s5, s{m}, 1"), &[5], &[m])?;
        sop(b, "s_mul_i32 s4, s4, s5", &[4], &[4, 5])?;
        sop(
            b,
            format!("{} s{RBASE}, s{RBASE}, s4", s_add_i32(a)),
            &[RBASE],
            &[RBASE, 4],
        )?;
    }
    // Y row length (BF16 gate/up H is M/2, everything else M) and per-tile Y step (16 rows).
    if g.handoff() {
        sop(b, format!("s_lshr_b32 s5, s{m}, 1"), &[5], &[m])?
    } else {
        sop(b, format!("s_mov_b32 s5, s{m}"), &[5], &[m])?
    }
    sop(
        b,
        format!("s_lshl_b32 s{YSTEP}, s5, {}", if g.acc { 6 } else { 5 }),
        &[YSTEP],
        &[5],
    )?;
    // Y offset: BF16 ((16*tile + lr)*row_len + col + 8*hi) * 2 (after the epilogue's lane exchange,
    // lane hi stores rows 8hi..8hi+8), F32 ((16*tile + lr)*M + col + hi) * 4 (lane (hi, lr) holds
    // rows 2j + hi, stored per j at `+ 8j`).
    op(
        b,
        format!("v_add_nc_u32_e32 v8, s{TILE16}, v3"),
        &[v(8)],
        &[s(TILE16), v(3)],
    )?;
    op(b, "v_mul_lo_u32 v8, v8, s5", &[v(8)], &[v(8), s(5)])?;
    if g.acc {
        op(
            b,
            format!("v_add_nc_u32_e32 v9, s{RBASE}, v4"),
            &[v(9)],
            &[s(RBASE), v(4)],
        )?;
    } else {
        op(
            b,
            format!("v_lshl_add_u32 v9, v4, 3, s{RBASE}"),
            &[v(9)],
            &[v(4), s(RBASE)],
        )?;
    }
    op(b, "v_add_nc_u32_e32 v8, v8, v9", &[v(8)], &[v(8), v(9)])?;
    op(
        b,
        format!(
            "v_lshlrev_b32_e32 v{YOFF}, {}, v8",
            if g.acc { 2 } else { 1 }
        ),
        &[v(YOFF)],
        &[v(8)],
    )?;
    if g.handoff() {
        // Up waves read the up rows M/2 further.
        sop(b, format!("s_lshr_b32 s4, s{WAVE}, 1"), &[4], &[WAVE])?;
        sop(b, "s_mul_i32 s4, s4, s5", &[4], &[4, 5])?;
        sop(
            b,
            format!("{} s{RBASE}, s{RBASE}, s4", s_add_i32(a)),
            &[RBASE],
            &[RBASE, 4],
        )?;
    }
    // Slots of the cnt tiles: sorted[16*(tile+j) + lr] through a
    // descriptor of cnt*64 bytes; dead tiles' slots read -1 below.
    sop(b, format!("s_lshl_b32 s4, s{WGY}, 6"), &[4], &[WGY])?;
    add64(b, SRD_S, 12, 4)?;
    sop(
        b,
        format!("s_lshl_b32 s{}, s{CNT}, 6", SRD_S + 2),
        &[SRD_S + 2],
        &[CNT],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, {}", SRD_S + 3, lit(SRD_WORD3)),
        &[SRD_S + 3],
        &[],
    )?;
    op(b, "v_lshlrev_b32_e32 v5, 2, v3", &[v(5)], &[v(3)])?;
    for j in 0..NT {
        bload(b, 1, 6 + j, 5, SRD_S, 64 * u32::from(j))?;
    }
    // Activation descriptors: epoch 0 (X0) and 1 (X1), x_src_rows*72 records.
    x_descriptors(b)?;
    sop(b, format!("s_lshl_b32 s{XS2}, s4, 1"), &[XS2], &[4])?;
    // Weight descriptors (codes and scales) at rows rbase.. of the expert.
    w_descriptors(b)?;
    // Lane offsets. Codes: lr*row_bytes (lanes r and r+16 read the same row).
    op(
        b,
        format!("v_mul_u32_u24_e32 v{WOFF}, s{ROWB}, v3"),
        &[v(WOFF)],
        &[s(ROWB), v(3)],
    )?;
    // Scale row: the row whose scale this lane's broadcast slot serves (C rows 2j + hi).
    op(b, "v_and_b32_e32 v14, 14, v3", &[v(14)], &[v(3)])?;
    op(b, "v_or_b32_e32 v14, v14, v4", &[v(14)], &[v(14), v(4)])?;
    op(
        b,
        format!("v_mul_u32_u24_e32 v{HOFF}, s{SROW}, v14"),
        &[v(HOFF)],
        &[s(SROW), v(14)],
    )?;
    load_w(b, &G { blk: 0, ..*g }, 0)?;
    // Per tile: dead tiles (j >= cnt) gather slot -1, then the activation
    // row offset (or the out-of-range offset of a padding slot).
    for j in 0..NT {
        let slot = 6 + j;
        let b = wg.isa();
        if j > 0 {
            let mk = MASKT[1];
            sop(b, format!("s_cmp_gt_u32 s{CNT}, {j}"), &[], &[CNT])?;
            sop(b, format!("s_cselect_b32 s{mk}, -1, 0"), &[mk], &[])?;
            sop(b, format!("s_mov_b32 s{}, 0", mk + 1), &[mk + 1], &[])?;
            op(
                b,
                format!("v_cndmask_b32_e64 v{slot}, -1, v{slot}, s{mk}"),
                &[v(slot)],
                &[v(slot), s(mk)],
            )?;
        }
        common::gather_offset(
            b,
            XOFF + j,
            slot,
            Some(20),
            iu4_fold::XBLK_BYTES,
            g.live(j),
            GatherTemps {
                v: [14, 15, 1, 2],
                mask: MASKT[0],
            },
        )?;
        load_x(b, g, j, SRD_X[0])?;
    }
    let b = wg.isa();
    for j in 0..8u8 {
        op(
            b,
            format!("v_mov_b32_e32 v{}, {}", MAGIC8 + j, lit(MAGIC)),
            &[v(MAGIC8 + j)],
            &[],
        )?;
    }
    for t in 0..NT {
        for j in 0..8u8 {
            op(
                b,
                format!("v_mov_b32_e32 v{}, 0", g.sum(t) + j),
                &[v(g.sum(t) + j)],
                &[],
            )?;
        }
    }
    Ok(())
}

/// Weight set `p` (epoch parity `p` of the current trip): the trip's scale pair of this lane's
/// header row, then one unaligned dword per 8-code chunk (byte `3c` of the epoch's 48-byte
/// group), as one clause (17 loads).
fn load_w(b: &mut Builder, g: &G, p: usize) -> R {
    b.clause(|b| {
        bload(b, 1, g.wh(p), HOFF, SRD_WS, 0)?;
        for c in 0..16u8 {
            bload(
                b,
                1,
                g.raw(p, c),
                WOFF,
                SRD_W,
                48 * p as u32 + 3 * u32::from(c),
            )?;
        }
        Ok(())
    })
}

/// Tile `j`'s activations of the epoch `srd` points at: d, then nibbles (four b128).
fn load_x(b: &mut Builder, g: &G, j: u8, srd: u8) -> R {
    b.clause(|b| {
        bload(b, 1, g.d(j), XOFF + j, srd, 0)?;
        for i in 0..4u8 {
            bload(b, 4, g.x(j) + 4 * i, XOFF + j, srd, 8 + 16 * u32::from(i))?;
        }
        Ok(())
    })
}

/// One chunk: the dword at the chunk's byte offset (its low 24 bits are the 8 codes), spread
/// and sign-finished into `dst` in eight VALU. `dst == code` is fine: `code` is last read by
/// the second instruction, `dst` first written by the third; `y` is the only temporary.
///
/// ```text
/// y   = code << 4;  y = bfi(0xfff, code, y)        fields 0..3 | fields 4..7 << 4
/// dst = y << 2;     y = bfi(0x003f003f, y, dst)    field pairs, junk between them
/// dst = y << 1;     y = (y << 2) | 0x11111111      odd fields at bits 5..7, bit 0 / 4 of each nibble = 1
/// dst = bfi(0x0e0e0e0e, dst, y)                    even fields at bits 1..3: nibble `2c + 1`, no junk
/// dst ^= 0x88888888                                signed nibble `2c - 7`
/// ```
fn unpack(b: &mut Builder, dst: u8, code: u8, y: u8) -> R {
    let bfi = |b: &mut Builder, d: u8, mask: u32, p: u8, q: u8| {
        op(
            b,
            format!("v_bfi_b32 v{d}, {}, v{p}, v{q}", lit(mask)),
            &[v(d)],
            &[v(p), v(q)],
        )
    };
    let shl = |b: &mut Builder, d: u8, n: u8, sr: u8| {
        op(
            b,
            format!("v_lshlrev_b32_e32 v{d}, {n}, v{sr}"),
            &[v(d)],
            &[v(sr)],
        )
    };
    shl(b, y, 4, code)?;
    bfi(b, y, 0xfff, code, y)?;
    shl(b, dst, 2, y)?;
    bfi(b, y, 0x003f_003f, y, dst)?;
    shl(b, dst, 1, y)?;
    op(
        b,
        format!("v_lshl_or_b32 v{y}, v{y}, 2, 0x11111111"),
        &[v(y)],
        &[v(y)],
    )?;
    bfi(b, dst, 0x0e0e_0e0e, dst, y)?;
    op(
        b,
        format!("v_xor_b32_e32 v{dst}, {}, v{dst}", lit(REBIAS)),
        &[v(dst)],
        &[v(dst)],
    )
}

/// Fold weight set `p` (one K128 epoch) into every live tile's sums, the weights dequantized once;
/// with `reload`, each tile's next-epoch activations are issued (through that descriptor) right
/// after its (maybe skipped) fold.
fn compute(
    w: &mut Wave<'_, Gfx1151, Builder>,
    g: &G,
    p: usize,
    site: &str,
    reload: Option<u8>,
) -> R {
    let b = w.isa();
    // Row scales: epoch parity selects the low (even) or high (odd) half of the scale pair (the
    // odd half is moved down in place first; the pair is dead afterwards), then the ds_swizzle
    // broadcast to the accumulator rows (C rows 2j + hi: slot 2j of each 16-lane half).
    let h = g.wh(p);
    if p == 1 {
        op(
            b,
            format!("v_lshrrev_b32_e32 v{h}, 16, v{h}"),
            &[v(h)],
            &[v(h)],
        )?;
    }
    op(
        b,
        format!("v_cvt_f32_f16_e64 v{HF}, v{h}.l"),
        &[v(HF)],
        &[v(h)],
    )?;
    for j in 0..8u8 {
        let text = format!(
            "ds_swizzle_b32 v{}, v{HF} offset:swizzle(BROADCAST,16,{})",
            SCF + j,
            2 * j
        );
        b.ds_crosslane(
            Instruction::new(text, vec![v(SCF + j)], vec![v(HF)]).memory(MemoryClass::DsLoad),
        )?;
    }
    for c in 0..16u8 {
        let r = g.raw(p, c);
        unpack(b, r, r, TMPN + (c % 2))?;
    }
    for j in 0..NT {
        let fold = |w: &mut Wave<'_, Gfx1151, Builder>, ()| -> R {
            let b = w.isa();
            for i in 0..8u8 {
                iu4_fold::wmma_step(
                    b,
                    arch(),
                    V::<8>(CACC),
                    V::<2>(g.raw(p, 2 * i)),
                    V::<2>(g.x(j) + 2 * i),
                    i == 0,
                    V::<8>(MAGIC8),
                )?;
            }
            iu4_fold::fold_pass(b, CACC, g.sum(j), SCF, g.d(j), TPROD)
        };
        if j > 0 {
            let b = w.isa();
            let x = Instruction::new(
                "",
                vec![],
                vec![
                    v(g.d(j)),
                    crate::reg::RegRef {
                        kind: crate::reg::Kind::V,
                        base: g.x(j),
                        len: 16,
                    },
                ],
            );
            for (c, n, _) in b.ledger.required(&x) {
                b.wait(c, n)?;
            }
            let live = w.scmp(Instruction::new(
                format!("s_cmp_gt_u32 s{CNT}, {j}"),
                vec![],
                vec![s(CNT)],
            ))?;
            w.skip_unless(live, &g.label(&format!("{site}_dead{j}")), (), fold)?;
        } else {
            fold(w, ())?;
        }
        if let Some(srd) = reload {
            load_x(w.isa(), g, j, srd)?;
        }
    }
    Ok(())
}

fn advance(b: &mut Builder) -> R {
    add64_imm(b, SRD_W, 96)?;
    add64_imm(b, SRD_WS, 4)?;
    for r in SRD_X {
        add64(b, r, r, XS2)?;
    }
    Ok(())
}

fn kloop(wg: &mut Wg<Gfx1151>, g: &G) -> R {
    wg.label(&g.label("k_begin"))?;
    let (head, done) = (g.label("k_loop"), g.label("k_loop_end"));
    let entry = wg.isa().ledger.shape();
    let no_trip = trips_cmp(wg, "s_cmp_eq_u32", TRIPS)?;
    wg.wg_skip_if(no_trip, &done, (), |wg, ()| {
        wg.loop_carried(&head, (), |wg, ()| {
            load_w(wg.isa(), g, 1)?;
            compute(wg, g, 0, "l0", Some(SRD_X[1]))?;
            advance(wg.isa())?;
            load_w(wg.isa(), g, 0)?;
            compute(wg, g, 1, "l1", Some(SRD_X[0]))?;
            let b = wg.isa();
            op(
                b,
                format!("{} s{TRIPS}, s{TRIPS}, -1", s_add_i32(b.spec.arch)),
                &[s(TRIPS)],
                &[s(TRIPS)],
            )?;
            Ok(((), trips_cmp(wg, "s_cmp_lg_u32", TRIPS)?))
        })?;
        if wg.isa().ledger.shape() != entry {
            return Err("k loop exit ledger differs from its entry".into());
        }
        Ok(())
    })?;
    match g.spec.kind {
        // K % 256 == 0: an even epoch count, two epochs left.
        Kind::GateUp => tail_pair(wg, g),
        // K % 128 == 0: one epoch left when K/128 is odd, else two.
        Kind::Down => {
            let odd = wg.scmp(Instruction::new(
                "s_bitcmp1_b32 s19, 7",
                vec![],
                vec![s(19)],
            ))?;
            wg.if_else(
                odd,
                &g.label("tail_odd"),
                &g.label("epilogue"),
                |w| tail_pair(w, g),
                |w| compute(w, g, 0, "t2", None),
            )
        }
    }
}

fn tail_pair(w: &mut Wave<'_, Gfx1151, Builder>, g: &G) -> R {
    load_w(w.isa(), g, 1)?;
    compute(w, g, 0, "t0", Some(SRD_X[1]))?;
    compute(w, g, 1, "t1", None)
}

fn perm(b: &mut Builder, dst: u8, hi_src: u8, lo_src: u8, sel: String) -> R {
    let mut uses = vec![v(hi_src), v(lo_src)];
    if sel.starts_with('v') {
        uses.push(v(sel[1..].parse::<u8>().map_err(|e| e.to_string())?));
    }
    op(
        b,
        format!("v_perm_b32 v{dst}, v{hi_src}, v{lo_src}, {sel}"),
        &[v(dst)],
        &uses,
    )
}

/// RNE BF16 bits of `vals` (8 rows, lane (hi, lr) holding rows `2j + hi`), +0 on padding slots, then
/// the half exchange so lane `hi` stores rows `8hi..8hi+8` of token `lr`, at the current `SRD_Y`
/// (`+128 * blk` bytes for row block `blk`).
fn store_rows(b: &mut Builder, g: &G, vals: u8, live: u8) -> R {
    for j in 0..8u8 {
        Bf16::rne_finite_passthrough(b, vals + j, RT_TMP + j, MASKT[usize::from(j % 2)], false)?;
    }
    for j in 0..8u8 {
        op(
            b,
            format!("v_cndmask_b32_e64 v{0}, 0, v{0}, s{live}", vals + j),
            &[v(vals + j)],
            &[v(vals + j), s(live)],
        )?;
    }
    let (own, send, recv, sel, pack, out) =
        (g.own(), g.send(), g.recv(), g.sel(), g.pack(), g.out());
    for k in 0..4u8 {
        op(
            b,
            format!(
                "v_cndmask_b32_e64 v{}, v{}, v{}, s{MASKA}",
                own + k,
                vals + k,
                vals + 4 + k
            ),
            &[v(own + k)],
            &[v(vals + k), v(vals + 4 + k), s(MASKA)],
        )?;
    }
    for m in 0..2u8 {
        perm(
            b,
            pack + 2 * m,
            vals + 5 + 2 * m,
            vals + 4 + 2 * m,
            lit(0x0706_0302),
        )?;
        perm(
            b,
            pack + 2 * m + 1,
            vals + 1 + 2 * m,
            vals + 2 * m,
            lit(0x0706_0302),
        )?;
        op(
            b,
            format!(
                "v_cndmask_b32_e64 v{}, v{}, v{}, s{MASKA}",
                send + m,
                pack + 2 * m,
                pack + 2 * m + 1
            ),
            &[v(send + m)],
            &[v(pack + 2 * m), v(pack + 2 * m + 1), s(MASKA)],
        )?;
        op(
            b,
            format!("v_mov_b32_e32 v{}, v{}", recv + m, send + m),
            &[v(recv + m)],
            &[v(send + m)],
        )?;
        op(
            b,
            format!(
                "v_permlanex16_b32 v{}, v{}, s{PSEL}, 0xfedcba98",
                recv + m,
                send + m
            ),
            &[v(recv + m)],
            &[v(recv + m), v(send + m), s(PSEL)],
        )?;
    }
    for (i, (lo, hi)) in [(0x0504_0302u32, 0x0302_0504u32), (0x0706_0302, 0x0302_0706)]
        .into_iter()
        .enumerate()
    {
        let r = sel + i as u8;
        op(b, format!("v_mov_b32_e32 v{r}, {}", lit(hi)), &[v(r)], &[])?;
        op(
            b,
            format!("v_cndmask_b32_e64 v{r}, {}, v{r}, s{MASKA}", lit(lo)),
            &[v(r)],
            &[v(r), s(MASKA)],
        )?;
    }
    for k in 0..4u8 {
        perm(
            b,
            out + k,
            recv + k / 2,
            own + k,
            format!("v{}", sel + k % 2),
        )?;
    }
    bstore_b128(b, out, YOFF, SRD_Y, 128 * u32::from(g.blk))
}

/// Raw single-dword buffer store at `v{voff} + offset` (soffset zero; `common` has the b128 one only).
fn bstore_b32(b: &mut Builder, data: u8, voff: u8, srd: u8, offset: u32) -> R {
    if offset > b.spec.arch.buffer_offset_max() {
        return Err(format!("buffer offset {offset} exceeds the field"));
    }
    let imm = if offset == 0 {
        String::new()
    } else {
        format!(" offset:{offset}")
    };
    common::mem(
        b,
        format!(
            "buffer_store_b32 v{data}, v{voff}, s[{srd}:{}], 0 offen{imm}",
            srd + 3
        ),
        &[],
        &[v(voff), v(data), common::sr(srd, 4)],
        MemoryClass::VmemStore,
    )
}

/// The F32 sums of tile `j`: padding slots +0, then accumulator `i` as one dword at `+ 8i` (row `2i + hi`).
fn store_acc(b: &mut Builder, g: &G, j: u8) -> R {
    let r = g.sum(j);
    for i in 0..8u8 {
        op(
            b,
            format!("v_cndmask_b32_e64 v{0}, 0, v{0}, s{1}", r + i, g.live(j)),
            &[v(r + i)],
            &[v(r + i), s(g.live(j))],
        )?;
    }
    for i in 0..8u8 {
        bstore_b32(b, r + i, YOFF, SRD_Y, 8 * u32::from(i))?;
    }
    Ok(())
}

fn y_descriptor(b: &mut Builder) -> R {
    sop(b, format!("s_mov_b32 s{SRD_Y}, s16"), &[SRD_Y], &[16])?;
    sop(
        b,
        format!("s_mov_b32 s{}, s17", SRD_Y + 1),
        &[SRD_Y + 1],
        &[17],
    )?;
    srd_tail(b, SRD_Y, None)?;
    // The gfx11 half exchange: lanes 16..31 are the high half; permlane selector 0x76543210.
    op(
        b,
        format!("s_mov_b32 s{MASKA}, 0xffff0000"),
        &[s(MASKA)],
        &[],
    )?;
    op(
        b,
        format!("s_mov_b32 s{}, 0", MASKA + 1),
        &[s(MASKA + 1)],
        &[],
    )?;
    op(b, format!("s_mov_b32 s{PSEL}, 0x76543210"), &[s(PSEL)], &[])
}

/// Stores of every live tile: tile 0 always, tile j while j < cnt (a dead tile leaves the kernel);
/// the descriptor base advances one tile of Y per tile. `store` writes tile `j`.
fn store_tiles(
    w: &mut Wave<'_, Gfx1151, Builder>,
    end: &End,
    store: impl Fn(&mut Wave<'_, Gfx1151, Builder>, u8) -> R,
) -> R {
    y_descriptor(w.isa())?;
    for j in 0..NT {
        if j > 0 {
            let live = w.scmp(Instruction::new(
                format!("s_cmp_gt_u32 s{CNT}, {j}"),
                vec![],
                vec![s(CNT)],
            ))?;
            w.exit_unless(live, end)?;
            add64(w.isa(), SRD_Y, SRD_Y, YSTEP)?;
        }
        store(w, j)?;
    }
    Ok(())
}

/// Stores of a non-final row block: a dead tile skips its store (the wave goes on to the next block)
/// in place of leaving the kernel.
fn store_tiles_mid(w: &mut Wave<'_, Gfx1151, Builder>, g: &G) -> R {
    y_descriptor(w.isa())?;
    for j in 0..NT {
        if j == 0 {
            store_rows(w.isa(), g, g.sum(0), g.live(0))?;
            continue;
        }
        let live = w.scmp(Instruction::new(
            format!("s_cmp_gt_u32 s{CNT}, {j}"),
            vec![],
            vec![s(CNT)],
        ))?;
        w.skip_unless(
            live,
            &g.label(&format!("store_dead{j}")),
            (),
            |w: &mut Wave<'_, Gfx1151, Builder>, ()| {
                add64(w.isa(), SRD_Y, SRD_Y, YSTEP)?;
                store_rows(w.isa(), g, g.sum(j), g.live(j))
            },
        )?;
    }
    Ok(())
}

/// Row block `g` to block `g + 1`: reset the descriptors and the trip count, issue the next block's
/// first-epoch weights, store this block's tiles, then reload the next block's first-epoch
/// activations and zero the sums. Pending stores are drained so the next K loop enters with the
/// ledger shape its back edge leaves.
fn transition(wg: &mut Wg<Gfx1151>, g: &G) -> R {
    let nx = g.at(g.blk + 1);
    wg.label(&g.label("epilogue"))?;
    let b = wg.isa();
    x_descriptors(b)?;
    sop(
        b,
        format!("{} s{RBASE}, s{RBASE}, 64", s_add_i32(arch())),
        &[RBASE],
        &[RBASE],
    )?;
    w_descriptors(b)?;
    set_trips(b)?;
    load_w(b, &nx, 0)?;
    store_tiles_mid(wg, g)?;
    wg.label(&g.label("xreload"))?;
    let b = wg.isa();
    for j in 0..NT {
        load_x(b, g, j, SRD_X[0])?;
    }
    for t in 0..NT {
        for j in 0..8u8 {
            op(
                b,
                format!("v_mov_b32_e32 v{}, 0", g.sum(t) + j),
                &[v(g.sum(t) + j)],
                &[],
            )?;
        }
    }
    b.wait(crate::ledger::Counter::Vs, 0)
}

/// Tile `j`'s gate publication (two b128 stores per lane) in the linear write token `out`.
fn publish_tile<T: Target, S: StoreTarget<T::Waits, Out = S>>(
    w: &mut Wave<'_, T, Builder>,
    g: &G,
    j: u8,
    mut out: S,
) -> Result<S, String> {
    for h in 0..2u8 {
        let off = u32::from(j) * NT_GATE_BYTES + 16 * u32::from(h);
        let d = crate::reg::RegRef {
            kind: crate::reg::Kind::V,
            base: g.sum(j) + 4 * h,
            len: 4,
        };
        let text = format!(
            "ds_store_b128 v{LDSA}, {d}{}",
            if off == 0 {
                String::new()
            } else {
                format!(" offset:{off}")
            }
        );
        out = w.ds_store(
            out,
            Instruction::new(text, vec![], vec![v(LDSA), d]).memory(MemoryClass::DsStore),
        )?;
    }
    Ok(out)
}

fn epilogue(wg: &mut Wg<Gfx1151>, g: &G, end: End, gate: Option<LdsRegion<Gate, Free>>) -> R {
    let b = wg.isa();
    b.label(&g.label("epilogue"))?;
    let Some(gate) = gate else {
        if g.acc {
            store_tiles(wg, &end, |w, j| store_acc(w.isa(), g, j))?;
        } else {
            store_tiles(wg, &end, |w, j| store_rows(w.isa(), g, g.sum(j), g.live(j)))?;
        }
        return wg.end(end);
    };
    // Every wave: BF16 round trip of its sums (gate or up).
    for j in 0..NT {
        for i in 0..8u8 {
            Bf16::rne_finite_passthrough(
                b,
                g.sum(j) + i,
                RT_TMP + i,
                MASKT[usize::from(i % 2)],
                true,
            )?;
        }
    }
    // Gate waves publish the rounded gate rows and leave; up waves meet them at their own barrier,
    // read the publication and store the tiles (wave scope, up to the kernel exit).
    let up = wg.scmp(Instruction::new(
        format!("s_cmp_ge_u32 s{WAVE}, 2"),
        vec![],
        vec![s(WAVE)],
    ))?;
    let gv = g.gval();
    wg.handoff(
        up,
        &g.label("up"),
        end,
        gate,
        |w, gate| {
            let mut out = w.begin_write(gate);
            for j in 0..NT {
                out = publish_tile(w, g, j, out)?;
            }
            Ok(out)
        },
        |w, gate, end| {
            store_tiles(w, end, |w, j| {
                for h in 0..2u8 {
                    let off = u32::from(j) * NT_GATE_BYTES + 16 * u32::from(h);
                    let d = crate::reg::RegRef {
                        kind: crate::reg::Kind::V,
                        base: gv + 4 * h,
                        len: 4,
                    };
                    let text = format!(
                        "ds_load_b128 {d}, v{LDSA}{}",
                        if off == 0 {
                            String::new()
                        } else {
                            format!(" offset:{off}")
                        }
                    );
                    w.ds_load(
                        &gate,
                        Instruction::new(text, vec![d], vec![v(LDSA)]).memory(MemoryClass::DsLoad),
                    )?;
                }
                for grp in 0..2u8 {
                    crate::kernels::gemm_uk::Epilogue::silu_dense(
                        w.isa(),
                        gv + SILU_N * grp,
                        g.sum(j) + SILU_N * grp,
                        SILU_TMP,
                        SILU_MASK,
                        SILU_N,
                    )?;
                }
                store_rows(w.isa(), g, gv, g.live(j))
            })
        },
    )
}

/// One gfx1151 entry (`acc`: the F32 qualification twin, `rr == 1` only).
pub(super) fn emit_gfx1151(spec: Spec, acc: bool) -> Result<Emitted, String> {
    validate(spec)?;
    if acc && spec.rr != 1 {
        return Err(
            "qwen4_moe_sym w3 nt4 gfx1151: the F32 qualification twin is built for rr 1 only"
                .into(),
        );
    }
    let g = G { spec, blk: 0, acc };
    let sfx = if acc { "_acc" } else { "" };
    let kspec = KernelSpec {
        kernel_id: "qwen4_moe_sym".into(),
        variant: format!("{}{sfx}", spec.variant()),
        arch: spec.arch,
        symbol: format!("{}{sfx}", spec.symbol()),
        kernargs: spec.kernargs(),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: true,
        workgroup_size: spec.threads() as u16,
        group_segment_fixed_size: if g.handoff() { spec.lds_bytes() } else { 0 },
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, g.plan()?);
    b.enable_delay_alu();
    body(&mut Workgroup::<Gfx1151, Builder>::new(&mut b)?, &g)?;
    b.finish()
}

fn body(wg: &mut Wg<Gfx1151>, g: &G) -> R {
    let gate = if g.handoff() {
        Some(wg.lds::<Gate>("swiglu_gate", 0, g.spec.lds_bytes())?)
    } else {
        None
    };
    let end = wg.exit(&g.exit_label())?;
    prologue(wg, g, &end)?;
    for blk in 0..g.rr() - 1 {
        let gb = g.at(blk);
        kloop(wg, &gb)?;
        transition(wg, &gb)?;
    }
    let last = g.at(g.rr() - 1);
    kloop(wg, &last)?;
    epilogue(wg, &last, end, gate)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn num_after(text: &str, key: &str) -> u32 {
        let at = text.find(key).unwrap_or_else(|| panic!("{key} missing"));
        text[at + key.len()..]
            .trim_start()
            .split(|c: char| !c.is_ascii_digit())
            .next()
            .unwrap()
            .parse()
            .unwrap()
    }

    #[test]
    fn module_specs_are_the_nt4_entries_and_the_down_row_repeats() {
        let symbols: Vec<_> = module_specs().iter().map(|s| s.symbol()).collect();
        assert_eq!(
            symbols,
            [
                "qwen4_moe_gate_up_silu_iu4_sym_w3_pm_gfx1151_nt4",
                "qwen4_moe_down_iu4_sym_w3_pm_gfx1151_nt4",
                "qwen4_moe_down_iu4_sym_w3_pm_gfx1151_nt4r2",
                "qwen4_moe_down_iu4_sym_w3_pm_gfx1151_nt4r4",
            ]
        );
        for s in module_specs() {
            validate(s).unwrap();
        }
    }

    #[test]
    fn only_the_built_subset_validates() {
        let spec = module_specs()[1];
        for bad in [
            Spec {
                arch: Arch::Gfx1201,
                ..spec
            },
            Spec {
                arch: Arch::Gfx1100,
                ..spec
            },
            Spec { nt: 1, ..spec },
            Spec { nt: 2, ..spec },
            Spec { nt: 8, ..spec },
            Spec { rr: 3, ..spec },
            Spec {
                kind: Kind::GateUp,
                rr: 2,
                ..spec
            },
            Spec {
                codec: WeightCodec::Mq4Sym,
                ..spec
            },
        ] {
            assert!(
                validate(bad).is_err() && emit_gfx1151(bad, false).is_err(),
                "{bad:?}"
            );
        }
        assert!(emit_gfx1151(Spec { rr: 2, ..spec }, true).is_err());
    }

    #[test]
    fn entries_emit_deterministically_with_the_planned_resources() {
        for spec in module_specs() {
            for acc in [false, true] {
                if acc && spec.rr != 1 {
                    continue;
                }
                let (a, b) = (
                    emit_gfx1151(spec, acc).unwrap(),
                    emit_gfx1151(spec, acc).unwrap(),
                );
                assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256);
                let sym = format!("{}{}", spec.symbol(), if acc { "_acc" } else { "" });
                assert!(a.s_text.contains(&sym), "{sym}");
                // 208 VGPRs: the MQ4 gfx1151 NT4 layout's; the two unpack temporaries fill its free v142..v143.
                assert_eq!(a.proof.next_free_vgpr, 208, "{sym}");
                assert!(a.shape.next_free_sgpr <= 104, "{sym}");
                let handoff = spec.kind == Kind::GateUp && !acc;
                assert_eq!(
                    num_after(&a.s_text, ".amdhsa_group_segment_fixed_size"),
                    if handoff { 8192 } else { 0 },
                    "{sym}"
                );
                assert_eq!(a.s_text.contains("s_barrier"), handoff, "{sym}");
                // gfx11 spellings and the planar W3 weight side: K16 WMMA, dword code loads, bit-field spread.
                assert!(
                    a.s_text.contains("v_wmma_i32_16x16x16_iu4")
                        && !a.s_text.contains("v_wmma_i32_16x16x32_iu4"),
                    "{sym}"
                );
                assert!(
                    a.s_text.contains("v_bfi_b32") && a.s_text.contains("buffer_load_b32"),
                    "{sym}"
                );
                assert!(
                    a.s_text.contains("s_waitcnt") && !a.s_text.contains("ttmp"),
                    "{sym}"
                );
            }
        }
    }
}
