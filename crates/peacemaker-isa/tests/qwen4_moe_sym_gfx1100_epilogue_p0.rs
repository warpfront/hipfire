// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! B4-P0 (SOL-B4-B5 §2.5): pure bit-routing proof of the gfx1100 K2 epilogue.
//!
//! The emitter is run on a gfx1100 `Builder`, its EMITTED INSTRUCTION TEXT is read back
//! and interpreted by a small wave32 model of exactly the instructions it uses
//! (`v_perm_b32` byte-select over `{src0, src1}`, `v_permlanex16_b32` row-swap, `v_cndmask_b32_e64`
//! lane-mask select, movs, and for `_acc` the scalar address arithmetic and `buffer_store_b32`).
//! Any other instruction in the interpreted region fails the test. Words are 4 byte-symbols:
//! numeric bytes (<256) or provenance tags (`256 + origin`), so a single run proves, for every
//! lane/j/byte, where each output byte came from, and a numeric run feeds arbitrary high-half
//! BF16 patterns and independent low-half junk.
//!
//! Mutation knob (CPU only): `B4P0_MUTANT=<name>` perturbs the selectors/row formula the main
//! tests use; every mutant must make `cargo test` fail. Names: send, recv_h0_even, recv_h0_odd,
//! recv_h1_even, recv_h1_odd, permlane_lo, permlane_hi, wrong_pack_selector, rowmap8.
//! `p0_mutants_are_rejected` asserts the same in-process.

#[path = "../src/kernels/qwen4_moe_sym_gfx1100/epilogue.rs"]
mod epilogue;
#[path = "../src/kernels/qwen4_moe_sym_gfx1100/epilogue_fixture.rs"]
mod epilogue_fixture;

use epilogue::{regs::*, AccRegs, PackSelectors};
use peacemaker_isa::{
    kernels::qwen4_moe_sym::Kind, reg::Live, Arch, Builder, KernargLayout, KernelSpec, RegPlan,
};

// ---------------------------------------------------------------- wave32 model

type W = [u32; 4];
const LANES: usize = 32;
const TAG: u32 = 256;

fn num(x: u32) -> W {
    [x & 255, (x >> 8) & 255, (x >> 16) & 255, x >> 24]
}
fn to_num(w: W) -> u32 {
    assert!(
        w.iter().all(|&b| b < TAG),
        "non-numeric word {w:?} used as a number"
    );
    w[0] | w[1] << 8 | w[2] << 16 | w[3] << 24
}

fn parse_lit(t: &str) -> Option<u32> {
    let t = t.trim();
    if let Some(h) = t.strip_prefix("0x") {
        return u32::from_str_radix(h, 16).ok();
    }
    if let Some(n) = t.strip_prefix('-') {
        return n.parse::<u32>().ok().map(|x| x.wrapping_neg());
    }
    t.parse::<u32>().ok()
}

fn sgpr_range(a: &str) -> (usize, usize) {
    if let Some(r) = a.strip_prefix("s[").and_then(|x| x.strip_suffix(']')) {
        let (lo, hi) = r.split_once(':').unwrap();
        let (lo, hi): (usize, usize) = (lo.parse().unwrap(), hi.parse().unwrap());
        return (lo, hi - lo + 1);
    }
    (a.strip_prefix('s').unwrap().parse().unwrap(), 1)
}
fn vgpr_range(a: &str) -> (usize, usize) {
    if let Some(r) = a.strip_prefix("v[").and_then(|x| x.strip_suffix(']')) {
        let (lo, hi) = r.split_once(':').unwrap();
        let (lo, hi): (usize, usize) = (lo.parse().unwrap(), hi.parse().unwrap());
        return (lo, hi - lo + 1);
    }
    (a.strip_prefix('v').unwrap().parse().unwrap(), 1)
}

struct Sim {
    v: Vec<[W; LANES]>,
    s: Vec<u32>,
    stores: Vec<(u64, u32)>,
    mem: std::collections::HashMap<u64, u8>,
}
impl Sim {
    fn new() -> Self {
        let mut v = vec![[[0u32; 4]; LANES]; 256];
        for (r, reg) in v.iter_mut().enumerate() {
            for (l, w) in reg.iter_mut().enumerate() {
                *w = [TAG + (1 << 20) + ((r as u32) << 5) + l as u32; 4];
            }
        }
        Self {
            v,
            s: vec![0xdead_beef; 128],
            stores: vec![],
            mem: Default::default(),
        }
    }
    fn rd32(&self, a: u64) -> u32 {
        (0..4).fold(0u32, |x, i| {
            x | u32::from(
                *self
                    .mem
                    .get(&(a + i))
                    .unwrap_or_else(|| panic!("read of unmapped byte {:#x}", a + i)),
            ) << (8 * i)
        })
    }
    fn wr32(&mut self, a: u64, x: u32) {
        for i in 0..4 {
            self.mem.insert(a + i, (x >> (8 * i)) as u8);
        }
    }
    fn vreg(a: &str) -> Option<usize> {
        a.strip_prefix('v').and_then(|n| n.parse().ok())
    }
    fn sreg(a: &str) -> Option<usize> {
        a.strip_prefix('s').and_then(|n| n.parse().ok())
    }
    fn src(&self, a: &str, l: usize) -> W {
        if let Some(r) = Self::vreg(a) {
            return self.v[r][l];
        }
        if let Some(r) = Self::sreg(a) {
            return num(self.s[r]);
        }
        num(parse_lit(a).unwrap_or_else(|| panic!("unparsed operand {a:?}")))
    }
    fn mask(&self, a: &str) -> u64 {
        let r = Self::sreg(a).expect("lane mask is an sgpr pair");
        u64::from(self.s[r]) | u64::from(self.s[r + 1]) << 32
    }
    fn exec(&mut self, text: &str) {
        let (m, rest) = text.split_once(char::is_whitespace).unwrap_or((text, ""));
        let a: Vec<&str> = rest.split(',').map(str::trim).collect();
        match m {
            "s_nop" | "s_delay_alu" | "s_clause" | "s_waitcnt" | "s_waitcnt_vscnt" | "s_endpgm" => {
            }
            l if l.ends_with(':') => {}
            "s_mov_b32" => {
                let v = match Self::sreg(a[1]) {
                    Some(r) => self.s[r],
                    None => parse_lit(a[1]).unwrap(),
                };
                self.s[Self::sreg(a[0]).unwrap()] = v
            }
            "s_lshl_b32" => {
                let x = match Self::sreg(a[1]) {
                    Some(r) => self.s[r],
                    None => parse_lit(a[1]).unwrap(),
                };
                self.s[Self::sreg(a[0]).unwrap()] = x << parse_lit(a[2]).unwrap();
            }
            "v_mov_b32_e32" => {
                let d = Self::vreg(a[0]).unwrap();
                for l in 0..LANES {
                    self.v[d][l] = self.src(a[1], l)
                }
            }
            "v_cndmask_b32_e64" => {
                let d = Self::vreg(a[0]).unwrap();
                let mk = self.mask(a[3]);
                let out: Vec<W> = (0..LANES)
                    .map(|l| {
                        if mk >> l & 1 == 1 {
                            self.src(a[2], l)
                        } else {
                            self.src(a[1], l)
                        }
                    })
                    .collect();
                for l in 0..LANES {
                    self.v[d][l] = out[l]
                }
            }
            "v_perm_b32" => {
                let d = Self::vreg(a[0]).unwrap();
                let out: Vec<W> = (0..LANES)
                    .map(|l| {
                        let (s0, s1) = (self.src(a[1], l), self.src(a[2], l));
                        let sel = to_num(self.src(a[3], l));
                        let c = [s1[0], s1[1], s1[2], s1[3], s0[0], s0[1], s0[2], s0[3]];
                        let mut o = [0u32; 4];
                        for i in 0..4 {
                            let sb = (sel >> (8 * i)) & 255;
                            o[i] = match sb {
                                0..=7 => c[sb as usize],
                                8..=11 => {
                                    let b = c[(2 * (sb - 8) + 1) as usize];
                                    assert!(b < TAG, "sign-select of a tagged byte");
                                    if b & 0x80 != 0 {
                                        255
                                    } else {
                                        0
                                    }
                                }
                                12 => 0,
                                _ => 255,
                            };
                        }
                        o
                    })
                    .collect();
                for l in 0..LANES {
                    self.v[d][l] = out[l]
                }
            }
            "v_permlanex16_b32" => {
                let d = Self::vreg(a[0]).unwrap();
                let s0 = Self::vreg(a[1]).expect("permlane data is a vgpr");
                let lo = to_num(self.src(a[2], 0));
                let hi = to_num(self.src(a[3], 0));
                let out: Vec<W> = (0..LANES)
                    .map(|l| {
                        let i = l & 15;
                        let nib = (if i < 8 { lo } else { hi } >> (4 * (i & 7))) & 15;
                        self.v[s0][((l ^ 16) & 16) | nib as usize]
                    })
                    .collect();
                for l in 0..LANES {
                    self.v[d][l] = out[l]
                }
            }
            // `_acc` address arithmetic.
            "v_and_b32_e32" | "v_lshrrev_b32_e32" | "v_lshlrev_b32_e32" | "v_add_nc_u32_e32"
            | "v_mul_lo_u32" => {
                let d = Self::vreg(a[0]).unwrap();
                for l in 0..LANES {
                    let (x, y) = (to_num(self.src(a[1], l)), to_num(self.src(a[2], l)));
                    let r = match m {
                        "v_and_b32_e32" => x & y,
                        "v_lshrrev_b32_e32" => y >> (x & 31),
                        "v_lshlrev_b32_e32" => y << (x & 31),
                        "v_add_nc_u32_e32" => x.wrapping_add(y),
                        _ => x.wrapping_mul(y),
                    };
                    self.v[d][l] = num(r);
                }
            }
            "buffer_store_b32" => {
                let off: u64 = a[3]
                    .split("offset:")
                    .nth(1)
                    .map(|o| o.trim().parse().unwrap())
                    .unwrap_or(0);
                for l in 0..LANES {
                    let addr = u64::from(to_num(self.src(a[1], l))) + off;
                    self.stores.push((addr, to_num(self.src(a[0], l))));
                }
            }
            "s_load_b32" | "s_load_b64" | "s_load_b128" => {
                let n = match m {
                    "s_load_b32" => 1,
                    "s_load_b64" => 2,
                    _ => 4,
                };
                let (d, _) = sgpr_range(a[0]);
                let (base, _) = sgpr_range(a[1]);
                let off = if a[2] == "null" {
                    0
                } else {
                    u64::from(parse_lit(a[2]).unwrap())
                };
                let addr = u64::from(self.s[base]) | u64::from(self.s[base + 1]) << 32;
                for i in 0..n {
                    self.s[d + i] = self.rd32(addr + off + 4 * i as u64);
                }
            }
            "v_readfirstlane_b32" => {
                self.s[Self::sreg(a[0]).unwrap()] = to_num(self.src(a[1], 0));
            }
            "v_cmp_le_i32_e64" => {
                let d = Self::sreg(a[0]).unwrap();
                let mk = (0..LANES).fold(0u64, |acc, l| {
                    acc | u64::from(
                        (to_num(self.src(a[1], l)) as i32) <= (to_num(self.src(a[2], l)) as i32),
                    ) << l
                });
                self.s[d] = mk as u32;
                self.s[d + 1] = (mk >> 32) as u32;
            }
            "v_cmp_class_f32_e64" => {
                let d = Self::sreg(a[0]).unwrap();
                let cls = parse_lit(a[2]).unwrap();
                let mk = (0..LANES).fold(0u64, |acc, l| {
                    let x = to_num(self.src(a[1], l));
                    let (neg, e, f) = (x >> 31 == 1, (x >> 23) & 0xff, x & 0x7f_ffff);
                    let bit = match (e, f, neg) {
                        (0xff, f, _) if f != 0 => {
                            if f >> 22 == 1 {
                                1
                            } else {
                                0
                            }
                        }
                        (0xff, _, true) => 2,
                        (0xff, _, false) => 9,
                        (0, 0, true) => 5,
                        (0, 0, false) => 6,
                        (0, _, true) => 4,
                        (0, _, false) => 7,
                        (_, _, true) => 3,
                        (_, _, false) => 8,
                    };
                    acc | u64::from(cls >> bit & 1) << l
                });
                self.s[d] = mk as u32;
                self.s[d + 1] = (mk >> 32) as u32;
            }
            "v_bfe_u32" | "v_add3_u32" | "v_lshl_add_u32" => {
                let d = Self::vreg(a[0]).unwrap();
                for l in 0..LANES {
                    let (x, y, z) = (
                        to_num(self.src(a[1], l)),
                        to_num(self.src(a[2], l)),
                        to_num(self.src(a[3], l)),
                    );
                    let r = match m {
                        "v_bfe_u32" => (x >> (y & 31)) & ((1u32 << (z & 31)) - 1),
                        "v_add3_u32" => x.wrapping_add(y).wrapping_add(z),
                        _ => (x << (y & 31)).wrapping_add(z),
                    };
                    self.v[d][l] = num(r);
                }
            }
            "s_lshr_b32" | "s_add_i32" => {
                let x = match Self::sreg(a[1]) {
                    Some(r) => self.s[r],
                    None => parse_lit(a[1]).unwrap(),
                };
                let y = match Self::sreg(a[2]) {
                    Some(r) => self.s[r],
                    None => parse_lit(a[2]).unwrap(),
                };
                self.s[Self::sreg(a[0]).unwrap()] = if m == "s_lshr_b32" {
                    x >> (y & 31)
                } else {
                    x.wrapping_add(y)
                };
            }
            "buffer_load_b32" => {
                let d = Self::vreg(a[0]).unwrap();
                let (sb, _) = sgpr_range(a[2]);
                let base = u64::from(self.s[sb]) | u64::from(self.s[sb + 1]) << 32;
                let off: u64 = a[3]
                    .split("offset:")
                    .nth(1)
                    .map(|o| o.trim().parse().unwrap())
                    .unwrap_or(0);
                for l in 0..LANES {
                    let addr = base + u64::from(to_num(self.src(a[1], l))) + off;
                    self.v[d][l] = num(self.rd32(addr));
                }
            }
            "buffer_store_b128" => {
                let (d, _) = vgpr_range(a[0]);
                let (sb, _) = sgpr_range(a[2]);
                let base = u64::from(self.s[sb]) | u64::from(self.s[sb + 1]) << 32;
                let off: u64 = a[3]
                    .split("offset:")
                    .nth(1)
                    .map(|o| o.trim().parse().unwrap())
                    .unwrap_or(0);
                for l in 0..LANES {
                    for k in 0..4 {
                        let addr = base + u64::from(to_num(self.src(a[1], l))) + off + 4 * k as u64;
                        let w = to_num(self.v[d + k][l]);
                        self.wr32(addr, w);
                    }
                }
            }
            other => panic!("instruction {other:?} is outside the P0 model: {text:?}"),
        }
    }
    fn run(&mut self, b: &Builder, from: usize) {
        for i in &b.program().instructions[from..] {
            self.exec(&i.text)
        }
    }
}

// ---------------------------------------------------------------- builder harness

fn plan(kind: Kind, acc: bool) -> RegPlan {
    let mut p = RegPlan::new(256, 104).unwrap();
    let whole = || Live::Whole;
    p.v::<1>("y_off", YOFF, whole()).unwrap();
    p.v::<8>("sum", SUM, whole()).unwrap();
    if kind == Kind::GateUp {
        p.v::<8>("sum_up", SUM + 8, whole()).unwrap();
    }
    p.s::<2>("y_base", Y_PTR, whole()).unwrap();
    p.s::<2>("live", LIVE, whole()).unwrap();
    for m in MASKT {
        p.s::<2>("cmp_mask", m, whole()).unwrap();
    }
    if acc {
        let r = AccRegs::default();
        p.s::<1>("m", r.m, whole()).unwrap();
        p.s::<1>("tile16", r.tile16, whole()).unwrap();
        p.s::<1>("rbase", r.rbase, whole()).unwrap();
        p.v::<1>("lane", r.lane, whole()).unwrap();
        epilogue::reserve_acc(&mut p, kind, "epilogue", "end").unwrap();
    } else {
        epilogue::reserve_bf16(&mut p, kind, "epilogue", "end").unwrap();
    }
    p
}

fn builder(kind: Kind, acc: bool) -> Builder {
    let spec = KernelSpec {
        kernel_id: "qwen4_moe_sym".into(),
        variant: "p0".into(),
        arch: Arch::Gfx1100,
        symbol: "p0_gfx1100".into(),
        kernargs: KernargLayout::new(60),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: true,
        workgroup_size: 64,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(spec, plan(kind, acc));
    b.enable_delay_alu();
    b.label("epilogue").unwrap();
    b
}

fn texts(b: &Builder) -> Vec<String> {
    b.program()
        .instructions
        .iter()
        .map(|i| i.text.clone())
        .collect()
}

/// Hazard/delay filler the builder inserts; ignored when comparing instruction streams.
fn is_filler(t: &str) -> bool {
    let m = t.split_whitespace().next().unwrap_or("");
    matches!(m, "s_nop" | "s_delay_alu" | "s_clause") || m.starts_with("s_wait") || m.ends_with(':')
}

// ---------------------------------------------------------------- the checks

type RowMap = fn(usize, usize) -> usize;
/// gfx11/gfx1100 C layout: accumulator `j` of lane half `hi` is row `2*j + hi`.
fn rows_gfx11(hi: usize, j: usize) -> usize {
    2 * j + hi
}
/// The gfx12 layout; wrong for this kernel.
fn rows_gfx12(hi: usize, j: usize) -> usize {
    8 * hi + j
}

fn source_of(row: usize, rowmap: RowMap) -> Result<(usize, usize), String> {
    let mut hit = None;
    for hi in 0..2 {
        for j in 0..8 {
            if rowmap(hi, j) == row {
                if hit.is_some() {
                    return Err(format!("row {row} has two sources"));
                }
                hit = Some((hi, j));
            }
        }
    }
    hit.ok_or_else(|| format!("row {row} has no source"))
}

fn origin(lane: usize, j: usize, byte: usize) -> u32 {
    TAG + ((lane << 5 | j << 2 | byte) as u32)
}

fn exchange_program(sel: &PackSelectors, live: bool) -> Builder {
    let mut b = builder(Kind::Down, false);
    if live {
        epilogue::emit_apply_live(&mut b).unwrap();
    }
    epilogue::emit_exchange(&mut b, sel).unwrap();
    b
}

/// Byte provenance of every output byte, lane/j/byte exact, against the row formula `rowmap`.
fn check_provenance(sel: &PackSelectors, rowmap: RowMap) -> Result<(), String> {
    let b = exchange_program(sel, false);
    let mut sim = Sim::new();
    for lane in 0..LANES {
        for j in 0..8 {
            sim.v[SUM as usize + j][lane] = [
                origin(lane, j, 0),
                origin(lane, j, 1),
                origin(lane, j, 2),
                origin(lane, j, 3),
            ];
        }
    }
    sim.run(&b, 0);
    let mut seen = std::collections::BTreeSet::new();
    for lane in 0..LANES {
        let (hi, lr) = (lane >> 4, lane & 15);
        for k in 0..4 {
            let got = sim.v[OUT as usize + k][lane];
            let mut want = [0u32; 4];
            for w in 0..2 {
                let row = 8 * hi + 2 * k + w;
                let (shi, sj) = source_of(row, rowmap)?;
                let src = shi * 16 + lr;
                want[2 * w] = origin(src, sj, 2);
                want[2 * w + 1] = origin(src, sj, 3);
            }
            if got != want {
                return Err(format!(
                    "lane {lane} (hi {hi}, lr {lr}) OUT[{k}] = {got:x?}, want {want:x?}"
                ));
            }
            for g in got {
                if !seen.insert(g) {
                    return Err(format!("origin {g:#x} written twice"));
                }
            }
        }
    }
    // 32 lanes * 8 j * (high-half bytes 2,3) = 512 distinct sources, each exactly once.
    if seen.len() != LANES * 8 * 2 {
        return Err(format!(
            "{} distinct byte origins written, want {}",
            seen.len(),
            LANES * 8 * 2
        ));
    }
    Ok(())
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

/// Arbitrary high halves (incl. NaN/inf/zero patterns) and independent low-half junk, numeric;
/// optionally with padding tokens (LIVE equal in lanes `lr` and `lr+16`).
fn check_numeric(sel: &PackSelectors, rowmap: RowMap, live: bool) -> Result<(), String> {
    let b = exchange_program(sel, live);
    let special = [
        0x0000u32, 0x8000, 0x7f80, 0xff80, 0x7fc0, 0xffff, 0x0001, 0x3f80, 0xbf80, 0x7f7f,
    ];
    let mut rng = Lcg(0x1234_5678_9abc_def1);
    for round in 0..256 {
        let mut input = [[0u32; 8]; LANES];
        for lane in 0..LANES {
            for j in 0..8 {
                let hi16 = if (round + lane + j) % 5 == 0 {
                    special[(rng.next() as usize) % special.len()]
                } else {
                    rng.next() & 0xffff
                };
                input[lane][j] = hi16 << 16 | (rng.next() & 0xffff);
            }
        }
        let live_tokens = rng.next() & 0xffff;
        let live_mask = if live {
            u64::from(live_tokens | live_tokens << 16)
        } else {
            0
        };
        let mut sim = Sim::new();
        for lane in 0..LANES {
            for j in 0..8 {
                sim.v[SUM as usize + j][lane] = num(input[lane][j]);
            }
        }
        sim.s[LIVE as usize] = live_mask as u32;
        sim.s[LIVE as usize + 1] = (live_mask >> 32) as u32;
        sim.run(&b, 0);
        for lane in 0..LANES {
            let (hi, lr) = (lane >> 4, lane & 15);
            for k in 0..4 {
                let h = |row: usize| -> Result<u32, String> {
                    let (shi, sj) = source_of(row, rowmap)?;
                    let alive = !live || live_tokens >> lr & 1 == 1;
                    Ok(if alive {
                        input[shi * 16 + lr][sj] >> 16
                    } else {
                        0
                    })
                };
                let want = h(8 * hi + 2 * k)? | h(8 * hi + 2 * k + 1)? << 16;
                let got = to_num(sim.v[OUT as usize + k][lane]);
                if got != want {
                    return Err(format!(
                        "round {round} lane {lane} OUT[{k}] = {got:#010x}, want {want:#010x}"
                    ));
                }
            }
        }
    }
    Ok(())
}

/// `_acc`: every F32 accumulator lands at `4*(p*M + rbase + row + t*M/2)`, padding +0, no collisions.
fn check_acc(kind: Kind, rowmap: RowMap) -> Result<(), String> {
    let tn = if kind == Kind::GateUp { 2 } else { 1 };
    let mut b = builder(kind, true);
    epilogue::emit_acc(&mut b, kind, &AccRegs::default()).unwrap();
    for (m, tile, rbase) in [
        (1280u32, 3u32, 32u32),
        (2560, 0, 192),
        (2560, 77, 2496),
        (1280, 1, 0),
    ] {
        let mut sim = Sim::new();
        let r = AccRegs::default();
        sim.s[r.m as usize] = m;
        sim.s[r.tile16 as usize] = 16 * tile;
        sim.s[r.rbase as usize] = rbase;
        let live_tokens = 0b1011_0110_1101_0011u32 ^ tile;
        let live_mask = u64::from((live_tokens & 0xffff) | (live_tokens & 0xffff) << 16);
        sim.s[LIVE as usize] = live_mask as u32;
        sim.s[LIVE as usize + 1] = 0;
        for lane in 0..LANES {
            sim.v[r.lane as usize][lane] = num(lane as u32);
            for t in 0..tn {
                for j in 0..8 {
                    sim.v[SUM as usize + 8 * t + j][lane] =
                        num(0x4000_0000 | (lane as u32) << 12 | (t as u32) << 8 | j as u32 | 0x80);
                }
            }
        }
        sim.run(&b, 0);
        let mut mem = std::collections::BTreeMap::new();
        for (addr, val) in &sim.stores {
            if mem.insert(*addr, *val).is_some() {
                return Err(format!("two stores to byte {addr}"));
            }
        }
        if mem.len() != 16 * 16 * tn {
            return Err(format!("{} stores, want {}", mem.len(), 16 * 16 * tn));
        }
        for lane in 0..LANES {
            let (hi, lr) = (lane >> 4, lane & 15);
            let p = 16 * tile as usize + lr;
            for t in 0..tn {
                for j in 0..8 {
                    let row = rowmap(hi, j);
                    let addr = 4 * (p * m as usize + rbase as usize + row + t * (m as usize / 2));
                    let want = if live_tokens >> lr & 1 == 1 {
                        0x4000_0000 | (lane as u32) << 12 | (t as u32) << 8 | j as u32 | 0x80
                    } else {
                        0
                    };
                    match mem.get(&(addr as u64)) {
                    Some(&got) if got == want => {}
                    other => return Err(format!("{kind:?} M {m} lane {lane} t {t} j {j}: byte {addr} = {other:x?}, want {want:#x}")),
                }
                }
            }
        }
    }
    Ok(())
}

fn check_all(sel: &PackSelectors, rowmap: RowMap) -> Result<(), String> {
    check_provenance(sel, rowmap).map_err(|e| format!("provenance: {e}"))?;
    check_numeric(sel, rowmap, false).map_err(|e| format!("numeric: {e}"))?;
    check_numeric(sel, rowmap, true).map_err(|e| format!("live: {e}"))?;
    for kind in [Kind::GateUp, Kind::Down] {
        check_acc(kind, rowmap).map_err(|e| format!("acc: {e}"))?;
    }
    Ok(())
}

// ---------------------------------------------------------------- mutants

fn mutate(name: &str) -> (PackSelectors, RowMap) {
    let mut s = PackSelectors::frozen();
    let mut r: RowMap = rows_gfx11;
    match name {
        "send" => s.send ^= 0x0000_0100,
        "recv_h0_even" => s.recv_h0[0] ^= 0x0000_0001,
        "recv_h0_odd" => s.recv_h0[1] ^= 0x0001_0000,
        "recv_h1_even" => s.recv_h1[0] ^= 0x0000_0100,
        "recv_h1_odd" => s.recv_h1[1] ^= 0x0100_0000,
        "permlane_lo" => s.permlane_lo ^= 1,
        "permlane_hi" => s.permlane_hi ^= 1,
        "wrong_pack_selector" => s = PackSelectors::wrong_pack_selector(),
        "rowmap8" => r = rows_gfx12,
        other => panic!("unknown B4P0_MUTANT {other}"),
    }
    (s, r)
}
const MUTANTS: [&str; 9] = [
    "send",
    "recv_h0_even",
    "recv_h0_odd",
    "recv_h1_even",
    "recv_h1_odd",
    "permlane_lo",
    "permlane_hi",
    "wrong_pack_selector",
    "rowmap8",
];

fn under_test() -> (PackSelectors, RowMap) {
    match std::env::var("B4P0_MUTANT") {
        Ok(n) if !n.is_empty() => mutate(&n),
        _ => (PackSelectors::frozen(), rows_gfx11),
    }
}

// ---------------------------------------------------------------- tests

#[test]
fn p0_byte_provenance_matches_row_formula() {
    let (s, r) = under_test();
    check_provenance(&s, r).unwrap();
}

#[test]
fn p0_numeric_high_halves_and_low_junk() {
    let (s, r) = under_test();
    check_numeric(&s, r, false).unwrap();
}

#[test]
fn p0_live_padding_is_zero_and_never_leaks() {
    let (s, r) = under_test();
    check_numeric(&s, r, true).unwrap();
}

#[test]
fn p0_acc_scalar_addresses_are_independent_of_the_pack() {
    let (_, r) = under_test();
    for kind in [Kind::GateUp, Kind::Down] {
        check_acc(kind, r).unwrap();
    }
}

/// Each single-selector mutation and `8*hi+j` in place of `2*j+hi` must be caught.
#[test]
fn p0_mutants_are_rejected() {
    check_all(&PackSelectors::frozen(), rows_gfx11).expect("frozen emitter must pass");
    for name in MUTANTS {
        let (s, r) = mutate(name);
        assert!(check_all(&s, r).is_err(), "mutant {name} was NOT rejected");
    }
}

/// `wrong_pack_selector()` changes exactly one emitted instruction.
#[test]
fn p0_wrong_pack_selector_changes_exactly_one_instruction() {
    let t = |s: &PackSelectors| -> Vec<String> {
        texts(&exchange_program(s, false))
            .into_iter()
            .filter(|x| !is_filler(x))
            .collect()
    };
    let (a, b) = (
        t(&PackSelectors::frozen()),
        t(&PackSelectors::wrong_pack_selector()),
    );
    assert_eq!(a.len(), b.len());
    assert_eq!(a.iter().zip(&b).filter(|(x, y)| x != y).count(), 1);
}

/// Both kinds and both entries build under the gfx1100 builder (access/lifetime/hazard checks),
/// EXEC is never written, and the full BF16 stream contains the exchange verbatim.
#[test]
fn p0_full_emitters_build_exec_untouched_and_embed_the_exchange() {
    let route: Vec<String> = texts(&exchange_program(&PackSelectors::frozen(), false))
        .into_iter()
        .filter(|x| !is_filler(x))
        .collect();
    for kind in [Kind::GateUp, Kind::Down] {
        let mut b = builder(kind, false);
        epilogue::emit_bf16(&mut b, kind, &PackSelectors::frozen()).unwrap();
        let full: Vec<String> = texts(&b).into_iter().filter(|x| !is_filler(x)).collect();
        assert!(
            full.windows(route.len()).any(|w| w == route.as_slice()),
            "{kind:?}: exchange not embedded verbatim"
        );
        assert!(
            full.iter()
                .all(|t| !t.to_ascii_lowercase().contains("exec")),
            "{kind:?}: EXEC referenced"
        );
        assert!(full.last().unwrap().starts_with("buffer_store_b128"));
        b.label("end").unwrap();
        b.control(peacemaker_isa::insn::Instruction::new(
            "s_endpgm",
            vec![],
            vec![],
        ))
        .unwrap();
        b.finish()
            .unwrap_or_else(|e| panic!("{kind:?} bf16 finish: {e}"));

        let mut a = builder(kind, true);
        epilogue::emit_acc(&mut a, kind, &AccRegs::default()).unwrap();
        assert!(texts(&a)
            .iter()
            .all(|t| !t.to_ascii_lowercase().contains("exec")));
        a.label("end").unwrap();
        a.control(peacemaker_isa::insn::Instruction::new(
            "s_endpgm",
            vec![],
            vec![],
        ))
        .unwrap();
        a.finish()
            .unwrap_or_else(|e| panic!("{kind:?} acc finish: {e}"));
    }
}

// ---------------------------------------------------------------- fixture (standalone epilogue kernel)

const KARG_BASE: u64 = 0x7000_0000;
const SUMS_BASE: u64 = 0x1_0000_0000;
const SLOTS_BASE: u64 = 0x2_0000_0000;
const Y_BASE: u64 = 0x3_0000_0000;

struct FxMem {
    mem: std::collections::HashMap<u64, u8>,
    m: u32,
    p: u32,
}
impl FxMem {
    fn new(m: u32, p: u32, sums: &[u32], slots: &[i32], y_poison: u16) -> Self {
        let mut s = Sim::new();
        s.mem.clear();
        for (i, x) in sums.iter().enumerate() {
            s.wr32(SUMS_BASE + 4 * i as u64, *x)
        }
        for (i, x) in slots.iter().enumerate() {
            s.wr32(SLOTS_BASE + 4 * i as u64, *x as u32)
        }
        s.wr32(KARG_BASE, SUMS_BASE as u32);
        s.wr32(KARG_BASE + 4, (SUMS_BASE >> 32) as u32);
        s.wr32(KARG_BASE + 8, SLOTS_BASE as u32);
        s.wr32(KARG_BASE + 12, (SLOTS_BASE >> 32) as u32);
        s.wr32(KARG_BASE + 16, Y_BASE as u32);
        s.wr32(KARG_BASE + 20, (Y_BASE >> 32) as u32);
        s.wr32(KARG_BASE + 24, m);
        for i in 0..(u64::from(p) * u64::from(m) * 2) {
            s.mem.insert(Y_BASE + i, (y_poison >> (8 * (i & 1))) as u8);
        }
        Self { mem: s.mem, m, p }
    }
    /// A wave of CTA `(x, y)` with the given wave index, run through `b`'s whole program.
    fn run_wave(&mut self, b: &Builder, x: u32, y: u32, wave: u32) -> Sim {
        let mut sim = Sim::new();
        sim.mem = std::mem::take(&mut self.mem);
        sim.s[0] = KARG_BASE as u32;
        sim.s[1] = (KARG_BASE >> 32) as u32;
        // gfx1100 places the workgroup ids after the DECLARED (padded) user SGPRs: s14/s15.
        sim.s[epilogue_fixture::WG_X_IN as usize] = x;
        sim.s[epilogue_fixture::WG_Y_IN as usize] = y;
        for lane in 0..LANES {
            sim.v[0][lane] = num(wave * 32 + lane as u32);
        }
        sim.run(b, 0);
        self.mem = std::mem::take(&mut sim.mem);
        sim
    }
    fn y16(&self, byte: u64) -> u16 {
        u16::from(self.mem[&(Y_BASE + byte)]) | u16::from(self.mem[&(Y_BASE + byte + 1)]) << 8
    }
}

fn fx_inputs(m: u32, p: u32, seed: u64) -> (Vec<u32>, Vec<i32>) {
    let special = [
        0u32,
        0x8000_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
        0x7f80_0001,
        0xffff_ffff,
        0x0000_0001,
        0x8000_0001,
        0x3f80_8000,
        0x3f81_8000,
        0x3f80_7fff,
        0x3f81_7fff,
        0x7f7f_ffff,
        0x7f7f_8000,
        0xff7f_8000,
    ];
    let mut rng = Lcg(seed);
    let sums = (0..p * m)
        .map(|_| {
            if rng.next() % 4 == 0 {
                special[(rng.next() as usize) % special.len()]
            } else {
                rng.next()
            }
        })
        .collect();
    let slots = (0..p)
        .map(|i| {
            if rng.next() % 3 == 0 {
                -1 - (i as i32 % 7)
            } else {
                (rng.next() % 4000) as i32
            }
        })
        .collect();
    (sums, slots)
}

fn rne_bits(u: u32) -> u32 {
    let finite = (u >> 23) & 0xff != 0xff;
    (if finite {
        u.wrapping_add(0x7fff).wrapping_add((u >> 16) & 1)
    } else {
        u
    }) >> 16
}

/// Whole Down fixture entry over every CTA/wave: every output u16 equals `LIVE ? rne(sum) : 0`
/// (CPU formula written here independently of the emitter), poison untouched nowhere.
fn check_fixture_down(wrong: bool) -> Result<(), String> {
    let (m, p) = (192u32, 32u32);
    let b = epilogue_fixture::emit_builder(Kind::Down, wrong).unwrap();
    for seed in 1..=4u64 {
        let (sums, slots) = fx_inputs(m, p, seed);
        let mut fx = FxMem::new(m, p, &sums, &slots, 0xBEEF);
        for y in 0..p / 16 {
            for x in 0..m / 64 {
                for w in 0..4 {
                    fx.run_wave(&b, x, y, w);
                }
            }
        }
        for tok in 0..p as usize {
            for col in 0..m as usize {
                let want = if slots[tok] >= 0 {
                    rne_bits(sums[tok * m as usize + col]) as u16
                } else {
                    0
                };
                let got = fx.y16(2 * (tok * m as usize + col) as u64);
                if got != want {
                    return Err(format!(
                        "seed {seed} token {tok} col {col}: {got:#06x}, want {want:#06x}"
                    ));
                }
            }
        }
    }
    Ok(())
}

#[test]
fn fixture_down_matches_cpu_rne_at_every_output_element() {
    check_fixture_down(false).unwrap();
}

#[test]
fn fixture_down_wrong_selector_control_is_detected() {
    assert!(check_fixture_down(true).is_err());
}

/// GateUp prologue (kernarg loads, ids, LIVE, YOFF, F32 sum loads of gate AND up rows) per lane.
#[test]
fn fixture_gate_up_prologue_loads_the_contract_registers() {
    let (m, p) = (128u32, 32u32);
    let b = epilogue_fixture::prologue_builder(Kind::GateUp).unwrap();
    let (sums, slots) = fx_inputs(m, p, 9);
    let mut fx = FxMem::new(m, p, &sums, &slots, 0);
    for y in 0..p / 16 {
        for x in 0..m / 64 {
            for w in 0..2u32 {
                let sim = fx.run_wave(&b, x, y, w);
                let rbase = 32 * x + 16 * w;
                let live_tokens: u32 = (0..16).fold(0, |a, lr| {
                    a | u32::from(slots[(16 * y + lr) as usize] >= 0) << lr
                });
                assert_eq!(
                    sim.s[LIVE as usize],
                    live_tokens | live_tokens << 16,
                    "LIVE x{x} y{y} w{w}"
                );
                for lane in 0..LANES as u32 {
                    let (hi, lr) = (lane >> 4, lane & 15);
                    let tok = 16 * y + lr;
                    for j in 0..8u32 {
                        let g = sums[(tok * m + rbase + 2 * j + hi) as usize];
                        let u = sums[(tok * m + m / 2 + rbase + 2 * j + hi) as usize];
                        assert_eq!(
                            to_num(sim.v[SUM as usize + j as usize][lane as usize]),
                            g,
                            "gate x{x} y{y} w{w} lane{lane} j{j}"
                        );
                        assert_eq!(
                            to_num(sim.v[SUM as usize + 8 + j as usize][lane as usize]),
                            u,
                            "up x{x} y{y} w{w} lane{lane} j{j}"
                        );
                    }
                    assert_eq!(
                        to_num(sim.v[YOFF as usize][lane as usize]),
                        2 * (tok * (m / 2) + rbase + 8 * hi),
                        "YOFF x{x} y{y} w{w} lane{lane}"
                    );
                }
            }
        }
    }
}

#[test]
fn fixture_module_emits_all_four_entries() {
    let (emitted, text) = epilogue_fixture::emit_module().unwrap();
    assert_eq!(emitted.len(), 4);
    for wrong in [false, true] {
        for kind in [Kind::GateUp, Kind::Down] {
            assert!(text.contains(&epilogue_fixture::symbol(kind, wrong)));
        }
    }
}
