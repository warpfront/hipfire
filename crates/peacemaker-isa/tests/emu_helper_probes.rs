// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Hardware-vs-emulator probes of the emulator's helper-call capabilities (item 21): `s_getpc/s_swappc/s_setpc` call
//! and return of leaf callees (two different callees called from the kernel in sequence, a divergent call site with
//! EXEC checked at entry and after, values live across calls), per-lane scratch frames, `flat_atomic_add_u32`
//! (returning and non-returning), the scalar f32 ops `s_sub_f32` / `s_cmp_{gt,neq,ngt}_f32` / `s_cvt_f32_i32` and
//! `v_cmpx_{gt,nge}_f32_e32`, plus two negative controls the hardware runs and the emulator must refuse.
//!
//! Source and build: `isa/emu/helper-probe/{helper_probe.hip,build.sh}`; committed objects
//! `peacemaker-ir/tests/fixtures/helper_probe.gfx120{1,0}.hsaco` (device-only hipcc code objects, unchanged by any
//! PeaceMaker pass). Every probe output word is independent of the load base (PCs are differences to a reference
//! `s_getpc_b64`; no function pointer is formed).
//!
//! * `helper_probes_hardware_vs_emulator` (GPU fabric): runs each probe on a gfx1201 device with the gfx1201 object and
//!   in the emulator with the gfx1201 object, and with the gfx1200 object under `gfx1200_as_gfx1201` (shared-ISA
//!   reasoning, NOT a gfx1200 hardware measurement; no gfx1200 hardware is in the fabric). Output words are compared
//!   word for word; `EMU_PROBE_RECEIPT_OUT=<dir>` writes the hardware outputs and `receipt.json`.
//! * `helper_probes_emulator_matches_committed_hardware_receipt` (CPU): the gfx1201 object in the emulator against
//!   the committed hardware outputs `peacemaker-ir/tests/fixtures/helper_probe_hw_gfx1201/<probe>.bin` (little-endian
//!   u32; a missing file fails the test).
//!
//! Scalar-f32 sweep refusals: the emulator may refuse an operand domain it has no measured model for (e.g. NaN). A
//! refused workgroup is re-run record by record, so the refusal is attributed to exact operand pairs; a refusal is
//! tolerated only for a pair with a NaN operand, is always printed as unmeasured, and never counts as agreement.
//!
//! VALU float-corner probes (`probe_v_*`, `Kind::Valu`): every float instruction of the committed `lane_leviathan`
//! objects (hipcc gfx1200/gfx1201 and the PM twin) the emulator had no measured model for or only a host-arithmetic
//! hypothesis (`v_subrev_f32`, the VOPD halves of `v_dual_{sub,fmamk,fmaak}_f32`, `v_frexp_{exp,mant}`, `v_cvt_*`,
//! `v_rcp_f32`, `v_exp_f32`, `v_ldexp_f32`, `v_rndne_f32`, `v_cndmask_b32` with abs/neg, `v_fma_f32` with neg,
//! `v_div_{fmas,fixup,scale}_f32`, `v_cmp_*_f32`, `v_cmpx_*_f32`, `v_cmp_class_f32`), each as ONE raw inline-asm
//! instruction over NaN (quiet/signalling, both signs, payloads), infinity, zero, denormal and random operands. The
//! hardware words are recorded raw; the emulator is compared field by field, and a record the emulator refuses is
//! reported as UNMEASURED and fails the probe. `helper_probe_emulator_census` (no hardware) lists those refusals and
//! `helper_probe_inputs_dump` writes the input records for offline analysis of a receipt.
use hip_bridge::HipRuntime;
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::helpers::{load_helpers, Helper};
use peacemaker_ir::inst::{Abi, Frontend};
use peacemaker_ir::Program;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

const GFX1201: &[u8] =
    include_bytes!("../../peacemaker-ir/tests/fixtures/helper_probe.gfx1201.hsaco");
const GFX1200: &[u8] =
    include_bytes!("../../peacemaker-ir/tests/fixtures/helper_probe.gfx1200.hsaco");
const RECEIPT_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../peacemaker-ir/tests/fixtures/helper_probe_hw_gfx1201"
);

const SENTINEL: u32 = 0xa5a5_a5a5;
const BLOCK: u32 = 64;
const KARG: u64 = 0x1_0000;
const IN: u64 = 0x100_0000;
const OUT: u64 = 0x800_0000;

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

fn to_bytes(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn to_words(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
fn sha256(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

// ---------------------------------------------------------------------------------------------------------------
// Probe definitions (deterministic inputs shared by the hardware run, the emulator runs and the committed receipt)
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// Hardware output is deterministic and must equal the emulator's word for word.
    Compare,
    /// Returning flat add, every lane on one dword: hardware runs, the emulator must refuse.
    Collide,
    /// Returning flat add through a generic pointer into LDS: hardware runs, the emulator must refuse.
    Lds,
    /// Per-lane VALU float probe (`probe_v_*`): `n` records of three raw words, `width` output words per record,
    /// decoded into `fields`; the emulator is compared field by field and a refused record is reported as unmeasured.
    Valu,
}

/// One decoded result of a VALU probe record: `(out[width * record + word] >> shift) & mask`.
#[derive(Clone)]
struct Field {
    name: String,
    word: usize,
    shift: u32,
    mask: u32,
}

#[derive(Clone)]
struct Probe {
    name: &'static str,
    kind: Kind,
    /// Output words per lane (error reports only; 0 = not lane-structured).
    stride: usize,
    input: Vec<u32>,
    /// Initial content of the `out` buffer; its length is the number of output words.
    out_init: Vec<u32>,
    /// The kernel's `n` argument.
    n: u32,
    groups: u32,
    /// `Kind::Valu` only: output words per record and the decoded fields.
    width: usize,
    fields: Vec<Field>,
}

const ONE: u32 = 0x3f80_0000;
const FLOAT_EDGES: [u32; 21] = [
    0x0000_0000, // +0
    0x8000_0000, // -0
    0x0000_0001, // +min denormal
    0x8000_0001, // -min denormal
    0x007f_ffff, // +max denormal
    0x807f_ffff, // -max denormal
    ONE,
    0xbf80_0000, // -1
    0x3fc0_0000, // 1.5
    0xbfc0_0000, // -1.5
    0x7f7f_ffff, // +max finite
    0xff7f_ffff, // -max finite
    0x7f80_0000, // +inf
    0xff80_0000, // -inf
    0x7fc0_0000, // qNaN
    0x7fc0_0001, // qNaN
    0xffc1_2345, // qNaN (negative, payload)
    0x7fff_ffff, // qNaN (all payload bits)
    0x7f80_0001, // sNaN
    0x7fa0_0000, // sNaN
    0xff81_2345, // sNaN (negative, payload)
];

fn int_edges() -> Vec<u32> {
    let mut v: Vec<u32> = vec![
        0,
        1,
        u32::MAX,
        (1 << 24) - 1,
        1 << 24,
        (1 << 24) + 1,
        ((1u32 << 24) - 1).wrapping_neg(),
        (1u32 << 24).wrapping_neg(),
        ((1u32 << 24) + 1).wrapping_neg(),
        i32::MIN as u32,
        i32::MAX as u32,
    ];
    for k in 0x7fff_ff80u32..=0x7fff_ffff {
        v.push(k);
        v.push(k.wrapping_neg());
    }
    v.sort_unstable();
    v.dedup();
    v
}

/// Records `(a, b, i)` of the scalar-f32 sweep: all pairs of the float edge list (`i = a`), the int edges for
/// `s_cvt_f32_i32` (with the benign float pair 1.0/1.0 so no NaN rides along), 65536 random bit-pattern pairs (fixed-
/// seed LCG), 8192 neighbour pairs (`b = a +- 8 ulps` in bit space), 1024 equal and 1024 sign-flipped pairs, padded
/// with random records to a multiple of the 64-record workgroup.
fn salu_records() -> Vec<[u32; 3]> {
    let mut r: Vec<[u32; 3]> = Vec::new();
    for &a in &FLOAT_EDGES {
        for &b in &FLOAT_EDGES {
            r.push([a, b, a]);
        }
    }
    for i in int_edges() {
        r.push([ONE, ONE, i]);
    }
    let mut rng = Lcg(0x5eed_1201);
    for _ in 0..65536 {
        let (a, b, i) = (rng.next(), rng.next(), rng.next());
        r.push([a, b, i]);
    }
    for _ in 0..8192 {
        let a = rng.next();
        let d = (rng.next() % 17) as i32 - 8;
        r.push([a, a.wrapping_add(d as u32), rng.next()]);
    }
    for _ in 0..1024 {
        let a = rng.next();
        r.push([a, a, rng.next()]);
        r.push([a, a ^ 0x8000_0000, rng.next()]);
    }
    while r.len() % 64 != 0 {
        r.push([rng.next(), rng.next(), rng.next()]);
    }
    r
}

fn salu_out_words(n: usize) -> usize {
    5 * n + n / 16
}

fn salu_probe(recs: &[[u32; 3]]) -> Probe {
    let n = recs.len();
    assert!(n % 64 == 0 && n > 0);
    Probe {
        name: "probe_salu_f32",
        kind: Kind::Compare,
        stride: 0,
        input: recs.iter().flatten().copied().collect(),
        out_init: vec![SENTINEL; salu_out_words(n)],
        n: n as u32,
        groups: (n / 64) as u32,
        width: 0,
        fields: Vec::new(),
    }
}

/// Operand edge list of the VALU probes: both zeros, denormals (min, mid, max), min normal, +-1, 1.5, 2, 0.5, finite
/// extremes, 2^126, 2^23/2^24/2^31/2^32 and their neighbours, +-inf, quiet NaNs of both signs with payloads (including
/// the all-ones word) and signalling NaNs of both signs with payloads (including the largest and smallest sNaN).
const VALU_EDGES: [u32; 52] = [
    0x0000_0000,
    0x8000_0000, // +-0
    0x0000_0001,
    0x8000_0001, // +-min denormal
    0x0040_0000,
    0x8040_0000, // +-mid denormal
    0x007f_ffff,
    0x807f_ffff, // +-max denormal
    0x0080_0000,
    0x8080_0000, // +-min normal
    0x0080_0001,
    0x8080_0001,
    0x3f80_0000,
    0xbf80_0000, // +-1
    0x3fc0_0000,
    0xbfc0_0000, // +-1.5
    0x4000_0000,
    0xc000_0000, // +-2
    0x3f00_0000,
    0xbf00_0000, // +-0.5
    0x3f7f_ffff,
    0xbf7f_ffff, // 1 - ulp
    0x7f7f_ffff,
    0xff7f_ffff, // +-max finite
    0x7e80_0000,
    0xfe80_0000, // +-2^126
    0x4b00_0000,
    0x4b80_0000, // 2^23, 2^24
    0x4f00_0000,
    0xcf00_0000, // 2^31, -2^31
    0x4f80_0000,
    0x4eff_ffff, // 2^32, just below 2^31
    0x7f80_0000,
    0xff80_0000, // +-inf
    0x7fc0_0000,
    0xffc0_0000, // default qNaN, -qNaN
    0x7fc0_0001,
    0xffc0_0001,
    0x7fc1_2345,
    0xffc1_2345,
    0x7fff_ffff,
    0xffff_ffff, // all-ones payload
    0x7f80_0001,
    0xff80_0001, // smallest sNaN
    0x7fa0_0000,
    0xffa0_0000,
    0x7fbf_ffff,
    0xffbf_ffff, // largest sNaN
    0x7f81_2345,
    0xff81_2345,
    0x7fc0_beef,
    0xffa0_cafe,
];

/// 27-value subset of [`VALU_EDGES`] for the three-operand probes (27^3 = 19683 triples).
const VALU_REDUCED: [u32; 27] = [
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x8000_0001,
    0x007f_ffff,
    0x0080_0000,
    0x3f80_0000,
    0xbf80_0000,
    0x4000_0000,
    0xc000_0000,
    0x3f00_0000,
    0x7f7f_ffff,
    0xff7f_ffff,
    0x7e80_0000,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0000,
    0xffc0_0000,
    0x7fc1_2345,
    0xffff_ffff,
    0x7f80_0001,
    0xff80_0001,
    0x7fa0_0000,
    0xffbf_ffff,
    0xff81_2345,
    0x0040_0000,
    0x4b80_0000,
];

/// Pair set of the two-operand VALU probes: every pair of [`VALU_EDGES`] (the third word is random), 8192 random
/// records, 2048 neighbour pairs (`b = a +- 8 ulps` in bit space), 2048 equal and sign-flipped pairs, 2048 records
/// whose `b` is a small signed integer (the `v_ldexp_f32` exponent), padded to a multiple of 64.
fn valu_pairs() -> Vec<[u32; 3]> {
    let mut rng = Lcg(0x7a1b_0001);
    let mut r: Vec<[u32; 3]> = Vec::new();
    for &a in &VALU_EDGES {
        for &b in &VALU_EDGES {
            r.push([a, b, rng.next()]);
        }
    }
    for _ in 0..8192 {
        r.push([rng.next(), rng.next(), rng.next()]);
    }
    for _ in 0..2048 {
        let a = rng.next();
        let d = (rng.next() % 17) as i32 - 8;
        r.push([a, a.wrapping_add(d as u32), rng.next()]);
    }
    for _ in 0..1024 {
        let a = rng.next();
        r.push([a, a, rng.next()]);
        r.push([a, a ^ 0x8000_0000, rng.next()]);
    }
    for _ in 0..2048 {
        let a = if rng.next() % 4 == 0 {
            VALU_EDGES[(rng.next() as usize) % VALU_EDGES.len()]
        } else {
            rng.next()
        };
        r.push([a, ((rng.next() % 701) as i32 - 350) as u32, rng.next()]);
    }
    while r.len() % 64 != 0 {
        r.push([rng.next(), rng.next(), rng.next()]);
    }
    r
}

/// Triple set of the three-operand VALU probes: every triple of [`VALU_REDUCED`], 4096 random triples, padded.
fn valu_triples() -> Vec<[u32; 3]> {
    let mut rng = Lcg(0x7a1b_0003);
    let mut r: Vec<[u32; 3]> = Vec::new();
    for &a in &VALU_REDUCED {
        for &b in &VALU_REDUCED {
            for &c in &VALU_REDUCED {
                r.push([a, b, c]);
            }
        }
    }
    for _ in 0..4096 {
        r.push([rng.next(), rng.next(), rng.next()]);
    }
    while r.len() % 64 != 0 {
        r.push([rng.next(), rng.next(), rng.next()]);
    }
    r
}

/// Single-operand set (`a`; `b`, `c` random): the edge list, the integer edges (the `a` of the integer-source
/// conversions), every exponent field of both signs with seven mantissas, 4096 denormals, 24576 random words.
fn valu_singles() -> Vec<[u32; 3]> {
    let mut rng = Lcg(0x7a1b_0002);
    let mut a_list: Vec<u32> = VALU_EDGES.to_vec();
    a_list.extend(int_edges());
    for sign in [0u32, 0x8000_0000] {
        for e in 0..256u32 {
            for m in [0, 1, 0x7f_ffff, 0x40_0000, 0x15_5555] {
                a_list.push(sign | e << 23 | m);
            }
            for _ in 0..2 {
                a_list.push(sign | e << 23 | (rng.next() & 0x7f_ffff));
            }
        }
    }
    for _ in 0..4096 {
        a_list.push(rng.next() & 0x807f_ffff);
    }
    for _ in 0..24576 {
        a_list.push(rng.next());
    }
    let mut r: Vec<[u32; 3]> = a_list
        .into_iter()
        .map(|a| [a, rng.next(), rng.next()])
        .collect();
    while r.len() % 64 != 0 {
        r.push([rng.next(), rng.next(), rng.next()]);
    }
    r
}

/// Literal list of the fmamk/fmaak kernels (`KLIST` in `helper_probe.hip`).
const KLIST: [u32; 8] = [
    0x32a5_705f,
    0x3f80_0000,
    0x0000_0000,
    0x8000_0000,
    0x7f80_0000,
    0x7fc0_0000,
    0x7f80_0001,
    0x0000_0001,
];
/// Predicate order of `PREDS` in `helper_probe.hip` (the `v_cmp_*_f32` predicates of the committed objects) with whether a
/// VOPC e32 form exists (`v_cmp_ge_f32` has only the e64 form: its e32 bit stays 0 and is not compared).
const PREDS: [(&str, bool); 9] = [
    ("eq", true),
    ("ge", false),
    ("gt", true),
    ("le", true),
    ("lt", true),
    ("o", true),
    ("ngt", true),
    ("neq", true),
    ("nlt", true),
];
/// Predicate order of `PREDSX` in `helper_probe.hip` (the `v_cmpx_*_f32_e32` predicates of the committed objects).
const PREDSX: [&str; 5] = ["ge", "gt", "lt", "nge", "nlt"];

/// One whole word per field name.
fn word_fields(names: Vec<String>) -> Vec<Field> {
    names
        .into_iter()
        .enumerate()
        .map(|(word, name)| Field {
            name,
            word,
            shift: 0,
            mask: u32::MAX,
        })
        .collect()
}

/// One bit per field name, 32 bits per word in slot order; a `None` slot is not probed (its bit is skipped).
fn bit_fields(names: Vec<Option<String>>) -> Vec<Field> {
    names
        .into_iter()
        .enumerate()
        .filter_map(|(i, name)| {
            name.map(|name| Field {
                name,
                word: i / 32,
                shift: (i % 32) as u32,
                mask: 1,
            })
        })
        .collect()
}

fn valu_probe(name: &'static str, recs: &[[u32; 3]], width: usize, fields: Vec<Field>) -> Probe {
    let n = recs.len();
    assert!(n % 64 == 0 && n > 0);
    assert!(
        fields.iter().all(|f| f.word < width),
        "{name}: field outside the record"
    );
    Probe {
        name,
        kind: Kind::Valu,
        stride: 0,
        input: recs.iter().flatten().copied().collect(),
        out_init: vec![SENTINEL; width * n],
        n: n as u32,
        groups: (n / 64) as u32,
        width,
        fields,
    }
}

fn names(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

/// The `probe_v_*` kernels of `helper_probe.hip`: per kernel the record set, the words per record and the decoded
/// fields (the order in the kernel's comment).
fn valu_probes() -> Vec<Probe> {
    let (pairs, triples, singles) = (valu_pairs(), valu_triples(), valu_singles());
    let mut v = Vec::new();
    v.push(valu_probe(
        "probe_v_sub",
        &pairs,
        2,
        word_fields(names(&["v_sub_f32_e32 (a,b)", "v_subrev_f32_e32 (a,b)"])),
    ));
    v.push(valu_probe(
        "probe_v_dual_sub",
        &pairs,
        2,
        word_fields(names(&["v_dual_sub_f32 X (a,b)", "v_dual_sub_f32 Y (b,a)"])),
    ));
    let k_fields = |op: &str| -> Vec<String> {
        KLIST
            .iter()
            .flat_map(|k| [format!("{op} X K={k:#010x}"), format!("{op} Y K={k:#010x}")])
            .collect()
    };
    v.push(valu_probe(
        "probe_v_dual_fmamk",
        &pairs,
        16,
        word_fields(k_fields("v_dual_fmamk_f32")),
    ));
    v.push(valu_probe(
        "probe_v_dual_fmaak",
        &pairs,
        16,
        word_fields(k_fields("v_dual_fmaak_f32")),
    ));
    v.push(valu_probe(
        "probe_v_fmamk",
        &pairs,
        8,
        word_fields(
            KLIST
                .iter()
                .map(|k| format!("v_fmamk_f32 K={k:#010x}"))
                .collect(),
        ),
    ));
    v.push(valu_probe(
        "probe_v_frexp",
        &singles,
        2,
        word_fields(names(&["v_frexp_exp_i32_f32_e32", "v_frexp_mant_f32_e32"])),
    ));
    v.push(valu_probe(
        "probe_v_cvt",
        &singles,
        4,
        word_fields(names(&[
            "v_cvt_i32_f32_e32",
            "v_cvt_f32_i32_e32",
            "v_cvt_f32_u32_e32",
            "v_rndne_f32_e32",
        ])),
    ));
    v.push(valu_probe(
        "probe_v_rcp",
        &singles,
        1,
        word_fields(names(&["v_rcp_f32_e32"])),
    ));
    v.push(valu_probe(
        "probe_v_exp",
        &singles,
        1,
        word_fields(names(&["v_exp_f32_e32"])),
    ));
    v.push(valu_probe(
        "probe_v_ldexp",
        &pairs,
        1,
        word_fields(names(&["v_ldexp_f32"])),
    ));
    v.push(valu_probe(
        "probe_v_cndmask",
        &pairs,
        8,
        word_fields(
            ["sel", "inv"]
                .iter()
                .flat_map(|s| {
                    ["|a|,b", "-a,b", "a,|b|", "a,-b"]
                        .iter()
                        .map(move |m| format!("v_cndmask_b32_e64 {m} mask={s}"))
                })
                .collect(),
        ),
    ));
    v.push(valu_probe(
        "probe_v_fma",
        &triples,
        4,
        word_fields(names(&[
            "v_fma_f32 (a,b,c)",
            "v_fma_f32 (-a,b,c)",
            "v_fma_f32 (a,b,-c)",
            "v_fma_f32 (-a,b,-c)",
        ])),
    ));
    v.push(valu_probe(
        "probe_v_div",
        &triples,
        5,
        word_fields(names(&[
            "v_div_fmas_f32 VCC=0",
            "v_div_fmas_f32 VCC=1",
            "v_div_fixup_f32",
            "v_div_scale_f32 D",
            "v_div_scale_f32 VCC",
        ])),
    ));
    const FORMS: [&str; 7] = [
        "e32",
        "e64",
        "e64 |a|",
        "e64 |b|",
        "e64 -a",
        "e64 -b",
        "e64 |a| |b|",
    ];
    v.push(valu_probe(
        "probe_v_cmp",
        &pairs,
        2,
        bit_fields(
            PREDS
                .iter()
                .flat_map(|&(p, e32)| {
                    FORMS
                        .iter()
                        .enumerate()
                        .map(move |(k, f)| (k > 0 || e32).then(|| format!("v_cmp_{p}_f32 {f}")))
                })
                .collect(),
        ),
    ));
    v.push(valu_probe(
        "probe_v_cmpx",
        &pairs,
        1,
        bit_fields(
            PREDSX
                .iter()
                .map(|p| Some(format!("v_cmpx_{p}_f32_e32")))
                .collect(),
        ),
    ));
    v.push(valu_probe(
        "probe_v_class",
        &pairs,
        1,
        bit_fields(
            ["mask=b", "mask=c&0x3ff", "mask=1<<(c%10)"]
                .iter()
                .flat_map(|m| {
                    ["e64", "e64 |a|", "e64 -a"]
                        .iter()
                        .map(move |f| Some(format!("v_cmp_class_f32 {f} {m}")))
                })
                .collect(),
        ),
    ));
    v
}

fn probes() -> Vec<Probe> {
    let mut rng = Lcg(0x1234_5678);
    let mut words = |n: usize| (0..n).map(|_| rng.next()).collect::<Vec<u32>>();
    let one_group = |name, kind, stride, input: Vec<u32>, out_init: Vec<u32>| Probe {
        name,
        kind,
        stride,
        input,
        out_init,
        n: 64,
        groups: 1,
        width: 0,
        fields: Vec::new(),
    };
    // probe_call: wave 0 has mixed parity (divergent call site), wave 1 has lanes 32..37 odd and the rest even.
    let mut call_in = words(64);
    for (t, x) in call_in.iter_mut().enumerate().skip(32) {
        *x = (*x & !1) | u32::from(t < 37);
    }
    let atomic_in = words(64);
    let mut atomic_out = vec![SENTINEL; 64];
    atomic_out.extend(words(64)); // data dwords
    atomic_out.push(0x1000); // counter
    let collide_in = words(64);
    let mut collide_out = vec![SENTINEL; 64];
    collide_out.push(7);
    let lds_in = words(64);
    let scratch_in = words(64);
    let mut v = vec![
        one_group(
            "probe_call",
            Kind::Compare,
            24,
            call_in,
            vec![SENTINEL; 64 * 24],
        ),
        one_group(
            "probe_scratch",
            Kind::Compare,
            2,
            scratch_in,
            vec![SENTINEL; 64 * 2],
        ),
        one_group("probe_atomic", Kind::Compare, 0, atomic_in, atomic_out),
        one_group(
            "probe_atomic_collide",
            Kind::Collide,
            0,
            collide_in,
            collide_out,
        ),
        one_group(
            "probe_atomic_lds",
            Kind::Lds,
            0,
            lds_in,
            vec![SENTINEL; 128],
        ),
    ];
    v.push(salu_probe(&salu_records()));
    v.extend(valu_probes());
    v
}

// ---------------------------------------------------------------------------------------------------------------
// Objects and runners
// ---------------------------------------------------------------------------------------------------------------

struct Object {
    label: &'static str,
    bytes: &'static [u8],
    program: Program,
    helpers: Vec<Helper>,
    as_gfx1201: bool,
}

impl Object {
    fn load(label: &'static str, bytes: &'static [u8], as_gfx1201: bool) -> Self {
        let program = peacemaker_lift::lift_object(
            bytes,
            peacemaker_lift::Options {
                frontend: Frontend::Hipcc,
            },
        )
        .unwrap_or_else(|e| panic!("lift {label}: {e}"))
        .program;
        let helpers =
            load_helpers(&program).unwrap_or_else(|e| panic!("load_helpers {label}: {e}"));
        Self {
            label,
            bytes,
            program,
            helpers,
            as_gfx1201,
        }
    }

    fn emulator(&self, symbol: &str) -> Result<Emulator<'_>, String> {
        let e = Emulator::with_helpers(&self.program, symbol, &self.helpers)?;
        Ok(if self.as_gfx1201 {
            e.gfx1200_as_gfx1201()
        } else {
            e
        })
    }

    /// The kernarg image of the probe signature `(const u32* in, u32* out, u32 n)`: the three explicit arguments at
    /// the offsets of the lifted kernel's metadata, hidden arguments zero (no probe reads them).
    fn kernarg(&self, symbol: &str, in_ptr: u64, out_ptr: u64, n: u32) -> Vec<u8> {
        let kernel = self
            .program
            .kernels
            .iter()
            .find(|k| k.symbol.0 == symbol)
            .unwrap_or_else(|| panic!("{}: kernel {symbol} absent", self.label));
        let Abi::Hsa { metadata, .. } = &kernel.abi else {
            panic!("{}: {symbol} has no HSA metadata", self.label);
        };
        let m = &metadata.parsed;
        let explicit: Vec<_> = m
            .args
            .iter()
            .filter(|a| a.value_kind == "global_buffer" || a.value_kind == "by_value")
            .collect();
        assert_eq!(
            explicit.len(),
            3,
            "{}: {symbol} must take (in, out, n), metadata args: {:?}",
            self.label,
            m.args
        );
        let mut v = vec![0u8; (m.kernarg_segment_size as usize).max(256)];
        for ((a, value), size) in explicit
            .iter()
            .zip([in_ptr, out_ptr, u64::from(n)])
            .zip([8, 8, 4])
        {
            assert_eq!(a.size, size, "{}: {symbol} arg {} size", self.label, a.name);
            let off = a.offset as usize;
            v[off..off + size as usize].copy_from_slice(&value.to_le_bytes()[..size as usize]);
        }
        v
    }
}

struct EmuRun {
    out: Vec<u32>,
    /// Per workgroup: `Ok` or the emulator's refusal text.
    wg: Vec<Result<(), String>>,
}

fn run_emulated(obj: &Object, emu: &Emulator, p: &Probe) -> EmuRun {
    let mut mem = Memory::default();
    mem.map("karg".into(), KARG, obj.kernarg(p.name, IN, OUT, p.n))
        .unwrap();
    mem.map("in".into(), IN, to_bytes(&p.input)).unwrap();
    mem.map("out".into(), OUT, to_bytes(&p.out_init)).unwrap();
    let wg = (0..p.groups)
        .map(|g| {
            emu.run_workgroup(
                &Launch {
                    kernarg_ptr: KARG,
                    workgroup: [g, 0, 0],
                    block: [BLOCK, 1, 1],
                    dynamic_lds: 0,
                },
                &mut mem,
            )
            .map_err(|e| format!("{e:#}"))
        })
        .collect();
    let out = to_words(&mem.regions.iter().find(|r| r.name == "out").unwrap().bytes);
    EmuRun { out, wg }
}

fn run_hardware(rt: &HipRuntime, obj: &Object, p: &Probe) -> Result<Vec<u32>, String> {
    let module = rt
        .module_load_data(obj.bytes)
        .map_err(|e| format!("hipModuleLoadData: {e}"))?;
    let f = rt
        .module_get_function(&module, p.name)
        .map_err(|e| format!("hipModuleGetFunction({}): {e}", p.name))?;
    std::mem::forget(module);
    let upload = |w: &[u32]| -> Result<_, String> {
        let b = to_bytes(w);
        let d = rt
            .malloc(b.len().max(16))
            .map_err(|e| format!("hipMalloc: {e}"))?;
        rt.memcpy_htod(&d, &b).map_err(|e| format!("upload: {e}"))?;
        Ok(d)
    };
    let din = upload(&p.input)?;
    let dout = upload(&p.out_init)?;
    let mut blob = obj.kernarg(p.name, din.as_ptr() as u64, dout.as_ptr() as u64, p.n);
    // SAFETY: `blob` is the kernel's kernarg image (explicit arguments at the lifted metadata offsets); both pointers
    // are live allocations sized for the probe.
    unsafe { rt.launch_kernel_blob(&f, [p.groups, 1, 1], [BLOCK, 1, 1], 0, None, &mut blob) }
        .map_err(|e| format!("launch {}: {e}", p.name))?;
    rt.device_synchronize()
        .map_err(|e| format!("synchronize {}: {e}", p.name))?;
    let mut b = vec![0u8; p.out_init.len() * 4];
    rt.memcpy_dtoh(&mut b, &dout)
        .map_err(|e| format!("readback {}: {e}", p.name))?;
    rt.free(din).map_err(|e| format!("free: {e}"))?;
    rt.free(dout).map_err(|e| format!("free: {e}"))?;
    Ok(to_words(&b))
}

// ---------------------------------------------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------------------------------------------

fn describe(p: &Probe, i: usize) -> String {
    if p.stride > 0 {
        format!("word {i} (lane {}, field {})", i / p.stride, i % p.stride)
    } else {
        format!("word {i}")
    }
}

/// Compare a generic probe's outputs word for word; prints the receipt line.
fn evaluate_words(obj: &Object, p: &Probe, hw: &[u32], failures: &mut Vec<String>) {
    let tag = format!("{} / {}", p.name, obj.label);
    let emu = match obj.emulator(p.name) {
        Ok(e) => e,
        Err(e) => {
            failures.push(format!("{tag}: emulator construction refused: {e}"));
            println!(
                "RECEIPT probe={} target=\"{}\" REFUSED at construction: {e}",
                p.name, obj.label
            );
            return;
        }
    };
    let run = run_emulated(obj, &emu, p);
    if let Some(Err(e)) = run.wg.iter().find(|r| r.is_err()) {
        failures.push(format!("{tag}: emulator refused: {e}"));
        println!(
            "RECEIPT probe={} target=\"{}\" REFUSED at run time: {e}",
            p.name, obj.label
        );
        return;
    }
    assert_eq!(hw.len(), run.out.len(), "{tag}: output size");
    let bad: Vec<usize> = (0..hw.len()).filter(|&i| hw[i] != run.out[i]).collect();
    println!(
        "RECEIPT probe={} target=\"{}\" words={} mismatches={}",
        p.name,
        obj.label,
        hw.len(),
        bad.len()
    );
    for &i in bad.iter().take(8) {
        println!(
            "  {}: hardware {:#010x} emulator {:#010x}",
            describe(p, i),
            hw[i],
            run.out[i]
        );
    }
    if !bad.is_empty() {
        failures.push(format!("{tag}: {} of {} words differ", bad.len(), hw.len()));
    }
}

const SALU_OPS: [&str; 7] = [
    "s_sub_f32",
    "s_cmp_gt_f32",
    "s_cmp_neq_f32",
    "s_cmp_ngt_f32",
    "s_cvt_f32_i32",
    "v_cmpx_gt_f32_e32",
    "v_cmpx_nge_f32_e32",
];

/// The seven results of record `i` as the kernel stored them (workgroup layout of `n` records).
fn salu_result(out: &[u32], n: usize, i: usize) -> [u32; 7] {
    let (c, l) = (i / 32, i % 32);
    [
        out[5 * i],
        out[5 * i + 1],
        out[5 * i + 2],
        out[5 * i + 3],
        out[5 * i + 4],
        (out[5 * n + 2 * c] >> l) & 1,
        (out[5 * n + 2 * c + 1] >> l) & 1,
    ]
}

/// Per record: the emulated seven results, or the refusal of that record. A refused workgroup is re-run per record
/// (the record replicated over a whole 64-record workgroup, so the refusal is that record's own).
fn salu_emulated(obj: &Object, emu: &Emulator, p: &Probe) -> Vec<Result<[u32; 7], String>> {
    let n = p.n as usize;
    let main = run_emulated(obj, emu, p);
    let mut res: Vec<Result<[u32; 7], String>> = Vec::with_capacity(n);
    for (g, r) in main.wg.iter().enumerate() {
        for i in g * 64..(g + 1) * 64 {
            match r {
                Ok(()) => res.push(Ok(salu_result(&main.out, n, i))),
                Err(_) => {
                    let rec = [p.input[3 * i], p.input[3 * i + 1], p.input[3 * i + 2]];
                    let single = salu_probe(&[rec; 64]);
                    let one = run_emulated(obj, emu, &single);
                    res.push(match &one.wg[0] {
                        Ok(()) => Ok(salu_result(&one.out, 64, 0)),
                        Err(e) => Err(e.clone()),
                    });
                }
            }
        }
    }
    res
}

fn is_nan(x: u32) -> bool {
    x & 0x7fff_ffff > 0x7f80_0000
}

fn evaluate_salu(obj: &Object, p: &Probe, hw: &[u32], failures: &mut Vec<String>) {
    let tag = format!("{} / {}", p.name, obj.label);
    let emu = match obj.emulator(p.name) {
        Ok(e) => e,
        Err(e) => {
            failures.push(format!("{tag}: emulator construction refused: {e}"));
            println!(
                "RECEIPT probe={} target=\"{}\" REFUSED at construction: {e}",
                p.name, obj.label
            );
            return;
        }
    };
    let n = p.n as usize;
    assert_eq!(hw.len(), salu_out_words(n), "{tag}: output size");
    let emulated = salu_emulated(obj, &emu, p);
    let mut compared = [0usize; 7];
    let mut mismatches: Vec<Vec<(usize, u32, u32)>> = vec![Vec::new(); 7];
    let mut refused: Vec<(usize, &String)> = Vec::new();
    for (i, e) in emulated.iter().enumerate() {
        match e {
            Err(why) => refused.push((i, why)),
            Ok(e) => {
                let h = salu_result(hw, n, i);
                for op in 0..7 {
                    compared[op] += 1;
                    if h[op] != e[op] {
                        mismatches[op].push((i, h[op], e[op]));
                    }
                }
            }
        }
    }
    for op in 0..7 {
        println!(
            "RECEIPT probe={} target=\"{}\" op={} records={} compared={} mismatches={}",
            p.name,
            obj.label,
            SALU_OPS[op],
            n,
            compared[op],
            mismatches[op].len()
        );
        if !mismatches[op].is_empty() {
            println!(
                "  mismatch table {}: operand bits | hardware | emulator",
                SALU_OPS[op]
            );
            for &(i, h, e) in mismatches[op].iter().take(40) {
                let r = &p.input[3 * i..3 * i + 3];
                if op == 4 {
                    println!("    i={:#010x} | {h:#010x} | {e:#010x}", r[2]);
                } else {
                    println!(
                        "    a={:#010x} b={:#010x} | {h:#010x} | {e:#010x}",
                        r[0], r[1]
                    );
                }
            }
            failures.push(format!(
                "{tag}: {} mismatching records for {}",
                mismatches[op].len(),
                SALU_OPS[op]
            ));
        }
    }
    if !refused.is_empty() {
        let (nan, other): (Vec<&(usize, &String)>, Vec<&(usize, &String)>) = refused
            .iter()
            .partition(|(i, _)| is_nan(p.input[3 * i]) || is_nan(p.input[3 * i + 1]));
        let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
        for (_, why) in &refused {
            let key: String = why.chars().take(160).collect();
            *reasons
                .entry(key.split(" | arch=").next().unwrap().to_string())
                .or_default() += 1;
        }
        println!(
            "RECEIPT probe={} target=\"{}\" UNMEASURED refused_records={} (NaN operand {}, no NaN operand {})",
            p.name,
            obj.label,
            refused.len(),
            nan.len(),
            other.len()
        );
        for (why, count) in &reasons {
            println!("  refused x{count}: {why}");
        }
        if !other.is_empty() {
            failures.push(format!(
                "{tag}: {} records without a NaN operand were refused by the emulator (first: {})",
                other.len(),
                other[0].1
            ));
        }
    }
}

/// Per-record re-runs the refusal attribution of one probe may spend before the remaining refused records are only
/// counted (the refusing workgroup is known, the exact record is not).
const RERUN_CAP: usize = 8192;

/// Per record: the `width` emulated words, or the refusal text of that record. A refused workgroup is re-run per
/// record (the record replicated over a whole 64-record workgroup, so the refusal is that record's own); identical
/// records share one re-run.
fn valu_emulated(obj: &Object, emu: &Emulator, p: &Probe) -> Vec<Result<Vec<u32>, String>> {
    let n = p.n as usize;
    let main = run_emulated(obj, emu, p);
    let mut memo: BTreeMap<[u32; 3], Result<Vec<u32>, String>> = BTreeMap::new();
    let mut res: Vec<Result<Vec<u32>, String>> = Vec::with_capacity(n);
    for (g, r) in main.wg.iter().enumerate() {
        for i in g * 64..(g + 1) * 64 {
            match r {
                Ok(()) => res.push(Ok(main.out[p.width * i..p.width * (i + 1)].to_vec())),
                Err(_) => {
                    let rec = [p.input[3 * i], p.input[3 * i + 1], p.input[3 * i + 2]];
                    if let Some(known) = memo.get(&rec) {
                        res.push(known.clone());
                    } else if memo.len() >= RERUN_CAP {
                        res.push(Err(
                            "workgroup refused; record not re-run (attribution cap)".into(),
                        ));
                    } else {
                        let single = Probe {
                            name: p.name,
                            kind: p.kind,
                            stride: 0,
                            input: rec.iter().copied().cycle().take(3 * 64).collect(),
                            out_init: vec![SENTINEL; p.width * 64],
                            n: 64,
                            groups: 1,
                            width: p.width,
                            fields: Vec::new(),
                        };
                        let one = run_emulated(obj, emu, &single);
                        let out = match &one.wg[0] {
                            Ok(()) => Ok(one.out[..p.width].to_vec()),
                            Err(e) => Err(e.clone()),
                        };
                        memo.insert(rec, out.clone());
                        res.push(out);
                    }
                }
            }
        }
    }
    res
}

/// The refused records of an emulated probe and, per refusal reason (text before the machine-state dump), how many
/// records it refused and the first such record.
fn refusal_table(
    emulated: &[Result<Vec<u32>, String>],
) -> (Vec<usize>, BTreeMap<String, (usize, usize)>) {
    let refused: Vec<usize> = (0..emulated.len())
        .filter(|&i| emulated[i].is_err())
        .collect();
    let mut reasons: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for &i in &refused {
        let why = emulated[i].as_ref().unwrap_err();
        let key: String = why
            .split(" | arch=")
            .next()
            .unwrap()
            .chars()
            .take(220)
            .collect();
        reasons.entry(key).or_insert((0, i)).0 += 1;
    }
    (refused, reasons)
}

/// Compares a `Kind::Valu` probe field by field. Every mismatch and every refused (unmeasured) record is printed with
/// its operand words; either one fails the probe.
fn evaluate_valu(obj: &Object, p: &Probe, hw: &[u32], failures: &mut Vec<String>) {
    let tag = format!("{} / {}", p.name, obj.label);
    let emu = match obj.emulator(p.name) {
        Ok(e) => e,
        Err(e) => {
            failures.push(format!("{tag}: emulator construction refused: {e}"));
            println!(
                "RECEIPT probe={} target=\"{}\" REFUSED at construction: {e}",
                p.name, obj.label
            );
            return;
        }
    };
    let n = p.n as usize;
    assert_eq!(hw.len(), p.width * n, "{tag}: output size");
    let emulated = valu_emulated(obj, &emu, p);
    let operands = |i: usize| {
        format!(
            "a={:#010x} b={:#010x} c={:#010x}",
            p.input[3 * i],
            p.input[3 * i + 1],
            p.input[3 * i + 2]
        )
    };
    let (refused_records, reasons) = refusal_table(&emulated);
    let mut bad_fields = 0usize;
    for f in &p.fields {
        let mut compared = 0usize;
        let mut mismatches: Vec<(usize, u32, u32)> = Vec::new();
        for i in 0..n {
            if let Ok(words) = &emulated[i] {
                compared += 1;
                let h = (hw[p.width * i + f.word] >> f.shift) & f.mask;
                let e = (words[f.word] >> f.shift) & f.mask;
                if h != e {
                    mismatches.push((i, h, e));
                }
            }
        }
        println!(
            "RECEIPT probe={} target=\"{}\" field=\"{}\" records={n} compared={compared} mismatches={} refused={}",
            p.name,
            obj.label,
            f.name,
            mismatches.len(),
            n - compared
        );
        if !mismatches.is_empty() {
            bad_fields += 1;
            println!(
                "  mismatch table {}: operand bits | hardware | emulator",
                f.name
            );
            for &(i, h, e) in mismatches.iter().take(24) {
                println!("    {} | {h:#010x} | {e:#010x}", operands(i));
            }
        }
    }
    if !refused_records.is_empty() {
        println!(
            "RECEIPT probe={} target=\"{}\" UNMEASURED refused_records={} of {n}",
            p.name,
            obj.label,
            refused_records.len()
        );
        for (why, (count, first)) in &reasons {
            println!("  refused x{count}: {why}  [first: {}]", operands(*first));
        }
        failures.push(format!(
            "{tag}: {} of {n} records were refused by the emulator (first: {})",
            refused_records.len(),
            emulated[refused_records[0]]
                .as_ref()
                .unwrap_err()
                .split(" | arch=")
                .next()
                .unwrap()
        ));
    }
    if bad_fields > 0 {
        failures.push(format!(
            "{tag}: {bad_fields} field(s) differ from the hardware"
        ));
    }
}

fn evaluate(obj: &Object, p: &Probe, hw: &[u32], failures: &mut Vec<String>) {
    match (p.kind, p.name) {
        (Kind::Valu, _) => evaluate_valu(obj, p, hw, failures),
        (_, "probe_salu_f32") => evaluate_salu(obj, p, hw, failures),
        _ => evaluate_words(obj, p, hw, failures),
    }
}

/// Negative controls: the emulator must refuse with the named reason (the hardware runs both probes).
fn check_negative(obj: &Object, p: &Probe, failures: &mut Vec<String>) {
    let want = match p.kind {
        Kind::Collide => "two active lanes meet in one dword",
        Kind::Lds => "LDS-aperture",
        Kind::Compare | Kind::Valu => unreachable!(),
    };
    let tag = format!("{} / {}", p.name, obj.label);
    let emu = match obj.emulator(p.name) {
        Ok(e) => e,
        Err(e) => {
            failures.push(format!("{tag}: emulator construction refused: {e}"));
            return;
        }
    };
    let run = run_emulated(obj, &emu, p);
    match &run.wg[0] {
        Err(e) if e.contains(want) => println!(
            "RECEIPT probe={} target=\"{}\" negative control REFUSED as required: {}",
            p.name,
            obj.label,
            e.chars().take(200).collect::<String>()
        ),
        Err(e) => failures.push(format!(
            "{tag}: refused for the wrong reason (want \"{want}\"): {e}"
        )),
        Ok(()) => failures.push(format!(
            "{tag}: the emulator executed a negative control it must refuse"
        )),
    }
}

/// What the hardware outputs of the negative controls must satisfy (order-independent facts only).
fn check_hardware_negative(p: &Probe, hw: &[u32], failures: &mut Vec<String>) {
    match p.kind {
        Kind::Collide => {
            let expect =
                p.out_init[64].wrapping_add(p.input.iter().fold(0u32, |a, &x| a.wrapping_add(x)));
            if hw[64] != expect {
                failures.push(format!(
                    "{}: hardware final dword {:#x} != {expect:#x}",
                    p.name, hw[64]
                ));
            }
        }
        Kind::Lds => {
            for t in 0..64 {
                let (old, new) = (p.input[t], p.input[t].wrapping_add(5 + t as u32));
                if hw[t] != old || hw[64 + t] != new {
                    failures.push(format!(
                        "{}: hardware lane {t}: old {:#x} new {:#x}, expected {old:#x} {new:#x}",
                        p.name,
                        hw[t],
                        hw[64 + t]
                    ));
                    break;
                }
            }
        }
        Kind::Compare | Kind::Valu => {}
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a gfx1201 GPU: run through the GPU fabric"]
fn helper_probes_hardware_vs_emulator() {
    let rt = HipRuntime::load().expect("HipRuntime::load");
    rt.set_device(0).expect("hipSetDevice");
    let arch = rt.get_arch(0).expect("arch");
    assert!(arch.starts_with("gfx1201"), "device is {arch}");
    let objects = [
        Object::load("gfx1201 object", GFX1201, false),
        Object::load(
            "gfx1200 object under gfx1200_as_gfx1201 (shared-ISA reasoning, not a gfx1200 hardware measurement)",
            GFX1200,
            true,
        ),
    ];
    let mut failures = Vec::new();
    // Phase 1: the hardware outputs of every probe (and the receipt files), so that a slow or refusing emulator can
    // never lose a hardware measurement.
    let mut hardware: Vec<(Probe, Vec<u32>)> = Vec::new();
    for p in probes() {
        let hw = run_hardware(&rt, &objects[0], &p)
            .unwrap_or_else(|e| panic!("hardware {}: {e}", p.name));
        println!(
            "RECEIPT probe={} hardware ran: words={} sha256={}",
            p.name,
            hw.len(),
            sha256(&to_bytes(&hw))
        );
        hardware.push((p, hw));
    }
    if let Ok(dir) = std::env::var("EMU_PROBE_RECEIPT_OUT") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).expect("receipt dir");
        let mut per_probe = serde_json::Map::new();
        for (p, hw) in &hardware {
            let bytes = to_bytes(hw);
            std::fs::write(dir.join(format!("{}.bin", p.name)), &bytes)
                .expect("write probe output");
            per_probe.insert(
                p.name.to_string(),
                serde_json::json!({
                    "kind": format!("{:?}", p.kind),
                    "words": hw.len(),
                    "output_sha256": sha256(&bytes),
                    // the collide probe's returned (old) values depend on the hardware lane order
                    "deterministic": p.kind != Kind::Collide,
                }),
            );
        }
        let receipt = serde_json::json!({
            "gpu_arch": arch,
            // hip-bridge exposes no device-name accessor; the fabric job may name the card here
            "gpu_name": std::env::var("EMU_PROBE_GPU_NAME").ok(),
            "objects": {
                "helper_probe.gfx1201.hsaco": sha256(GFX1201),
                "helper_probe.gfx1200.hsaco": sha256(GFX1200),
            },
            "hardware_object": "helper_probe.gfx1201.hsaco",
            "probes": per_probe,
        });
        std::fs::write(
            dir.join("receipt.json"),
            serde_json::to_string_pretty(&receipt).unwrap() + "\n",
        )
        .expect("write receipt.json");
    }
    // Phase 2: hardware vs emulator.
    for (p, hw) in &hardware {
        match p.kind {
            Kind::Compare | Kind::Valu => {
                for obj in &objects {
                    evaluate(obj, p, hw, &mut failures);
                }
            }
            Kind::Collide | Kind::Lds => {
                check_hardware_negative(p, hw, &mut failures);
                for obj in &objects {
                    check_negative(obj, p, &mut failures);
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "probe failures:\n{}",
        failures.join("\n")
    );
}

#[test]
fn helper_probes_emulator_matches_committed_hardware_receipt() {
    let obj = Object::load("gfx1201 object", GFX1201, false);
    let mut failures = Vec::new();
    for p in probes() {
        match p.kind {
            Kind::Compare | Kind::Valu => {
                let path = format!("{RECEIPT_DIR}/{}.bin", p.name);
                let bytes = std::fs::read(&path).unwrap_or_else(|e| {
                    panic!("committed hardware receipt {path} is missing: {e}")
                });
                assert_eq!(
                    bytes.len(),
                    p.out_init.len() * 4,
                    "committed hardware receipt {path}: size"
                );
                evaluate(&obj, &p, &to_words(&bytes), &mut failures);
            }
            Kind::Collide | Kind::Lds => check_negative(&obj, &p, &mut failures),
        }
    }
    assert!(
        failures.is_empty(),
        "probe failures:\n{}",
        failures.join("\n")
    );
}

/// Writes every probe's input records (little-endian u32, three words per record for the VALU probes) to
/// `$EMU_PROBE_INPUTS_OUT`, for offline analysis of a hardware receipt next to its `<probe>.bin`.
#[test]
#[ignore = "analysis helper: set EMU_PROBE_INPUTS_OUT"]
fn helper_probe_inputs_dump() {
    let dir = std::path::PathBuf::from(
        std::env::var("EMU_PROBE_INPUTS_OUT").expect("EMU_PROBE_INPUTS_OUT"),
    );
    std::fs::create_dir_all(&dir).expect("inputs dir");
    for p in probes() {
        std::fs::write(dir.join(format!("{}.in.bin", p.name)), to_bytes(&p.input))
            .expect("write inputs");
    }
}

/// Development aid, no hardware: the emulator alone runs every `probe_v_*` kernel and prints which records it refuses
/// and why (the unmeasured rows of the emulator). `EMU_PROBE_OBJECT_DIR` names a directory holding
/// `helper_probe.gfx120{1,0}.hsaco` to use instead of the committed fixtures (a probe under development).
#[test]
#[ignore = "development aid: emulator-only refusal census of the VALU probes"]
fn helper_probe_emulator_census() {
    let load = |label: &'static str, file: &str, committed: &'static [u8], as_gfx1201: bool| {
        match std::env::var("EMU_PROBE_OBJECT_DIR") {
            Ok(dir) => {
                let bytes: &'static [u8] = Box::leak(
                    std::fs::read(format!("{dir}/{file}"))
                        .expect("object")
                        .into_boxed_slice(),
                );
                Object::load(label, bytes, as_gfx1201)
            }
            Err(_) => Object::load(label, committed, as_gfx1201),
        }
    };
    let objects = [
        load(
            "gfx1201 object",
            "helper_probe.gfx1201.hsaco",
            GFX1201,
            false,
        ),
        load(
            "gfx1200 object as gfx1201",
            "helper_probe.gfx1200.hsaco",
            GFX1200,
            true,
        ),
    ];
    for obj in &objects {
        for p in probes().into_iter().filter(|p| p.kind == Kind::Valu) {
            let emu = match obj.emulator(p.name) {
                Ok(e) => e,
                Err(e) => {
                    println!(
                        "CENSUS {} / {}: construction refused: {e}",
                        p.name, obj.label
                    );
                    continue;
                }
            };
            let emulated = valu_emulated(obj, &emu, &p);
            let (refused, reasons) = refusal_table(&emulated);
            println!(
                "CENSUS {} / {}: {} of {} records refused",
                p.name,
                obj.label,
                refused.len(),
                p.n
            );
            for (why, (count, first)) in &reasons {
                println!(
                    "  x{count}: {why}  [first: a={:#010x} b={:#010x} c={:#010x}]",
                    p.input[3 * first],
                    p.input[3 * first + 1],
                    p.input[3 * first + 2]
                );
            }
        }
    }
}
