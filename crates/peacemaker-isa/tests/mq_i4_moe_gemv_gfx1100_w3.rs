// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1100 W3 (`Mq3I4`, planar MQ3, qt131/qt132) K1 decode GEMV entries
//! `mq_i4_moe_gemv_mq3i4_{r256_gate_up,r128_down}[_t8]_pm_gfx1100` of the generator
//! `kernels/mq_i4_moe_gemv_gfx1100.rs` (module `mq_i4_moe_gemv_mq3_pm_gfx1100`).
//!
//! * Structure: symbols, the frozen 96-byte ABI, wave32 / no LDS / no barrier, the weight-load census,
//!   and that the existing MQ4-only gfx1100 object (the landed K1) is byte-identical to its committed text.
//! * Numerics: the interpreter refuses gfx1100 (`peacemaker-ir` `emu/alu.rs`, "unqualified numerical
//!   architecture"), so the emitted gfx1100 text is retargeted to gfx1151 (same RDNA3-family ISA; the only
//!   textual delta is the `amdgcn_target`/symbol suffix) and executed on the CPU interpreter against an
//!   independent model of the CPU planar W3 decoder law (`hipfire_cpu::quant::decode_planar_group`:
//!   `w = f32(d) * f32(2c - 7)`, codes the little-endian 3-bit stream, eight per three bytes) and the frozen
//!   F4 tree. Every output word is compared bit for bit (tolerance 0).
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;
use peacemaker_isa::kernels::mq_i4_moe_gemv_gfx1100 as k;
use peacemaker_isa::kernels::{iu4_gemm, mq_i4_moe_gemv};
use peacemaker_isa::{native, Arch, Emitted};

const KA: u64 = 0x1_0000;
const TABLE: u64 = 0x10_0000;
const IDS: u64 = 0x20_0000;
const OFFS: u64 = 0x21_0000;
const SLOTS: u64 = 0x22_0000;
// Tile descriptors straddle the low-dword boundary so descriptor additions must carry.
const DESC: u64 = 0x2_ffff_ff00;
const EXPERT: u64 = 0x1000_0000;
const EXPERT_STRIDE: u64 = 0x10_0000;
const XBASE: u64 = 0x8000_0000;
const Y0: u64 = 0x9000_0000;
const Y1: u64 = 0xa000_0000;
const N_EXPERTS: usize = 512;
const GUARD: usize = 64;
const POISON: u8 = 0xa5;
const TILE_WORDS: usize = 482;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn gate(spec: k::Spec) -> bool {
    spec.kind == k::Kind::GateUp
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15);
    let exp = u32::from((bits >> 10) & 0x1f);
    let man = u32::from(bits & 0x3ff);
    f32::from_bits(match exp {
        0 if man == 0 => sign << 31,
        0 => (man as f32 * f32::from_bits(0x3380_0000)).to_bits() | (sign << 31),
        0x1f => (sign << 31) | 0x7f80_0000 | (man << 13),
        _ => (sign << 31) | ((exp + 112) << 23) | (man << 13),
    })
}

/// Finite f16 scale bits: normal band, extremes, signed zero, subnormals, negatives.
fn scale_bits(rng: &mut Rng, i: usize) -> u16 {
    match i % 11 {
        0 => 0x0400,
        1 => 0x7bff,
        2 => 0x0000,
        3 => 0x8000,
        4 => 0x0001 + rng.below(0x3ff) as u16,
        5 => 0x8400 + rng.below(0x3bff) as u16,
        _ => {
            let e = 8 + rng.below(13) as u16;
            (e << 10) | (rng.next() as u16 & 0x3ff) | ((rng.next() as u16 & 1) << 15)
        }
    }
}

/// One expert tensor: `m` rows of `k` weights as planar W3: per row `k*3/8` code bytes (48 per 128-group, the
/// little-endian 3-bit stream), then the f16 scale plane `2*(row*k/128 + g)` at `m*k*3/8`. Returns the tensor
/// bytes and the decoded f32 weights, row-major (the CPU `decode_planar_group` law).
fn w3_tensor(rng: &mut Rng, m: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let (groups, row_bytes) = (k / 128, k * 3 / 8);
    let mut bytes = vec![0u8; m * row_bytes + 2 * m * groups];
    let mut weights = vec![0f32; m * k];
    for r in 0..m {
        for g in 0..groups {
            let base = r * row_bytes + g * 48;
            // Row 0 group 0 is all code 0, row 1 group 0 all code 7: the two extreme planes.
            let fixed = (g == 0 && r < 2).then_some(if r == 0 { 0 } else { 7 });
            let scale = scale_bits(rng, r * groups + g);
            let off = m * row_bytes + 2 * (r * groups + g);
            bytes[off..off + 2].copy_from_slice(&scale.to_le_bytes());
            let d = f16_to_f32(scale);
            for j in 0..128 {
                let code = fixed.unwrap_or_else(|| rng.below(8) as u32);
                for b in 0..3 {
                    if (code >> b) & 1 == 1 {
                        bytes[base + (3 * j + b) / 8] |= 1 << ((3 * j + b) % 8);
                    }
                }
                weights[r * k + g * 128 + j] = d * (2 * code as i32 - 7) as f32;
            }
        }
    }
    (bytes, weights)
}

/// Cross-check of the model against the CPU decoder's exact byte layout: three bytes decode eight codes.
#[test]
fn bitstream_layout_is_the_three_byte_eight_code_chunk() {
    let mut rng = Rng(0x5eed_0003);
    let (bytes, weights) = w3_tensor(&mut rng, 3, 256);
    let d =
        |scale_at: usize| f16_to_f32(u16::from_le_bytes([bytes[scale_at], bytes[scale_at + 1]]));
    let scale_plane = 3 * 256 * 3 / 8;
    for r in 0..3 {
        for g in 0..2 {
            let dd = d(scale_plane + 2 * (r * 2 + g));
            let codes = &bytes[r * 96 + g * 48..r * 96 + g * 48 + 48];
            for (c, chunk) in codes.chunks_exact(3).enumerate() {
                let acc =
                    u32::from(chunk[0]) | u32::from(chunk[1]) << 8 | u32::from(chunk[2]) << 16;
                for j in 0..8 {
                    let code = ((acc >> (3 * j)) & 7) as i32;
                    let want = dd * (2 * code - 7) as f32;
                    assert_eq!(
                        weights[r * 256 + g * 128 + c * 8 + j].to_bits(),
                        want.to_bits()
                    );
                }
            }
        }
    }
}

/// The frozen F4 tree of one (slot, row): lane `l` owns `g*128 + 4l ..+3` of every group, `p = RN(w0*x0)` then
/// `fma(w_j, x_j, p)`, chains `acc[g&3]` (gate/up) or one chain (down), fold `(a0+a1)+(a2+a3)`, then the
/// snapshot shuffle-down at 16, 8, 4, 2, 1 with out-of-range lanes unchanged.
fn twin_row(gate: bool, k: usize, x: &[f32], w: &[f32]) -> f32 {
    let mut lane = [0f32; 32];
    for (l, out) in lane.iter_mut().enumerate() {
        let mut acc = [0f32; 4];
        for g in 0..k / 128 {
            let base = g * 128 + 4 * l;
            let mut p = w[base] * x[base];
            for j in 1..4 {
                p = w[base + j].mul_add(x[base + j], p);
            }
            acc[if gate { g & 3 } else { 0 }] += p;
        }
        *out = if gate {
            (acc[0] + acc[1]) + (acc[2] + acc[3])
        } else {
            acc[0]
        };
    }
    for off in [16usize, 8, 4, 2, 1] {
        let snap = lane;
        for l in 0..32 {
            if l + off < 32 {
                lane[l] = snap[l] + snap[l + off];
            }
        }
    }
    lane[0]
}

fn lift(elf: &[u8]) -> peacemaker_ir::Program {
    peacemaker_lift::lift_object(
        elf,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    )
    .unwrap()
    .program
}

fn all_entries() -> Vec<(k::Spec, bool, Emitted)> {
    let mut v = Vec::new();
    for tiled in [false, true] {
        for spec in k::mq3_specs(Arch::Gfx1100) {
            let e = if tiled {
                k::emit_rowtiles(spec).unwrap()
            } else {
                k::emit(spec).unwrap()
            };
            v.push((spec, tiled, e));
        }
    }
    v
}

fn symbol(spec: k::Spec, tiled: bool) -> String {
    if tiled {
        format!("{}_t8", spec.symbol())
    } else {
        spec.symbol()
    }
}

/// The module text exactly as the landing emitter builds it (narrow gate, narrow down, then the t8 pair).
fn module_text(entries: &[(k::Spec, bool, Emitted)], name: &str) -> String {
    let emitted: Vec<Emitted> = entries.iter().map(|(_, _, e)| e.clone()).collect();
    iu4_gemm::module(&emitted, name).unwrap().0
}

#[test]
fn w3_entries_export_the_four_symbols_with_the_frozen_96_byte_abi() {
    let entries = all_entries();
    let names: Vec<String> = entries.iter().map(|(s, t, _)| symbol(*s, *t)).collect();
    assert_eq!(
        names,
        [
            "mq_i4_moe_gemv_mq3i4_r256_gate_up_pm_gfx1100",
            "mq_i4_moe_gemv_mq3i4_r128_down_pm_gfx1100",
            "mq_i4_moe_gemv_mq3i4_r256_gate_up_pm_gfx1100_t8",
            "mq_i4_moe_gemv_mq3i4_r128_down_pm_gfx1100_t8",
        ]
    );
    assert_eq!(
        k::Spec::module_mq3(Arch::Gfx1100),
        "mq_i4_moe_gemv_mq3_pm_gfx1100"
    );
    for (spec, tiled, e) in &entries {
        let t = &e.s_text;
        assert!(t.contains(".amdgcn_target \"amdgcn-amd-amdhsa--gfx1100\""));
        assert!(t.contains(".amdhsa_kernarg_size 96\n"));
        assert!(t.contains(".amdhsa_wavefront_size32 1"));
        assert!(t.contains(".amdhsa_group_segment_fixed_size 0"));
        assert!(t.contains(".amdhsa_private_segment_fixed_size 0"));
        // 16 user + system SGPRs on gfx1100 (`FeatureUserSGPRInit16Bug`): the ids follow 14 declared user SGPRs
        // for the narrow entries, 13 for the tiles that also enable z.
        assert!(t.contains(if *tiled {
            ".amdhsa_user_sgpr_count 13"
        } else {
            ".amdhsa_user_sgpr_count 14"
        }));
        assert_eq!(e.shape.barriers, 0);
        assert!(e.proof.lds_slots.is_empty());
        // The unsigned `expert_id >= n_experts` exit (`n_experts@88` loaded to s4 / s5 for the tiles) precedes the
        // table index `expert_id * 8` and the table load.
        let nexp = if *tiled { 5 } else { 4 };
        let guard = t
            .find(&format!("s_cmp_ge_u32 s34, s{nexp}"))
            .expect("expert-range guard");
        let index = t.find("s_lshl_b32 s51, s34, 3").expect("table index");
        assert!(
            guard < index,
            "{}: guard after the table index",
            symbol(*spec, *tiled)
        );
        assert!(t.contains(&format!("s_load_b32 s{nexp}, s[0:1], 0x58")));
        // Census: per group instance two byte loads and one scale load, four codes unpacked once.
        let gi = spec.group_instances() as usize;
        let count = |m: &str| {
            t.lines()
                .filter(|l| l.split_whitespace().next() == Some(m))
                .count()
        };
        assert_eq!(count("global_load_u8"), 2 * gi);
        assert_eq!(count("global_load_u16"), gi);
        // The single `global_load_b32` is the prologue's slot-id gather; no weight load is a dword load.
        assert_eq!(count("global_load_b32"), 1);
        assert_eq!(count("v_bfe_u32"), 4 * gi);
        assert_eq!(count("v_cvt_f32_i32_e32"), 4 * gi);
        assert_eq!(count("v_lshl_or_b32"), gi);
        assert_eq!(count("buffer_load_b128"), 8 * gi);
        assert_eq!(count("global_store_b32"), 8);
        // No MQ4 / W2 arithmetic in the W3 body.
        assert_eq!(count("v_fma_f32"), 0);
        let code_stride = t.contains("s_mul_i32 s55, s53, 48");
        assert!(code_stride, "48-byte group stride");
        assert!(t.contains("s_mul_i32 s49, s49, 3"), "K*3/8 row bytes");
    }
}

/// The landed MQ4-only gfx1100 K1 object is untouched by the W3 codec: regenerating its module from the shared
/// generator reproduces the committed text byte for byte (the W2/MQ4 arithmetic is shared code).
#[test]
fn existing_mq4_only_gfx1100_module_is_byte_identical() {
    let mut emitted = Vec::new();
    for tiled in [false, true] {
        for spec in k::module_specs(Arch::Gfx1100)
            .into_iter()
            .filter(|s| s.codec == k::Codec::Mq4Sym)
        {
            emitted.push(if tiled {
                k::emit_rowtiles(spec).unwrap()
            } else {
                k::emit(spec).unwrap()
            });
        }
    }
    let (text, _) = iu4_gemm::module(&emitted, &k::Spec::module(Arch::Gfx1100)).unwrap();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/mq_i4_moe_gemv_mq4only_pm_gfx1100.s");
    assert_eq!(text, std::fs::read_to_string(path).unwrap());
}

/// The W3 module is not part of the W2/MQ4 module and its arithmetic equals the gfx1201 W3 entry's, instruction for
/// instruction, apart from the gfx1100-forced addressing/hazard deltas: the unpack and chain/fold/shuffle VALU
/// stream (the numerics) is identical.
#[test]
fn w3_arithmetic_stream_equals_the_gfx1201_w3_entry() {
    // Register-agnostic: the two generators assign VGPRs differently (the gfx1201 network keeps a separate second
    // packet and shift register); the mnemonic/immediate sequence is the arithmetic.
    let numeric = |text: &str| -> Vec<String> {
        let strip = |l: &str| -> String {
            let mut out = String::new();
            let mut chars = l.chars().peekable();
            while let Some(ch) = chars.next() {
                out.push(ch);
                if ch == 'v' && chars.peek().is_some_and(|c| c.is_ascii_digit()) {
                    while chars.peek().is_some_and(|c| c.is_ascii_digit()) {
                        chars.next();
                    }
                }
            }
            out
        };
        text.lines()
            .map(str::trim)
            .filter(|l| {
                [
                    "v_mul_f32",
                    "v_fmac_f32",
                    "v_add_f32",
                    "v_cvt_f32_i32",
                    "v_cvt_f32_f16",
                    "v_bfe_u32",
                ]
                .iter()
                .any(|p| l.starts_with(p))
            })
            .map(strip)
            .collect()
    };
    for kind in [k::Kind::GateUp, k::Kind::Down] {
        let spec = k::Spec {
            arch: Arch::Gfx1100,
            codec: k::Codec::Mq3I4,
            kind,
        };
        let ours = k::emit(spec).unwrap();
        let theirs = mq_i4_moe_gemv::emit(mq_i4_moe_gemv::Spec {
            arch: Arch::Gfx1201,
            codec: mq_i4_moe_gemv::Codec::Mq3I4,
            kind: match kind {
                k::Kind::GateUp => mq_i4_moe_gemv::Kind::GateUp,
                k::Kind::Down => mq_i4_moe_gemv::Kind::Down,
            },
            cap: mq_i4_moe_gemv::Cap::C8,
        })
        .unwrap();
        // The VALU register numbering differs only where the arch plan does (none here); the stream is equal.
        let (a, b) = (numeric(&ours.s_text), numeric(&theirs.s_text));
        assert!(!a.is_empty());
        assert_eq!(a, b, "{kind:?}: W3 arithmetic stream differs from gfx1201");
    }
}

/// gfx1100 text retargeted to gfx1151 for the interpreter (see the module comment).
fn retarget(entries: &[(k::Spec, bool, Emitted)]) -> peacemaker_ir::Program {
    let text = module_text(entries, "mq3k1_gfx1151_retarget").replace("gfx1100", "gfx1151");
    lift(&native::assemble(&text, Arch::Gfx1151).unwrap())
}

struct Case {
    spec: k::Spec,
    m: usize,
    k: usize,
    /// Tokens (`C`); `10*C` slots.
    c: usize,
    /// Pool size of the routed experts (all distinct within a token).
    pool: usize,
    pad: usize,
    seed: u64,
}

struct Built {
    mem: Memory,
    route: Vec<u32>,
    weights: Vec<Vec<f32>>,
    xs: Vec<f32>,
    ys: usize,
    xrs: usize,
    xsrc: usize,
}

fn kernarg(c: &Case, u: usize, xsrc: usize, xrs: usize, ys: usize, tiled: bool) -> Vec<u8> {
    let (ids, offs, slots) = if tiled {
        (DESC, DESC + 160 * 4, DESC + 322 * 4)
    } else {
        (IDS, OFFS, SLOTS)
    };
    let mut a = Vec::new();
    for p in [
        TABLE,
        ids,
        offs,
        slots,
        XBASE,
        Y0 + GUARD as u64,
        if gate(c.spec) { Y1 + GUARD as u64 } else { 0 },
    ] {
        a.extend(p.to_le_bytes());
    }
    for v in [
        c.m,
        c.k,
        u,
        c.c,
        if gate(c.spec) { 10 } else { 1 },
        xsrc,
        xrs,
        ys,
        N_EXPERTS,
        0,
    ] {
        a.extend((v as u32).to_le_bytes());
    }
    assert_eq!(a.len(), 96);
    a
}

fn u32s(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn set(mem: &mut Memory, name: &str, base: u64, bytes: Vec<u8>) {
    if let Some(r) = mem.regions.iter_mut().find(|r| r.name == name) {
        r.bytes = bytes;
    } else {
        mem.map(name.into(), base, bytes).unwrap();
    }
}

fn build(c: &Case) -> Built {
    let mut rng = Rng(c.seed | 1);
    let slots = 10 * c.c;
    let (xsrc, xrs) = (if gate(c.spec) { c.c } else { slots }, c.k + c.pad);
    let ys = (if gate(c.spec) { c.m / 2 } else { c.m }) + c.pad;
    // Ten distinct experts per token from a shared pool.
    let mut route = Vec::with_capacity(slots);
    for t in 0..c.c {
        let mut ids: Vec<u32> = (0..c.pool as u32)
            .map(|i| (i * 37 + 11) % N_EXPERTS as u32)
            .collect();
        for i in (1..ids.len()).rev() {
            ids.swap(i, rng.below(i + 1));
        }
        let _ = t;
        route.extend_from_slice(&ids[..10]);
    }
    let mut mem = Memory::default();
    let mut table = vec![0u8; N_EXPERTS * 8];
    let mut weights = vec![Vec::new(); N_EXPERTS];
    for &id in &route {
        if !weights[id as usize].is_empty() {
            continue;
        }
        let (bytes, w) = w3_tensor(&mut rng, c.m, c.k);
        let addr = EXPERT + u64::from(id) * EXPERT_STRIDE;
        assert!(bytes.len() as u64 <= EXPERT_STRIDE);
        table[id as usize * 8..id as usize * 8 + 8].copy_from_slice(&addr.to_le_bytes());
        mem.map(format!("expert{id}"), addr, bytes).unwrap();
        weights[id as usize] = w;
    }
    mem.map("table".into(), TABLE, table).unwrap();
    // Wide-exponent full-mantissa activations; the stride padding is a NaN.
    let xs: Vec<f32> = (0..xsrc * xrs)
        .map(|i| {
            if i % xrs >= c.k {
                f32::NAN
            } else {
                let (s, e, m) = (
                    (rng.next() & 1) as u32,
                    118 + rng.below(19) as u32,
                    (rng.next() & 0x7f_ffff) as u32,
                );
                f32::from_bits((s << 31) | (e << 23) | m)
            }
        })
        .collect();
    mem.map(
        "x".into(),
        XBASE,
        xs.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect(),
    )
    .unwrap();
    let y_bytes = GUARD * 2 + slots * ys * 4;
    mem.map("y0".into(), Y0, vec![POISON; y_bytes]).unwrap();
    mem.map("y1".into(), Y1, vec![POISON; y_bytes]).unwrap();
    Built {
        mem,
        route,
        weights,
        xs,
        ys,
        xrs,
        xsrc,
    }
}

/// First-occurrence experts over `range`, ascending absolute slots.
fn groups(route: &[u32], range: std::ops::Range<usize>) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut ids: Vec<u32> = Vec::new();
    for s in range.clone() {
        if !ids.contains(&route[s]) {
            ids.push(route[s]);
        }
    }
    let (mut offs, mut slots) = (Vec::new(), Vec::new());
    for &id in &ids {
        offs.push(slots.len() as u32);
        slots.extend(range.clone().filter(|&s| route[s] == id).map(|s| s as u32));
    }
    offs.push(slots.len() as u32);
    (ids, offs, slots)
}

fn launch(emu: &Emulator<'_>, mem: &mut Memory, wg: [u32; 3], block: u32) {
    emu.run_workgroup(
        &Launch {
            kernarg_ptr: KA,
            workgroup: wg,
            block: [block, 1, 1],
            dynamic_lds: 0,
        },
        mem,
    )
    .unwrap_or_else(|e| panic!("workgroup {wg:?}: {e}"));
}

/// Every payload word of Y0/Y1 equals the twin bit for bit; every other byte is untouched poison; no input is written.
fn compare(c: &Case, b: &Built, mem: &Memory) {
    let out_rows = if gate(c.spec) { c.m / 2 } else { c.m };
    let slots = 10 * c.c;
    for (part, name) in ["y0", "y1"].into_iter().enumerate() {
        let r = mem.regions.iter().find(|r| r.name == name).unwrap();
        let live_part = part == 0 || gate(c.spec);
        for byte in 0..r.bytes.len() {
            let payload = byte >= GUARD && byte < GUARD + slots * b.ys * 4;
            let j = byte.saturating_sub(GUARD) / 4;
            let live = payload && live_part && j % b.ys < out_rows;
            assert_eq!(
                r.written.contains(byte),
                live,
                "{name} byte {byte}: ownership"
            );
            if !live {
                assert_eq!(r.bytes[byte], POISON, "{name} byte {byte}: poison");
            }
        }
        if !live_part {
            continue;
        }
        for slot in 0..slots {
            let xr = slot / if gate(c.spec) { 10 } else { 1 };
            let x = &b.xs[xr * b.xrs..xr * b.xrs + c.k];
            let w = &b.weights[b.route[slot] as usize];
            for row in 0..out_rows {
                let wrow = part * out_rows + row;
                let want = twin_row(gate(c.spec), c.k, x, &w[wrow * c.k..(wrow + 1) * c.k]);
                let at = GUARD + (slot * b.ys + row) * 4;
                let got = u32::from_le_bytes(r.bytes[at..at + 4].try_into().unwrap());
                assert_eq!(
                    got,
                    want.to_bits(),
                    "{:?} M={} K={} slot={slot} {name} row={row}: got {got:#010x} want {:#010x}",
                    c.spec,
                    c.m,
                    c.k,
                    want.to_bits()
                );
            }
        }
    }
    for r in &mem.regions {
        if r.name != "y0" && r.name != "y1" {
            assert_eq!(r.written.count(), 0, "{} input modified", r.name);
        }
    }
}

fn run_narrow(program: &peacemaker_ir::Program, c: &Case) {
    let b = build(c);
    let (ids, offs, slots) = groups(&b.route, 0..10 * c.c);
    let mut mem = b.mem.clone();
    set(&mut mem, "ids", IDS, u32s(&ids));
    set(&mut mem, "offs", OFFS, u32s(&offs));
    set(&mut mem, "slots", SLOTS, u32s(&slots));
    set(
        &mut mem,
        "kernarg",
        KA,
        kernarg(c, ids.len(), b.xsrc, b.xrs, b.ys, false),
    );
    let emu = Emulator::new(
        program,
        &symbol(c.spec, false).replace("gfx1100", "gfx1151"),
    )
    .unwrap();
    for y in 0..ids.len() as u32 {
        for x in 0..c.m as u32 {
            launch(&emu, &mut mem, [x, y, 0], 32);
        }
    }
    compare(c, &b, &mem);
}

fn run_tiled(program: &peacemaker_ir::Program, c: &Case) {
    let b = build(c);
    let slots = 10 * c.c;
    let tiles = c.c.div_ceil(8);
    let mut words = Vec::new();
    for t in 0..tiles {
        let (ids, offs, sl) = groups(&b.route, 80 * t..(80 * t + 80).min(slots));
        let mut d = vec![u32::MAX; TILE_WORDS];
        d[..ids.len()].copy_from_slice(&ids);
        d[160..160 + offs.len()].copy_from_slice(&offs);
        let tail = *offs.last().unwrap();
        for w in &mut d[160 + offs.len()..160 + 81] {
            *w = tail;
        }
        d[322..322 + sl.len()].copy_from_slice(&sl);
        words.extend(d);
    }
    let mut mem = b.mem.clone();
    set(&mut mem, "descriptor", DESC, u32s(&words));
    set(
        &mut mem,
        "kernarg",
        KA,
        kernarg(c, 80, b.xsrc, b.xrs, b.ys, true),
    );
    let emu = Emulator::new(program, &symbol(c.spec, true).replace("gfx1100", "gfx1151")).unwrap();
    for z in 0..tiles as u32 {
        for y in 0..80u32 {
            for x in 0..c.m.div_ceil(8) as u32 {
                launch(&emu, &mut mem, [x, y, z], 256);
            }
        }
    }
    compare(c, &b, &mem);
}

fn cases(spec: k::Spec) -> Vec<Case> {
    let g = gate(spec);
    // (m, k): production shapes plus K tails of every group count class (quad loop remainder 0..3) and odd row counts.
    let shapes: &[(usize, usize)] = if g {
        &[
            (4, 2560),
            (2, 128),
            (6, 256),
            (4, 384),
            (2, 640),
            (2, 1152),
            (6, 1280),
        ]
    } else {
        &[(3, 640), (1, 128), (2, 256), (3, 384), (1, 1024), (2, 1152)]
    };
    shapes
        .iter()
        .enumerate()
        .map(|(i, &(m, k))| Case {
            spec,
            m,
            k,
            c: 1 + i % 3,
            pool: [10, 14, 33][i % 3],
            pad: [0, 3][i % 2],
            seed: 0x3a_0000 + (i as u64) * 7919 + u64::from(g),
        })
        .collect()
}

#[test]
fn narrow_entries_match_the_cpu_w3_law_bit_for_bit_in_the_interpreter() {
    let entries = all_entries();
    let program = retarget(&entries);
    for (spec, tiled, _) in &entries {
        if *tiled {
            continue;
        }
        for c in cases(*spec) {
            run_narrow(&program, &c);
        }
    }
}

#[test]
fn rowtile_entries_match_the_cpu_w3_law_bit_for_bit_in_the_interpreter() {
    let entries = all_entries();
    let program = retarget(&entries);
    for (spec, tiled, _) in &entries {
        if !*tiled {
            continue;
        }
        // C spans one full tile, a tile boundary and a ragged last tile.
        for (i, &c) in [1usize, 8, 9].iter().enumerate() {
            let (m, kk) = if gate(*spec) { (6, 384) } else { (3, 256) };
            run_tiled(
                &program,
                &Case {
                    spec: *spec,
                    m,
                    k: kk,
                    c,
                    pool: 33,
                    pad: [0, 3, 0][i],
                    seed: 0x71_0000 + c as u64,
                },
            );
        }
    }
}

/// `expert_id >= n_experts` (including a negative id), a zero or over-eight slot count and `offsets[u+1] > 10*C`
/// exit without a store and before any table dereference: the weight table is unmapped, so a dereference faults.
#[test]
fn guards_exit_before_the_table_and_store_nothing() {
    let entries = all_entries();
    let program = retarget(&entries);
    let spec = k::Spec {
        arch: Arch::Gfx1100,
        codec: k::Codec::Mq3I4,
        kind: k::Kind::Down,
    };
    let c = Case {
        spec,
        m: 2,
        k: 256,
        c: 1,
        pool: 10,
        pad: 0,
        seed: 9,
    };
    let emu = Emulator::new(&program, &symbol(spec, false).replace("gfx1100", "gfx1151")).unwrap();
    // (expert id, offsets[0], offsets[1])
    for (id, o0, o1) in [
        (N_EXPERTS as u32, 0u32, 1u32),
        (0x8000_0000, 0, 1),
        (u32::MAX, 0, 1),
        (3, 0, 0),
        (3, 0, 9),
        (3, 0, 11),
    ] {
        let mut mem = Memory::default();
        mem.map("ids".into(), IDS, u32s(&[id])).unwrap();
        mem.map("offs".into(), OFFS, u32s(&[o0, o1])).unwrap();
        mem.map("slots".into(), SLOTS, u32s(&[0; 16])).unwrap();
        mem.map("x".into(), XBASE, vec![0; 4 * 10 * 260]).unwrap();
        mem.map("y0".into(), Y0, vec![POISON; GUARD * 2 + 10 * 2 * 4])
            .unwrap();
        mem.map("kernarg".into(), KA, kernarg(&c, 1, 10, 256, 2, false))
            .unwrap();
        for x in 0..2u32 {
            launch(&emu, &mut mem, [x, 0, 0], 32);
        }
        for r in &mem.regions {
            assert_eq!(
                r.written.count(),
                0,
                "id {id:#x} slots {o0}..{o1}: {} written",
                r.name
            );
        }
    }
}

/// Negative controls: the comparator MUST detect (a) one flipped weight code bit in device memory after the model
/// was built, (b) one negated modelled weight, and (c) a wrong
/// W3 level law (`2c - 5`): each makes the bit-exact comparison panic, so a pass above is not vacuous.
#[test]
fn negative_controls_are_detected() {
    let entries = all_entries();
    let program = retarget(&entries);
    let spec = k::Spec {
        arch: Arch::Gfx1100,
        codec: k::Codec::Mq3I4,
        kind: k::Kind::GateUp,
    };
    let c = Case {
        spec,
        m: 4,
        k: 384,
        c: 1,
        pool: 10,
        pad: 0,
        seed: 0xc0de,
    };
    let detects = |mutate: &dyn Fn(&mut Built)| {
        let mut b = build(&c);
        mutate(&mut b);
        let (ids, offs, slots) = groups(&b.route, 0..10 * c.c);
        let mut mem = b.mem.clone();
        set(&mut mem, "ids", IDS, u32s(&ids));
        set(&mut mem, "offs", OFFS, u32s(&offs));
        set(&mut mem, "slots", SLOTS, u32s(&slots));
        set(
            &mut mem,
            "kernarg",
            KA,
            kernarg(&c, ids.len(), b.xsrc, b.xrs, b.ys, false),
        );
        let emu =
            Emulator::new(&program, &symbol(spec, false).replace("gfx1100", "gfx1151")).unwrap();
        for y in 0..ids.len() as u32 {
            for x in 0..c.m as u32 {
                launch(&emu, &mut mem, [x, y, 0], 32);
            }
        }
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| compare(&c, &b, &mem))).is_err()
    };
    // Control: the untouched build passes.
    assert!(!detects(&|_| {}));
    // (a) the MSB of code 40 of row 2 group 1 (byte 2*144 + 48 + 15, bit 2) of expert `route[0]` flipped in device
    // memory only; a random-scale group, so the change is far above the f32 rounding of the row sum.
    assert!(detects(&|b| {
        let id = b.route[0];
        let r = b
            .mem
            .regions
            .iter_mut()
            .find(|r| r.name == format!("expert{id}"))
            .unwrap();
        r.bytes[2 * 144 + 48 + 15] ^= 0x04;
    }));
    // (b) the model's weight of one expert (row 3, column 200, a random-scale group) negated.
    assert!(detects(&|b| {
        let id = b.route[3] as usize;
        let w = &mut b.weights[id][3 * 384 + 200];
        assert!(*w != 0.0);
        *w = -*w;
    }));
    // (c) the wrong level law everywhere (`2c - 5` for `2c - 7`): scale every weight by the wrong shift.
    assert!(detects(&|b| {
        for w in b.weights.iter_mut().flatten() {
            *w *= 1.0 + 1.0 / 7.0;
        }
    }));
}

/// Waits/hazards (incl. the gfx1100 `VALUTransUseHazard` guard) replay independently of the builder's ledger, and M7
/// lifts the natively linked object byte-exactly with no outstanding obligation for every export.
#[cfg(feature = "toolchain")]
#[test]
fn every_w3_export_passes_wait_hazard_replay_and_m7() {
    use peacemaker_isa::{ledger_replay, pm_check};
    let entries = all_entries();
    for (spec, tiled, e) in &entries {
        ledger_replay::replay_waits(&e.s_text, Arch::Gfx1100)
            .unwrap_or_else(|err| panic!("{}: {err}", symbol(*spec, *tiled)));
    }
    let text = module_text(&entries, &k::Spec::module_mq3(Arch::Gfx1100));
    ledger_replay::replay_waits(&text, Arch::Gfx1100).unwrap();
    let dir = std::env::temp_dir().join(format!("peacemaker-isa-mq3k1-m7-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let co = dir.join("module.co");
    std::fs::write(&co, native::assemble(&text, Arch::Gfx1100).unwrap()).unwrap();
    for (spec, tiled, _) in &entries {
        let sym = symbol(*spec, *tiled);
        let m7 =
            pm_check::m7(&co, "gfx1100", sym.as_str()).unwrap_or_else(|e| panic!("{sym}: {e}"));
        assert_eq!(
            (m7["lift"].as_str(), &m7["obligations"]),
            (Some("byte-exact"), &serde_json::json!({})),
            "{sym}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
