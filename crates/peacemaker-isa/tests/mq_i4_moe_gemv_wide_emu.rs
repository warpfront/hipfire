// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Fixed-eight-row K1 descriptor entries on the CPU interpreter, gfx1151 and gfx1201.
//! Every total row count 1..=64 executes the real z grid and compares all output bytes
//! against independent C=1 launches of the original PM narrow entry. This proves PM
//! scheduling/descriptor parity, not parity with the shipped singleton object: lifting
//! that pinned object and hardware qualification remain separate gates.
//!
//! Descriptor tiles have 482 u32 words (1928 bytes): ids[0..160], offsets[160..322],
//! slots[322..482]. Bucket pointer adjustments are 80, 81, 80 words respectively.
//! Expert order is first occurrence; slot ids are absolute, while X/Y retain whole-window
//! bases. Both bucket placements, every tile boundary through slot639, padded strides,
//! inactive output waves, untouched inputs/tails and malformed-domain exits are covered.
//! Guard groups map descriptor+kernarg only: the weight table, tensors, X and Y are
//! unmapped, so any table dereference before the offset/slot guards is a CPU fault;
//! success proves pre-deref exits. gfx1151 Cn uses its production K2560/K640; gfx1201
//! also covers legal K loop tails.
//! Singleton outputs are cached across the width sweep, avoiding redundant reference work.
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;
use peacemaker_isa::kernels::mq_i4_moe_gemv::{self, Codec, Kind, Spec};
use peacemaker_isa::Arch;

const SENTINEL: u8 = 0xa5;
const GUARD: usize = 64;
const KERNARG: u64 = 0x1_0000;
const TABLE: u64 = 0x10_0000;
const IDS: u64 = 0x20_0000;
const REF_IDS: u64 = 0x50_0000;
const REF_OFFS: u64 = 0x60_0000;
const REF_SLOTS: u64 = 0x70_0000;
const EXPERTS: u64 = 0x1000_0000;
const EXPERT_STRIDE: u64 = 0x10_0000;
const XBASE: u64 = 0x8000_0000;
const Y0: u64 = 0x9000_0000;
const Y1: u64 = 0xa000_0000;
const R0: u64 = 0xb000_0000;
const R1: u64 = 0xc000_0000;
const N_EXPERTS: usize = 512;
const TILE_CAP: usize = 80;
const TILE_WORDS: usize = 482;
const TILE_BYTES: usize = TILE_WORDS * 4;

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
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }
}

/// Normal f16 scale with exponent field 8..=20: `-8 * sc` is then exactly the symmetric MQ4 zero.
fn scale_bits(rng: &mut Rng) -> u16 {
    (((8 + rng.below(13)) as u16) << 10) | (rng.next() as u16 & 0x3ff)
}
fn sym_zero_bits(sc: u16) -> u16 {
    0x8000 | (sc + (3 << 10))
}

/// Planar W3 tensor: per row `K*3/8` code bytes (128 3-bit codes per 48-byte group, little-endian bit stream),
/// then the f16 scale plane `2 * (row * K/128 + g)` at `M*K*3/8`.
fn w3_tensor(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    let (groups, row_bytes) = (k / 128, k * 3 / 8);
    let mut out = vec![0u8; m * row_bytes + 2 * m * groups];
    for r in 0..m {
        for g in 0..groups {
            let base = r * row_bytes + g * 48;
            for j in 0..128 {
                let code = rng.below(8) as u32;
                for b in 0..3 {
                    if (code >> b) & 1 == 1 {
                        out[base + (3 * j + b) / 8] |= 1 << ((3 * j + b) % 8);
                    }
                }
            }
            let off = m * row_bytes + 2 * (r * groups + g);
            out[off..off + 2].copy_from_slice(&scale_bits(rng).to_le_bytes());
        }
    }
    out
}

/// Pointer-addressed MQ4: gate G256 groups of 136 B (`[sc0 zp0 sc1 zp1]`, 128 payload bytes at +8), down G128
/// groups of 68 B (`[sc zp]`, 64 payload bytes at +4).
fn mq4_tensor(rng: &mut Rng, kind: Kind, m: usize, k: usize) -> Vec<u8> {
    let (cols, stride, halves, payload_at, payload) = match kind {
        Kind::GateUp => (256, 136, 2, 8, 128),
        Kind::Down => (128, 68, 1, 4, 64),
    };
    let groups = k / cols;
    let mut out = vec![0u8; m * groups * stride];
    for r in 0..m {
        for g in 0..groups {
            let base = (r * groups + g) * stride;
            for h in 0..halves {
                let sc = scale_bits(rng);
                out[base + 4 * h..base + 4 * h + 2].copy_from_slice(&sc.to_le_bytes());
                out[base + 4 * h + 2..base + 4 * h + 4]
                    .copy_from_slice(&sym_zero_bits(sc).to_le_bytes());
            }
            for byte in &mut out[base + payload_at..base + payload_at + payload] {
                *byte = rng.next() as u8;
            }
        }
    }
    out
}

/// Full-mantissa activations over a wide exponent band, random signs; the stride padding is a NaN.
fn x_rows(rng: &mut Rng, rows: usize, k: usize, stride: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * stride * 4);
    for _ in 0..rows {
        for i in 0..stride {
            let bits = if i < k {
                let (s, e, m) = (
                    (rng.next() & 1) as u32,
                    118 + rng.below(19) as u32,
                    (rng.next() & 0x7f_ffff) as u32,
                );
                (s << 31) | (e << 23) | m
            } else {
                0x7fc0_0000
            };
            out.extend(bits.to_le_bytes());
        }
    }
    out
}

#[derive(Clone, Copy, Debug)]
enum Routing {
    /// Each lane draws ten distinct experts from a pool of this size.
    Pool(usize),
    /// Every lane routes to the same ten experts (each expert has one slot per lane).
    Identical,
    /// Ten fresh experts per lane while ids last (`10 * C` distinct, capped at the table).
    Disjoint,
}

struct Case {
    c: usize,
    m: usize,
    k: usize,
    routing: Routing,
    /// Extra padding words of the X row stride and the Y slot stride.
    pad: bool,
    seed: u64,
}

fn route(rng: &mut Rng, case: &Case) -> Vec<u32> {
    let mut out = Vec::with_capacity(10 * case.c);
    match case.routing {
        Routing::Pool(n) => {
            let mut pool: Vec<u32> = (0..n).map(|i| ((i * 37 + 11) % N_EXPERTS) as u32).collect();
            pool.sort();
            pool.dedup();
            assert!(pool.len() >= 10);
            for _ in 0..case.c {
                let mut p = pool.clone();
                rng.shuffle(&mut p);
                out.extend_from_slice(&p[..10]);
            }
        }
        Routing::Identical => {
            let mut lane: Vec<u32> = (0..10).map(|i| (i * 53 + 7) as u32).collect();
            rng.shuffle(&mut lane);
            for _ in 0..case.c {
                out.extend_from_slice(&lane);
            }
        }
        Routing::Disjoint => {
            let mut ids: Vec<u32> = (0..N_EXPERTS as u32).collect();
            rng.shuffle(&mut ids);
            for l in 0..case.c {
                out.extend_from_slice(
                    &ids[10 * (l % (N_EXPERTS / 10))..10 * (l % (N_EXPERTS / 10)) + 10],
                );
            }
        }
    }
    out
}

/// One codec bucket, in first-occurrence expert order and ascending original slots.
struct Bucket {
    ids: Vec<u32>,
    offs: Vec<u32>,
    slots: Vec<u32>,
}
fn bucket(route: &[u32]) -> Bucket {
    bucket_slots(route.iter().enumerate().map(|(s, &e)| (s as u32, e)))
}
fn bucket_slots(slots: impl IntoIterator<Item = (u32, u32)>) -> Bucket {
    let entries: Vec<_> = slots.into_iter().collect();
    let mut b = Bucket {
        ids: vec![],
        offs: vec![],
        slots: vec![],
    };
    for &(_, e) in &entries {
        if !b.ids.contains(&e) {
            b.ids.push(e);
        }
    }
    for &id in &b.ids {
        b.offs.push(b.slots.len() as u32);
        b.slots
            .extend(entries.iter().filter(|&&(_, e)| e == id).map(|&(s, _)| s));
    }
    b.offs.push(b.slots.len() as u32);
    b
}

fn u32s(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn kernarg(
    ptrs: [u64; 7],
    m: usize,
    k: usize,
    u: usize,
    c: usize,
    xdiv: usize,
    xsrc: usize,
    xrs: usize,
    ys: usize,
) -> Vec<u8> {
    let mut a = Vec::with_capacity(96);
    for p in ptrs {
        a.extend(p.to_le_bytes());
    }
    for v in [m, k, u, c, xdiv, xsrc, xrs, ys] {
        a.extend((v as u32).to_le_bytes());
    }
    // `n_experts@88`: the extent of the weight table, above every expert id of every route; `reserved@92` is zero.
    a.extend((N_EXPERTS as u32).to_le_bytes());
    a.extend(0u32.to_le_bytes());
    assert_eq!(a.len(), 96);
    a
}

struct Rig {
    arch: Arch,
    program: peacemaker_ir::Program,
}
impl Rig {
    fn new(arch: Arch) -> Self {
        // Deliberately independent of production registration and legacy C16/C32 exports.
        let mut emitted = Vec::new();
        for spec in mq_i4_moe_gemv::narrow_specs(arch) {
            emitted.push(mq_i4_moe_gemv::emit(spec).unwrap());
            emitted.push(mq_i4_moe_gemv::emit_rowtiles(spec).unwrap());
        }
        let (text, _) = peacemaker_isa::kernels::iu4_gemm::module(
            &emitted,
            &format!("k1_rowtile_test_{}", arch.name()),
        )
        .unwrap();
        let elf = peacemaker_isa::native::assemble(&text, arch).unwrap();
        let program = peacemaker_lift::lift_object(
            &elf,
            peacemaker_lift::Options {
                frontend: Frontend::Builder,
            },
        )
        .unwrap()
        .program;
        Rig { arch, program }
    }
    fn emu(&self, spec: Spec) -> Emulator<'_> {
        Emulator::new(&self.program, &spec.symbol()).unwrap()
    }
    fn tile_emu(&self, codec: Codec, kind: Kind) -> Emulator<'_> {
        Emulator::new(&self.program, &tile_symbol(self.arch, codec, kind)).unwrap()
    }
}

fn set_region(mem: &mut Memory, name: &str, bytes: Vec<u8>) {
    mem.regions
        .iter_mut()
        .find(|r| r.name == name)
        .unwrap_or_else(|| panic!("region {name}"))
        .bytes = bytes;
}
fn region<'a>(mem: &'a Memory, name: &str) -> &'a [u8] {
    &mem.regions.iter().find(|r| r.name == name).unwrap().bytes
}

fn launch(emu: &Emulator, mem: &mut Memory, tag: &str, grid: [u32; 3], block: u32) {
    for gz in 0..grid[2] {
        for gy in 0..grid[1] {
            for gx in 0..grid[0] {
                run_group(emu, mem, tag, [gx, gy, gz], block);
            }
        }
    }
}

fn run_group(emu: &Emulator, mem: &mut Memory, tag: &str, workgroup: [u32; 3], block: u32) {
    emu.run_workgroup(
        &Launch {
            kernarg_ptr: KERNARG,
            workgroup,
            block: [block, 1, 1],
            dynamic_lds: 0,
        },
        mem,
    )
    .unwrap_or_else(|e| panic!("{tag} wg{workgroup:?}: {e}"));
}

fn tile_symbol(arch: Arch, codec: Codec, kind: Kind) -> String {
    let prefix = if arch == Arch::Gfx1151 { "cn_" } else { "" };
    format!("mq_i4_moe_gemv_{prefix}{}_{}_t8", codec.tag(), kind.tag())
}

fn bucket_ptrs(b: usize) -> [u64; 3] {
    [
        IDS + (b * 80 * 4) as u64,
        IDS + ((160 + b * 81) * 4) as u64,
        IDS + ((322 + b * 80) * 4) as u64,
    ]
}

/// Reproduce the planner's physical layout, not three separately packed arrays.
/// Both buckets are live; each codec is tested at both pointer placements. In this
/// geometry oracle every expert tensor has the selected codec, so the same entry
/// consumes both buckets (real mixed-codec dispatch has separate qualification).
fn descriptors(route: &[u32], uniform_bucket: Option<usize>) -> Vec<u32> {
    let mut words = vec![0; route.len().div_ceil(TILE_CAP) * TILE_WORDS];
    for (t, tile) in route.chunks(TILE_CAP).enumerate() {
        for b in 0..2 {
            let group = bucket_slots(
                tile.iter()
                    .enumerate()
                    .filter(|&(_, &e)| uniform_bucket.unwrap_or(e as usize % 2) == b)
                    .map(|(s, &e)| ((t * TILE_CAP + s) as u32, e)),
            );
            let base = t * TILE_WORDS;
            words[base + b * 80..base + b * 80 + group.ids.len()].copy_from_slice(&group.ids);
            // Equal terminal offsets give every capacity-tail expert n=0.
            words[base + 160 + b * 81..base + 160 + (b + 1) * 81].fill(*group.offs.last().unwrap());
            words[base + 160 + b * 81..base + 160 + b * 81 + group.offs.len()]
                .copy_from_slice(&group.offs);
            // Poison unused slot words: clamped loads must not consume these.
            words[base + 322 + b * 80..base + 322 + (b + 1) * 80].fill(u32::MAX);
            words[base + 322 + b * 80..base + 322 + b * 80 + group.slots.len()]
                .copy_from_slice(&group.slots);
            assert!(group.offs.windows(2).all(|o| o[1] - o[0] <= 8));
        }
    }
    assert_eq!(words.len() * 4, route.len().div_ceil(80) * TILE_BYTES);
    words
}

fn input_snapshot(mem: &Memory) -> Vec<(String, Vec<u8>)> {
    mem.regions
        .iter()
        .filter(|r| !matches!(r.name.as_str(), "kernarg" | "y0" | "y1" | "r0" | "r1"))
        .map(|r| (r.name.clone(), r.bytes.clone()))
        .collect()
}

fn assert_inputs(mem: &Memory, before: &[(String, Vec<u8>)], tag: &str) {
    for (name, bytes) in before {
        assert_eq!(
            region(mem, name),
            bytes.as_slice(),
            "{tag}: modified input {name}"
        );
        let mapped = mem.regions.iter().find(|r| r.name == *name).unwrap();
        assert_eq!(mapped.written.count(), 0, "{tag}: wrote input {name}");
    }
}

/// Cache original PM singleton rows once, not once for every prefix width.
struct Fixture {
    codec: Codec,
    kind: Kind,
    case: Case,
    route: Vec<u32>,
    mem: Memory,
    xdiv: usize,
    xrs: usize,
    ys: usize,
    out_rows: usize,
}

impl Fixture {
    fn new(rig: &Rig, codec: Codec, kind: Kind, case: Case) -> Self {
        let mut rng = Rng(case.seed | 1);
        let route = route(&mut rng, &case);
        let (c, m, k) = (case.c, case.m, case.k);
        let out_rows = if kind == Kind::GateUp { m / 2 } else { m };
        let (xrows, xdiv) = if kind == Kind::GateUp {
            (c, 10)
        } else {
            (10 * c, 1)
        };
        let (xrs, ys) = if case.pad {
            (k + 4, out_rows + 3)
        } else {
            (k, out_rows)
        };
        let mut mem = Memory::default();
        let mut table = vec![0u8; N_EXPERTS * 8];
        for id in bucket(&route).ids {
            let bytes = match codec {
                Codec::Mq3I4 => w3_tensor(&mut rng, m, k),
                Codec::Mq4Sym => mq4_tensor(&mut rng, kind, m, k),
            };
            let base = EXPERTS + u64::from(id) * EXPERT_STRIDE;
            assert!((bytes.len() as u64) < EXPERT_STRIDE);
            table[id as usize * 8..id as usize * 8 + 8].copy_from_slice(&base.to_le_bytes());
            mem.map(format!("e{id}"), base, bytes).unwrap();
        }
        mem.map("table".into(), TABLE, table).unwrap();
        mem.map("x".into(), XBASE, x_rows(&mut rng, xrows, k, xrs))
            .unwrap();
        let ybytes = 10 * c * ys * 4;
        for (name, base) in [("y0", Y0), ("y1", Y1), ("r0", R0), ("r1", R1)] {
            mem.map(name.into(), base, vec![SENTINEL; ybytes + 2 * GUARD])
                .unwrap();
        }
        mem.map("descriptor".into(), IDS, u32s(&descriptors(&route, None)))
            .unwrap();
        for (name, base) in [
            ("ref_ids", REF_IDS),
            ("ref_offs", REF_OFFS),
            ("ref_slots", REF_SLOTS),
        ] {
            mem.map(name.into(), base, vec![0; 4 * 16]).unwrap();
        }
        mem.map("kernarg".into(), KERNARG, vec![0; 96]).unwrap();
        let immutable = input_snapshot(&mem);
        let nspec = Spec::narrow(rig.arch, codec, kind);
        let nemu = rig.emu(nspec);
        for l in 0..c {
            let lb = bucket(&route[10 * l..10 * l + 10]);
            set_region(&mut mem, "ref_ids", u32s(&lb.ids));
            set_region(&mut mem, "ref_offs", u32s(&lb.offs));
            set_region(&mut mem, "ref_slots", u32s(&lb.slots));
            let (xoff, xsrc) = if kind == Kind::GateUp {
                (l * xrs * 4, 1)
            } else {
                (l * 10 * xrs * 4, 10)
            };
            let yoff = l * 10 * ys * 4;
            set_region(
                &mut mem,
                "kernarg",
                kernarg(
                    [
                        TABLE,
                        REF_IDS,
                        REF_OFFS,
                        REF_SLOTS,
                        XBASE + xoff as u64,
                        R0 + GUARD as u64 + yoff as u64,
                        if kind == Kind::GateUp {
                            R1 + GUARD as u64 + yoff as u64
                        } else {
                            0
                        },
                    ],
                    m,
                    k,
                    lb.ids.len(),
                    1,
                    xdiv,
                    xsrc,
                    xrs,
                    ys,
                ),
            );
            let inputs = input_snapshot(&mem);
            launch(
                &nemu,
                &mut mem,
                &format!("{} lane {l}", nspec.symbol()),
                [m as u32, lb.ids.len() as u32, 1],
                32,
            );
            assert_inputs(&mem, &inputs, &nspec.symbol());
        }
        // The reference descriptor regions are deliberately host-rewritten above.
        let fixed: Vec<_> = immutable
            .into_iter()
            .filter(|(name, _)| !name.starts_with("ref_"))
            .collect();
        assert_inputs(&mem, &fixed, &nspec.symbol());
        for name in ["r0", "r1"] {
            let bytes = region(&mem, name);
            assert!(
                bytes[..GUARD]
                    .iter()
                    .chain(&bytes[GUARD + ybytes..])
                    .all(|&b| b == SENTINEL),
                "{}: singleton guard {name}",
                nspec.symbol()
            );
        }
        Self {
            codec,
            kind,
            case,
            route,
            mem,
            xdiv,
            xrs,
            ys,
            out_rows,
        }
    }
}

/// One whole-window tiled launch per bucket, always with full X/Y bases and U=80.
fn run_case(rig: &Rig, f: &Fixture, c: usize, uniform_bucket: Option<usize>) {
    assert!((1..=f.case.c).contains(&c));
    let (m, k, ys) = (f.case.m, f.case.k, f.ys);
    let tag = format!(
        "{} C={c} M={m} K={k}",
        tile_symbol(rig.arch, f.codec, f.kind)
    );
    let mut mem = f.mem.clone();
    let xsrc = if f.kind == Kind::GateUp { c } else { 10 * c };
    // No mapped extra row: a bad token-tail load is an interpreter memory error.
    set_region(
        &mut mem,
        "x",
        region(&f.mem, "x")[..xsrc * f.xrs * 4].to_vec(),
    );
    set_region(
        &mut mem,
        "descriptor",
        u32s(&descriptors(&f.route[..10 * c], uniform_bucket)),
    );
    let ybytes = 10 * c * ys * 4;
    for name in ["y0", "y1"] {
        set_region(&mut mem, name, vec![SENTINEL; ybytes + 2 * GUARD]);
    }
    let inputs = input_snapshot(&mem);
    let emu = rig.tile_emu(f.codec, f.kind);
    for b in 0..2 {
        let [ids, offs, slots] = bucket_ptrs(b);
        set_region(
            &mut mem,
            "kernarg",
            kernarg(
                [
                    TABLE,
                    ids,
                    offs,
                    slots,
                    XBASE,
                    Y0 + GUARD as u64,
                    if f.kind == Kind::GateUp {
                        Y1 + GUARD as u64
                    } else {
                        0
                    },
                ],
                m,
                k,
                TILE_CAP,
                c,
                f.xdiv,
                xsrc,
                f.xrs,
                ys,
            ),
        );
        launch(
            &emu,
            &mut mem,
            &tag,
            [m.div_ceil(8) as u32, TILE_CAP as u32, c.div_ceil(8) as u32],
            256,
        );
        assert_inputs(&mem, &inputs, &tag);
        // Check after EACH bucket: an entry must not touch the other bucket's
        // output even if its later launch would overwrite the same value.
        for (name, reference) in [("y0", "r0"), ("y1", "r1")] {
            let output = mem.regions.iter().find(|r| r.name == name).unwrap();
            let w = output.bytes.as_slice();
            assert!(
                w[..GUARD]
                    .iter()
                    .chain(&w[GUARD + ybytes..])
                    .all(|&v| v == SENTINEL),
                "{tag}: guard {name}"
            );
            let live_slots = f.route[..10 * c]
                .iter()
                .filter(|&&e| uniform_bucket.unwrap_or(e as usize % 2) <= b)
                .count();
            let live_bytes = if f.kind == Kind::Down && name == "y1" {
                0
            } else {
                live_slots * f.out_rows * 4
            };
            assert_eq!(
                output.written.count(),
                live_bytes,
                "{tag}: unexpected writes {name}"
            );
            if f.kind == Kind::Down && name == "y1" {
                assert!(w.iter().all(|&v| v == SENTINEL), "{tag}: down wrote Y1");
                continue;
            }
            let r = &region(&f.mem, reference)[GUARD..GUARD + ybytes];
            for (i, (actual, singleton)) in w[GUARD..GUARD + ybytes]
                .chunks_exact(4)
                .zip(r.chunks_exact(4))
                .enumerate()
            {
                let actual = u32::from_le_bytes(actual.try_into().unwrap());
                let singleton = u32::from_le_bytes(singleton.try_into().unwrap());
                let (slot, row) = (i / ys, i % ys);
                let live = uniform_bucket.unwrap_or(f.route[slot] as usize % 2) <= b;
                let expected = if live {
                    singleton
                } else {
                    u32::from_le_bytes([SENTINEL; 4])
                };
                assert_eq!(
                    actual, expected,
                    "{tag} bucket{b}: {name}[slot{slot}][row{row}]"
                );
                if row < f.out_rows {
                    assert_ne!(
                        singleton,
                        u32::from_le_bytes([SENTINEL; 4]),
                        "{tag}: singleton did not write slot{slot} row{row}"
                    );
                } else {
                    assert_eq!(
                        actual,
                        u32::from_le_bytes([SENTINEL; 4]),
                        "{tag}: output stride padding was written"
                    );
                }
            }
        }
    }
}

fn production_k(kind: Kind) -> usize {
    if kind == Kind::GateUp {
        2560
    } else {
        640
    }
}

fn run_arch(arch: Arch) {
    let rig = Rig::new(arch);
    for codec in Codec::ALL {
        for kind in Kind::ALL {
            let k = if arch == Arch::Gfx1151 {
                production_k(kind)
            } else if codec == Codec::Mq4Sym && kind == Kind::GateUp {
                1280
            } else {
                640
            };
            let sweep = Fixture::new(
                &rig,
                codec,
                kind,
                Case {
                    c: 64,
                    m: if kind == Kind::GateUp { 2 } else { 3 },
                    k,
                    routing: Routing::Pool(17),
                    pad: true,
                    seed: 0x5a17,
                },
            );
            for c in 1..=64 {
                run_case(&rig, &sweep, c, None);
            }
            // n=8 with fewer than eight active outputs requires inactive waves
            // to stage live slots and participate in all barriers. Also cover
            // a second output block and its ragged last waves.
            for (i, (c, m, routing, pad)) in [
                (9, 8, Routing::Identical, false),
                (
                    17,
                    if kind == Kind::GateUp { 10 } else { 9 },
                    Routing::Disjoint,
                    true,
                ),
                (
                    8,
                    if kind == Kind::GateUp { 6 } else { 1 },
                    Routing::Identical,
                    true,
                ),
                (
                    1,
                    if kind == Kind::GateUp { 4 } else { 7 },
                    Routing::Pool(14),
                    false,
                ),
            ]
            .into_iter()
            .enumerate()
            {
                let k = if arch == Arch::Gfx1151 {
                    production_k(kind)
                } else {
                    let group = if codec == Codec::Mq4Sym && kind == Kind::GateUp {
                        256
                    } else {
                        128
                    };
                    group * [1, 11, 4, 7][i]
                };
                let extra = Fixture::new(
                    &rig,
                    codec,
                    kind,
                    Case {
                        c,
                        m,
                        k,
                        routing,
                        pad,
                        seed: 0x781 + i as u64,
                    },
                );
                if matches!(routing, Routing::Disjoint) {
                    // Full 80-expert/80-slot bucket: exercise u=79 and the
                    // inclusive terminal offset80 at BOTH pointer placements.
                    run_case(&rig, &extra, c, Some(0));
                    run_case(&rig, &extra, c, Some(1));
                } else {
                    run_case(&rig, &extra, c, None);
                }
            }
        }
    }
}

/// Guard tests map descriptor+kernarg only: the weight table, tensors, X and Y are
/// all unmapped. Returning successfully is evidence that the whole invalid group
/// exited before any such memory access (in particular before the table dereference);
/// even a zero-valued store or an otherwise harmless invalid-wave load fails, and any
/// table-before-guard ordering faults on the unmapped table address.
///
/// `slot_ids: None` maps only the live id word and offset pair as two tiny regions
/// and leaves slots unmapped, so success additionally proves the offset guard fires
/// before any slot access; `Some` maps the full physical tile.
fn guarded_group(
    rig: &Rig,
    codec: Codec,
    kind: Kind,
    b: usize,
    tag: &str,
    c: usize,
    u: usize,
    wg: [u32; 3],
    eid: u32,
    off: [u32; 2],
    slot_ids: Option<&[u32]>,
) {
    let mut mem = Memory::default();
    let [ids, offs, slots] = bucket_ptrs(b);
    match slot_ids {
        Some(sids) => {
            let mut words = vec![0u32; TILE_WORDS];
            words[b * 80] = eid;
            words[160 + b * 81..162 + b * 81].copy_from_slice(&off);
            for (i, &sid) in sids.iter().enumerate() {
                words[322 + b * 80 + i] = sid;
            }
            // If the y guard is missing, u=80 sees a live, valid trap group in the
            // neighbouring physical fields rather than accidentally exiting for n=0.
            if wg[1] == 80 {
                words[b * 80 + 80] = 0;
                words[160 + b * 81 + 80] = 0;
                words[160 + b * 81 + 81] = 1;
            }
            mem.map("descriptor".into(), IDS, u32s(&words)).unwrap();
        }
        None => {
            mem.map("ids".into(), ids, u32s(&[eid])).unwrap();
            mem.map("offs".into(), offs, u32s(&off)).unwrap();
            if wg[1] == 80 {
                mem.map("trap_ids".into(), ids + 320, u32s(&[0])).unwrap();
                mem.map("trap_offs".into(), offs + 320, u32s(&[0, 1]))
                    .unwrap();
            }
        }
    }
    // No table mapping: TABLE stays unmapped alongside tensors/X/Y, so a table
    // load before the offset/slot guards faults instead of accidentally succeeding.
    mem.map(
        "kernarg".into(),
        KERNARG,
        kernarg(
            [TABLE, ids, offs, slots, XBASE, Y0, Y1],
            if kind == Kind::GateUp { 2 } else { 1 },
            production_k(kind),
            u,
            c,
            if kind == Kind::GateUp { 10 } else { 1 },
            640,
            production_k(kind) + 4,
            5,
        ),
    )
    .unwrap();
    let before = mem.clone();
    run_group(&rig.tile_emu(codec, kind), &mut mem, tag, wg, 256);
    for r in &before.regions {
        assert_eq!(
            region(&mem, &r.name),
            r.bytes.as_slice(),
            "{tag}: modified {}",
            r.name
        );
        assert_eq!(
            mem.regions
                .iter()
                .find(|m| m.name == r.name)
                .unwrap()
                .written
                .count(),
            0,
            "{tag}: issued a store to {}",
            r.name
        );
    }
}

fn run_guards(arch: Arch) {
    let rig = Rig::new(arch);
    for codec in Codec::ALL {
        for kind in Kind::ALL {
            for b in 0..2 {
                let tag = tile_symbol(arch, codec, kind);
                let check = |name: &str, c, u, wg, eid, off, slots: &[u32]| {
                    guarded_group(
                        &rig,
                        codec,
                        kind,
                        b,
                        &format!("{tag} bucket{b} {name}"),
                        c,
                        u,
                        wg,
                        eid,
                        off,
                        Some(slots),
                    );
                };
                for c in [0, 65, u32::MAX as usize] {
                    check("C domain", c, 80, [0, 0, 0], 0, [0, 1], &[0]);
                }
                for u in [0, 79, 81, u32::MAX as usize] {
                    check("U domain", 8, u, [0, 0, 0], 0, [0, 1], &[0]);
                }
                for (c, z) in [(1, 1), (8, 1), (9, 2), (64, 8)] {
                    check("inactive z", c, 80, [0, 0, z], 0, [0, 1], &[0]);
                }
                check("inactive x", 8, 80, [1, 0, 0], 0, [0, 1], &[0]);
                check("inactive y", 8, 80, [0, 80, 0], 0, [0, 1], &[0]);
                for off in [[0, 0], [0, 9], [79, 81], [81, 82], [3, 2], [u32::MAX, 0]] {
                    check("count/offset", 8, 80, [0, 0, 0], 0, off, &[0; 9]);
                }
                for eid in [N_EXPERTS as u32, u32::MAX] {
                    check("expert extent", 8, 80, [0, 0, 0], eid, [0, 1], &[0]);
                }
                for bad in [80, u32::MAX] {
                    check("global slot n1", 8, 80, [0, 0, 0], 0, [0, 1], &[bad]);
                    // Last live slot must be checked, not just the staging wave's first.
                    let mut slots = [0, 10, 20, 30, 40, 50, 60, 70];
                    slots[7] = bad;
                    check("global slot n8", 8, 80, [0, 0, 0], 0, [0, 8], &slots);
                }
            }
        }
    }
}

#[test]
fn tiled_k1_matches_singleton_rows_1_through_64_on_gfx1151() {
    run_arch(Arch::Gfx1151);
}

#[test]
fn tiled_k1_matches_singleton_rows_1_through_64_on_gfx1201() {
    run_arch(Arch::Gfx1201);
}

#[test]
fn tiled_k1_rejects_malformed_descriptors_and_geometry_on_gfx1151() {
    run_guards(Arch::Gfx1151);
}

#[test]
fn tiled_k1_rejects_malformed_descriptors_and_geometry_on_gfx1201() {
    run_guards(Arch::Gfx1201);
}
/// R1 span guard before any downstream dereference: C=1, offsets [8,11] (count 3
/// is capacity-valid but off1=11 exceeds 10*C=10). Maps ONLY the live id word and
/// offset pair; slots stay unmapped, so success proves the offset guard itself fires
/// before any slot or table access. Pre-fix table-before-guard ordering faults on
/// TABLE; post-fix the offset exit returns cleanly with immutable bytes and zero writes.
fn run_r1_bad_span_gfx1201() {
    let rig = Rig::new(Arch::Gfx1201);
    for codec in Codec::ALL {
        for kind in Kind::ALL {
            for b in 0..2 {
                let tag = format!(
                    "{} bucket{b} r1 bad span",
                    tile_symbol(Arch::Gfx1201, codec, kind)
                );
                guarded_group(
                    &rig,
                    codec,
                    kind,
                    b,
                    &tag,
                    1,
                    80,
                    [0, 0, 0],
                    0,
                    [8, 11],
                    None,
                );
            }
        }
    }
}

/// Per-slot guard before the table dereference: C=64, count 8, each position of an
/// otherwise valid [0,10,..,70] replaced by 640 (=10*C, out of window) and u32::MAX,
/// both buckets and all four codec/kind specs, table genuinely unmapped. Pre-fix the
/// table load faults; post-fix the slot exit returns cleanly.
fn run_each_invalid_slot_gfx1201() {
    let rig = Rig::new(Arch::Gfx1201);
    for codec in Codec::ALL {
        for kind in Kind::ALL {
            for b in 0..2 {
                for pos in 0..8 {
                    for bad in [640u32, u32::MAX] {
                        let mut slots = [0u32, 10, 20, 30, 40, 50, 60, 70];
                        slots[pos] = bad;
                        let tag = format!(
                            "{} bucket{b} bad slot pos{pos}={bad}",
                            tile_symbol(Arch::Gfx1201, codec, kind)
                        );
                        guarded_group(
                            &rig,
                            codec,
                            kind,
                            b,
                            &tag,
                            64,
                            80,
                            [0, 0, 0],
                            0,
                            [0, 8],
                            Some(&slots),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn tiled_k1_rejects_r1_bad_span_before_downstream_dereference_on_gfx1201() {
    run_r1_bad_span_gfx1201();
}

#[test]
fn tiled_k1_rejects_each_invalid_slot_before_table_dereference_on_gfx1201() {
    run_each_invalid_slot_gfx1201();
}

/// Dead-z boundary (CORRECTED fixture, PERMANENT): exactly one all-zero 1928 B
/// tile mapped — NO extra prefix/suffix guards — db = logical allocation base,
/// next tile unmapped, table/X NULL, real poisoned Y outputs. z0 (empty counts)
/// and z1 (past C=8) must both exit silently with bytes and written sets
/// unchanged, across z0, z1, reverse and repeat continuation.
fn dead_z_silent(rig: &Rig, codec: Codec, kind: Kind, b: usize, tag: &str) {
    dead_z_silent_emu(&rig.tile_emu(codec, kind), kind, b, tag);
}

fn dead_z_silent_emu(emu: &Emulator, kind: Kind, b: usize, tag: &str) {
    let mut mem = Memory::default();
    mem.map("descriptor".into(), IDS, vec![0u8; TILE_BYTES])
        .unwrap();
    assert!(
        mem.read(IDS + TILE_BYTES as u64, 4).is_err(),
        "{tag}: next tile must stay unmapped"
    );
    let gate = kind == Kind::GateUp;
    let (m, k, ys) = (
        if gate { 1280 } else { 2560 },
        production_k(kind),
        if gate { 640 } else { 2560 },
    );
    let [ids, offs, slots] = bucket_ptrs(b);
    mem.map(
        "kernarg".into(),
        KERNARG,
        kernarg(
            [
                0,
                ids,
                offs,
                slots,
                0,
                Y0 + GUARD as u64,
                if gate { Y1 + GUARD as u64 } else { 0 },
            ],
            m,
            k,
            80,
            8,
            if gate { 10 } else { 1 },
            8,
            k,
            ys,
        ),
    )
    .unwrap();
    // Native exact output extent: y_len_for(650, ys) = 650*ys*4 + 2*GUARD,
    // FILL_A 0xa5 poison, pointer Ybase+64; both Y mapped for gate, Y1
    // NULL/unmapped for down.
    let y_len = 650 * ys * 4 + 2 * GUARD;
    mem.map("y0".into(), Y0, vec![SENTINEL; y_len]).unwrap();
    if gate {
        mem.map("y1".into(), Y1, vec![SENTINEL; y_len]).unwrap();
    }
    let before = mem.clone();
    for wg in [[0, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 0]] {
        run_group(&emu, &mut mem, &format!("{tag} z{}", wg[2]), wg, 256);
        for r in &before.regions {
            assert_eq!(
                region(&mem, &r.name),
                r.bytes.as_slice(),
                "{tag} z{}: modified {}",
                wg[2],
                r.name
            );
            assert_eq!(
                mem.regions
                    .iter()
                    .find(|m| m.name == r.name)
                    .unwrap()
                    .written
                    .count(),
                0,
                "{tag} z{}: issued a store to {}",
                wg[2],
                r.name
            );
        }
    }
}

#[test]
fn dead_z_corrected_stays_silent_on_gfx1201() {
    let rig = Rig::new(Arch::Gfx1201);
    for codec in Codec::ALL {
        for kind in Kind::ALL {
            for b in 0..2 {
                dead_z_silent(
                    &rig,
                    codec,
                    kind,
                    b,
                    &format!(
                        "{} bucket{b} dead-z",
                        tile_symbol(Arch::Gfx1201, codec, kind)
                    ),
                );
            }
        }
    }
}
