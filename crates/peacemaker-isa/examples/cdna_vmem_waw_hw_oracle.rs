// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx942 (MI300X) hardware ordering oracle for VMEM load -> VMEM load write-after-write (WAW) on
//! overlapping VGPR destinations with NO `s_waitcnt` between the two loads.
//!
//! Claim under test: vector-memory loads of one wave of the same VMEM type class (untyped
//! GLOBAL and MUBUF buffer loads, LLVM's non-sampler `VMEM_READ_ACCESS` class) return in issue order, so when an
//! older and a newer load share a destination dword the NEWER (last-issued) value is what remains for
//! every lane the newer load was active, and lanes active only in the older load (or in neither) keep
//! the older load's value (or the preserved prior value). Lane masks are stress only; the ledger rule
//! does not track lanes.
//!
//! Families: `g` = `global_load_*` (`off`), `b` = `buffer_load_*` (MUBUF, `offen`, raw descriptor),
//! `f` = `flat_load_*` to global addresses. Pairs are same-family and mixed (g->b, b->g, ...). The flat
//! rows are INFORMATIONAL and non-gating: FLAT may route to LDS (vmcnt+lgkmcnt), LLVM does not treat it
//! as in-order, and the LDS-aperture case that would break the order is deliberately not exercised
//! (unsafe/negative). Image/sampler/BVH VMEM classes and scratch are not covered (no descriptors, no
//! scratch setup); the receipt lists them under `not_covered`.
//!
//! Race design: scenarios 128..255 deliberately pit a slow older load (fresh addresses spread over a
//! 248 MiB cold region that is re-picked per launch, host-side 256 MiB write thrash before each launch,
//! sc0/sc1/nt bypass policies) against a fast newer load on a few repeated, deliberately primed hot lines
//! (primed by a full-exec load + `s_waitcnt vmcnt(0)` before the pair), and the reverse. Cold-ness is best
//! effort; it is evidenced, not assumed, by a load-latency calibration (`s_memtime` around one wave load,
//! 256 trials: randomized cold lines vs primed lines, default vs sc0/sc1 bypass) that gates `status`
//! (median cold >= 1.25x median primed hit) and is bound in the receipt together with the cold region size,
//! the cold/primed line counts per launch and the cold-older/primed-newer lane-pair counts. Every kernel is
//! launched in 512 blocks (one wave each), twice (two output poisons / two old-value seeds), with
//! per-launch re-randomized scenarios.
//!
//! Harness: hipcc compiles one HIP kernel per (pair, queue distance, cache policy). Each kernel is a
//! single `asm volatile` block that pins staging VGPRs `v32..v38`/`v64..` by explicit clobbers and
//! issues `OLDER; [D filler global_load_dword]; NEWER` with `s_mov_b64 exec` lane-mask switches but no
//! wait in between, then one full drain, then stores the destination group. Before the pair the staging
//! group is loaded from a per-lane "old" value (inactive-lane preservation) and drained.
//!
//! Proof obligations (any miss => `status: FAIL`):
//! * the device ELF is extracted from the hipcc bundle with `BundleCodec`, hashed, lifted by the
//!   PeaceMaker gfx942 codec (`peacemaker_lift::cdna::lift_object`, byte-identity checked) and that same
//!   ELF is loaded and launched;
//! * the typed disassembly of every kernel has the exact pair (mnemonics, destination tuples, cache bits,
//!   `offen`), exactly `D` filler loads and only `s_mov_b64 exec` between them, a full drain between the
//!   old-value preload and the older load, and the first wait after the newer load is the full drain
//!   before the stores;
//! * device output equals the host last-issued-wins model byte for byte (padding, guards and filler
//!   checksums included);
//! * mutated and reversed-order expectations are detected, and output/expected hashes are bound in the
//!   receipt.
//!
//! Run only inside the MI300X fabric wrapper:
//! `cargo run -p peacemaker-isa --features lab --example cdna_vmem_waw_hw_oracle -- OUT_DIR`
//! (writes `OUT_DIR/receipt.json`, `waw.hip`, `waw.hsaco`, `waw.elf`).
use hip_bridge::HipRuntime;
use peacemaker_ir::envelope::Bundle;
use peacemaker_lift::bundle::{BundleCodec, HIP_DEVICE_TRIPLE_PREFIX};
use peacemaker_lift::cdna::lift_object;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const BLOCKS: usize = 512; // one wave64 per block
const CAL_BLOCKS: usize = 256; // latency-calibration trials per configuration
const NSCN: usize = 256; // 16 mask classes x 16 address configurations (8 hot, 8 cold/hot races)
const HOT: usize = 8 << 20; // stored host-side
const TOTAL: usize = 256 << 20; // device data buffer; bytes past HOT are generated, never stored
const GUARD: usize = 64;
const LANE_BYTES: usize = 32; // 4 value dwords, 1 filler checksum dword, 3 untouched pad dwords
const POISONS: [u8; 2] = [0x5a, 0xa5];
const DISTS: [usize; 3] = [0, 8, 32];
const SEED: u64 = 0x5eed_0fa1_1c0d_e7e5;
const MAX_FAIL_NOTES: usize = 32;
const EF_AMDGPU_MACH_GFX942: u32 = 0x4c;
const EDGES: [u32; 18] = [
    0x7fc0_0000,
    0xffc0_0000,
    0x7f80_0000,
    0xff80_0000,
    0x7f80_0001,
    0xff80_0001,
    0x7fff_ffff,
    0xffff_ffff,
    0x0000_0001,
    0x8000_0001,
    0x007f_ffff,
    0x807f_ffff,
    0x0000_0000,
    0x8000_0000,
    0x3f80_0000,
    0xbf80_0000,
    0x7f7f_ffff,
    0xff7f_ffff,
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Fam {
    G,
    B,
    F,
}
impl Fam {
    fn pre(self) -> &'static str {
        match self {
            Fam::G => "global",
            Fam::B => "buffer",
            Fam::F => "flat",
        }
    }
    fn ch(self) -> char {
        match self {
            Fam::G => 'g',
            Fam::B => 'b',
            Fam::F => 'f',
        }
    }
}

#[derive(Clone, Copy)]
struct Kind {
    tag: &'static str,
    bytes: u32,
    w: u32,
    signed: bool,
}
const UB: Kind = Kind {
    tag: "ubyte",
    bytes: 1,
    w: 1,
    signed: false,
};
const SB: Kind = Kind {
    tag: "sbyte",
    bytes: 1,
    w: 1,
    signed: true,
};
const US: Kind = Kind {
    tag: "ushort",
    bytes: 2,
    w: 1,
    signed: false,
};
const SS: Kind = Kind {
    tag: "sshort",
    bytes: 2,
    w: 1,
    signed: true,
};
const DW: Kind = Kind {
    tag: "dword",
    bytes: 4,
    w: 1,
    signed: false,
};
const X2: Kind = Kind {
    tag: "dwordx2",
    bytes: 8,
    w: 2,
    signed: false,
};
const X4: Kind = Kind {
    tag: "dwordx4",
    bytes: 16,
    w: 4,
    signed: false,
};
fn mnem(f: Fam, k: &Kind) -> String {
    format!("{}_load_{}", f.pre(), k.tag)
}

/// Older load, its destination offset (dwords) inside the `v[32:35]` staging group, newer load, same.
/// gfx942 requires even-aligned 64/128-bit VGPR tuples: x2 only at 0/2, x4 only at 0.
struct Pair {
    of: Fam,
    older: Kind,
    ooff: u32,
    nf: Fam,
    newer: Kind,
    noff: u32,
}
impl Pair {
    fn overlaps(&self) -> bool {
        self.ooff < self.noff + self.newer.w && self.noff < self.ooff + self.older.w
    }
    fn tag(&self) -> String {
        format!(
            "{}{}{}_{}{}{}",
            self.of.ch(),
            self.older.tag,
            self.ooff,
            self.nf.ch(),
            self.newer.tag,
            self.noff
        )
    }
    fn class(&self) -> String {
        format!("{}_{}", self.of.pre(), self.nf.pre())
    }
    fn gating(&self) -> bool {
        self.of != Fam::F && self.nf != Fam::F
    }
}
const fn pr(of: Fam, older: Kind, ooff: u32, nf: Fam, newer: Kind, noff: u32) -> Pair {
    Pair {
        of,
        older,
        ooff,
        nf,
        newer,
        noff,
    }
}
const NGLOBAL: usize = 26;
static PAIRS: [Pair; 49] = [
    // ---- global -> global ----
    pr(Fam::G, DW, 0, Fam::G, DW, 0),
    pr(Fam::G, DW, 1, Fam::G, DW, 1),
    pr(Fam::G, DW, 3, Fam::G, DW, 3),
    pr(Fam::G, UB, 0, Fam::G, UB, 0),
    pr(Fam::G, US, 2, Fam::G, US, 2),
    pr(Fam::G, SB, 1, Fam::G, SB, 1),
    pr(Fam::G, SS, 0, Fam::G, SS, 0),
    pr(Fam::G, US, 0, Fam::G, DW, 0),
    pr(Fam::G, DW, 0, Fam::G, UB, 0),
    pr(Fam::G, UB, 0, Fam::G, US, 0),
    pr(Fam::G, SB, 2, Fam::G, DW, 2),
    pr(Fam::G, X2, 0, Fam::G, X2, 0),
    pr(Fam::G, X2, 2, Fam::G, X2, 2),
    pr(Fam::G, X2, 0, Fam::G, DW, 0),
    pr(Fam::G, X2, 0, Fam::G, DW, 1),
    pr(Fam::G, DW, 1, Fam::G, X2, 0),
    pr(Fam::G, DW, 0, Fam::G, X2, 0),
    pr(Fam::G, X2, 2, Fam::G, UB, 3),
    pr(Fam::G, X4, 0, Fam::G, X4, 0),
    pr(Fam::G, X4, 0, Fam::G, DW, 0),
    pr(Fam::G, X4, 0, Fam::G, DW, 3),
    pr(Fam::G, DW, 2, Fam::G, X4, 0),
    pr(Fam::G, X4, 0, Fam::G, X2, 2),
    pr(Fam::G, X2, 0, Fam::G, X4, 0),
    pr(Fam::G, US, 1, Fam::G, X2, 0),
    // Disjoint-destination control (no WAW): both results must survive.
    pr(Fam::G, X2, 0, Fam::G, X2, 2),
    // ---- buffer -> buffer ----
    pr(Fam::B, DW, 0, Fam::B, DW, 0),
    pr(Fam::B, UB, 0, Fam::B, US, 0),
    pr(Fam::B, X2, 0, Fam::B, DW, 1),
    pr(Fam::B, DW, 1, Fam::B, X2, 0),
    pr(Fam::B, X4, 0, Fam::B, DW, 3),
    pr(Fam::B, DW, 2, Fam::B, X4, 0),
    pr(Fam::B, X2, 2, Fam::B, X2, 2),
    // ---- global -> buffer ----
    pr(Fam::G, DW, 0, Fam::B, DW, 0),
    pr(Fam::G, DW, 1, Fam::B, X2, 0),
    pr(Fam::G, X2, 0, Fam::B, DW, 1),
    pr(Fam::G, US, 0, Fam::B, DW, 0),
    // ---- buffer -> global ----
    pr(Fam::B, DW, 0, Fam::G, DW, 0),
    pr(Fam::B, DW, 1, Fam::G, X2, 0),
    pr(Fam::B, X2, 0, Fam::G, DW, 1),
    pr(Fam::B, US, 0, Fam::G, DW, 0),
    // ---- flat (informational, non-gating) ----
    pr(Fam::F, DW, 0, Fam::F, DW, 0),
    pr(Fam::F, X2, 0, Fam::F, DW, 1),
    pr(Fam::F, DW, 1, Fam::F, X2, 0),
    pr(Fam::F, UB, 0, Fam::F, US, 0),
    pr(Fam::G, DW, 0, Fam::F, DW, 0),
    pr(Fam::G, X2, 0, Fam::F, DW, 1),
    pr(Fam::F, DW, 0, Fam::G, DW, 0),
    pr(Fam::F, X2, 0, Fam::G, DW, 1),
];

/// (tag, cache bits of the older load, cache bits of the newer load)
const POLICIES: [(&str, &str, &str); 4] = [
    ("dd", "", ""),
    ("bd", "sc0 sc1", ""),
    ("db", "", "sc0 sc1"),
    ("nn", "nt", "nt"),
];

struct Spec {
    name: String,
    pair: usize,
    dist: usize,
    pol: usize,
}

fn specs() -> Vec<Spec> {
    let mut v = Vec::new();
    for (pi, p) in PAIRS.iter().enumerate() {
        for &dist in &DISTS {
            for (qi, q) in POLICIES.iter().enumerate() {
                // The full grid for global pairs; the other families skip the `nn` policy.
                if pi >= NGLOBAL && qi == 3 {
                    continue;
                }
                v.push(Spec {
                    name: format!("waw_{pi:02}_{}_d{dist}_{}", p.tag(), q.0),
                    pair: pi,
                    dist,
                    pol: qi,
                });
            }
        }
    }
    v
}

fn sha(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(mix(seed) | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

fn reg(base: u32, w: u32) -> String {
    if w <= 1 {
        format!("v{base}")
    } else {
        format!("v[{base}:{}]", base + w - 1)
    }
}

/// The full-drain wait: flat loads also count on lgkmcnt, so pairs with a flat side drain both.
fn drain(p: &Pair) -> &'static str {
    if p.of == Fam::F || p.nf == Fam::F {
        "s_waitcnt vmcnt(0) lgkmcnt(0)"
    } else {
        "s_waitcnt vmcnt(0)"
    }
}

// ───────────────────────────── HIP source ─────────────────────────────

/// One pair load. Operands: %1/%2 global pointers (older/newer), %10/%11 buffer voffsets, %12 resource.
fn mem_op(f: Fam, k: &Kind, dst: u32, newer: bool, pol: &str) -> String {
    let m = mnem(f, k);
    let d = reg(dst, k.w);
    match f {
        Fam::G => format!("{m} {d}, {}, off {pol}", if newer { "%2" } else { "%1" }),
        Fam::F => format!("{m} {d}, {} {pol}", if newer { "%2" } else { "%1" }),
        Fam::B => format!(
            "{m} {d}, {}, %12, 0 offen {pol}",
            if newer { "%11" } else { "%10" }
        ),
    }
}

fn asm_lines(sp: &Spec) -> Vec<String> {
    let p = &PAIRS[sp.pair];
    let pol = POLICIES[sp.pol];
    // Operands: %0 = saved exec (out); %1 pa %2 pb %3 old %4 filler %5 out %10 oa %11 ob (v);
    // %6 maskA %7 maskB %8 warmA %9 warmB %12 rsrc (s).
    let mut a: Vec<String> = vec![
        "s_mov_b64 %0, exec".into(),
        // Cache priming of the pair's lines (masked all-or-nothing); results discarded.
        "s_mov_b64 exec, %8".into(),
        "global_load_ubyte v37, %1, off".into(),
        "s_mov_b64 exec, %9".into(),
        "global_load_ubyte v38, %2, off".into(),
        "s_mov_b64 exec, %0".into(),
        // Prior value of the staging group (what inactive lanes must keep), then a full drain.
        "global_load_dwordx4 v[32:35], %3, off".into(),
        drain(p).into(),
        // ---- the pair: no wait from here to the next full drain ----
        "s_mov_b64 exec, %6".into(),
        mem_op(p.of, &p.older, 32 + p.ooff, false, pol.1),
        "s_mov_b64 exec, %0".into(),
    ];
    for k in 0..sp.dist {
        a.push(format!(
            "global_load_dword v{}, %4, off offset:{}",
            64 + k,
            k * 64
        ));
    }
    a.push("s_mov_b64 exec, %7".into());
    a.push(mem_op(p.nf, &p.newer, 32 + p.noff, true, pol.2));
    a.push("s_mov_b64 exec, %0".into());
    // ---- first wait after the pair ----
    a.push(drain(p).into());
    a.push("v_mov_b32 v36, 0".into());
    for k in 0..sp.dist {
        a.push(format!("v_xor_b32 v36, v36, v{}", 64 + k));
    }
    a.push("global_store_dwordx4 %5, v[32:35], off".into());
    a.push("global_store_dword %5, v36, off offset:16".into());
    a.push("s_waitcnt vmcnt(0)".into());
    a
}

fn source(specs: &[Spec]) -> String {
    let mut s = String::from(
        "#include <hip/hip_runtime.h>\ntypedef unsigned u4 __attribute__((ext_vector_type(4)));\n",
    );
    for sp in specs {
        let asm = asm_lines(sp).join("\\n\\t");
        let mut clob = String::from("\"memory\"");
        for r in 32..=38 {
            clob.push_str(&format!(", \"v{r}\""));
        }
        for k in 0..sp.dist {
            clob.push_str(&format!(", \"v{}\"", 64 + k));
        }
        s.push_str(&format!(
            "extern \"C\" __global__ void __launch_bounds__(64) {name}(const char* data, const unsigned* old, const unsigned* tbl, const unsigned long long* msk, char* out) {{\n\
             const unsigned b = blockIdx.x, l = threadIdx.x;\n\
             const unsigned oa = tbl[b * 128u + l], ob = tbl[b * 128u + 64u + l];\n\
             const char* pa = data + oa;\n\
             const char* pb = data + ob;\n\
             const char* po = (const char*)old + (size_t)(b * 64u + l) * 16u;\n\
             const char* pf = data + msk[b * 5u + 4u] + (size_t)l * 4u;\n\
             char* pw = out + (size_t)(b * 64u + l) * {stride}u;\n\
             const unsigned long long ma = msk[b * 5u], mb = msk[b * 5u + 1u], wa = msk[b * 5u + 2u], wb = msk[b * 5u + 3u];\n\
             const unsigned long long db = (unsigned long long)data;\n\
             const u4 rs = {{(unsigned)db, (unsigned)(db >> 32) & 0xffffu, {total}u, 0x00020000u}};\n\
             unsigned long long sv;\n\
             asm volatile(\"{asm}\" : \"=&s\"(sv) : \"v\"(pa), \"v\"(pb), \"v\"(po), \"v\"(pf), \"v\"(pw), \"s\"(ma), \"s\"(mb), \"s\"(wa), \"s\"(wb), \"v\"(oa), \"v\"(ob), \"s\"(rs) : {clob});\n\
             }}\n",
            name = sp.name,
            stride = LANE_BYTES,
            total = TOTAL,
        ));
    }
    // Load-latency calibration kernels: prime (masked all-or-nothing), drain, then one timed wave load.
    for (tag, bits) in [("dd", ""), ("bp", "sc0 sc1")] {
        s.push_str(&format!(
            "extern \"C\" __global__ void __launch_bounds__(64) cal_{tag}(const char* data, const unsigned* tbl, const unsigned long long* msk, unsigned long long* res) {{\n\
             const unsigned b = blockIdx.x, l = threadIdx.x;\n\
             const char* pa = data + tbl[b * 64u + l];\n\
             const unsigned long long wm = msk[b];\n\
             unsigned long long sv, t0, t1;\n\
             asm volatile(\"s_mov_b64 %0, exec\\n\\ts_mov_b64 exec, %4\\n\\tglobal_load_ubyte v37, %3, off\\n\\ts_mov_b64 exec, %0\\n\\ts_waitcnt vmcnt(0)\\n\\ts_memtime %1\\n\\ts_waitcnt lgkmcnt(0)\\n\\tglobal_load_dword v38, %3, off {bits}\\n\\ts_waitcnt vmcnt(0)\\n\\ts_memtime %2\\n\\ts_waitcnt lgkmcnt(0)\" : \"=&s\"(sv), \"=&s\"(t0), \"=&s\"(t1) : \"v\"(pa), \"s\"(wm) : \"memory\", \"v37\", \"v38\");\n\
             if (l == 0) {{ res[b * 2u] = t0; res[b * 2u + 1u] = t1; }}\n\
             }}\n"
        ));
    }
    s
}

// ───────────────────────────── device image + disassembly proof ─────────────────────────────

/// The gfx942 device ELF of `bytes`: the single `hipv4-amdgcn-amd-amdhsa--gfx942` entry of the offload
/// bundle (read with `BundleCodec`), or the bytes themselves when they are already a bare ELF.
fn device_image(bytes: &[u8]) -> Result<&[u8], String> {
    let elf: &[u8] = if Bundle::is_bundle(bytes) {
        let (bundle, payloads) = Bundle::read(bytes).map_err(|e| format!("bundle: {e}"))?;
        let exact = format!("{HIP_DEVICE_TRIPLE_PREFIX}gfx942");
        let hits: Vec<usize> = bundle
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                e.triple
                    .strip_prefix(&exact)
                    .is_some_and(|rest| rest.is_empty() || rest.starts_with(':'))
            })
            .map(|(i, _)| i)
            .collect();
        match hits[..] {
            [i] => payloads[i],
            _ => {
                return Err(format!(
                    "bundle has {} `{exact}` entries (want 1); entries {:?}",
                    hits.len(),
                    bundle.entries.iter().map(|e| &e.triple).collect::<Vec<_>>()
                ))
            }
        }
    } else {
        bytes
    };
    if elf.len() < 64 || &elf[..4] != b"\x7fELF" || elf[4] != 2 || elf[5] != 1 {
        return Err("device image is not a 64-bit little-endian ELF".into());
    }
    let machine = u16::from_le_bytes([elf[18], elf[19]]);
    let flags = u32::from_le_bytes([elf[48], elf[49], elf[50], elf[51]]);
    if machine != 224 || flags & 0xff != EF_AMDGPU_MACH_GFX942 {
        return Err(format!(
            "device image is not an AMDGPU gfx942 code object (e_machine {machine}, e_flags {flags:#x})"
        ));
    }
    Ok(elf)
}

/// A parsed `global_load_*` / `flat_load_*` / `buffer_load_*` text line.
struct Ld {
    mnem: String,
    dst: String,
    addr: String,
    mods: Vec<String>,
    offset: Option<String>,
}
fn parse_ld(t: &str) -> Option<Ld> {
    let tk: Vec<&str> = t.split_whitespace().collect();
    if tk.len() < 3 {
        return None;
    }
    // Index of the first modifier token: global `dst, addr, off`; buffer `dst, voff, rsrc, soff`; flat `dst, addr`.
    let first = if tk[0].starts_with("global_load") {
        if tk.len() < 4 || tk[3] != "off" {
            return None;
        }
        4
    } else if tk[0].starts_with("buffer_load") {
        if tk.len() < 5 {
            return None;
        }
        5
    } else if tk[0].starts_with("flat_load") {
        3
    } else {
        return None;
    };
    let mut mods = Vec::new();
    let mut offset = None;
    for m in &tk[first..] {
        if m.starts_with("offset:") {
            offset = Some((*m).to_string());
        } else {
            mods.push((*m).to_string());
        }
    }
    mods.sort_unstable();
    Some(Ld {
        mnem: tk[0].to_string(),
        dst: tk[1].trim_end_matches(',').to_string(),
        addr: tk[2].trim_end_matches(',').to_string(),
        mods,
        offset,
    })
}
fn vstart(t: &str) -> Option<u32> {
    let t = t.strip_prefix('v')?;
    let t = t.strip_prefix('[').unwrap_or(t);
    let n: String = t.chars().take_while(|c| c.is_ascii_digit()).collect();
    n.parse().ok()
}
fn mod_set(s: &str, extra: &[&str]) -> Vec<String> {
    let mut v: Vec<String> = s
        .split_whitespace()
        .chain(extra.iter().copied())
        .map(str::to_string)
        .collect();
    v.sort_unstable();
    v
}

struct Proof {
    window: Vec<String>,
    disasm_sha256: String,
    l0: usize,
    l1: usize,
    l2: usize,
    instructions: usize,
}

fn prove(sp: &Spec, texts: &[String]) -> Result<Proof, String> {
    let p = &PAIRS[sp.pair];
    let pol = POLICIES[sp.pol];
    let wait = drain(p);
    let loads: Vec<(usize, Ld)> = texts
        .iter()
        .enumerate()
        .filter_map(|(i, t)| parse_ld(t).map(|l| (i, l)))
        .filter(|(_, l)| vstart(&l.dst).is_some_and(|s| (32..36).contains(&s)))
        .collect();
    if loads.len() < 3 {
        return Err(format!(
            "{}: only {} loads into v32..v35",
            sp.name,
            loads.len()
        ));
    }
    let (l0, ld0) = &loads[loads.len() - 3];
    let (l1, ld1) = &loads[loads.len() - 2];
    let (l2, ld2) = &loads[loads.len() - 1];
    let (l0, l1, l2) = (*l0, *l1, *l2);
    let want =
        |ld: &Ld, fam: Fam, k: &Kind, dst: String, bits: &str, what: &str| -> Result<(), String> {
            let extra: &[&str] = if fam == Fam::B { &["offen"] } else { &[] };
            let (m, ms) = (mnem(fam, k), mod_set(bits, extra));
            if ld.mnem != m || ld.dst != dst || ld.mods != ms {
                return Err(format!(
                    "{}: {what} is `{} {} {:?}`, want `{m} {dst} {ms:?}`",
                    sp.name, ld.mnem, ld.dst, ld.mods
                ));
            }
            let a = vstart(&ld.addr)
                .ok_or_else(|| format!("{}: {what} address `{}`", sp.name, ld.addr))?;
            if (32..96).contains(&a) || (32..96).contains(&(a + 1)) {
                return Err(format!(
                    "{}: {what} address {} collides with the staging VGPRs",
                    sp.name, ld.addr
                ));
            }
            if ld.offset.is_some() {
                return Err(format!("{}: {what} has an unexpected offset", sp.name));
            }
            Ok(())
        };
    want(ld0, Fam::G, &X4, reg(32, 4), "", "old-value preload")?;
    want(
        ld1,
        p.of,
        &p.older,
        reg(32 + p.ooff, p.older.w),
        pol.1,
        "older load",
    )?;
    want(
        ld2,
        p.nf,
        &p.newer,
        reg(32 + p.noff, p.newer.w),
        pol.2,
        "newer load",
    )?;
    // Preload -> drain -> older: only the full drain and exec switches.
    let mut drains = 0;
    for t in &texts[l0 + 1..l1] {
        if t == wait {
            drains += 1;
        } else if !t.starts_with("s_mov_b64 exec,") {
            return Err(format!("{}: `{t}` between preload and older load", sp.name));
        }
    }
    if drains != 1 {
        return Err(format!(
            "{}: {drains} waits between preload and older load (want 1 `{wait}`)",
            sp.name
        ));
    }
    // Older -> newer: no wait of any kind; only exec switches and exactly `dist` filler loads.
    let mut k = 0usize;
    for t in &texts[l1 + 1..l2] {
        if t.starts_with("s_mov_b64 exec,") {
            continue;
        }
        let f = parse_ld(t)
            .ok_or_else(|| format!("{}: `{t}` between the pair (wait or other)", sp.name))?;
        let off = if k == 0 {
            None
        } else {
            Some(format!("offset:{}", k * 64))
        };
        let a = vstart(&f.addr).unwrap_or(0);
        if f.mnem != "global_load_dword"
            || f.dst != format!("v{}", 64 + k)
            || !f.mods.is_empty()
            || f.offset != off
            || (32..96).contains(&a)
        {
            return Err(format!("{}: unexpected filler #{k}: `{t}`", sp.name));
        }
        k += 1;
    }
    if k != sp.dist {
        return Err(format!(
            "{}: {k} fillers between the pair, want {}",
            sp.name, sp.dist
        ));
    }
    // After the newer load: first non-exec instruction is the full drain, then the stores.
    let mut j = l2 + 1;
    while j < texts.len() && texts[j].starts_with("s_mov_b64 exec,") {
        j += 1;
    }
    if texts.get(j).map(String::as_str) != Some(wait) {
        return Err(format!(
            "{}: first instruction after the pair is `{:?}`, want `{wait}`",
            sp.name,
            texts.get(j)
        ));
    }
    let tail = &texts[j + 1..];
    let st4 = tail
        .iter()
        .any(|t| t.starts_with("global_store_dwordx4") && t.contains("v[32:35]"));
    let st1 = tail
        .iter()
        .any(|t| t.starts_with("global_store_dword ") && t.contains("v36"));
    if !st4 || !st1 {
        return Err(format!(
            "{}: result stores missing after the drain",
            sp.name
        ));
    }
    Ok(Proof {
        window: texts[l1..=l2].to_vec(),
        disasm_sha256: sha(texts.join("\n").as_bytes()),
        l0,
        l1,
        l2,
        instructions: texts.len(),
    })
}

// ───────────────────────────── data, scenarios, host model ─────────────────────────────

/// Dword `i` of the whole data buffer (hot and cold regions): unique-ish random values with
/// NaN/Inf/denormal/sign-edge patterns (and payload variants) sprinkled in.
fn dw(i: usize) -> u32 {
    let mut x = (mix(SEED ^ i as u64) >> 16) as u32;
    if i % 4099 == 17 {
        let q = i / 4099;
        let (e, round) = (EDGES[q % EDGES.len()], q / EDGES.len());
        x = if round == 0 {
            e
        } else {
            e ^ (1 << (round % 23))
        };
    }
    x
}
struct Data {
    hot: Vec<u8>,
}
impl Data {
    fn new() -> Self {
        let mut hot = Vec::with_capacity(HOT);
        for i in 0..HOT / 4 {
            hot.extend(dw(i).to_le_bytes());
        }
        Data { hot }
    }
    fn dword(&self, a: usize) -> u32 {
        if a + 4 <= HOT {
            u32::from_le_bytes([
                self.hot[a],
                self.hot[a + 1],
                self.hot[a + 2],
                self.hot[a + 3],
            ])
        } else {
            dw(a / 4)
        }
    }
    fn byte(&self, a: usize) -> u8 {
        if a < HOT {
            self.hot[a]
        } else {
            dw(a / 4).to_le_bytes()[a % 4]
        }
    }
}
fn make_old(pass: usize) -> Vec<u32> {
    (0..BLOCKS * 64 * 4)
        .map(|i| {
            let x = (mix(SEED ^ ((pass as u64 + 1) << 56) ^ i as u64) >> 8) as u32;
            if i % 5 == 0 {
                EDGES[(i / 5 + pass) % EDGES.len()] ^ (x & 0xff)
            } else {
                x
            }
        })
        .collect()
}
fn bytes_of(x: &[u32]) -> Vec<u8> {
    x.iter().flat_map(|x| x.to_le_bytes()).collect()
}

struct Scn {
    ma: u64,
    mb: u64,
    line_a: [u32; 64],
    line_b: [u32; 64],
    sub_a: u32,
    sub_b: u32,
    fill: u32,
    warm: u32,
    race: bool,
}

fn mask_pair(c: usize, r: &mut Rng) -> (u64, u64) {
    let sparse = |r: &mut Rng| {
        let mut m = 0u64;
        for _ in 0..1 + r.next() % 3 {
            m |= 1 << (r.next() % 64);
        }
        m
    };
    match c {
        0 => (!0, !0),
        1 => (0, 0),
        2 => (!0, 0),
        3 => (0, !0),
        4 => (1, 1),
        5 => (1 << 63, 1 << 63),
        6 => (1, 1 << 63),
        7 => (0xffff_ffff, !0xffff_ffffu64),
        8 => (0x5555_5555_5555_5555, 0xaaaa_aaaa_aaaa_aaaa),
        9 => (0x5555_5555_5555_5555, 0x5555_5555_5555_5555),
        10 => {
            let m = r.next();
            (m, m)
        }
        11 => (r.next(), r.next()),
        12 => {
            let a = r.next();
            (a, a & r.next())
        }
        13 => {
            let b = r.next();
            (b & r.next(), b)
        }
        14 => {
            let a = r.next();
            (a, !a)
        }
        _ => (sparse(r), sparse(r)),
    }
}

/// Per-lane 16-byte-aligned line offsets. Modes: 0 coalesced, 1 stride 64 B, 2 stride 4112 B,
/// 3 random in 256 KiB, 4 broadcast, 5 four repeated lines, 6 random across 64 MiB.
fn lines(mode: u32, base: u32, r: &mut Rng) -> [u32; 64] {
    let mut v = [0u32; 64];
    for (l, x) in v.iter_mut().enumerate() {
        let l = l as u32;
        *x = base
            + match mode {
                0 => l * 16,
                1 => l * 64,
                2 => l * 4112,
                3 => (r.next() % 16384) as u32 * 16,
                5 => (l % 4) * 64,
                6 => (r.next() % (1 << 22)) as u32 * 16,
                _ => 0,
            };
    }
    v
}

/// Scenario of block `b` in launch `rot`: 16 mask classes x 16 address configurations; configurations
/// 8..15 race a cold (fresh, wide) line set against a primed hot line set, in both directions.
fn scn(b: usize, rot: usize) -> Scn {
    let s = b % NSCN;
    let (c, k) = (s % 16, s / 16);
    let mut r = Rng::new(
        SEED ^ (s as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ (rot as u64 + 1).wrapping_mul(0xd1b5_4a32_d192_ed03),
    );
    let (ma, mb) = mask_pair(c, &mut r);
    let hot_span = (HOT - 0x8_0000) as u64;
    let cold_span = (TOTAL - HOT - (64 << 20) - 0x8_0000) as u64;
    let hot = |r: &mut Rng| ((r.next() % hot_span) as u32) & !0xfff;
    let cold = |r: &mut Rng| (HOT as u32 + (r.next() % cold_span) as u32) & !0xfff;
    let (ba, bb) = (hot(&mut r), hot(&mut r));
    let sub_a = (r.next() % 16) as u32;
    let sub_rand = (r.next() % 16) as u32;
    let (line_a, line_b, sub_b);
    let mut warm = (mix(SEED ^ 0xabcd ^ (b as u64) ^ ((rot as u64) << 20)) >> 7) as u32 & 3;
    match k {
        0 => {
            line_a = lines(0, ba, &mut r);
            line_b = line_a;
            sub_b = sub_a;
        }
        1 => {
            line_a = lines(0, ba, &mut r);
            line_b = line_a;
            sub_b = (sub_a + 1 + sub_rand % 15) % 16;
        }
        2 => {
            line_a = lines(0, ba, &mut r);
            line_b = lines(0, bb, &mut r);
            sub_b = sub_rand;
        }
        3 => {
            line_a = lines(1, ba, &mut r);
            line_b = lines(2, bb, &mut r);
            sub_b = sub_rand;
        }
        4 => {
            line_a = lines(3, ba, &mut r);
            line_b = lines(3, bb, &mut r);
            sub_b = sub_rand;
        }
        5 => {
            line_a = lines(4, ba, &mut r);
            line_b = line_a;
            sub_b = sub_a;
        }
        6 => {
            line_a = lines(4, ba, &mut r);
            line_b = lines(0, bb, &mut r);
            sub_b = sub_rand;
        }
        7 => {
            line_a = lines(0, ba, &mut r);
            line_b = lines(0, ba + 64, &mut r);
            sub_b = sub_a;
        }
        _ => {
            // Race: `slow` is cold and fresh, `fast` is a few repeated primed lines (warm bit of the fast side).
            let (slow_mode, fast_mode) = match (k - 8) % 4 {
                0 => (2, 5),
                1 => (0, 4),
                2 => (6, 5),
                _ => (1, 5),
            };
            let slow = lines(slow_mode, cold(&mut r), &mut r);
            let fast = lines(fast_mode, hot(&mut r), &mut r);
            sub_b = sub_rand;
            if k < 12 {
                warm = 2; // newer (fast) primed, older cold
                line_a = slow;
                line_b = fast;
            } else {
                warm = 1; // older primed hot, newer cold
                line_a = fast;
                line_b = slow;
            }
        }
    }
    let fill = ((r.next() % (HOT as u64 - 8192)) as u32) & !3;
    Scn {
        ma,
        mb,
        line_a,
        line_b,
        sub_a,
        sub_b,
        fill,
        warm,
        race: k >= 8,
    }
}

fn addr(s: &Scn, l: usize, k: &Kind, newer: bool) -> usize {
    let (line, sub) = if newer {
        (s.line_b[l], s.sub_b)
    } else {
        (s.line_a[l], s.sub_a)
    };
    (line + (sub & !(k.bytes - 1))) as usize
}

fn apply(r: &mut [u32; 4], k: &Kind, off: u32, data: &Data, a: usize) {
    if k.w == 1 {
        r[off as usize] = match (k.bytes, k.signed) {
            (1, false) => u32::from(data.byte(a)),
            (1, true) => data.byte(a) as i8 as i32 as u32,
            (2, false) => u32::from(u16::from_le_bytes([data.byte(a), data.byte(a + 1)])),
            (2, true) => i16::from_le_bytes([data.byte(a), data.byte(a + 1)]) as i32 as u32,
            _ => data.dword(a),
        };
    } else {
        for i in 0..k.w as usize {
            r[off as usize + i] = data.dword(a + 4 * i);
        }
    }
}

/// Last-issued-wins model. `swap` applies newer before older (the reversed-order control).
fn payload(sp: &Spec, scns: &[Scn], old: &[u32], data: &Data, poison: u8, swap: bool) -> Vec<u8> {
    let p = &PAIRS[sp.pair];
    let mut out = vec![poison; BLOCKS * 64 * LANE_BYTES];
    for (b, s) in scns.iter().enumerate() {
        for l in 0..64 {
            let o = (b * 64 + l) * 4;
            let mut r = [old[o], old[o + 1], old[o + 2], old[o + 3]];
            let ops = [
                (
                    s.ma >> l & 1 == 1,
                    &p.older,
                    p.ooff,
                    addr(s, l, &p.older, false),
                ),
                (
                    s.mb >> l & 1 == 1,
                    &p.newer,
                    p.noff,
                    addr(s, l, &p.newer, true),
                ),
            ];
            for i in if swap { [1, 0] } else { [0, 1] } {
                let (on, k, off, a) = ops[i];
                if on {
                    apply(&mut r, k, off, data, a);
                }
            }
            let mut aux = 0u32;
            for f in 0..sp.dist {
                aux ^= data.dword(s.fill as usize + l * 4 + f * 64);
            }
            let w = (b * 64 + l) * LANE_BYTES;
            for (dst, value) in out[w..w + 16].chunks_exact_mut(4).zip(r) {
                dst.copy_from_slice(&value.to_le_bytes());
            }
            out[w + 16..w + 20].copy_from_slice(&aux.to_le_bytes());
        }
    }
    out
}

fn tables(sp: &Spec, scns: &[Scn]) -> Result<(Vec<u8>, Vec<u8>), String> {
    let p = &PAIRS[sp.pair];
    let mut tbl = Vec::with_capacity(BLOCKS * 128 * 4);
    let mut msk = Vec::with_capacity(BLOCKS * 5 * 8);
    for s in scns {
        for (newer, k) in [(false, &p.older), (true, &p.newer)] {
            for l in 0..64 {
                let a = addr(s, l, k, newer);
                if a + 16 > TOTAL {
                    return Err(format!(
                        "{}: address {a:#x} out of the data buffer",
                        sp.name
                    ));
                }
                tbl.extend((a as u32).to_le_bytes());
            }
        }
        let all = |on: bool| if on { !0u64 } else { 0 };
        for v in [
            s.ma,
            s.mb,
            all(s.warm & 1 != 0),
            all(s.warm & 2 != 0),
            u64::from(s.fill),
        ] {
            msk.extend(v.to_le_bytes());
        }
    }
    Ok((tbl, msk))
}

// ───────────────────────────── run ─────────────────────────────

/// The calibration kernel must time exactly one default/bypass `global_load_dword v38` between two `s_memtime`s
/// with only `lgkmcnt` waits before it and the full `vmcnt(0)` drain right after it.
fn prove_cal(name: &str, bits: &str, texts: &[String]) -> Result<(), String> {
    let t: Vec<usize> = texts
        .iter()
        .enumerate()
        .filter(|(_, s)| s.starts_with("s_memtime"))
        .map(|(i, _)| i)
        .collect();
    if t.len() != 2 {
        return Err(format!(
            "{name}: {} s_memtime instructions, want 2",
            t.len()
        ));
    }
    let mid = &texts[t[0] + 1..t[1]];
    let li = mid
        .iter()
        .position(|s| parse_ld(s).is_some())
        .ok_or_else(|| format!("{name}: no timed load between the s_memtime pair"))?;
    let ld = parse_ld(&mid[li]).ok_or_else(|| format!("{name}: unparsable timed load"))?;
    if mid.iter().filter(|s| parse_ld(s).is_some()).count() != 1
        || ld.mnem != "global_load_dword"
        || ld.dst != "v38"
        || ld.mods != mod_set(bits, &[])
        || ld.offset.is_some()
    {
        return Err(format!(
            "{name}: timed load is not exactly `global_load_dword v38 {bits}`"
        ));
    }
    if !mid[..li].iter().all(|s| s == "s_waitcnt lgkmcnt(0)")
        || mid.get(li + 1).map(String::as_str) != Some("s_waitcnt vmcnt(0)")
    {
        return Err(format!(
            "{name}: unexpected waits around the timed load: {mid:?}"
        ));
    }
    Ok(())
}

#[derive(Default)]
struct RaceStats {
    both_active: u64,
    cold_older_pairs: u64,
    cold_newer_pairs: u64,
    cold_lines: usize,
    primed_lines: usize,
}
/// Per-launch accounting of the cold/hot race: distinct 128 B cold lines fetched by active older/newer loads
/// (addresses past the hot region), distinct lines primed by the full-exec priming loads, and the lane pairs
/// where both loads were active with a cold older and primed newer (or the reverse).
fn race_stats(p: &Pair, scns: &[Scn]) -> RaceStats {
    let (mut cold, mut primed) = (HashSet::new(), HashSet::new());
    let mut st = RaceStats::default();
    for s in scns.iter().filter(|s| s.race) {
        for l in 0..64 {
            let (aa, ab) = (addr(s, l, &p.older, false), addr(s, l, &p.newer, true));
            let (on_a, on_b) = (s.ma >> l & 1 == 1, s.mb >> l & 1 == 1);
            if s.warm & 1 != 0 {
                primed.insert(aa >> 7);
            }
            if s.warm & 2 != 0 {
                primed.insert(ab >> 7);
            }
            if on_a && aa >= HOT {
                cold.insert(aa >> 7);
            }
            if on_b && ab >= HOT {
                cold.insert(ab >> 7);
            }
            if on_a && on_b {
                st.both_active += 1;
                st.cold_older_pairs += u64::from(aa >= HOT && ab < HOT && s.warm & 2 != 0);
                st.cold_newer_pairs += u64::from(ab >= HOT && aa < HOT && s.warm & 1 != 0);
            }
        }
    }
    st.cold_lines = cold.len();
    st.primed_lines = primed.len();
    st
}

#[derive(Default)]
struct ClassStat {
    kernels: usize,
    failing: usize,
}

fn run(dir: &Path) -> Result<Value, String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let specs = specs();
    let src = source(&specs);
    let src_path = dir.join("waw.hip");
    let co_path = dir.join("waw.hsaco");
    std::fs::write(&src_path, &src).map_err(|e| e.to_string())?;
    let cc = Command::new("/opt/rocm/bin/hipcc")
        .args([
            "--genco",
            "--offload-arch=gfx942",
            "-O3",
            "--no-offload-compress",
            "-fuse-cuid=none",
            "-o",
            co_path.to_str().ok_or("non-UTF8 output path")?,
            src_path.to_str().ok_or("non-UTF8 source path")?,
        ])
        .output()
        .map_err(|e| e.to_string())?;
    if !cc.status.success() {
        return Err(format!("hipcc: {}", String::from_utf8_lossy(&cc.stderr)));
    }
    let container = std::fs::read(&co_path).map_err(|e| e.to_string())?;
    let elf = device_image(&container)?;
    std::fs::write(dir.join("waw.elf"), elf).map_err(|e| e.to_string())?;

    // Typed disassembly of the exact native ELF that is loaded below.
    let lifted = lift_object(elf).map_err(|e| format!("lift_object(device ELF): {e}"))?;
    let by_name: HashMap<&str, usize> = lifted
        .kernels
        .iter()
        .enumerate()
        .map(|(i, k)| (k.image.name.as_str(), i))
        .collect();
    let mut proofs = Vec::with_capacity(specs.len());
    for sp in &specs {
        let &ki = by_name
            .get(sp.name.as_str())
            .ok_or_else(|| format!("{}: kernel missing from the device ELF", sp.name))?;
        let texts: Vec<String> = lifted.kernels[ki]
            .instructions
            .iter()
            .map(|i| i.inst.to_string())
            .collect();
        proofs.push(prove(sp, &texts)?);
    }

    let rt = HipRuntime::load().map_err(|e| e.to_string())?;
    rt.set_device(0).map_err(|e| e.to_string())?;
    let arch = rt.get_arch(0).map_err(|e| e.to_string())?;
    if arch != "gfx942" {
        return Err(format!("requires gfx942, got {arch}"));
    }
    let module = rt.module_load_data(elf).map_err(|e| e.to_string())?;
    let data = Data::new();
    let olds = [make_old(0), make_old(1)];
    let total = GUARD * 2 + BLOCKS * 64 * LANE_BYTES;
    let ddata = rt.malloc(TOTAL).map_err(|e| e.to_string())?;
    let dthrash = rt.malloc(TOTAL).map_err(|e| e.to_string())?;
    let dold = rt.malloc(BLOCKS * 64 * 16).map_err(|e| e.to_string())?;
    let dtbl = rt.malloc(BLOCKS * 128 * 4).map_err(|e| e.to_string())?;
    let dmsk = rt.malloc(BLOCKS * 5 * 8).map_err(|e| e.to_string())?;
    let dout = rt.malloc(total).map_err(|e| e.to_string())?;
    // Stream the generated 256 MiB data buffer up in 16 MiB chunks, hashing as it goes.
    let mut data_hash = Sha256::new();
    const CHUNK: usize = 16 << 20;
    for c in 0..TOTAL / CHUNK {
        let mut chunk = Vec::with_capacity(CHUNK);
        for i in c * CHUNK / 4..(c + 1) * CHUNK / 4 {
            chunk.extend(dw(i).to_le_bytes());
        }
        data_hash.update(&chunk);
        rt.memcpy_htod_offset(&ddata, c * CHUNK, &chunk)
            .map_err(|e| e.to_string())?;
    }

    // ---- Load-latency calibration (honest miss/hit evidence, not an assertion) ----
    // One timed wave load (`s_memtime` around load + `s_waitcnt vmcnt(0)`), 256 trials per configuration:
    // randomized cold lines (lane lines spread over a 64 MiB window of the cold region, thrashed beforehand)
    // vs four repeated lines primed by a preceding load + drain, each with default and sc0/sc1 bypass policy.
    for (name, bits) in [("cal_dd", ""), ("cal_bp", "sc0 sc1")] {
        let &ki = by_name
            .get(name)
            .ok_or_else(|| format!("{name}: kernel missing from the device ELF"))?;
        let texts: Vec<String> = lifted.kernels[ki]
            .instructions
            .iter()
            .map(|i| i.inst.to_string())
            .collect();
        prove_cal(name, bits, &texts)?;
    }
    let dctbl = rt.malloc(CAL_BLOCKS * 64 * 4).map_err(|e| e.to_string())?;
    let dcmsk = rt.malloc(CAL_BLOCKS * 8).map_err(|e| e.to_string())?;
    let dcres = rt.malloc(CAL_BLOCKS * 16).map_err(|e| e.to_string())?;
    let mut cal_rows = Vec::new();
    let mut cal_med: HashMap<String, u64> = HashMap::new();
    let mut cfg = 0u64;
    for (name, label) in [("cal_dd", "default"), ("cal_bp", "bypass_sc0_sc1")] {
        let f = rt
            .module_get_function(&module, name)
            .map_err(|e| e.to_string())?;
        for hot in [false, true] {
            cfg += 1;
            let (mut tbl, mut msk) = (Vec::new(), Vec::new());
            let hot_span = (HOT - 0x8_0000) as u64;
            let cold_span = (TOTAL - HOT - (64 << 20) - 0x8_0000) as u64;
            for b in 0..CAL_BLOCKS {
                let mut r = Rng::new(SEED ^ 0xca11_ca11 ^ (cfg << 40) ^ b as u64);
                let ls = if hot {
                    lines(5, ((r.next() % hot_span) as u32) & !0xfff, &mut r)
                } else {
                    lines(
                        6,
                        (HOT as u32 + (r.next() % cold_span) as u32) & !0xfff,
                        &mut r,
                    )
                };
                for x in ls {
                    tbl.extend((x + (b as u32 % 4) * 4).to_le_bytes());
                }
                msk.extend((if hot { !0u64 } else { 0 }).to_le_bytes());
            }
            rt.memcpy_htod(&dctbl, &tbl).map_err(|e| e.to_string())?;
            rt.memcpy_htod(&dcmsk, &msk).map_err(|e| e.to_string())?;
            rt.memset(&dthrash, (cfg as i32 * 11) & 0xff, TOTAL)
                .map_err(|e| e.to_string())?;
            rt.device_synchronize().map_err(|e| e.to_string())?;
            let mut blob = Vec::new();
            for ptr in [
                ddata.as_ptr() as usize as u64,
                dctbl.as_ptr() as usize as u64,
                dcmsk.as_ptr() as usize as u64,
                dcres.as_ptr() as usize as u64,
            ] {
                blob.extend(ptr.to_le_bytes());
            }
            for x in [CAL_BLOCKS as u32, 1, 1] {
                blob.extend(x.to_le_bytes());
            }
            for x in [64u16, 1, 1] {
                blob.extend(x.to_le_bytes());
            }
            blob.resize(256, 0);
            // SAFETY: live allocations, one wave per block, in-range tables, full hidden argument storage.
            unsafe {
                rt.launch_kernel_blob(
                    &f,
                    [CAL_BLOCKS as u32, 1, 1],
                    [64, 1, 1],
                    0,
                    None,
                    &mut blob,
                )
            }
            .map_err(|e| e.to_string())?;
            rt.device_synchronize().map_err(|e| e.to_string())?;
            let mut res = vec![0u8; CAL_BLOCKS * 16];
            rt.memcpy_dtoh(&mut res, &dcres)
                .map_err(|e| e.to_string())?;
            let mut dts: Vec<u64> = res
                .chunks_exact(16)
                .map(|c| {
                    let t0 = u64::from_le_bytes(c[..8].try_into().unwrap());
                    let t1 = u64::from_le_bytes(c[8..].try_into().unwrap());
                    t1.wrapping_sub(t0)
                })
                .collect();
            dts.sort_unstable();
            let n = dts.len();
            let key = format!("{label}_{}", if hot { "primed_hit" } else { "cold_miss" });
            cal_med.insert(key.clone(), dts[n / 2]);
            cal_rows.push(json!({"config":key, "trials":n, "clock":"s_memtime (shader clock ticks)",
                "min":dts[0], "p10":dts[n / 10], "median":dts[n / 2], "p90":dts[n * 9 / 10], "max":dts[n - 1],
                "table_sha256":sha(&tbl), "result_sha256":sha(&res)}));
        }
    }
    for d in [dctbl, dcmsk, dcres] {
        rt.free(d).map_err(|e| e.to_string())?;
    }
    let (cold, hit, byp) = (
        cal_med["default_cold_miss"],
        cal_med["default_primed_hit"],
        cal_med["bypass_sc0_sc1_primed_hit"],
    );
    // A race needs the older load to be clearly slower than the newer: >= 1.25x median latency.
    let cal_ok = hit > 0 && cold * 4 >= hit * 5 && byp * 4 >= hit * 5;
    let cal_value = json!({"ok":cal_ok, "rule":"median(default cold miss) >= 1.25 x median(default primed hit) and median(bypass sc0/sc1 on primed line) >= 1.25 x median(default primed hit)",
        "cold_over_hit":cold as f64 / hit.max(1) as f64, "bypass_over_hit":byp as f64 / hit.max(1) as f64,
        "rows":cal_rows});

    let mut rows = Vec::with_capacity(specs.len());
    let mut notes: Vec<String> = Vec::new();
    let mut info_notes: Vec<String> = Vec::new();
    let mut classes: BTreeMap<String, ClassStat> = BTreeMap::new();
    let (mut fail_rows, mut info_fail_rows) = (0usize, 0usize);
    let (mut controls, mut controls_expected) = (0usize, 0usize);
    let mut launches = 4usize; // the four calibration launches
    let (mut race_both_active, mut race_cold_older, mut race_cold_newer) = (0u64, 0u64, 0u64);
    let (mut cold_lines_total, mut cold_lines_max) = (0usize, 0usize);
    let (mut primed_lines_total, mut primed_lines_max) = (0usize, 0usize);
    let mut digest = Sha256::new();
    for (si, (sp, proof)) in specs.iter().zip(&proofs).enumerate() {
        let p = &PAIRS[sp.pair];
        let f = rt
            .module_get_function(&module, &sp.name)
            .map_err(|e| e.to_string())?;
        let mut passes = Vec::new();
        let mut row_bad = false;
        for (pi, &poison) in POISONS.iter().enumerate() {
            let scns: Vec<Scn> = (0..BLOCKS).map(|b| scn(b, si * 2 + pi)).collect();
            let (tbl, msk) = tables(sp, &scns)?;
            rt.memcpy_htod(&dtbl, &tbl).map_err(|e| e.to_string())?;
            rt.memcpy_htod(&dmsk, &msk).map_err(|e| e.to_string())?;
            let old = &olds[pi];
            let old_bytes = bytes_of(old);
            rt.memcpy_htod(&dold, &old_bytes)
                .map_err(|e| e.to_string())?;
            // Evict caches (best effort) so the cold older lines really are misses.
            rt.memset(&dthrash, (si as i32 * 7 + pi as i32) & 0xff, TOTAL)
                .map_err(|e| e.to_string())?;
            rt.memset(&dout, i32::from(poison), total)
                .map_err(|e| e.to_string())?;
            rt.device_synchronize().map_err(|e| e.to_string())?;
            let mut blob = Vec::new();
            for ptr in [
                ddata.as_ptr() as usize as u64,
                dold.as_ptr() as usize as u64,
                dtbl.as_ptr() as usize as u64,
                dmsk.as_ptr() as usize as u64,
                dout.as_ptr() as usize as u64 + GUARD as u64,
            ] {
                blob.extend(ptr.to_le_bytes());
            }
            // Same explicit/hidden dispatch layout as cdna_nor_hw_oracle: five aligned pointers.
            for x in [BLOCKS as u32, 1, 1] {
                blob.extend(x.to_le_bytes());
            }
            for x in [64u16, 1, 1] {
                blob.extend(x.to_le_bytes());
            }
            blob.resize(256, 0);
            // SAFETY: live allocations, one wave per block, in-range tables, full hidden argument storage.
            unsafe {
                rt.launch_kernel_blob(&f, [BLOCKS as u32, 1, 1], [64, 1, 1], 0, None, &mut blob)
            }
            .map_err(|e| e.to_string())?;
            rt.device_synchronize().map_err(|e| e.to_string())?;
            launches += 1;
            let mut got = vec![0u8; total];
            rt.memcpy_dtoh(&mut got, &dout).map_err(|e| e.to_string())?;
            let guards_ok = got[..GUARD]
                .iter()
                .chain(&got[total - GUARD..])
                .all(|&v| v == poison);
            let out = &got[GUARD..total - GUARD];
            let mut want = payload(sp, &scns, old, &data, poison, false);
            let rev = payload(sp, &scns, old, &data, poison, true);
            let bad = out.iter().zip(&want).filter(|(a, b)| a != b).count();
            let disc = (0..BLOCKS * 64)
                .filter(|i| {
                    want[i * LANE_BYTES..i * LANE_BYTES + 16]
                        != rev[i * LANE_BYTES..i * LANE_BYTES + 16]
                })
                .count();
            // Cold/hot race accounting: cold older lines, primed newer lines, lane pairs, per launch.
            let rs = race_stats(p, &scns);
            race_both_active += rs.both_active;
            race_cold_older += rs.cold_older_pairs;
            race_cold_newer += rs.cold_newer_pairs;
            cold_lines_total += rs.cold_lines;
            cold_lines_max = cold_lines_max.max(rs.cold_lines);
            primed_lines_total += rs.primed_lines;
            primed_lines_max = primed_lines_max.max(rs.primed_lines);
            let mut pass_bad = !guards_ok || bad != 0;
            if pass_bad {
                let at = out.iter().zip(&want).position(|(a, b)| a != b);
                let loc = at.map(|i| {
                    let (blk, lane, byte) =
                        (i / (64 * LANE_BYTES), i / LANE_BYTES % 64, i % LANE_BYTES);
                    format!(
                        "block {blk} lane {lane} byte {byte}: got {:#04x} want {:#04x} (older_mask {:#x} newer_mask {:#x} race {}; reversed-order model {})",
                        out[i],
                        want[i],
                        scns[blk].ma,
                        scns[blk].mb,
                        scns[blk].race,
                        if rev[i] == out[i] { "MATCHES" } else { "differs" }
                    )
                });
                let list = if p.gating() {
                    &mut notes
                } else {
                    &mut info_notes
                };
                if list.len() < MAX_FAIL_NOTES {
                    list.push(format!(
                        "{} poison {poison:#x}: guards_ok={guards_ok} bad_bytes={bad} first {loc:?}",
                        sp.name
                    ));
                }
            }
            // Corruption controls: single-bit mutants of the expectation and the reversed-order model
            // must be detected by the comparison.
            let n = want.len();
            let (mut detected, mut ctl_expected) = (0, 0);
            for pos in [
                (0usize, 0x01u8),
                (n - LANE_BYTES + 16, 0x80),
                ((n / 2) & !31, 0x80),
            ] {
                want[pos.0] ^= pos.1;
                ctl_expected += 1;
                detected += usize::from(out != want.as_slice());
                want[pos.0] ^= pos.1;
            }
            if disc > 0 {
                ctl_expected += 1;
                detected += usize::from(out != rev.as_slice());
            } else if p.overlaps() {
                ctl_expected += 1; // overlap pair that cannot tell the two orders apart: undetected
            }
            if detected != ctl_expected {
                pass_bad = true;
                let list = if p.gating() {
                    &mut notes
                } else {
                    &mut info_notes
                };
                if list.len() < MAX_FAIL_NOTES {
                    list.push(format!(
                        "{} poison {poison:#x}: {detected}/{ctl_expected} controls detected (discriminating lanes {disc})",
                        sp.name
                    ));
                }
            }
            if p.gating() {
                controls += detected;
                controls_expected += ctl_expected;
            }
            row_bad |= pass_bad;
            let (osha, esha, rsha) = (sha(out), sha(&want), sha(&rev));
            digest.update(sp.name.as_bytes());
            for h in [&osha, &esha, &rsha] {
                digest.update(h.as_bytes());
            }
            passes.push(json!({"poison":poison, "old_seed_pass":pi, "guards_ok":guards_ok, "bad_bytes":bad,
                "discriminating_lanes":disc, "controls_detected":detected, "controls_expected":ctl_expected,
                "old_sha256":sha(&old_bytes), "table_sha256":sha(&tbl), "mask_sha256":sha(&msk),
                "output_sha256":osha, "expected_sha256":esha, "reversed_order_model_sha256":rsha}));
        }
        let cs = classes.entry(p.class()).or_default();
        cs.kernels += 1;
        cs.failing += usize::from(row_bad);
        if row_bad {
            if p.gating() {
                fail_rows += 1;
            } else {
                info_fail_rows += 1;
            }
        }
        rows.push(json!({"kernel":sp.name, "class":p.class(), "gating":p.gating(),
            "older":mnem(p.of, &p.older), "older_dst": reg(32 + p.ooff, p.older.w),
            "newer":mnem(p.nf, &p.newer), "newer_dst": reg(32 + p.noff, p.newer.w), "overlap":p.overlaps(),
            "queue_distance":sp.dist, "policy":POLICIES[sp.pol].0, "row_ok":!row_bad,
            "disasm_sha256":proof.disasm_sha256, "instructions":proof.instructions,
            "pair_instruction_index":[proof.l1, proof.l2], "preload_instruction_index":proof.l0,
            "pair_window": if sp.dist == 0 { json!(proof.window) } else { Value::Null },
            "passes":passes}));
    }
    for d in [ddata, dthrash, dold, dtbl, dmsk, dout] {
        rt.free(d).map_err(|e| e.to_string())?;
    }
    let ok = fail_rows == 0 && controls == controls_expected && cal_ok;
    let mut receipt = json!({
        "status": if ok {"PASS"} else {"FAIL"},
        "informational_flat_status": if info_fail_rows == 0 {"PASS"} else {"FAIL"},
        "oracle":"cdna_vmem_waw_hw_oracle", "arch":arch,
        "claim":"gfx942 untyped VMEM loads (global/buffer, same non-sampler VMEM_READ_ACCESS type class), no wait between: last-issued write wins per active lane/dword, inactive lanes keep the prior value (hardware in-order VGPR return); lane masks are stress only; flat rows informational",
        "not_covered":["image/sampler/BVH VMEM classes", "scratch loads (the ledger may model them as the same class; this harness has no scratch setup)", "flat to the LDS aperture", "d16/lds-bit/atomic-return/store forms"],
        "latency_calibration":cal_value, "latency_calibration_ok":cal_ok,
        "cold_region_bytes":TOTAL - HOT, "hot_region_bytes":HOT, "cold_random_window_bytes":64usize << 20, "thrash_bytes_per_launch":TOTAL,
        "primed_distinct_hot_lines_note":"line = 128 B; cold older lines are counted per launch over active older lanes at addresses >= hot_region_bytes",
        "race_cold_older_primed_newer_lane_pairs":race_cold_older, "race_primed_older_cold_newer_lane_pairs":race_cold_newer,
        "cold_lines_total":cold_lines_total, "cold_lines_max_per_launch":cold_lines_max,
        "primed_lines_total":primed_lines_total, "primed_lines_max_per_launch":primed_lines_max,
    });
    let bindings = json!({
        "kernels":specs.len(), "launches":launches, "blocks_per_launch":BLOCKS, "scenarios":NSCN,
        "race_both_active_lane_pairs":race_both_active, "queue_distances":DISTS,
        "policies":POLICIES.iter().map(|p| json!({"tag":p.0, "older":p.1, "newer":p.2})).collect::<Vec<_>>(),
        "classes":classes.iter().map(|(k, v)| json!({"class":k, "kernels":v.kernels, "failing_kernels":v.failing})).collect::<Vec<_>>(),
        "failing_rows":fail_rows, "informational_failing_rows":info_fail_rows,
        "controls_detected":controls, "controls_expected":controls_expected,
        "failure_notes":notes, "informational_failure_notes":info_notes, "seed":SEED,
        "hip_source_sha256":sha(src.as_bytes()), "code_object_container_sha256":sha(&container),
        "device_elf_sha256":sha(elf), "device_elf_bytes":elf.len(),
        "data_bytes":TOTAL, "data_sha256":format!("{:x}", data_hash.finalize()),
        "old_sha256":[sha(&bytes_of(&olds[0])), sha(&bytes_of(&olds[1]))],
        "results_digest_sha256":format!("{:x}", digest.finalize()),
        "oracle_source_sha256":sha(include_bytes!("cdna_vmem_waw_hw_oracle.rs")),
        "lift_source_sha256":sha(include_bytes!("../../peacemaker-lift/src/cdna.rs")),
        "bundle_codec_source_sha256":sha(include_bytes!("../../peacemaker-lift/src/bundle.rs")),
        "flat_codec_source_sha256":sha(include_bytes!("../../peacemaker-cdna/src/enc/flat.rs")),
        "mubuf_codec_source_sha256":sha(include_bytes!("../../peacemaker-cdna/src/enc/mubuf.rs")),
        "rows":rows});
    // Move the binding fields into the receipt without a larger macro recursion limit.
    if let (Value::Object(receipt), Value::Object(bindings)) = (&mut receipt, bindings) {
        receipt.extend(bindings);
    }
    Ok(receipt)
}

fn main() -> ExitCode {
    let mut args = std::env::args_os().skip(1);
    let Some(dir) = args.next().map(PathBuf::from) else {
        eprintln!("usage: cdna_vmem_waw_hw_oracle OUT_DIR");
        return ExitCode::FAILURE;
    };
    if args.next().is_some() {
        eprintln!("unexpected argument");
        return ExitCode::FAILURE;
    }
    let result = run(&dir);
    let v = match &result {
        Ok(v) => v.clone(),
        Err(e) => json!({"status":"FAIL", "oracle":"cdna_vmem_waw_hw_oracle", "error":e}),
    };
    let path = dir.join("receipt.json");
    let saved = std::fs::create_dir_all(&dir)
        .map_err(|e| e.to_string())
        .and_then(|()| serde_json::to_vec_pretty(&v).map_err(|e| e.to_string()))
        .and_then(|b| std::fs::write(&path, b).map_err(|e| e.to_string()));
    // The full receipt is large (one row per kernel); stdout carries the summary only.
    let mut summary = v.clone();
    if let Some(o) = summary.as_object_mut() {
        o.remove("rows");
    }
    println!("{summary}");
    if v["status"] == "PASS" && saved.is_ok() {
        ExitCode::SUCCESS
    } else {
        eprintln!("oracle result: {:?}; receipt write: {saved:?}", v["status"]);
        ExitCode::FAILURE
    }
}
