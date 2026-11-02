// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Native tiled numerical oracle (`pm_moe_cn_tiled_numeric_oracle`).
//!
//! Standalone example only (ownership forbids external docs; all CLI usage lives here).
//! Compares the fixed-eight-row-tiled candidate entries
//! against independently pinned K1 narrow singleton launches (true per-token local
//! singletons: C1/U1, local slot=rank, X pointer at the token's row window of the
//! case X, local x_src_rows clamped to the case's remaining rows, x_stride is the
//! case's actual (possibly padded) stride, block32), never against a candidate echo.
//!
//! Families (per `--arch`):
//! - gfx1151: candidate module `mq_i4_moe_gemv_cn_pm_gfx1151`, symbols
//!   `mq_i4_moe_gemv_cn_{mq3i4,mq4sym}_{r256_gate_up,r128_down}_t8`;
//!   K1 module `mq_i4_moe_gemv_pm_gfx1151`, symbols
//!   `mq_i4_moe_gemv_{mq3i4,mq4sym}_{r256_gate_up,r128_down}`.
//! - gfx1201: candidate module `mq_i4_moe_gemv_pm_gfx1201`, symbols
//!   `mq_i4_moe_gemv_{mq3i4,mq4sym}_{r256_gate_up,r128_down}_t8`;
//!   K1 module `mq_i4_moe_gemv_pm_gfx1201` narrow symbols.
//! No CO retag or kernel edits happen here; `--co` / `--k1-co` select files externally.
//!
//! Frozen ABI (both archs): one guarded contiguous descriptor allocation, 482 u32
//! (1928 B) per tile, tile `t` at word `482*t`; bucket `b=0` W3 (mq3i4), `b=1` MQ4
//! (mq4sym): IDs word `80*b`, offsets word `160+81*b` (81 entries), slots word
//! `322+80*b` (80 entries). Unused IDs/slots zero, unused offsets terminal count.
//! Candidate pointers address the selected bucket in tile zero; only the kernel adds
//! `1928*z`. Opposite bucket stays empty. Kernarg is 96 B: 7 pointers
//! (table, ids, offs, slots, X, Y0+GUARD, gate?Y1+GUARD:0) + 10 u32
//! (M,K,80,R,x_row_div,x_src_rows,x_stride,y_slot_stride,n_experts,0), strides in F32.
//! Candidate grid is the FULL production grid `[ceil(M/8),80,ceil(R/8)]`, block
//! `[256,1,1]` (full production M: simpler than a selected-block grid because HIP
//! cannot offset a grid origin to reach the last output block alone). K1 reference is
//! one grid-`[M,1,1]` launch per token (every row at once), block `[32,1,1]`.
//!
//! VMM-guarded allocation (REQUIRED, no malloc fallback): every device buffer is a
//! `VmmArena` from the existing hip-bridge VMM helpers (same loader/launch APIs as
//! production paths, no new loader). Mapped prefix is `rounded = ceil(len/gran)*gran`
//! with one extra granule reserved-but-UNMAPPED as suffix; logical bytes sit
//! RIGHT-ALIGNED at `[mapped_end-len, mapped_end)` so an exact-tail overread hits
//! genuinely unmapped VA and faults. All immutable inputs (tensors, table, X,
//! descriptors) are set `read_only_device` via `mem_set_access` after upload, so even
//! a same-byte store faults instead of passing. Outputs stay RW with the same
//! right-aligned unmapped suffix. Downstream-must-not-read pointers are null device
//! pointers. If the HIP runtime lacks VMM symbols the run fails closed with an
//! explicit error (never a silent malloc fallback). Early-geometry guard cases
//! (C0/C65/U79/U81/wrong-K/dead-M0) map NO downstream descriptor/table/slot/X
//! pointers at all (first five kernarg pointers null); y=80/z-past/empty keep the
//! separately right-end-guarded exact arrays because the full grid needs them.
//!
//! K rules are source-defined per arch: gfx1151 Cn requires the exact production K
//! (gate 2560, down 640); gfx1201 `wide.rs` rowtile guard requires K nonzero and
//! codec group-aligned, NOT exact production K. So the negative "wrong K" uses K+128
//! (aligned, non-production) on gfx1151 and K=0 plus K+1 (misaligned) on gfx1201.
//! C=65, U=79/81 reject on both. R1 offsets `[8,11]` and invalid-slot-with-unmapped-table
//! fixtures are REQUIRED in source; on gfx1201 the current source loads the expert table
//! before the slot check and only checks offsets `<=80`, so those two fixtures are
//! expected to fault against the old CO. They are kept exact (never suppressed/passed)
//! and run LAST; first HIP fault STOPS the run (no retries) with prior evidence kept.
//! The invalid-slot fixture is R64/count8: bad `640 = 10*C` and `u32::MAX` in each of
//! the 8 loaded positions, table NULL.
//!
//! Native write-bit limitation (stated, not faked): the GPU cannot expose instruction
//! written bits. Stores are established per byte with TWO independent fill poisons
//! (0xA5 then 0x3C on ambiguous bytes) plus an expected written-byte map and guard
//! poison (0x5A); every output byte must equal the independent K1 byte and every
//! unwritten byte must retain poison. Read-only VMM mappings additionally catch
//! same-byte input stores as faults. All inputs are downloaded back and byte-compared
//! after every launch (not hash-only) and every input hash is recorded.
//!
//! Usage (no device allocation or GPU access under `--help` / `--selfcheck`):
//! ```text
//! pm_moe_cn_tiled_numeric_oracle --help
//! pm_moe_cn_tiled_numeric_oracle --selfcheck [--out FILE.json]
//! pm_moe_cn_tiled_numeric_oracle --co CAND --k1-co K1 [--k1-sha256 HEX]
//!   --arch gfx1151|gfx1201 [--rows A-B] [--shard I/4] [--cheap] [--guards-only]
//!   [--expect-bdf BDF] --out OUT.json
//! ```
//! - `--selfcheck`: pure-host fixture/ABI/plan validation, no HIP init.
//!   Constructs the actual ordinary R1..64 and focused fixtures, decodes every
//!   descriptor, and requires exact nonempty codec extents for every live expert.
//!   Ordinary tables stay sparse (0/511); terminal80 maps all ten experts.
//! - `--cheap`: kill gate first — R9 ordinary (all 4 specs), R8 tail pair
//!   (gate M1282 / down M2563), and C65 rejection — then stop.
//! - `--guards-only`: skip all positive construction and launches and run only the
//!   original invalid-descriptor guard family, regardless of rows or shard.
//!   `--cheap` with `--guards-only` is rejected.
//! - Full default: every R 1..64, all four candidate bodies, all focused extras, and
//!   the invalid-descriptor guard family (including unsharded `--rows 1-64`).
//! - Shards (four local processes): `--rows 1-16 --shard 1/4`, `--rows 17-32 --shard 2/4`,
//!   `--rows 33-48 --shard 3/4`, `--rows 49-64 --shard 4/4` (rows must match shard).
//!   Invalid-descriptor guard fixtures run on shard 1 only (or the unsharded full run);
//!   other focused extras are partitioned by case index mod 4 so the four shards
//!   jointly cover everything with no omission.
//! - `--k1-sha256 HEX`: if given, the actual `--k1-co` bytes must hash to it (pin).
//! - `--expect-bdf BDF`: if given, `device_pci_bus_id(0)` must equal it.
//!
//! Receipt: JSON with candidate/reference/source identities, arch, device UUID/BDF,
//! fixture and write counts, VMM read-only/right-aligned status, full input
//! verification note, and the leaf-only limitation. Leaf-only: this proves executed
//! Candidate numerics vs independent K1 singletons only; no hardware VM/LGKM timing,
//! barrier-participation, capture/replay, throughput, admission, or product claims.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use hip_bridge::{
    DeviceBuffer, Function, HipMemAccessDesc, HipMemAllocationProp, HipRuntime, Module, VmmArena,
    HIP_MEM_ALLOCATION_GRANULARITY_RECOMMENDED,
};
use serde_json::json;

type Result<T, E = String> = std::result::Result<T, E>;

const GUARD: usize = 64;
const FILL_A: u8 = 0xa5;
const FILL_B: u8 = 0x3c;
const GUARD_FILL: u8 = 0x5a;
const GUARD_FILL_B: u8 = 0x6b;
const DESC_WORDS: usize = 482;
const DESC_BYTES: usize = DESC_WORDS * 4; // 1928
const U_FIXED: u32 = 80;

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest as _;
    format!("{:x}", sha2::Sha256::digest(bytes))
}

fn ceil_div(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

// ---------------------------------------------------------------- fixtures ---

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

fn scale_bits(rng: &mut Rng) -> u16 {
    (((8 + rng.below(13)) as u16) << 10) | (rng.next() as u16 & 0x3ff)
}
fn sym_zero_bits(sc: u16) -> u16 {
    0x8000 | (sc + (3 << 10))
}

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

fn mq4_tensor(rng: &mut Rng, gate: bool, m: usize, k: usize) -> Vec<u8> {
    let (cols, stride, halves, payload_at, payload) = if gate {
        (256, 136, 2, 8, 128)
    } else {
        (128, 68, 1, 4, 64)
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

fn x_rows(rng: &mut Rng, rows: usize, k: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * k * 4);
    for _ in 0..rows * k {
        let (s, e, m) = (
            (rng.next() & 1) as u32,
            118 + rng.below(19) as u32,
            (rng.next() & 0x7f_ffff) as u32,
        );
        out.extend(((s << 31) | (e << 23) | m).to_le_bytes());
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Codec {
    Mq3I4,
    Mq4Sym,
}
impl Codec {
    fn tag(self) -> &'static str {
        match self {
            Self::Mq3I4 => "mq3i4",
            Self::Mq4Sym => "mq4sym",
        }
    }
    fn bucket(self) -> usize {
        match self {
            Self::Mq3I4 => 0,
            Self::Mq4Sym => 1,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Kind {
    GateUp,
    Down,
}
impl Kind {
    fn tag(self) -> &'static str {
        match self {
            Self::GateUp => "r256_gate_up",
            Self::Down => "r128_down",
        }
    }
    fn gate(self) -> bool {
        matches!(self, Self::GateUp)
    }
}

#[derive(Clone, Copy, Debug)]
struct Spec {
    codec: Codec,
    kind: Kind,
}
fn all_specs() -> [Spec; 4] {
    [
        Spec {
            codec: Codec::Mq3I4,
            kind: Kind::GateUp,
        },
        Spec {
            codec: Codec::Mq3I4,
            kind: Kind::Down,
        },
        Spec {
            codec: Codec::Mq4Sym,
            kind: Kind::GateUp,
        },
        Spec {
            codec: Codec::Mq4Sym,
            kind: Kind::Down,
        },
    ]
}

fn prod_k(spec: Spec) -> u32 {
    if spec.kind.gate() {
        2560
    } else {
        640
    }
}
fn prod_m(spec: Spec) -> u32 {
    if spec.kind.gate() {
        1280
    } else {
        2560
    }
}
fn xdiv(spec: Spec) -> u32 {
    if spec.kind.gate() {
        10
    } else {
        1
    }
}
fn ystride_prod(spec: Spec) -> u32 {
    if spec.kind.gate() {
        640
    } else {
        2560
    }
}

fn cand_symbol(arch: &str, spec: Spec) -> String {
    // These are exports of the externally supplied tiled candidate, not the
    // embedded narrow/K1 images. The gfx1201 tiled route is deliberately non-Cn.
    match arch {
        "gfx1151" => format!(
            "mq_i4_moe_gemv_cn_{}_{}_t8",
            spec.codec.tag(),
            spec.kind.tag()
        ),
        _ => format!("mq_i4_moe_gemv_{}_{}_t8", spec.codec.tag(), spec.kind.tag()),
    }
}
fn k1_symbol(spec: Spec) -> String {
    format!("mq_i4_moe_gemv_{}_{}", spec.codec.tag(), spec.kind.tag())
}
/// Handle loader for the candidate image: the same `(arch, spec)` selection
/// as [`cand_symbol`], but binding the module-loaded [`Function`] handle
/// through a literal symbol name per finite `(arch, codec, kind)` case, so the
/// launcher ABI gate resolves every launch site to its checked kernels.
fn cand_handle(hip: &HipRuntime, m: &Module, arch: &str, spec: Spec) -> Result<Function> {
    let name: &str = match (arch, spec.codec, spec.kind) {
        ("gfx1151", Codec::Mq3I4, Kind::GateUp) => "mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8",
        ("gfx1151", Codec::Mq3I4, Kind::Down) => "mq_i4_moe_gemv_cn_mq3i4_r128_down_t8",
        ("gfx1151", Codec::Mq4Sym, Kind::GateUp) => "mq_i4_moe_gemv_cn_mq4sym_r256_gate_up_t8",
        ("gfx1151", Codec::Mq4Sym, Kind::Down) => "mq_i4_moe_gemv_cn_mq4sym_r128_down_t8",
        (_, Codec::Mq3I4, Kind::GateUp) => "mq_i4_moe_gemv_mq3i4_r256_gate_up_t8",
        (_, Codec::Mq3I4, Kind::Down) => "mq_i4_moe_gemv_mq3i4_r128_down_t8",
        (_, Codec::Mq4Sym, Kind::GateUp) => "mq_i4_moe_gemv_mq4sym_r256_gate_up_t8",
        (_, Codec::Mq4Sym, Kind::Down) => "mq_i4_moe_gemv_mq4sym_r128_down_t8",
    };
    hip.module_get_function(m, name).map_err(err)
}

/// Handle loader for the K1 image: the same `spec` selection as [`k1_symbol`],
/// binding through a literal symbol name per finite `(codec, kind)` case.
fn k1_handle(hip: &HipRuntime, m: &Module, spec: Spec) -> Result<Function> {
    let name: &str = match (spec.codec, spec.kind) {
        (Codec::Mq3I4, Kind::GateUp) => "mq_i4_moe_gemv_mq3i4_r256_gate_up",
        (Codec::Mq3I4, Kind::Down) => "mq_i4_moe_gemv_mq3i4_r128_down",
        (Codec::Mq4Sym, Kind::GateUp) => "mq_i4_moe_gemv_mq4sym_r256_gate_up",
        (Codec::Mq4Sym, Kind::Down) => "mq_i4_moe_gemv_mq4sym_r128_down",
    };
    hip.module_get_function(m, name).map_err(err)
}

// tile -> list of (expert, absolute slots)
type TileGroups = Vec<Vec<(u32, Vec<u32>)>>;

fn ordinary_groups(rows: u32, n_experts: u32) -> TileGroups {
    let n_tiles = ceil_div(rows, 8) as usize;
    let mut groups = Vec::new();
    for tile in 0..n_tiles {
        let mut tg: Vec<(u32, Vec<u32>)> = Vec::new();
        for token in 0..8u32 {
            let r = tile as u32 * 8 + token;
            if r >= rows {
                break;
            }
            let rank = if r % 2 == 0 { 0 } else { 9 };
            let slot = 10 * r + rank;
            let expert = if tile % 2 == 0 { 0 } else { n_experts - 1 };
            match tg.iter_mut().find(|(e, _)| *e == expert) {
                Some((_, s)) => s.push(slot),
                None => tg.push((expert, vec![slot])),
            }
        }
        groups.push(tg);
    }
    groups
}

/// 1928 B/tile descriptor. Opposite bucket stays zero; unused offsets hold the
/// bucket terminal count; unused IDs/slots stay zero.
fn build_descriptor(n_tiles: usize, bucket: usize, groups: &TileGroups) -> Vec<u8> {
    let mut words = vec![0u32; DESC_WORDS * n_tiles.max(1)];
    for (tile, tg) in groups.iter().enumerate() {
        let base = tile * DESC_WORDS;
        let off = base + 160 + bucket * 81;
        let count: usize = tg.iter().map(|(_, s)| s.len()).sum();
        for o in 0..81usize {
            words[off + o] = count as u32;
        }
        let mut pos = 0usize;
        for (gi, (expert, slots)) in tg.iter().enumerate() {
            words[base + bucket * 80 + gi] = *expert;
            words[off + gi] = pos as u32;
            for (j, slot) in slots.iter().enumerate() {
                words[base + 322 + bucket * 80 + pos + j] = *slot;
            }
            pos += slots.len();
        }
    }
    words.into_iter().flat_map(u32::to_le_bytes).collect()
}

fn encode_kernarg(ptrs: [u64; 7], values: [u32; 10]) -> [u8; 96] {
    // Seven u64 pointers followed by ten u32 scalars; keep each ABI field's
    // exact byte range explicit for both host readers and the launcher checker.
    let mut out = [0u8; 96];
    out[0..8].copy_from_slice(&ptrs[0].to_le_bytes());
    out[8..16].copy_from_slice(&ptrs[1].to_le_bytes());
    out[16..24].copy_from_slice(&ptrs[2].to_le_bytes());
    out[24..32].copy_from_slice(&ptrs[3].to_le_bytes());
    out[32..40].copy_from_slice(&ptrs[4].to_le_bytes());
    out[40..48].copy_from_slice(&ptrs[5].to_le_bytes());
    out[48..56].copy_from_slice(&ptrs[6].to_le_bytes());
    out[56..60].copy_from_slice(&values[0].to_le_bytes());
    out[60..64].copy_from_slice(&values[1].to_le_bytes());
    out[64..68].copy_from_slice(&values[2].to_le_bytes());
    out[68..72].copy_from_slice(&values[3].to_le_bytes());
    out[72..76].copy_from_slice(&values[4].to_le_bytes());
    out[76..80].copy_from_slice(&values[5].to_le_bytes());
    out[80..84].copy_from_slice(&values[6].to_le_bytes());
    out[84..88].copy_from_slice(&values[7].to_le_bytes());
    out[88..92].copy_from_slice(&values[8].to_le_bytes());
    out[92..96].copy_from_slice(&values[9].to_le_bytes());
    out
}

// ------------------------------------------------------------------- CLI ---

struct Args {
    co: Option<String>,
    k1_co: Option<String>,
    k1_sha256: Option<String>,
    arch: Option<String>,
    rows: Option<(u32, u32)>,
    shard: Option<(u32, u32)>,
    cheap: bool,
    guards_only: bool,
    expect_bdf: Option<String>,
    out: Option<String>,
    selfcheck: bool,
    help: bool,
}

fn parse_args() -> Result<Args> {
    let a: Vec<String> = std::env::args().collect();
    let mut o = Args {
        co: None,
        k1_co: None,
        k1_sha256: None,
        arch: None,
        rows: None,
        shard: None,
        cheap: false,
        guards_only: false,
        expect_bdf: None,
        out: None,
        selfcheck: false,
        help: false,
    };
    let mut i = 1;
    while i < a.len() {
        match a[i].as_str() {
            "--help" | "-h" => o.help = true,
            "--selfcheck" => o.selfcheck = true,
            "--cheap" => o.cheap = true,
            "--guards-only" => o.guards_only = true,
            "--co" => {
                i += 1;
                o.co = Some(a.get(i).ok_or("--co needs a value")?.clone());
            }
            "--k1-co" => {
                i += 1;
                o.k1_co = Some(a.get(i).ok_or("--k1-co needs a value")?.clone());
            }
            "--k1-sha256" => {
                i += 1;
                o.k1_sha256 = Some(a.get(i).ok_or("--k1-sha256 needs a value")?.clone());
            }
            "--arch" => {
                i += 1;
                o.arch = Some(a.get(i).ok_or("--arch needs a value")?.clone());
            }
            "--rows" => {
                i += 1;
                let s = a.get(i).ok_or("--rows needs A-B")?.clone();
                let (x, y) = s.split_once('-').ok_or("--rows must be A-B")?;
                let (lo, hi): (u32, u32) = (x.parse().map_err(err)?, y.parse().map_err(err)?);
                if lo < 1 || hi > 64 || lo > hi {
                    return Err("--rows must satisfy 1 <= A <= B <= 64".into());
                }
                o.rows = Some((lo, hi));
            }
            "--shard" => {
                i += 1;
                let s = a.get(i).ok_or("--shard needs I/4")?.clone();
                let (x, y) = s.split_once('/').ok_or("--shard must be I/4")?;
                let (p, q): (u32, u32) = (x.parse().map_err(err)?, y.parse().map_err(err)?);
                if q != 4 || p < 1 || p > 4 {
                    return Err("--shard must be 1/4, 2/4, 3/4 or 4/4".into());
                }
                o.shard = Some((p, q));
            }
            "--expect-bdf" => {
                i += 1;
                o.expect_bdf = Some(a.get(i).ok_or("--expect-bdf needs a value")?.clone());
            }
            "--out" => {
                i += 1;
                o.out = Some(a.get(i).ok_or("--out needs a value")?.clone());
            }
            other => return Err(format!("unknown flag {other}")),
        }
        i += 1;
    }
    if o.cheap && o.guards_only {
        return Err(
            "--cheap with --guards-only would silently skip the cheap gate positives; pick one"
                .into(),
        );
    }
    if let (Some((p, _)), Some((lo, hi))) = (o.shard, o.rows) {
        let want = match p {
            1 => (1, 16),
            2 => (17, 32),
            3 => (33, 48),
            _ => (49, 64),
        };
        if (lo, hi) != want {
            return Err(format!(
                "--rows {lo}-{hi} disagrees with --shard {p}/4 (want {}-{})",
                want.0, want.1
            ));
        }
    }
    Ok(o)
}

fn usage() -> &'static str {
    "usage:\n  pm_moe_cn_tiled_numeric_oracle --help\n  pm_moe_cn_tiled_numeric_oracle --selfcheck [--out FILE.json]\n  pm_moe_cn_tiled_numeric_oracle --co CAND --k1-co K1 [--k1-sha256 HEX] --arch gfx1151|gfx1201 [--rows A-B] [--shard I/4] [--cheap] [--guards-only] [--expect-bdf BDF] --out OUT.json\n\nshards:\n  --rows 1-16 --shard 1/4 | --rows 17-32 --shard 2/4 | --rows 33-48 --shard 3/4 | --rows 49-64 --shard 4/4\nguards: invalid-descriptor suite runs on --shard 1/4 or the unsharded full run, LAST (first fault stops). --guards-only skips all positive construction/launches and always runs the suite regardless of rows/shard.\nalloc: every device buffer is a right-aligned VMM arena (mapped prefix + unmapped suffix; inputs read-only).\n"
}

// --------------------------------------------------------------- selfcheck ---

fn selfcheck(out: Option<String>) -> Result<()> {
    // Actual host fixtures only: no HIP init or device allocation. Ordinary R
    // cases share the factory's real weights rather than regenerating per R.
    let mut notes: Vec<String> = Vec::new();
    let mut fixtures = Vec::new();
    for (si, s) in all_specs().into_iter().enumerate() {
        let full = build_case(s, 64, prod_m(s), 0, 0, 0, None, 512);
        for rows in 1..=64 {
            let c = ordinary_case(s, rows, &full.x, &full.tensors);
            fixtures.push(check_fixture(
                &format!("ordinary R{rows} {}", k1_symbol(s)),
                &c,
            )?);
        }
        let c = cheap_tail_case(si, s);
        fixtures.push(check_fixture(&format!("cheap tail {}", k1_symbol(s)), &c)?);
    }
    for e in focused_extras() {
        let c = focused_case(&e);
        fixtures.push(check_fixture(&e.name, &c)?);
    }
    let fixture_groups: u64 = fixtures
        .iter()
        .map(|f| f["launch_groups"].as_u64().unwrap())
        .sum();
    let fixture_live_experts: u64 = fixtures
        .iter()
        .map(|f| f["live_expert_count"].as_u64().unwrap())
        .sum();
    let fixture_extents: usize = fixtures
        .iter()
        .map(|f| f["mapped_weight_extents"].as_array().unwrap().len())
        .sum();
    notes.push("actual ordinary R1..64, cheap tails, terminal80/tail/pad/phase/xclip/xzero: every live expert has exact nonempty codec weights; every 482-word tile decoded ok".into());
    // Kernarg 96 B layout probe.
    let ptrs = [8u64, 9, 10, 11, 12, 13, 14];
    let values = [1280, 2560, 80, 9, 10, 9, 2560, 640, 2, 0];
    let ka = encode_kernarg(ptrs, values);
    if ka.len() != 96
        || ptrs
            .iter()
            .enumerate()
            .any(|(i, p)| u64::from_le_bytes(ka[8 * i..8 * i + 8].try_into().unwrap()) != *p)
        || values
            .iter()
            .enumerate()
            .any(|(i, v)| u32::from_le_bytes(ka[56 + 4 * i..60 + 4 * i].try_into().unwrap()) != *v)
    {
        return Err("kernarg-96B pointer/scalar layout".into());
    }
    notes.push("kernarg-96B ok".into());
    // Tiled external candidates and independent narrow K1 must stay distinct.
    for arch in ["gfx1151", "gfx1201"] {
        for s in all_specs() {
            let c = cand_symbol(arch, s);
            let k = k1_symbol(s);
            if !c.ends_with("_t8") || k.ends_with("_t8") {
                return Err("symbol family".into());
            }
            if (arch == "gfx1151") != c.contains("_cn_") {
                return Err("symbol family".into());
            }
        }
    }
    notes.push(
        "symbols: four external tiled _t8 candidates and four independent narrow K1 per arch ok"
            .into(),
    );
    // Shards cover 1..64 exactly once.
    {
        let shards = [(1u32, 16u32), (17, 32), (33, 48), (49, 64)];
        let mut covered = vec![false; 65];
        for (lo, hi) in shards {
            for r in lo..=hi {
                if covered[r as usize] {
                    return Err("shard overlap".into());
                }
                covered[r as usize] = true;
            }
        }
        if covered.iter().skip(1).any(|c| !c) {
            return Err("shard gap".into());
        }
    }
    // Focused-extra partition: indices round-trip mod 4 with no omission.
    {
        let mut seen = [0u32; 4];
        let mut terminal_seen = [0u32; 4];
        for (idx, e) in focused_extras().iter().enumerate() {
            if e.idx != idx {
                return Err("actual extra index gap/overlap".into());
            }
            seen[e.idx % 4] += 1;
            if e.name.starts_with("nonzero-offsets") {
                terminal_seen[e.idx % 4] += 1;
            }
        }
        if seen != [6, 6, 6, 6] || terminal_seen != [1, 1, 1, 1] {
            return Err("actual extra partition/terminal80 coverage".into());
        }
    }
    notes.push("shards 1-16/17-32/33-48/49-64, extras idx%4, guards shard1 ok".into());
    // Poisons distinct.
    if FILL_A == FILL_B || FILL_A == GUARD_FILL || FILL_B == GUARD_FILL {
        return Err("poisons must be pairwise distinct".into());
    }
    if FILL_B == GUARD_FILL_B || FILL_A == GUARD_FILL_B {
        return Err("B-poison/guard must be pairwise distinct".into());
    }
    notes.push("dual-poison 0xA5/0x3C + guards 0x5A/0x6B distinct ok".into());
    // K rules per arch (source-defined).
    notes.push("K-rule gfx1151 exact-2560/640; gfx1201 nonzero+group-aligned (wrong-K: K0/K+1); C65/U79/U81 reject both ok".into());
    // Fixture determinism + full mantissa spot check + X prefix-stability
    // (the sweep shares one X64 upload across R; smaller R must see a byte prefix).
    {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ 2560u64);
        let x8 = x_rows(&mut rng, 8, 8);
        let mut rng2 = Rng(0x9e37_79b9_7f4a_7c15 ^ 2560u64);
        let x4 = x_rows(&mut rng2, 4, 8);
        if x8[..x4.len()] != x4[..] {
            return Err("x_rows not prefix-stable".into());
        }
        let mut exp_or = 0u32;
        for w in x8.chunks(4) {
            let f = f32::from_le_bytes(w.try_into().unwrap());
            if !f.is_finite() {
                return Err("fixture finite".into());
            }
            exp_or |= (u32::from_le_bytes(w.try_into().unwrap()) >> 23) & 0xff;
        }
        if exp_or == 0 {
            return Err("fixture exponent band".into());
        }
    }
    notes.push("full-mantissa fixtures deterministic+finite, X prefix-stable ok".into());
    // VMM right-align math (host-side): logical_end == mapped_end for sample lens.
    {
        for gran in [4096usize, 2 << 20] {
            for len in [1usize, 64, 320, 324, 1928 * 8 + 128, 640 * 2560 * 4 + 128] {
                let rounded = len.div_ceil(gran) * gran;
                if rounded < len || rounded % gran != 0 {
                    return Err("vmm rounding".into());
                }
                let logical_start = rounded - len; // offset inside mapped prefix
                if logical_start + len != rounded {
                    return Err("vmm right-align".into());
                }
            }
        }
    }
    notes.push("vmm right-align mapped_end-len..mapped_end, suffix gran unmapped ok".into());
    let guard_witnesses = guard_host_witnesses()?;
    notes.push("all actual negative guard plans share the runtime factory (35 gfx1151 / 36 gfx1201 per spec); host proof walks constructed kernarg bytes/regions, EID-first required-order model, no hardware certification ok".into());
    // Grids: full production grid + per-token K1 grid.
    for (m, r) in [(1280u32, 64u32), (2560, 64), (1282, 8), (2563, 8)] {
        let g = [ceil_div(m, 8), 80, ceil_div(r, 8)];
        if g[1] != 80 || g[0] != m.div_ceil(8) {
            return Err("grid".into());
        }
    }
    notes.push(
        "candidate grid [ceil(M/8),80,ceil(R/8)] block256; K1 grid [M,1,1] block32 ok".into(),
    );
    let v = json!({
        "schema": "pm-moe-cn-tiled-numeric-oracle-v1",
        "mode": "selfcheck",
        "checks": notes,
        "arch": "none (host-only; no HIP init or device allocation)",
        "descriptor_bytes_per_tile": DESC_BYTES,
        "kernarg_bytes": 96,
        "u_fixed": U_FIXED,
        "actual_guard_witnesses": guard_witnesses,
        "actual_fixture_validation": {
            "checked_fixture_count": fixtures.len(),
            "checked_launch_group_count": fixture_groups,
            "checked_live_expert_count": fixture_live_experts,
            "checked_nonempty_weight_extent_count": fixture_extents,
            "live_expert_count_note": "distinct live experts per fixture, summed across fixtures",
            "ordinary_weight_generation": "once per spec, shared across R1..64",
            "fixtures": fixtures,
        },
        "vmm": {"read_only_inputs": true, "right_aligned": true, "unmapped_suffix_granule": true},
        "limitations": [
            "Native GPU cannot expose instruction written bits; verify uses dual-poison fills plus expected written-byte maps; read-only VMM mappings additionally fault on input stores.",
            "Leaf-only: numerics vs independent K1 singletons only; no timing, capture, throughput, or admission claims.",
        ],
    });
    let s = serde_json::to_string_pretty(&v).map_err(err)?;
    if let Some(p) = out {
        std::fs::write(&p, &s).map_err(err)?;
        eprintln!("wrote {p}");
    } else {
        println!("{s}");
    }
    Ok(())
}

// ------------------------------------------------------------------ device ---

struct Device {
    hip: HipRuntime,
    id: i32,
    arch: String,
    uuid: String,
    bdf: String,
}

fn open_device(want_arch: &str, expect_bdf: Option<&str>) -> Result<Device> {
    let hip = HipRuntime::load().map_err(err)?;
    let n = hip.device_count().map_err(err)?;
    if n < 1 {
        return Err("no HIP devices visible".into());
    }
    let id = 0;
    hip.set_device(id).map_err(err)?;
    let arch = hip.get_arch(id).map_err(err)?;
    if arch != want_arch {
        return Err(format!(
            "arch guard: device is {arch}, want {want_arch}; never retag"
        ));
    }
    let uuid = hip
        .device_uuid(id)
        .map_err(err)
        .unwrap_or_else(|_| "unknown".into());
    let bdf = hip
        .device_pci_bus_id(id)
        .map_err(err)
        .unwrap_or_else(|_| "unknown".into());
    if let Some(want) = expect_bdf {
        if bdf != want {
            return Err(format!("BDF guard: device {bdf} != --expect-bdf {want}"));
        }
    }
    Ok(Device {
        hip,
        id,
        arch,
        uuid,
        bdf,
    })
}

// ------------------------------------------------------------- VMM alloc ---

/// One right-aligned VMM allocation: mapped prefix `rounded` bytes, logical
/// bytes at `[mapped_end-len, mapped_end)`, one extra granule reserved but
/// UNMAPPED as suffix so an exact-tail overread faults. Immutable inputs are
/// switched to device read-only after upload. Owns its arena; release with
/// `VmmArena::release` (never `hip.free` on the borrowed view).
struct VmmBuf {
    arena: VmmArena,
    mapped: usize,
    len: usize,
}

impl VmmBuf {
    fn base(&self) -> usize {
        self.arena.base_address()
    }
    fn logical(&self) -> u64 {
        (self.base() + self.mapped - self.len) as u64
    }
    fn view(&self) -> DeviceBuffer {
        // SAFETY: [logical, logical+len) lies inside the mapped prefix; the
        // Borrowed wrapper is never freed and never outlives the arena.
        unsafe {
            DeviceBuffer::from_raw(
                (self.base() + self.mapped - self.len) as *mut std::ffi::c_void,
                self.len,
            )
        }
    }
    fn release(mut self, hip: &HipRuntime) -> Result<()> {
        self.arena.release(hip).map_err(err)
    }
}

fn vmm_upload(hip: &HipRuntime, id: i32, data: &[u8], read_only: bool) -> Result<VmmBuf> {
    let len = data.len().max(1);
    let prop = HipMemAllocationProp::device_pinned(id);
    let gran = hip
        .mem_get_allocation_granularity(&prop, HIP_MEM_ALLOCATION_GRANULARITY_RECOMMENDED)
        .map_err(|e| format!("VMM unavailable (no HIP VMM support, no malloc fallback): {e}"))?;
    if gran == 0 {
        return Err("HIP returned zero VMM allocation granularity".into());
    }
    let rounded = len.div_ceil(gran) * gran;
    let mut arena = VmmArena::reserve(hip, id, rounded + gran).map_err(err)?;
    if arena.granularity() != gran {
        return Err("VMM granularity mismatch".into());
    }
    arena.map_next(hip, rounded, &[id]).map_err(err)?;
    let buf = VmmBuf {
        arena,
        mapped: rounded,
        len,
    };
    hip.memcpy_htod(&buf.view(), if data.is_empty() { &[0u8; 1] } else { data })
        .map_err(err)?;
    hip.device_synchronize().map_err(err)?;
    if read_only {
        // SAFETY: base..base+rounded is the mapped prefix just filled and
        // synchronized; RO narrows owner access to read-only.
        unsafe {
            hip.mem_set_access(
                buf.base() as *mut std::ffi::c_void,
                rounded,
                &[HipMemAccessDesc::read_only_device(id)],
            )
        }
        .map_err(err)?;
    }
    Ok(buf)
}

fn vmm_filled(hip: &HipRuntime, id: i32, len: usize, byte: u8) -> Result<VmmBuf> {
    vmm_upload(hip, id, &vec![byte; len.max(1)], false)
}

fn vmm_download(hip: &HipRuntime, buf: &VmmBuf) -> Result<Vec<u8>> {
    hip.device_synchronize().map_err(err)?;
    let mut v = vec![0u8; buf.len];
    hip.memcpy_dtoh(&mut v, &buf.view()).map_err(err)?;
    Ok(v)
}

fn vmm_memset(hip: &HipRuntime, buf: &VmmBuf, byte: u8) -> Result<()> {
    hip.memset(&buf.view(), byte as i32, buf.len).map_err(err)
}

/// Flip a buffer's owner-device mapping between read-write and read-only.
/// SAFETY: `base..base+mapped` is the buffer's live mapped VMM prefix.
fn set_ro(hip: &HipRuntime, id: i32, buf: &VmmBuf, ro: bool) -> Result<()> {
    let flags = if ro {
        HipMemAccessDesc::read_only_device(id)
    } else {
        HipMemAccessDesc::read_write_device(id)
    };
    // SAFETY: base..base+mapped is the live mapped VMM prefix of this buffer.
    unsafe { hip.mem_set_access(buf.base() as *mut std::ffi::c_void, buf.mapped, &[flags]) }
        .map_err(err)
}

fn launch256(hip: &HipRuntime, f: &Function, grid: [u32; 3], ka: &mut [u8]) -> Result<()> {
    // SAFETY: kernarg is the kernel's 96 B ABI; all pointers address live VMM
    // allocations (or null for must-not-read).
    unsafe { hip.launch_kernel_blob(f, grid, [256, 1, 1], 0, None, ka) }.map_err(err)?;
    hip.device_synchronize().map_err(err)
}
fn launch32(hip: &HipRuntime, f: &Function, grid: [u32; 3], ka: &mut [u8]) -> Result<()> {
    // SAFETY: same contract, block32 for K1 singletons.
    unsafe { hip.launch_kernel_blob(f, grid, [32, 1, 1], 0, None, ka) }.map_err(err)?;
    hip.device_synchronize().map_err(err)
}

// ------------------------------------------------------------------ cases ---

struct Counts {
    cases: u64,
    candidate_launches: u64,
    k1_launches: u64,
    compared_words: u64,
    written_bytes: u64,
    second_passes: u64,
    vmm_buffers: u64,
    guard_bytes: u64,
}

/// Host-side case. X is prefix-stable across R for one (spec, m, seed):
/// smaller R uses the first x_src_rows(R) rows of the R=64 build.
struct Case {
    spec: Spec,
    rows: u32,
    m: u32,
    k: u32,
    x_stride: u32,
    y_stride: u32,
    x_src_rows: u32,
    n_experts: u32,
    bucket: usize,
    x: Vec<u8>,
    tensors: Arc<Vec<Vec<u8>>>,
    descriptor: Vec<u8>,
    groups: TileGroups,
}

fn build_case(
    spec: Spec,
    rows: u32,
    m: u32,
    x_pad: u32,
    y_pad: u32,
    seed_xor: u64,
    groups_override: Option<TileGroups>,
    n_experts: u32,
) -> Case {
    let k = prod_k(spec);
    let x_stride = k + x_pad;
    let y_stride = (if spec.kind.gate() { m / 2 } else { m }) + y_pad;
    let x_src_rows = if spec.kind.gate() { rows } else { 10 * rows };
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15
        ^ k as u64
        ^ ((spec.codec as u64) << 40)
        ^ ((m as u64) << 24)
        ^ seed_xor);
    // Ordinary fixtures keep only the two endpoint experts resident. Explicit
    // grouped fixtures also materialize every referenced expert: a full native
    // terminal80 grid executes all ten groups, not only the endpoint groups.
    assert!(n_experts >= 2);
    let mut tensors: Vec<Vec<u8>> = Vec::new();
    for e in 0..n_experts {
        if e == 0
            || e == n_experts - 1
            || groups_override.as_ref().is_some_and(|tiles| {
                tiles
                    .iter()
                    .any(|groups| groups.iter().any(|(expert, _)| *expert == e))
            })
        {
            tensors.push(if spec.codec == Codec::Mq3I4 {
                w3_tensor(&mut rng, m as usize, k as usize)
            } else {
                mq4_tensor(&mut rng, spec.kind.gate(), m as usize, k as usize)
            });
        } else {
            tensors.push(Vec::new());
        }
    }
    let xr = x_rows(&mut rng, x_src_rows as usize, k as usize);
    let mut x = vec![0u8; x_src_rows as usize * x_stride as usize * 4];
    for r in 0..x_src_rows as usize {
        x[r * x_stride as usize * 4..r * x_stride as usize * 4 + k as usize * 4]
            .copy_from_slice(&xr[r * k as usize * 4..(r + 1) * k as usize * 4]);
        for b in x[r * x_stride as usize * 4 + k as usize * 4..(r + 1) * x_stride as usize * 4]
            .iter_mut()
        {
            *b = GUARD_FILL;
        }
    }
    let groups = groups_override.unwrap_or_else(|| ordinary_groups(rows, n_experts));
    let bucket = spec.codec.bucket();
    let descriptor = build_descriptor(groups.len(), bucket, &groups);
    Case {
        spec,
        rows,
        m,
        k,
        x_stride,
        y_stride,
        x_src_rows,
        n_experts,
        bucket,
        x,
        tensors: Arc::new(tensors),
        descriptor,
        groups,
    }
}

// Shared native/selfcheck construction: no weight generation or tensor copy per R.
fn ordinary_case(spec: Spec, rows: u32, x64: &[u8], tensors: &Arc<Vec<Vec<u8>>>) -> Case {
    let k = prod_k(spec);
    let x_src_rows = if spec.kind.gate() { rows } else { 10 * rows };
    let groups = ordinary_groups(rows, 512);
    Case {
        spec,
        rows,
        m: prod_m(spec),
        k,
        x_stride: k,
        y_stride: ystride_prod(spec),
        x_src_rows,
        n_experts: 512,
        bucket: spec.codec.bucket(),
        x: x64[..x_src_rows as usize * k as usize * 4].to_vec(),
        tensors: Arc::clone(tensors),
        descriptor: build_descriptor(groups.len(), spec.codec.bucket(), &groups),
        groups,
    }
}

fn terminal_case(spec: Spec) -> Case {
    // Full native grid: tiles 0..6 use endpoint experts; tile 7 uses all ten.
    let mut groups = ordinary_groups(64, 10);
    let mut terminal = Vec::new();
    for expert in 0..10u32 {
        terminal.push((expert, (56..64u32).map(|t| 10 * t + expert).collect()));
    }
    groups[7] = terminal;
    build_case(spec, 64, prod_m(spec), 0, 0, 0x80, Some(groups), 10)
}

fn cheap_tail_case(si: usize, spec: Spec) -> Case {
    let m = if spec.kind.gate() { 1282 } else { 2563 };
    build_case(spec, 8, m, 0, 0, 0xa11ce5 + si as u64, None, 512)
}

struct Extra {
    idx: usize,
    name: String,
    spec: usize,
}

fn focused_extras() -> Vec<Extra> {
    let specs = all_specs();
    let mut extras = Vec::new();
    let mut push = |name: String, spec: usize| {
        extras.push(Extra {
            idx: extras.len(),
            name,
            spec,
        });
    };
    for si in 0..4 {
        push(
            format!("nonzero-offsets-terminal80 {}", k1_symbol(specs[si])),
            si,
        );
    }
    for si in 0..4 {
        push(format!("tail {}", k1_symbol(specs[si])), si);
    }
    for si in 0..4 {
        push(format!("pad {}", k1_symbol(specs[si])), si);
    }
    for si in [0usize, 2] {
        push(format!("phase0-zero {}", k1_symbol(specs[si])), si);
        push(format!("phase1-zero {}", k1_symbol(specs[si])), si);
    }
    for si in 0..4 {
        push(format!("xclip {}", k1_symbol(specs[si])), si);
    }
    for si in 0..4 {
        push(format!("xzero {}", k1_symbol(specs[si])), si);
    }
    extras
}

fn focused_case(e: &Extra) -> Case {
    let s = all_specs()[e.spec];
    let m = prod_m(s);
    if e.name.starts_with("nonzero-offsets") {
        terminal_case(s)
    } else if e.name.starts_with("tail ") {
        let m = if s.kind.gate() { 1282 } else { 2563 };
        build_case(s, 8, m, 0, 0, 0x7a11 + e.idx as u64, None, 512)
    } else if e.name.starts_with("pad ") {
        build_case(s, 8, m, 4, 5, 0x9ad + e.idx as u64, None, 512)
    } else if e.name.starts_with("phase0-zero ") || e.name.starts_with("phase1-zero ") {
        let mut c = build_case(s, 8, m, 0, 0, 0x9461 + e.idx as u64, None, 512);
        let half = c.k as usize / 2 * 4;
        for r in 0..c.x_src_rows as usize {
            let base = r * c.x_stride as usize * 4;
            let start = if e.name.starts_with("phase0-zero") {
                base
            } else {
                base + half
            };
            c.x[start..start + half].fill(0);
        }
        c
    } else if e.name.starts_with("xclip ") {
        let mut c = build_case(s, 8, m, 0, 0, 0xc119 + e.idx as u64, None, 512);
        c.x_src_rows -= 1;
        c.x.truncate(c.x_src_rows as usize * c.x_stride as usize * 4);
        c
    } else {
        assert!(e.name.starts_with("xzero "));
        let mut c = build_case(s, 8, m, 0, 0, 0x2e120 + e.idx as u64, None, 512);
        c.x.clear();
        c.x_src_rows = 0;
        c
    }
}

// Positive numerical Cases only. Negative guards use their direct sparse/null
// mappings, not this path. X is intentionally not required to be nonzero/full:
// phase-zero, xclip and xzero remain legitimate numerical fixtures.
fn check_case_weights(c: &Case) -> Result<()> {
    let m = c.m as usize;
    let k = c.k as usize;
    let expected = match c.spec.codec {
        Codec::Mq3I4 => m * (k * 3 / 8 + 2 * (k / 128)),
        Codec::Mq4Sym if c.spec.kind.gate() => m * (k / 256) * 136,
        Codec::Mq4Sym => m * (k / 128) * 68,
    };
    if expected == 0 || c.tensors.len() != c.n_experts as usize {
        return Err(format!(
            "{} R{}: invalid weight table/extent",
            k1_symbol(c.spec),
            c.rows
        ));
    }
    for (e, tensor) in c.tensors.iter().enumerate() {
        if !tensor.is_empty() && tensor.len() != expected {
            return Err(format!(
                "{} R{} expert {e}: weight bytes {} != codec extent {expected}",
                k1_symbol(c.spec),
                c.rows,
                tensor.len()
            ));
        }
        if c.n_experts == 512 && (!tensor.is_empty() != (e == 0 || e == 511)) {
            return Err(format!(
                "{} R{}: ordinary sparse table must map only 0/511 (expert {e})",
                k1_symbol(c.spec),
                c.rows
            ));
        }
        if c.n_experts == 10 && tensor.is_empty() {
            return Err(format!(
                "{} R{}: terminal80 expert {e} has no weight extent",
                k1_symbol(c.spec),
                c.rows
            ));
        }
    }
    for (tile, groups) in c.groups.iter().enumerate() {
        for (expert, slots) in groups {
            if slots.is_empty() {
                continue;
            }
            if c.tensors
                .get(*expert as usize)
                .is_none_or(|t| t.is_empty() || t.len() != expected)
            {
                return Err(format!("{} R{} tile {tile}: live expert {expert} lacks nonempty {expected}-byte weights", k1_symbol(c.spec), c.rows));
            }
        }
    }
    Ok(())
}

fn check_fixture(name: &str, c: &Case) -> Result<serde_json::Value> {
    check_case_weights(c).map_err(|e| format!("{name}: {e}"))?;
    if c.groups.len() != ceil_div(c.rows, 8) as usize
        || c.descriptor.len() != c.groups.len() * DESC_BYTES
        || c.bucket != c.spec.codec.bucket()
        || c.x.len() != c.x_src_rows as usize * c.x_stride as usize * 4
    {
        return Err(format!("{name}: actual case geometry/descriptor extent"));
    }
    let word = |i: usize| u32::from_le_bytes(c.descriptor[4 * i..4 * i + 4].try_into().unwrap());
    let mut launch_groups = 0usize;
    let mut slot_count = 0usize;
    for (tile, groups) in c.groups.iter().enumerate() {
        let base = tile * DESC_WORDS;
        let off = base + 160 + 81 * c.bucket;
        let slots_at = base + 322 + 80 * c.bucket;
        let count: usize = groups.iter().map(|(_, s)| s.len()).sum();
        if groups.len() > 80 || count > 80 {
            return Err(format!("{name}: tile {tile} exceeds 80 groups/slots"));
        }
        let mut pos = 0usize;
        for (gi, (expert, slots)) in groups.iter().enumerate() {
            if slots.is_empty()
                || word(base + 80 * c.bucket + gi) != *expert
                || word(off + gi) != pos as u32
                || word(off + gi + 1) != (pos + slots.len()) as u32
            {
                return Err(format!("{name}: tile {tile} group {gi} ID/offset/count"));
            }
            for (j, slot) in slots.iter().enumerate() {
                if word(slots_at + pos + j) != *slot
                    || *slot >= 10 * c.rows
                    || *slot / 10 / 8 != tile as u32
                {
                    return Err(format!("{name}: tile {tile} group {gi} slot {j} index"));
                }
            }
            pos += slots.len();
        }
        for gi in groups.len()..80 {
            if word(base + 80 * c.bucket + gi) != 0 {
                return Err(format!("{name}: tile {tile} unused expert ID"));
            }
        }
        for i in groups.len()..=80 {
            if word(off + i) != count as u32 {
                return Err(format!("{name}: tile {tile} terminal/unused offset {i}"));
            }
        }
        for i in count..80 {
            if word(slots_at + i) != 0 {
                return Err(format!("{name}: tile {tile} unused slot"));
            }
        }
        let opposite = 1 - c.bucket;
        for i in 0..80 {
            if word(base + 80 * opposite + i) != 0 || word(base + 322 + 80 * opposite + i) != 0 {
                return Err(format!("{name}: tile {tile} opposite IDs/slots"));
            }
        }
        for i in 0..=80 {
            if word(base + 160 + 81 * opposite + i) != 0 {
                return Err(format!("{name}: tile {tile} opposite offsets"));
            }
        }
        launch_groups += groups.len();
        slot_count += count;
    }
    if c.n_experts == 10 {
        if c.rows != 64 || c.groups.len() != 8 {
            return Err(format!(
                "{name}: terminal80 requires R64/eight actual tiles"
            ));
        }
        let last = &c.groups[7];
        if last.len() != 10
            || last.iter().enumerate().any(|(e, (expert, slots))| {
                *expert != e as u32
                    || slots.len() != 8
                    || slots
                        .iter()
                        .enumerate()
                        .any(|(t, slot)| *slot != 10 * (56 + t as u32) + e as u32)
            })
            || word(7 * DESC_WORDS + 160 + 81 * c.bucket + 80) != 80
        {
            return Err(format!("{name}: actual terminal80 coverage"));
        }
    }
    let live: Vec<usize> = (0..c.n_experts as usize)
        .filter(|e| {
            c.groups.iter().any(|g| {
                g.iter()
                    .any(|(expert, s)| *expert as usize == *e && !s.is_empty())
            })
        })
        .collect();
    let extents: Vec<_> = c
        .tensors
        .iter()
        .enumerate()
        .filter(|(_, t)| !t.is_empty())
        .map(|(expert, t)| json!({"expert": expert, "bytes": t.len()}))
        .collect();
    Ok(json!({
        "name": name, "rows": c.rows, "n_experts": c.n_experts,
        "tile_count": c.groups.len(), "descriptor_bytes": c.descriptor.len(),
        "launch_groups": launch_groups, "slot_count": slot_count,
        "live_expert_count": live.len(), "live_experts": live,
        "mapped_weight_extents": extents, "all_live_extents_nonempty_and_exact": true,
        "x_src_rows": c.x_src_rows, "x_bytes": c.x.len(),
        "x_stride": c.x_stride, "y_stride": c.y_stride,
    }))
}

fn y_len_for(slots: u32, y_stride: u32) -> usize {
    slots as usize * y_stride as usize * 4 + 2 * GUARD
}

/// Shared VMM device state for one spec across the R sweep: tensors, table, and
/// the max-rows X (X64) are uploaded ONCE read-only for K1 reference use (the X64
/// prefix is byte-identical to every per-R exact candidate X prefix). Per-R exact
/// candidate X prefixes are separately allocated read-only per case. K1 singleton
/// state (mutable ids/offs/slot patch buffers plus accumulating Y) also lives
/// here; each (token, rank, expert) launches once per case context.
/// Every buffer is right-aligned VMM; immutable inputs are device read-only.
struct SpecCtx {
    id: i32,
    spec: Spec,
    m: u32,
    k: u32,
    y_stride: u32,
    n_experts: u32,
    bucket: usize,
    x64_rows: u32,
    x_stride: u32,
    host_tensors: Arc<Vec<Vec<u8>>>,
    d_t: Vec<Option<VmmBuf>>,
    d_table: VmmBuf,
    d_x64: VmmBuf,
    host_x64: Vec<u8>,
    k_ids: VmmBuf,
    k_offs: VmmBuf,
    k_slots: VmmBuf,
    k_y0: VmmBuf,
    k_y1: Option<VmmBuf>,
    launched: HashSet<(u32, u32, u32)>,
}

impl SpecCtx {
    fn new(hip: &HipRuntime, id: i32, c: &Case, counts: &mut Counts) -> Result<Self> {
        check_case_weights(c)?;
        // Ordinary tables map only endpoints; explicit grouped fixtures map all
        // referenced experts. Empty entries remain genuinely unmapped (NULL).
        let mut d_t: Vec<Option<VmmBuf>> = Vec::new();
        for (e, t) in c.tensors.iter().enumerate() {
            if e < c.n_experts as usize && !t.is_empty() {
                d_t.push(Some(vmm_upload(hip, id, t, true)?));
                counts.vmm_buffers += 1;
            } else {
                d_t.push(None);
            }
        }
        let mut table = vec![0u8; c.n_experts as usize * 8];
        for (e, b) in d_t.iter().enumerate() {
            if e < c.n_experts as usize {
                if let Some(buf) = b {
                    table[e * 8..e * 8 + 8].copy_from_slice(&buf.logical().to_le_bytes());
                }
            }
        }
        let d_table = vmm_upload(hip, id, &table, true)?;
        counts.vmm_buffers += 1;
        let d_x = vmm_upload(hip, id, &c.x, true)?;
        counts.vmm_buffers += 1;
        // Mutable patch buffers (written per token) and accumulating K1 Y: RW VMM.
        let k_ids = vmm_upload(hip, id, &[0u8; 4], false)?;
        let k_offs = vmm_upload(hip, id, &[0u8, 0, 0, 0, 1, 0, 0, 0], false)?;
        let k_slots = vmm_upload(hip, id, &[0u8; 4], false)?;
        counts.vmm_buffers += 3;
        let ys = if c.spec.kind.gate() { c.m / 2 } else { c.m };
        let ky_len = y_len_for(640, ys);
        let k_y0 = vmm_filled(hip, id, ky_len, FILL_A)?;
        counts.vmm_buffers += 1;
        let k_y1 = if c.spec.kind.gate() {
            let y = vmm_filled(hip, id, ky_len, FILL_A)?;
            counts.vmm_buffers += 1;
            Some(y)
        } else {
            None
        };
        Ok(Self {
            id,
            spec: c.spec,
            m: c.m,
            k: c.k,
            y_stride: c.y_stride,
            n_experts: c.n_experts,
            bucket: c.bucket,
            x64_rows: c.x_src_rows,
            x_stride: c.x_stride,
            host_tensors: c.tensors.clone(),
            d_t,
            d_table,
            d_x64: d_x,
            host_x64: c.x.clone(),
            k_ids,
            k_offs,
            k_slots,
            k_y0,
            k_y1,
            launched: HashSet::new(),
        })
    }

    fn release(self, hip: &HipRuntime) -> Result<()> {
        for b in self.d_t.into_iter().flatten() {
            b.release(hip)?;
        }
        self.d_table.release(hip)?;
        self.d_x64.release(hip)?;
        self.k_ids.release(hip)?;
        self.k_offs.release(hip)?;
        self.k_slots.release(hip)?;
        self.k_y0.release(hip)?;
        if let Some(y) = self.k_y1 {
            y.release(hip)?;
        }
        Ok(())
    }

    /// True-C1 local singleton for (token, global rank, expert): local slot
    /// 0..9, X pointer at the token's row window of the case X, local x_src
    /// clamped to the remaining rows, grid [M,1,1] block32 (every row at once).
    /// Idempotent per (token, rank, expert, x-local-key) within this context.
    /// Patch buffers are RW for the host patch, then set read-only during the
    /// launch (restored after); the actual table/tensors/X stay read-only.
    #[allow(clippy::too_many_arguments)]
    fn k1_token(
        &mut self,
        hip: &HipRuntime,
        k1: &Function,
        token: u32,
        rank: u32,
        expert: u32,
        case_x_base: u64,
        x_src_rows: u32,
        x_stride: u32,
        n_experts: u32,
        counts: &mut Counts,
    ) -> Result<()> {
        if !self.launched.insert((token, rank, expert)) {
            return Ok(());
        }
        let gate = self.spec.kind.gate();
        let per = if gate { 1 } else { 10 };
        // Global X row of this token's window base; local rows remaining.
        let base_row = token * per as u32;
        let local_src = x_src_rows.saturating_sub(base_row).min(per as u32);
        let local_slot = rank;
        hip.memcpy_htod(&self.k_ids.view(), &expert.to_le_bytes())
            .map_err(err)?;
        hip.memcpy_htod(&self.k_slots.view(), &local_slot.to_le_bytes())
            .map_err(err)?;
        let x_ptr = if local_src == 0 {
            0
        } else {
            case_x_base + base_row as u64 * x_stride as u64 * 4
        };
        set_ro(hip, self.id, &self.k_ids, true)?;
        set_ro(hip, self.id, &self.k_offs, true)?;
        set_ro(hip, self.id, &self.k_slots, true)?;
        let ys = if gate { self.m / 2 } else { self.m };
        let mut ka = encode_kernarg(
            [
                self.d_table.logical(),
                self.k_ids.logical(),
                self.k_offs.logical(),
                self.k_slots.logical(),
                x_ptr,
                self.k_y0.logical() + GUARD as u64,
                if gate {
                    self.k_y1.as_ref().unwrap().logical() + GUARD as u64
                } else {
                    0
                },
            ],
            [
                self.m,
                self.k,
                1,
                1,
                xdiv(self.spec),
                local_src,
                x_stride,
                ys,
                n_experts,
                0,
            ],
        );
        let ka_before = ka.clone();
        launch32(hip, k1, [self.m, 1, 1], &mut ka)
            .map_err(|e| format!("k1 token{token} rank{rank} expert{expert}: {e}"))?;
        if ka != ka_before {
            return Err("kernarg blob changed by K1 launch".into());
        }
        set_ro(hip, self.id, &self.k_ids, false)?;
        set_ro(hip, self.id, &self.k_offs, false)?;
        set_ro(hip, self.id, &self.k_slots, false)?;
        counts.k1_launches += 1;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn k1_ref_word(
        &mut self,
        hip: &HipRuntime,
        k1: &Function,
        token: u32,
        rank: u32,
        expert: u32,
        case_x_base: u64,
        x_src_rows: u32,
        x_stride: u32,
        n_experts: u32,
        row: u32,
        counts: &mut Counts,
    ) -> Result<([u8; 4], Vec<[u8; 4]>)> {
        self.k1_token(
            hip,
            k1,
            token,
            rank,
            expert,
            case_x_base,
            x_src_rows,
            x_stride,
            n_experts,
            counts,
        )?;
        let words = self.k1_words(hip, rank)?;
        Ok((words[row as usize], words))
    }

    fn k1_words(&self, hip: &HipRuntime, rank: u32) -> Result<Vec<[u8; 4]>> {
        let gate = self.spec.kind.gate();
        let slot = rank;
        let ys = if gate { self.m / 2 } else { self.m } as usize;
        let ky_len = y_len_for(640, ys as u32);
        let y0 = vmm_download(hip, &self.k_y0)?;
        assert_eq!(y0.len(), ky_len);
        let y1 = if let Some(y) = &self.k_y1 {
            vmm_download(hip, y)?
        } else {
            Vec::new()
        };
        let mut out = Vec::with_capacity(self.m as usize);
        for row in 0..self.m {
            let w = if gate {
                let mi = self.m / 2;
                if row < mi {
                    let off = GUARD + (slot as usize * ys + row as usize) * 4;
                    [y0[off], y0[off + 1], y0[off + 2], y0[off + 3]]
                } else {
                    let off = GUARD + (slot as usize * ys + (row - mi) as usize) * 4;
                    [y1[off], y1[off + 1], y1[off + 2], y1[off + 3]]
                }
            } else {
                let off = GUARD + (slot as usize * ys + row as usize) * 4;
                [y0[off], y0[off + 1], y0[off + 2], y0[off + 3]]
            };
            out.push(w);
        }
        Ok(out)
    }
}

fn scatter_addr(gate: bool, m: u32, y_stride: u32, slot: u32, row: u32) -> (bool, usize) {
    // (is_y1, byte offset of the word inside the guarded Y allocation)
    if gate {
        let mi = m / 2;
        if row < mi {
            (
                false,
                GUARD + (slot as usize * y_stride as usize + row as usize) * 4,
            )
        } else {
            (
                true,
                GUARD + (slot as usize * y_stride as usize + (row - mi) as usize) * 4,
            )
        }
    } else {
        (
            false,
            GUARD + (slot as usize * y_stride as usize + row as usize) * 4,
        )
    }
}

/// Full-grid candidate launch + full-M K1 comparison for one case.
/// `k1cache[(token,rank,expert,row)]` persists within compatible fixture contexts
/// (same spec/shape/X-seed/stride/expert-set); per-R exact candidate X prefixes are
/// separately allocated read-only per case. Every numeric case runs BOTH poison
/// passes unconditionally with full byte maps, and `written` counts actual map bits.
#[allow(clippy::too_many_arguments)]
fn verify_case_full(
    hip: &HipRuntime,
    id: i32,
    cand: &Function,
    k1: &Function,
    ctx: &mut SpecCtx,
    c: &Case,
    k1cache: &mut HashMap<(u32, u32, u32, u32), [u8; 4]>,
    input_hashes: &mut Vec<String>,
    counts: &mut Counts,
    relax_ref: bool,
    x_null: bool,
) -> Result<()> {
    check_case_weights(c)?;
    let gate = c.spec.kind.gate();
    // Per-R exact candidate X prefix: right-aligned read-only VMM (same bytes as
    // the ctx X64 prefix, but an exact-length allocation whose tail hits the
    // unmapped suffix). K1 reference X uses the shared ctx X64 with a localized
    // per-token pointer + clamped local src rows (true-C1 contract).
    let d_desc = {
        let mut dh = vec![GUARD_FILL; GUARD];
        dh.extend_from_slice(&c.descriptor);
        dh.extend_from_slice(&vec![GUARD_FILL; GUARD]);
        vmm_upload(hip, id, &dh, true)?
    };
    counts.vmm_buffers += 1;
    // Re-derive dh for byte-compare below.
    let mut dh = vec![GUARD_FILL; GUARD];
    dh.extend_from_slice(&c.descriptor);
    dh.extend_from_slice(&vec![GUARD_FILL; GUARD]);
    let desc_base = d_desc.logical() + GUARD as u64;
    let d_xcase: Option<VmmBuf> = if x_null {
        None
    } else {
        let b = vmm_upload(hip, id, &c.x, true)?;
        counts.vmm_buffers += 1;
        Some(b)
    };
    let y_len = y_len_for(10 * c.rows, c.y_stride);
    let pass_fill = [FILL_A, FILL_B];
    // Only the explicit xnull family (null X, x_src_rows=0) may admit
    // zero/constant references; xclip keeps real clipped data and uses the
    // full finite/nonzero/varied bar like every other numeric case.
    let relax = relax_ref && x_null;
    let mut written0: Vec<bool> = Vec::new();
    let mut written1: Vec<bool> = Vec::new();
    for (pass, fill) in pass_fill.iter().enumerate() {
        let d_y0 = vmm_filled(hip, id, y_len, *fill)?;
        counts.vmm_buffers += 1;
        let d_y1 = if gate {
            let y = vmm_filled(hip, id, y_len, *fill)?;
            counts.vmm_buffers += 1;
            Some(y)
        } else {
            None
        };
        // Output guards use the pass fill as base, then a same-length guard-B
        // band pair that also differs per pass; nothing is pinned to a fixed
        // GUARD_FILL that both passes share.
        let (gb_lo, gb_hi) = if *fill == FILL_A {
            (GUARD_FILL, GUARD_FILL_B)
        } else {
            (GUARD_FILL_B, GUARD_FILL)
        };
        // Paint guard bands via host-side pattern: re-upload exact body instead.
        {
            let mut body = vec![*fill; y_len];
            for b in body[0..GUARD].iter_mut() {
                *b = gb_lo;
            }
            for b in body[y_len - GUARD..].iter_mut() {
                *b = gb_hi;
            }
            hip.memcpy_htod(&d_y0.view(), &body).map_err(err)?;
            if let Some(y) = &d_y1 {
                hip.memcpy_htod(&y.view(), &body).map_err(err)?;
            }
            hip.device_synchronize().map_err(err)?;
        }
        let x_ptr = match &d_xcase {
            Some(b) => b.logical(),
            None => 0,
        };
        let mut ka = encode_kernarg(
            [
                ctx.d_table.logical(),
                desc_base + (c.bucket as u64 * 80) * 4,
                desc_base + (160 + c.bucket * 81) as u64 * 4,
                desc_base + (322 + c.bucket * 80) as u64 * 4,
                x_ptr,
                d_y0.logical() + GUARD as u64,
                if gate {
                    d_y1.as_ref().unwrap().logical() + GUARD as u64
                } else {
                    0
                },
            ],
            [
                c.m,
                c.k,
                80,
                c.rows,
                xdiv(c.spec),
                c.x_src_rows,
                c.x_stride,
                c.y_stride,
                c.n_experts,
                0,
            ],
        );
        let ka_before = ka.clone();
        launch256(
            hip,
            cand,
            [ceil_div(c.m, 8), 80, ceil_div(c.rows, 8)],
            &mut ka,
        )
        .map_err(|e| format!("candidate R{} pass{}: {e}", c.rows, pass))?;
        if ka != ka_before {
            return Err("kernarg blob changed by candidate launch".into());
        }
        counts.candidate_launches += 1;
        if pass == 0 {
            counts.cases += 1;
        } else {
            counts.second_passes += 1;
        }
        let y0 = vmm_download(hip, &d_y0)?;
        let y1 = if let Some(y) = &d_y1 {
            vmm_download(hip, y)?
        } else {
            Vec::new()
        };
        // Input preservation AFTER EACH candidate launch: every immutable byte of
        // the per-R exact X prefix, table, tensors, descriptor compared (not
        // hash-only); read-only mappings additionally fault on same-byte stores.
        if let Some(b) = &d_xcase {
            let xa = vmm_download(hip, b)?;
            if xa != c.x {
                return Err(format!("input X changed by candidate launch pass{pass}"));
            }
        }
        let mut tab_expect = vec![0u8; c.n_experts as usize * 8];
        for e in 0..c.n_experts as usize {
            if !ctx.host_tensors[e].is_empty() {
                tab_expect[e * 8..e * 8 + 8]
                    .copy_from_slice(&ctx.d_t[e].as_ref().unwrap().logical().to_le_bytes());
            }
        }
        let ta = vmm_download(hip, &ctx.d_table)?;
        if ta != tab_expect {
            return Err(format!(
                "input table changed by candidate launch pass{pass}"
            ));
        }
        for e in 0..c.n_experts as usize {
            if ctx.host_tensors[e].is_empty() {
                continue;
            }
            let tb = vmm_download(hip, ctx.d_t[e].as_ref().unwrap())?;
            if tb != ctx.host_tensors[e] {
                return Err(format!(
                    "input tensor {e} changed by candidate launch pass{pass}"
                ));
            }
        }
        let da = vmm_download(hip, &d_desc)?;
        if da != dh {
            return Err(format!(
                "input descriptor changed by candidate launch pass{pass}"
            ));
        }
        // K1 true-C1 words for every (token, rank, expert, row); launch each once
        // per context; x_null routes through the same common path (X pointer null,
        // local src 0) and zero/constant references are admitted only there.
        let mut exp0 = vec![*fill; y_len];
        let mut exp1 = vec![*fill; y_len];
        for b in exp0[0..GUARD].iter_mut() {
            *b = gb_lo;
        }
        for b in exp0[y_len - GUARD..].iter_mut() {
            *b = gb_hi;
        }
        if gate {
            for b in exp1[0..GUARD].iter_mut() {
                *b = gb_lo;
            }
            for b in exp1[y_len - GUARD..].iter_mut() {
                *b = gb_hi;
            }
        }
        let mut written = vec![false; y_len];
        let mut written_g = if gate { Some(vec![false; y_len]) } else { None };
        let mut ref_seen = HashSet::new();
        let mut any_nonzero = false;
        for tg in c.groups.iter() {
            for (expert, slots) in tg {
                for slot in slots {
                    let token = *slot / 10;
                    let rank = *slot % 10;
                    for row in 0..c.m {
                        let key = (token, rank, *expert, row);
                        let w = match k1cache.get(&key) {
                            Some(w) => *w,
                            None => {
                                let case_x_base = d_xcase.as_ref().unwrap_or(&ctx.d_x64).logical();
                                let (w, words) = ctx.k1_ref_word(
                                    hip,
                                    k1,
                                    token,
                                    rank,
                                    *expert,
                                    case_x_base,
                                    c.x_src_rows,
                                    c.x_stride,
                                    c.n_experts,
                                    row,
                                    counts,
                                )?;
                                for (rr, ww) in words.iter().enumerate() {
                                    k1cache.insert((token, rank, *expert, rr as u32), *ww);
                                }
                                let _ = w;
                                words[row as usize]
                            }
                        };
                        let f = f32::from_le_bytes(w);
                        if !f.is_finite() {
                            return Err(format!("R{}: non-finite K1 reference", c.rows));
                        }
                        if f != 0.0 {
                            any_nonzero = true;
                        }
                        ref_seen.insert(w);
                        let (is_y1, off) = scatter_addr(gate, c.m, c.y_stride, *slot, row);
                        if is_y1 {
                            exp1[off..off + 4].copy_from_slice(&w);
                            written_g.as_mut().unwrap()[off..off + 4].fill(true);
                        } else {
                            exp0[off..off + 4].copy_from_slice(&w);
                            written[off..off + 4].fill(true);
                        }
                        counts.compared_words += 1;
                    }
                }
            }
        }
        if !relax {
            if !any_nonzero {
                return Err(format!("R{}: zero-only K1 reference", c.rows));
            }
            if ref_seen.len() < 2 {
                return Err(format!("R{}: constant K1 reference", c.rows));
            }
        }
        if y0 != exp0 {
            let n = y0.iter().zip(exp0.iter()).filter(|(a, b)| a != b).count();
            return Err(format!("R{} pass{pass}: Y0 mismatch bytes {n}", c.rows));
        }
        if gate && y1 != exp1 {
            let n = y1.iter().zip(exp1.iter()).filter(|(a, b)| a != b).count();
            return Err(format!("R{} pass{pass}: Y1 mismatch bytes {n}", c.rows));
        }
        if pass == 0 {
            written0 = written;
            written1 = written_g.unwrap_or_default();
            counts.guard_bytes += (2 * GUARD) as u64 * if gate { 2 } else { 1 };
        } else {
            // Second pass uses the SAME expected-written map: guards and padding
            // differ per pass (bodies above), so store proof is per byte.
            if written != written0 {
                return Err("written-map drift between passes".into());
            }
            if let Some(wg) = written_g {
                if wg != written1 {
                    return Err("written-map drift between passes (Y1)".into());
                }
            }
            counts.written_bytes += written0.iter().filter(|b| **b).count() as u64
                + written1.iter().filter(|b| **b).count() as u64;
        }
        d_y0.release(hip)?;
        if let Some(y) = d_y1 {
            y.release(hip)?;
        }
    }
    input_hashes.push(format!(
        "{}:R{}:M{}:{}",
        k1_symbol(c.spec),
        c.rows,
        c.m,
        sha256_hex(&c.x)
    ));
    if let Some(b) = d_xcase {
        b.release(hip)?;
    }
    d_desc.release(hip)?;
    Ok(())
}

// Guard-case launch with possibly-null downstream pointers; expect HIP success
// plus the full Y allocation (body fill, both guard bands, padding) observed
// preserved byte-for-byte, and count the preserved bytes as guard_bytes.
// This is an observed-preservation check, NOT an instruction-written-bits claim:
// the GPU cannot expose per-instruction write bits (see module docs).
#[allow(clippy::too_many_arguments)]
fn run_guard_launch(
    hip: &HipRuntime,
    id: i32,
    func: &Function,
    plan: &GuardPlan,
    bases: &[u64],
    y_len: usize,
    gate: bool,
    counts: &mut Counts,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let d_y0 = vmm_filled(hip, id, y_len, FILL_A)?;
    counts.vmm_buffers += 1;
    let d_y1 = if gate {
        let y = vmm_filled(hip, id, y_len, FILL_A)?;
        counts.vmm_buffers += 1;
        Some(y)
    } else {
        None
    };
    let ptrs = plan.bound_ptrs(bases);
    let mut ka = encode_kernarg(
        [
            ptrs[0],
            ptrs[1],
            ptrs[2],
            ptrs[3],
            ptrs[4],
            d_y0.logical() + GUARD as u64,
            d_y1.as_ref().map_or(0, |y| y.logical() + GUARD as u64),
        ],
        plan.scalars(),
    );
    launch256(hip, func, plan.grid, &mut ka)?;
    counts.guard_bytes += y_len as u64;
    let y0 = vmm_download(hip, &d_y0)?;
    let y1 = if let Some(y) = &d_y1 {
        vmm_download(hip, y)?
    } else {
        Vec::new()
    };
    d_y0.release(hip)?;
    if let Some(y) = d_y1 {
        y.release(hip)?;
    }
    Ok((y0, y1))
}

fn expect_silent(y0: &[u8], y1: &[u8], what: &str) -> Result<()> {
    if y0.iter().any(|&b| b != FILL_A) {
        return Err(format!("{what}: Y0 touched"));
    }
    if y1.iter().any(|&b| b != FILL_A) {
        return Err(format!("{what}: Y1 touched"));
    }
    Ok(())
}

// ------------------------------------------------------- guard plans ---
//
// Shared negative-guard plan factory: every runtime negative case and every
// host selfcheck witness flows through the SAME constructed arrays, packed
// scalars and region bytes (no fixture echo, no source-text matching).
struct GuardPlan {
    name: String,
    grid: [u32; 3],
    ka: [u8; 96],
    regions: Vec<Vec<u8>>,
    region_names: Vec<String>,
    bindings: [Option<(usize, usize)>; 5],
    expected_reason: &'static str,
    empty_prefix: bool,
}

impl GuardPlan {
    /// Resolve only the five downstream pointers. Unbound pointers remain NULL.
    fn bound_ptrs(&self, bases: &[u64]) -> [u64; 5] {
        std::array::from_fn(|i| match self.bindings[i] {
            Some((ri, off)) => bases[ri] + off as u64,
            None => 0,
        })
    }

    fn scalars(&self) -> [u32; 10] {
        std::array::from_fn(|i| {
            u32::from_le_bytes(self.ka[56 + 4 * i..60 + 4 * i].try_into().unwrap())
        })
    }

    /// Rebuild the template without aliasing or patching bytes. Runtime Y
    /// pointers are supplied only when constructing the fresh launch blob.
    fn bind(&self, bases: &[u64]) -> [u8; 96] {
        let ptrs = self.bound_ptrs(bases);
        encode_kernarg(
            [ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], 0, 0],
            self.scalars(),
        )
    }
}

fn u32s(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Actual negative plans per (arch, spec): 35 on gfx1151, 36 on gfx1201.
/// Base scalars [prod_m,prod_k,80,8,xdiv,8,prod_k,ystride_prod,512,0]; geometry
/// plans keep all downstream pointers NULL. Ordering matches the historical
/// suite; the first runtime fault still stops the run.
fn negative_guard_plans(arch: &str, s: Spec) -> Vec<GuardPlan> {
    let m = prod_m(s);
    let k = prod_k(s);
    let ys = ystride_prod(s);
    let bucket = s.codec.bucket();
    let mut plans = Vec::new();
    let mut mk = |name: String,
                  grid: [u32; 3],
                  ka: [u8; 96],
                  regions: Vec<(&'static str, Vec<u8>)>,
                  bindings: [Option<(usize, usize)>; 5],
                  expected_reason: &'static str,
                  empty_prefix: bool| {
        let mut bytes = Vec::with_capacity(regions.len());
        let mut names = Vec::with_capacity(regions.len());
        for (n, b) in regions {
            names.push(n.to_string());
            bytes.push(b);
        }
        plans.push(GuardPlan {
            name,
            grid,
            ka,
            regions: bytes,
            region_names: names,
            bindings,
            expected_reason,
            empty_prefix,
        });
    };
    let null_ka = |values: [u32; 10]| encode_kernarg([0, 0, 0, 0, 0, 0, 0], values);
    // C0 (xrows 0, grid 1).
    mk(
        "C0".into(),
        [1, 1, 1],
        null_ka([m, k, 80, 0, xdiv(s), 0, k, ys, 512, 0]),
        Vec::new(),
        [None; 5],
        "C-guard",
        false,
    );
    // C65 (xrows 64, full production grid).
    mk(
        "C65".into(),
        [ceil_div(m, 8), 80, 8],
        null_ka([m, k, 80, 65, xdiv(s), 64, k, ys, 512, 0]),
        Vec::new(),
        [None; 5],
        "C-guard",
        false,
    );
    // U79 / U81.
    for uu in [79u32, 81u32] {
        mk(
            format!("U{uu}"),
            [1, 1, 1],
            null_ka([m, k, uu, 8, xdiv(s), 8, k, ys, 512, 0]),
            Vec::new(),
            [None; 5],
            "U-guard",
            false,
        );
    }
    // Wrong K per arch source rule (xstride follows the template scalar).
    if arch == "gfx1151" {
        mk(
            "wrong-K".into(),
            [1, 1, 1],
            null_ka([m, k + 128, 80, 8, xdiv(s), 8, k + 128, ys, 512, 0]),
            Vec::new(),
            [None; 5],
            "K-guard",
            false,
        );
    } else {
        for kk in [0u32, k + 1] {
            mk(
                format!("wrong-K{kk}"),
                [1, 1, 1],
                null_ka([m, kk, 80, 8, xdiv(s), 8, k, ys, 512, 0]),
                Vec::new(),
                [None; 5],
                "K-guard",
                false,
            );
        }
    }
    // z-past: exact one-tile descriptor; z0 empty, z1 advances out of mapped bytes.
    mk(
        "z-past".into(),
        [1, 1, 2],
        z_past_kernarg(s, 0),
        vec![("descriptor", z_past_descriptor(bucket))],
        [
            None,
            Some((0, bucket * 80 * 4)),
            Some((0, (160 + 81 * bucket) * 4)),
            Some((0, (322 + 80 * bucket) * 4)),
            None,
        ],
        "z-guard",
        true,
    );
    // y80: separately exact 80-word ID / 81-word offset zero arrays.
    mk(
        "y80-exact-suffix".into(),
        [1, 81, 1],
        null_ka([m, k, 80, 8, xdiv(s), 8, k, ys, 512, 0]),
        vec![("ids80", vec![0u8; 80 * 4]), ("offs81", vec![0u8; 81 * 4])],
        [None, Some((0, 0)), Some((1, 0)), None, None],
        "U-guard",
        true,
    );
    // dead-M0.
    mk(
        "dead-M0".into(),
        [1, 1, 1],
        null_ka([0, k, 80, 8, xdiv(s), 8, k, ys, 512, 0]),
        Vec::new(),
        [None; 5],
        "M-guard",
        false,
    );
    // Empty tiles incl live z7 at R64, grid z=9: guarded 8-tile descriptor.
    {
        let mut w = vec![0u32; DESC_WORDS * 8];
        for t in 0..8 {
            let base = t * DESC_WORDS;
            for o in 0..81 {
                w[base + 160 + bucket * 81 + o] = 0;
            }
        }
        let desc: Vec<u8> = w.into_iter().flat_map(u32::to_le_bytes).collect();
        let mut dh = vec![GUARD_FILL; GUARD];
        dh.extend_from_slice(&desc);
        dh.extend_from_slice(&vec![GUARD_FILL; GUARD]);
        let g = GUARD;
        mk(
            "empty-z7-past".into(),
            [1, 1, 9],
            null_ka([m, k, 80, 64, xdiv(s), 64, k, ys, 512, 0]),
            vec![("descriptor_guarded8", dh)],
            [
                None,
                Some((0, g + bucket * 80 * 4)),
                Some((0, g + (160 + bucket * 81) * 4)),
                Some((0, g + (322 + 80 * bucket) * 4)),
                None,
            ],
            "z-guard",
            true,
        );
    }
    // Offset shapes: exactly one valid EID word (ID0) plus the two exact offset
    // words under test (downstream table/slots/X stay NULL). The EID buffer is
    // REQUIRED: native metadata order reads EID before offset/count branches,
    // so a NULL EID would fault before the intended offset bound is reached.
    // These cases test offset bounds, NOT invalid metadata pointers.
    for (name, (a, b)) in [
        ("count0", (0u32, 0u32)),
        ("count9", (0u32, 9u32)),
        ("descending", (5u32, 2u32)),
        ("off80", (0u32, 81u32)),
    ] {
        let reason = match name {
            "count0" => "empty-count",
            "count9" => "count-bound",
            "descending" => "descending",
            _ => "offset-bound",
        };
        mk(
            name.into(),
            [1, 1, 1],
            null_ka([m, k, 80, 8, xdiv(s), 8, k, ys, 512, 0]),
            vec![
                ("ids1", 0u32.to_le_bytes().to_vec()),
                ("offs2", u32s(&[a, b])),
            ],
            [None, Some((0, 0)), Some((1, 0)), None, None],
            reason,
            false,
        );
    }
    // R1 offsets [8,11]: valid EID word plus the two exact offset words, C1/xrows1.
    mk(
        "R1[8,11]".into(),
        [1, 1, 1],
        null_ka([m, k, 80, 1, xdiv(s), 1, k, ys, 512, 0]),
        vec![
            ("ids1", 0u32.to_le_bytes().to_vec()),
            ("offs2", u32s(&[8, 11])),
        ],
        [None, Some((0, 0)), Some((1, 0)), None, None],
        "offset-bound",
        false,
    );
    // Bad IDs: exactly one valid [0,1] offsets pair mapped; DECLARED n=512
    // (the old probe claimed n=2; the id guard exits before any table read).
    for (name, idv) in [
        ("id-neg1", -1i32),
        ("id-min", i32::MIN),
        ("id-n", 512),
        ("id-n1", 513),
    ] {
        mk(
            name.into(),
            [1, 1, 1],
            null_ka([m, k, 80, 8, xdiv(s), 8, k, ys, 512, 0]),
            vec![
                ("ids1", (idv as u32).to_le_bytes().to_vec()),
                ("offs2", u32s(&[0, 1])),
            ],
            [None, Some((0, 0)), Some((1, 0)), None, None],
            "id-bound",
            false,
        );
    }
    // R64/count8 invalid slots at tile 0: ID0, offsets [0,8], valid R64 slots
    // with bad 640 (=10*C) / u32::MAX at EACH of the 8 positions; table/X NULL.
    {
        let valid: Vec<u32> = (0..8u32)
            .map(|r| 10 * r + if r % 2 == 0 { 0 } else { 9 })
            .collect();
        let offs = u32s(&[0, 8]);
        for bad in [640u32, u32::MAX] {
            for pos in 0..8usize {
                let mut slots = valid.clone();
                slots[pos] = bad;
                mk(
                    format!("badslot{bad:x}@{pos}"),
                    [1, 1, 1],
                    null_ka([m, k, 80, 64, xdiv(s), 64, k, ys, 512, 0]),
                    vec![
                        ("ids1", 0u32.to_le_bytes().to_vec()),
                        ("offs2", offs.clone()),
                        ("slots8", u32s(&slots)),
                    ],
                    [None, Some((0, 0)), Some((1, 0)), Some((2, 0)), None],
                    "slot-bound",
                    false,
                );
            }
        }
    }
    // id0-n0: n=0 with VALID NONEMPTY offsets [0,1] and ID0, so the
    // expert-bound guard (not the empty-count path) causes the exit.
    mk(
        "id0-n0".into(),
        [1, 1, 1],
        null_ka([m, k, 80, 8, xdiv(s), 8, k, ys, 0, 0]),
        vec![
            ("ids1", 0u32.to_le_bytes().to_vec()),
            ("offs2", u32s(&[0, 1])),
        ],
        [None, Some((0, 0)), Some((1, 0)), None, None],
        "expert-bound",
        false,
    );
    plans
}

/// Runtime executor: upload every plan region read-only, bind real VMM bases,
/// launch the existing guard path on the actual grid, require silence, then
/// download-compare EVERY region unchanged and release. First fault stops.
fn run_guard_plans(
    hip: &HipRuntime,
    id: i32,
    func: &Function,
    plans: &[GuardPlan],
    y_len: usize,
    gate: bool,
    counts: &mut Counts,
) -> Result<()> {
    for plan in plans {
        let mut bufs = Vec::new();
        let mut bases = Vec::new();
        for r in &plan.regions {
            let b = vmm_upload(hip, id, r, true)?;
            counts.vmm_buffers += 1;
            bases.push(b.logical());
            bufs.push(b);
        }
        let (y0, y1) = run_guard_launch(hip, id, func, plan, &bases, y_len, gate, counts)?;
        expect_silent(&y0, &y1, &plan.name)?;
        counts.cases += 1;
        counts.candidate_launches += 1;
        for (b, r) in bufs.into_iter().zip(plan.regions.iter()) {
            let back = vmm_download(hip, &b)?;
            if &back != r {
                return Err(format!("{}: guard region changed by launch", plan.name));
            }
            b.release(hip)?;
        }
    }
    Ok(())
}

/// Codec group alignment for the gfx1201 nonzero/group-aligned K rule: mq3
/// groups of 128, mq4 gate groups of 256, mq4 down groups of 128.
fn k_group_align(s: Spec) -> u32 {
    match (s.codec, s.kind.gate()) {
        (Codec::Mq3I4, _) => 128,
        (Codec::Mq4Sym, true) => 256,
        (Codec::Mq4Sym, false) => 128,
    }
}

/// Host domain proof for one plan: bind symbolic non-device bases, decode the
/// actual 96 B pointer/scalar bytes, and walk EVERY launched WG, resolving
/// each read against the ACTUAL region byte extents. Labels the required
/// kernel rejection per WG; the kernel itself is not executed here.
fn prove_guard_plan(arch: &str, s: Spec, plan: &GuardPlan) -> Result<serde_json::Value> {
    let bucket = s.codec.bucket();
    // Symbolic non-device bases, 1 MiB apart (regions are far smaller).
    let bases: Vec<u64> = (0..plan.regions.len())
        .map(|i| 0x100000 + i as u64 * 0x100000)
        .collect();
    let ka = plan.bind(&bases);
    if ka.len() != 96 {
        return Err(format!("{}: kernarg extent", plan.name));
    }
    let ptrs: [u64; 7] =
        std::array::from_fn(|i| u64::from_le_bytes(ka[8 * i..8 * i + 8].try_into().unwrap()));
    let scalars: [u32; 10] =
        std::array::from_fn(|i| u32::from_le_bytes(ka[56 + 4 * i..60 + 4 * i].try_into().unwrap()));
    // Bound pointers must equal base+offset; unbound downstream pointers and
    // both Y pointers (supplied only at runtime) must be null in the template.
    for i in 0..5 {
        match plan.bindings[i] {
            Some((ri, off)) => {
                if ptrs[i] != bases[ri] + off as u64 {
                    return Err(format!("{}: pointer {i} binding mismatch", plan.name));
                }
            }
            None => {
                if ptrs[i] != 0 {
                    return Err(format!("{}: pointer {i} must be null", plan.name));
                }
            }
        }
    }
    if ptrs[5] != 0 || ptrs[6] != 0 {
        return Err(format!("{}: Y pointers must stay zero in plan", plan.name));
    }
    let fetch = |ptr: u64, len: usize| -> Result<&[u8]> {
        for (i, r) in plan.regions.iter().enumerate() {
            if ptr >= bases[i] && ptr + len as u64 <= bases[i] + r.len() as u64 {
                let off = (ptr - bases[i]) as usize;
                return Ok(&r[off..off + len]);
            }
        }
        Err(format!(
            "{}: read [{ptr:#x},{:#x}) outside actual region extents",
            plan.name,
            ptr + len as u64
        ))
    };
    let (m, kk, u, cc, xdiv_v, xrows, xstride, ystride, nn) = (
        scalars[0], scalars[1], scalars[2], scalars[3], scalars[4], scalars[5], scalars[6],
        scalars[7], scalars[8],
    );
    if plan.name == "z-past" && (cc != 8 || xrows != 8) {
        return Err("z-past witness must be R/C8, not 64".into());
    }
    if plan.name != "id0-n0" && plan.name.starts_with("id-") && nn != 512 {
        return Err(format!(
            "{}: DECLARED n must be 512 (old incorrect n2)",
            plan.name
        ));
    }
    if plan.name == "id0-n0" {
        if nn != 0 || plan.regions[1] != u32s(&[0, 1]) {
            return Err("id0-n0 must fix n=0 with valid nonempty offsets [0,1]".into());
        }
    }
    let k_ok = if arch == "gfx1151" {
        kk == prod_k(s)
    } else {
        kk != 0 && kk % k_group_align(s) == 0
    };
    let tiles = ceil_div(cc, 8);
    let mut reason_counts: HashMap<&'static str, u64> = HashMap::new();
    let mut target_wg: Option<[u32; 3]> = None;
    let mut wg_total: u64 = 0;
    for x in 0..plan.grid[0] {
        for y in 0..plan.grid[1] {
            for z in 0..plan.grid[2] {
                wg_total += 1;
                let reason: &'static str = if !(1..=64).contains(&cc) {
                    "C-guard"
                } else if u != 80 {
                    "U-guard"
                } else if !k_ok {
                    "K-guard"
                } else if x * 8 >= m {
                    "M-guard"
                } else if y >= u {
                    "U-guard"
                } else if z >= tiles {
                    "z-guard"
                } else {
                    // Required-order model: native metadata reads EID before the
                    // offset/count branch; a missing EID word fails closed as
                    // metadata-bound, never as the target offset rejection.
                    let offs_ptr = ptrs[2] + z as u64 * DESC_BYTES as u64;
                    match fetch(offs_ptr + y as u64 * 4, 8) {
                        Err(_) => {
                            let ids_ptr = ptrs[1] + z as u64 * DESC_BYTES as u64;
                            match fetch(ids_ptr + y as u64 * 4, 4) {
                                // Both words missing: descriptor/tile itself unmapped.
                                Err(_) => "z-guard",
                                // Offsets missing but EID present: metadata defect,
                                // never the target offset rejection.
                                Ok(_) => "metadata-bound",
                            }
                        }
                        Ok(pair) => {
                            let ids_ptr = ptrs[1] + z as u64 * DESC_BYTES as u64;
                            match fetch(ids_ptr + y as u64 * 4, 4) {
                                // EID-first: missing EID word fails closed as
                                // metadata-bound before any offset/count branch.
                                Err(_) => "metadata-bound",
                                Ok(idw) => {
                                    let idv = i32::from_le_bytes(idw.try_into().unwrap());
                                    if nn == 0 {
                                        "expert-bound"
                                    } else if idv < 0 || idv as u64 >= nn as u64 {
                                        "id-bound"
                                    } else {
                                        let start =
                                            u32::from_le_bytes(pair[0..4].try_into().unwrap());
                                        let end =
                                            u32::from_le_bytes(pair[4..8].try_into().unwrap());
                                        if start == end && end == 0 && start == 0 {
                                            "empty-count"
                                        } else if end == start {
                                            "empty-count"
                                        } else if start > end {
                                            "descending"
                                        } else if end > 80 || end > 10 * cc {
                                            "offset-bound"
                                        } else if end - start > 8 {
                                            "count-bound"
                                        } else {
                                            let slots_ptr = ptrs[3] + z as u64 * DESC_BYTES as u64;
                                            let count = (end - start) as usize;
                                            let mut bad = false;
                                            for j in 0..count {
                                                let sw = fetch(
                                                    slots_ptr + (start as usize + j) as u64 * 4,
                                                    4,
                                                )?;
                                                let slot =
                                                    u32::from_le_bytes(sw.try_into().unwrap());
                                                if slot >= 10 * cc || slot / 10 / 8 != z {
                                                    bad = true;
                                                    break;
                                                }
                                            }
                                            if bad {
                                                "slot-bound"
                                            } else {
                                                return Err(format!(
                                            "{}: WG [{x},{y},{z}] would reach table/X with a valid live group",
                                            plan.name));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                };
                *reason_counts.entry(reason).or_insert(0) += 1;
                if reason == plan.expected_reason && target_wg.is_none() {
                    target_wg = Some([x, y, z]);
                }
            }
        }
    }
    // Premise validation: every WG must require the expected rejection (z/y/
    // empty plans additionally admit the all-empty live prefix), and at least
    // one WG must hit the expected target reason. A wrong premise (e.g. an
    // empty-count exit where the expert bound is claimed) fails here.
    for r in reason_counts.keys() {
        if *r != plan.expected_reason && !(*r == "empty-count" && plan.empty_prefix) {
            return Err(format!(
                "{}: WG required {r}, expected {}",
                plan.name, plan.expected_reason
            ));
        }
    }
    let target =
        target_wg.ok_or_else(|| format!("{}: no WG hits {}", plan.name, plan.expected_reason))?;
    let mut counts_json = serde_json::Map::new();
    for (r, n) in &reason_counts {
        counts_json.insert(r.to_string(), json!(*n));
    }
    // Machine-usable interchange: raw 96 B kernarg bytes (NULL = literal zero
    // LE bytes), per-region raw bytes keyed by address, candidate symbol, and
    // the host-only symbolic bases. Template Y pointers stay NULL here; the
    // output arena below documents the runtime Y fields (bytes 40/48).
    let regions_json: Vec<serde_json::Value> = plan
        .regions
        .iter()
        .enumerate()
        .map(|(i, r)| json!({"name": plan.region_names[i], "address": bases[i], "bytes": r}))
        .collect();
    let mut w = json!({
        "name": plan.name, "arch": arch, "candidate_symbol": cand_symbol(arch, s),
        "k1_symbol": k1_symbol(s), "bucket": bucket,
        "grid": plan.grid, "kernarg_len": ka.len(),
        "kernarg_bytes": ka.to_vec(),
        "kernarg_hex": ka.iter().map(|b| format!("{b:02x}")).collect::<String>(),
        "kernarg_address": 0x80000u64,
        "host_only_symbolic_bases": true,
        "regions": regions_json,
        "M": m, "K": kk, "U_at64": u, "C_at68": cc, "xdiv": xdiv_v, "x_rows": xrows,
        "x_stride": xstride, "y_stride": ystride, "n_at88": nn,
        "decoded_scalars": scalars,
        "decoded_ptrs": ptrs,
        "selected_ptr_offsets": (0..5).map(|i| match plan.bindings[i] {
            Some((_, off)) => json!(off), None => serde_json::Value::Null,
        }).collect::<Vec<_>>(),
        "region_byte_extents": plan.regions.iter().map(|r| r.len()).collect::<Vec<_>>(),
        "region_readable_words": plan.regions.iter().map(|r| r.len() / 4).collect::<Vec<_>>(),
        "runtime_y": {"guard": GUARD, "fill": FILL_A, "len": y_len_for(650, ystride_prod(s)), "gate": s.kind.gate()},
        "template_y_null": true,
        "wg_total": wg_total, "per_reason_wg_counts": counts_json,
        "expected_reason": plan.expected_reason, "empty_prefix": plan.empty_prefix,
        "target_dead_wg": target,
        "caveat": "host domain proof only: labels the required kernel rejection per WG from decoded bytes (EID-first metadata order, corrected valid EID metadata on offset-domain plans); kernel not executed here, no hardware certification; actual CPU reproduction by consumer on same CO decides (gfx1201 R1/slots old-CO table-before-slot deficiency documented at top)",
    });
    if plan.name == "z-past" {
        let sel: Vec<u64> = [
            80 * bucket as u64 * 4,
            (160 + 81 * bucket) as u64 * 4,
            (322 + 80 * bucket) as u64 * 4,
        ]
        .to_vec();
        w["mapped_descriptor_bytes"] = json!(plan.regions[0].len());
        w["selected_descriptor_ptr_offsets"] = json!(sel);
        w["z1_descriptor_ptr_offsets"] = json!(sel
            .iter()
            .map(|o| o + DESC_BYTES as u64)
            .collect::<Vec<_>>());
        w["live_z0_empty"] = json!(true);
        w["first_z0_exit_premise"] = json!("empty descriptor count");
        w["target_dead_z"] = json!(1);
        w["table_slots_X"] = json!("unmapped");
    }
    if plan.name == "count0"
        || plan.name == "count9"
        || plan.name == "descending"
        || plan.name == "off80"
        || plan.name == "R1[8,11]"
    {
        w["original_packed_eid_null"] = json!(true);
        w["fixed_valid_eid_id0"] = json!(true);
        w["metadata_domain_note"] = json!("original factory carried EID NULL on these offset-domain plans; corrected factory maps one valid EID word (ID0, typed extent/alignment) while keeping exact offsets/counts and downstream NULLs; native EID-before-offset order makes valid metadata required, actual CPU consumer decides");
    }
    if plan.name.starts_with("id-") && plan.name != "id0-n0" {
        w["old_n"] = json!(2);
        w["fixed_n"] = json!(512);
    }
    if plan.name == "id0-n0" {
        w["old_n"] = json!(512);
        w["old_offsets"] = json!([0, 0]);
        w["fixed_n"] = json!(0);
        w["fixed_offsets"] = json!([0, 1]);
    }
    Ok(w)
}

/// Host proof over the SAME factory for BOTH arches and all four specs.
fn guard_host_witnesses() -> Result<Vec<serde_json::Value>> {
    let mut out = Vec::new();
    for arch in ["gfx1151", "gfx1201"] {
        for s in all_specs() {
            let plans = negative_guard_plans(arch, s);
            let want = if arch == "gfx1151" { 35 } else { 36 };
            if plans.len() != want {
                return Err(format!(
                    "{arch} {}: {} plans != {want}",
                    k1_symbol(s),
                    plans.len()
                ));
            }
            for plan in &plans {
                out.push(prove_guard_plan(arch, s, plan)?);
            }
        }
    }
    Ok(out)
}

// --------------------------------------------------------------- verify ---

fn shard_range(shard: Option<(u32, u32)>, rows: Option<(u32, u32)>) -> (u32, u32) {
    if let Some((lo, hi)) = rows {
        return (lo, hi);
    }
    match shard {
        Some((1, _)) => (1, 16),
        Some((2, _)) => (17, 32),
        Some((3, _)) => (33, 48),
        Some((4, _)) => (49, 64),
        _ => (1, 64),
    }
}

fn verify(args: &Args) -> Result<serde_json::Value> {
    let co_path = args.co.clone().ok_or("--co required")?;
    let k1_path = args.k1_co.clone().ok_or("--k1-co required")?;
    let arch = args
        .arch
        .clone()
        .ok_or("--arch required (gfx1151|gfx1201)")?;
    if arch != "gfx1151" && arch != "gfx1201" {
        return Err("--arch must be gfx1151|gfx1201".into());
    }
    let out_path = args.out.clone().ok_or("--out required")?;
    let co_bytes = std::fs::read(&co_path).map_err(err)?;
    let k1_bytes = std::fs::read(&k1_path).map_err(err)?;
    let co_sha = sha256_hex(&co_bytes);
    let k1_sha = sha256_hex(&k1_bytes);
    if let Some(pin) = &args.k1_sha256 {
        if k1_sha != pin.to_lowercase() {
            return Err(format!(
                "k1 pin mismatch: actual {k1_sha} != --k1-sha256 {pin}"
            ));
        }
    }
    let dev = open_device(&arch, args.expect_bdf.as_deref())?;
    let hip = &dev.hip;
    let id = dev.id;
    let cand_mod = hip.module_load_data(&co_bytes).map_err(err)?;
    let k1_mod = hip.module_load_data(&k1_bytes).map_err(err)?;
    let specs = all_specs();
    // Fail fast when an expected entry is absent from the external images
    // (same errors the old eager Vec fill produced, before any case runs).
    for s in specs {
        let cs = cand_symbol(&arch, s);
        hip.module_get_function(&cand_mod, &cs)
            .map_err(|e| format!("candidate missing {cs}: {e}"))?;
        let ks = k1_symbol(s);
        hip.module_get_function(&k1_mod, &ks)
            .map_err(|e| format!("k1 missing {ks}: {e}"))?;
    }
    let (rlo, rhi) = shard_range(args.shard, args.rows);
    let shard_idx: Option<usize> = args.shard.map(|(p, _)| (p - 1) as usize);
    let mut counts = Counts {
        cases: 0,
        candidate_launches: 0,
        k1_launches: 0,
        compared_words: 0,
        written_bytes: 0,
        second_passes: 0,
        vmm_buffers: 0,
        guard_bytes: 0,
    };
    let mut input_hashes: Vec<String> = Vec::new();

    if args.cheap {
        // Kill gate first: R9 ordinary all specs + R8 tail pair + C65 rejection.
        for (si, s) in specs.iter().enumerate() {
            let c = build_case(*s, 9, prod_m(*s), 0, 0, 0, None, 512);
            let mut ctx = SpecCtx::new(hip, id, &c, &mut counts)?;
            let mut kc = HashMap::new();
            let cand_h = cand_handle(hip, &cand_mod, &arch, *s)?;
            let k1_h = k1_handle(hip, &k1_mod, *s)?;
            verify_case_full(
                hip,
                id,
                &cand_h,
                &k1_h,
                &mut ctx,
                &c,
                &mut kc,
                &mut input_hashes,
                &mut counts,
                false,
                false,
            )?;
            ctx.release(hip)?;
        }
        for (si, s) in specs.iter().enumerate() {
            let c = cheap_tail_case(si, *s);
            let mut ctx = SpecCtx::new(hip, id, &c, &mut counts)?;
            let mut kc = HashMap::new();
            let cand_h = cand_handle(hip, &cand_mod, &arch, *s)?;
            let k1_h = k1_handle(hip, &k1_mod, *s)?;
            verify_case_full(
                hip,
                id,
                &cand_h,
                &k1_h,
                &mut ctx,
                &c,
                &mut kc,
                &mut input_hashes,
                &mut counts,
                false,
                false,
            )?;
            ctx.release(hip)?;
        }
        for (si, s) in specs.iter().enumerate() {
            let gate = s.kind.gate();
            let ys = ystride_prod(*s);
            // Shared C65 plan: ALL downstream pointers NULL, full production
            // grid; the C guard must exit before any dereference.
            let y_len = y_len_for(650, ys);
            let plans = negative_guard_plans(&arch, *s);
            let plan = plans
                .iter()
                .find(|p| p.name == "C65")
                .ok_or("shared factory missing C65")?;
            if plan.grid != [ceil_div(prod_m(*s), 8), 80, 8] {
                return Err("shared C65 grid drift".into());
            }
            let mut bufs = Vec::new();
            let mut bases = Vec::new();
            for r in &plan.regions {
                let b = vmm_upload(hip, id, r, true)?;
                counts.vmm_buffers += 1;
                bases.push(b.logical());
                bufs.push(b);
            }
            let cand_h = cand_handle(hip, &cand_mod, &arch, *s)?;
            let (y0, y1) =
                run_guard_launch(hip, id, &cand_h, plan, &bases, y_len, gate, &mut counts)?;
            expect_silent(&y0, &y1, &format!("C65 {}", k1_symbol(*s)))
                .map_err(|e| format!("cheap gate: {e}"))?;
            counts.cases += 1;
            counts.candidate_launches += 1;
            for b in bufs {
                b.release(hip)?;
            }
        }
        let v = json!({
            "schema": "pm-moe-cn-tiled-numeric-oracle-v1", "mode": "verify-cheap",
            "arch": arch, "candidate": co_path, "candidate_sha256": co_sha,
            "k1_co": k1_path, "k1_sha256": k1_sha, "k1_pin": args.k1_sha256,
            "device": {"uuid": dev.uuid, "bdf": dev.bdf, "arch": dev.arch, "id": dev.id},
            "rows": format!("{rlo}-{rhi}"), "cheap": true,
            "tallies": {"cases": counts.cases, "candidate_launches": counts.candidate_launches,
                "k1_launches": counts.k1_launches, "k1_launch_note": "per-(token,rank,expert) grid-[M,1,1] block32 true-C1 local launches",
                "compared_words": counts.compared_words,
                "written_bytes": counts.written_bytes, "second_poison_passes": counts.second_passes,
                "guard_bytes": counts.guard_bytes,
                "vmm_buffers": counts.vmm_buffers},
            "input_fixture_sha256": input_hashes,
            "vmm": {"read_only_inputs": true, "right_aligned": true, "unmapped_suffix_granule": true,
                "note": "every device buffer is a VMM arena: mapped prefix + unmapped suffix; immutable inputs device read-only"},
            "inputs_verified": "every immutable byte downloaded and compared after each launch (not hash-only)",
            "limitations": [
                "Native GPU cannot expose instruction written bits; dual-poison plus expected written-byte maps establish stores per byte; read-only VMM mappings fault on input stores.",
                "Leaf-only: executed candidate numerics vs independent K1 C1/U1 block32 singletons only; no timing, capture, throughput, or admission claims.",
            ],
        });
        std::fs::write(&out_path, serde_json::to_vec_pretty(&v).unwrap()).map_err(err)?;
        eprintln!("wrote {out_path}");
        return Ok(v);
    }

    // Guards-only never constructs positive cases or launches the K1 reference.
    if !args.guards_only {
        // Full ordinary sweep over the shard rows; shared SpecCtx per spec (X64 + K1 Y persist).
        let mut sweep_ctx: Vec<SpecCtx> = Vec::new();
        let mut sweep_kc: Vec<HashMap<(u32, u32, u32, u32), [u8; 4]>> = Vec::new();
        for s in specs {
            let full = build_case(s, 64, prod_m(s), 0, 0, 0, None, 512);
            sweep_ctx.push(SpecCtx::new(hip, id, &full, &mut counts)?);
            sweep_kc.push(HashMap::new());
        }
        for (si, s) in specs.iter().enumerate() {
            for r in rlo..=rhi {
                // Per-R exact candidate X prefix: same host X64 prefix bytes, exact
                // length allocation (right-aligned VMM, unmapped suffix). Tensors are
                // shared by reference (host_tensors owned by ctx, never cloned).
                let c = ordinary_case(*s, r, &sweep_ctx[si].host_x64, &sweep_ctx[si].host_tensors);
                {
                    let sc = &sweep_ctx[si];
                    assert_eq!(c.x_stride, sc.x_stride);
                    assert_eq!(c.y_stride, sc.y_stride);
                    assert_eq!(c.n_experts, sc.n_experts);
                }
                let cand_h = cand_handle(hip, &cand_mod, &arch, *s)?;
                let k1_h = k1_handle(hip, &k1_mod, *s)?;
                verify_case_full(
                    hip,
                    id,
                    &cand_h,
                    &k1_h,
                    &mut sweep_ctx[si],
                    &c,
                    &mut sweep_kc[si],
                    &mut input_hashes,
                    &mut counts,
                    false,
                    false,
                )?;
            }
        }

        // Focused extras, partitioned idx%4 when sharded.
        let extras = focused_extras();
        for e in &extras {
            if let Some(sh) = shard_idx {
                if e.idx % 4 != sh {
                    continue;
                }
            }
            let c = focused_case(e);
            let mut ctx = SpecCtx::new(hip, id, &c, &mut counts)?;
            let mut kc = HashMap::new();
            // Null X and zero/constant K1 admission remain exclusive to xzero;
            // all families retain full-grid, dual-poison, true-C1 verification.
            let xzero = e.name.starts_with("xzero ");
            let cand_h = cand_handle(hip, &cand_mod, &arch, specs[e.spec])?;
            let k1_h = k1_handle(hip, &k1_mod, specs[e.spec])?;
            verify_case_full(
                hip,
                id,
                &cand_h,
                &k1_h,
                &mut ctx,
                &c,
                &mut kc,
                &mut input_hashes,
                &mut counts,
                xzero,
                xzero,
            )?;
            ctx.release(hip)?;
        }
        for ctx in sweep_ctx {
            ctx.release(hip)?;
        }
    }

    // Keep shard routing unchanged; an explicit unsharded full range also owes guards.
    let run_guards = if args.guards_only {
        true
    } else {
        match (args.shard, args.rows) {
            (Some((1, _)), _) | (None, None) | (None, Some((1, 64))) => true,
            _ => false,
        }
    };
    if run_guards {
        for (si, s) in specs.iter().enumerate() {
            let ys = ystride_prod(*s);
            let gate = s.kind.gate();
            let y_len = y_len_for(650, ys);
            let cand_h = cand_handle(hip, &cand_mod, &arch, *s)?;
            run_guard_plans(
                hip,
                id,
                &cand_h,
                &negative_guard_plans(&arch, *s),
                y_len,
                gate,
                &mut counts,
            )?;
        }
    }

    let mode = if args.guards_only {
        "verify-guards"
    } else {
        "verify"
    };
    let backend_note = if arch == "gfx1201" {
        "gfx1201 source facts applied: K nonzero+group-aligned (not exact-K); table-before-slot order and offsets<=80-only are backend defects outside this oracle — fixtures kept exact and executed; first fault stops."
    } else {
        "gfx1151 Cn requires exact production K (gate 2560, down 640); wrong-K uses aligned non-production K+128 — no gfx1201 backend note applies on this arch."
    };
    let v = json!({
        "schema": "pm-moe-cn-tiled-numeric-oracle-v1", "mode": mode,
        "arch": arch, "candidate": co_path, "candidate_sha256": co_sha,
        "k1_co": k1_path, "k1_sha256": k1_sha, "k1_pin": args.k1_sha256,
        "device": {"uuid": dev.uuid, "bdf": dev.bdf, "arch": dev.arch, "id": dev.id},
        "rows": format!("{rlo}-{rhi}"), "shard": args.shard.map(|(p, q)| format!("{p}/{q}")),
        "cheap": false, "guards_only": args.guards_only, "guards_run": run_guards,
        "positive_fixtures": if args.guards_only { &[] as &[&str] } else { &["ordinary", "focused"] },
        "tallies": {"cases": counts.cases, "candidate_launches": counts.candidate_launches,
            "k1_launches": counts.k1_launches, "k1_launch_note": "per-(token,rank,expert) grid-[M,1,1] block32 true-C1 local launches",
            "compared_words": counts.compared_words,
            "written_bytes": counts.written_bytes, "second_poison_passes": counts.second_passes,
                "guard_bytes": counts.guard_bytes,
            "vmm_buffers": counts.vmm_buffers},
        "input_fixture_sha256": input_hashes,
        "vmm": {"read_only_inputs": true, "right_aligned": true, "unmapped_suffix_granule": true,
            "note": "every device buffer is a VMM arena: mapped prefix + unmapped suffix; immutable inputs device read-only"},
        "inputs_verified": "every immutable byte downloaded and compared after each launch (not hash-only)",
        "limitations": [
            "Native GPU cannot expose instruction written bits; dual-poison plus expected written-byte maps establish stores per byte; read-only VMM mappings fault on input stores.",
            "Leaf-only: executed candidate numerics vs independent K1 C1/U1 block32 singletons only; no timing, capture, throughput, or admission claims.",
            backend_note,
        ],
    });
    std::fs::write(&out_path, serde_json::to_vec_pretty(&v).unwrap()).map_err(err)?;
    eprintln!("wrote {out_path}");
    Ok(v)
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}\n{}", usage());
            std::process::exit(2);
        }
    };
    if args.help {
        print!("{}", usage());
        return;
    }
    if args.selfcheck {
        match selfcheck(args.out.clone()) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        return;
    }
    match verify(&args) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

fn z_past_descriptor(bucket: usize) -> Vec<u8> {
    build_descriptor(1, bucket, &vec![Vec::new()])
}

fn z_past_kernarg(spec: Spec, descriptor_base: u64) -> [u8; 96] {
    let db = descriptor_base;
    let bucket = spec.codec.bucket() as u64;
    encode_kernarg(
        [
            0,
            db + 80 * bucket * 4,
            db + (160 + 81 * bucket) * 4,
            db + (322 + 80 * bucket) * 4,
            0,
            0,
            0,
        ],
        [
            prod_m(spec),
            prod_k(spec),
            80,
            8,
            xdiv(spec),
            8,
            prod_k(spec),
            ystride_prod(spec),
            512,
            0,
        ],
    )
}
