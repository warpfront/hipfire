// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! K-E1 E8P NT1 **G1 oracle (synthetic + real-layer samples; gfx1201 and gfx1151)**: the E8P-on-IU4 grouped symmetric MoE GEMM gate/up
//! (`qwen4_moe_gate_up_silu_iu4_sym_e8p_pm_gfx1201`, `[1280, 2560]`, `H = [P][640]`) and down
//! (`qwen4_moe_down_iu4_sym_e8p_pm_gfx1201`, `[2560, 640]`, `Y = [P][2560]`), raw PM module launch (never runtime
//! dispatch), compared **byte for byte** against the frozen K-E0 arithmetic of `hipfire_runtime::e8p_i4_reference`.
//!
//! Build (parent runs it; this file never builds/runs itself):
//!   cargo build -p hipfire-runtime --release --features lab --example qwen4_moe_sym_e8p_g1_oracle
//! Probe code object (the BF16 intrinsic hook; same recipe/object as the real-layer K2 G1,
//! `crates/hipfire-arch-qwen4/examples/oracle_kernels/qwen4_moe_sym_g1_probe.gfx1201.hip`, contraction off):
//!   hipcc --genco --offload-arch=gfx1201 -O3 --no-offload-compress -fuse-cuid=none -mcode-object-version=6 \
//!       -ffp-contract=off -o qwen4_moe_sym_g1_probe_gfx1201.co qwen4_moe_sym_g1_probe.gfx1201.hip
//!   (committed copy: /home/kaden/qcal/release-0.4.3/subq4-week/artifacts/g1-real/qwen4_moe_sym_g1_probe_gfx1201.co)
//! Run (only on a card granted by the fabric scheduler, class `correct`):
//!   /home/kaden/qcal/locks/fabric/fabric-run.sh --class correct --card <UUID|label> --mem-gb N --owner NAME -- \
//!     target/release/examples/qwen4_moe_sym_e8p_g1_oracle --e8p kernels/<e8p module> \
//!       --probe-codeobject <probe .co> --out RECEIPT.json --card <label> --expect-bdf 0000:.. [--seed N] \
//!       [--negative-control 1|2] [--overwrite]
//! Exit code: 0 all exact, 3 ran but a check mismatched (receipt still written), 1 error (usage, device, I/O).
//!
//! # Wire / K-E0 inputs
//! `expert_weight_ptrs[e]` is the planar tensor `[words: M*K/8 u16 LE][scales: M*K/128 f16 LE]` (`words` row-major,
//! word `c` holds weights `8c..8c+8` of the flat `[M, K]` plane, `e8p_i4_reference::pack_word` fields; scale `i` is the
//! FP16 `d` of flat group `i`, `v = 4·codeword`, weight `(f32(d)·0.25)·v`). Every scale passes `writer_guard`.
//!
//! # Checks (per scenario, per projection)
//!  (1) `_acc` entry: every F32 accumulator `Y[P][M]` (gate rows then up rows; padding slots `+0`; sentinel tile
//!      slots keep the `0xa5a5a5a5` poison) == the K-E0 twin bit for bit. The twin composes only frozen K-E0 pieces:
//!      the integer chain `C = Σ o·q` per K128 epoch over `unpack_word_ref` nibbles `o`, seeded `MAGIC`
//!      (`a = MAGIC + C`), `fold128(a, s)`, then `epoch_sum(d_act, d, a', sum)` from `+0` over ascending epochs.
//!      No tolerance anywhere.
//!  (2) shipped entry BF16 `H`/`Y` == `mq_i4_reference::grouped_prefill_epilogue_with` over the twin F32 sums with the
//!      three SiLU intrinsics (`v_exp_f32`, `v_ldexp_f32`, IEEE divide expansion) answered by the DEVICE probe kernels
//!      (NOT `PortableEpilogue`); the three results of each element depend on each other so the DAG is replayed until a
//!      pass is answered entirely from device results for exactly the requested inputs. The probe's divide is
//!      cross-checked against hipcc's `__fdiv_rn` and CPU IEEE division for every divide the twin consumed, and a
//!      known-answer test (denormal/huge/scaled-denominator operands + 65536 random pairs) runs first. Padding slots
//!      are `+0`, sentinel tiles keep the `0xa5a5` poison. Down BF16 is `round(sum) >> 16` (no intrinsics).
//!  (3) K-E0 self-check before any launch: `unpack_word_swar == unpack_word_ref` over all 65536 wire words.
//! Experts sit in one slab at 256-byte offsets with 0xFF guards before/between/after; `sentinel_*` scenarios put
//! 0xFF/0x00 low bytes into the scale plane's first scale (bytes a last-row chunk's wide load spills into).
//!
//! # Pattern coverage
//! `random`; `every_word` (every one of the 65536 wire words appears, both shift values and both parities, in each
//! 65536-word block, bijectively scrambled per block); `extreme` (max `|v|` abs rows, all-negative/all-positive
//! signs, shift both ways, guard-boundary/huge scales); `parity` (odd and even abs rows, `popcount(sign7)` both
//! parities, shift alternating); scales include the guard boundary `0x0400` and low normals where FP16 `d/4` is not
//! exact (`0x0401`, `0x0403`, ...); activations include all `-8`, all `+7`, alternating and random blocks (nonzero
//! `s` of both signs and extreme `|2C+s|`), `d_act` in `{1e-20 .. 65536, 0}`.
//!
//! # gfx1151 (Strix Halo) and real-layer samples
//! `--expect-arch gfx1151` (default `gfx1201`, which keeps every existing behaviour: arch check, symbols, probe, synthetic-only receipt) binds the gfx1151
//! module (`kernels/qwen4_moe_iu4_sym_e8p_pm_gfx1151.hxaco`, the gfx1100 arm of `qwen4_moe_sym_gfx1100/` with only arch-forced deltas) through the symbols
//! `qwen4_moe_{gate_up_silu,down}_iu4_sym_e8p_pm_gfx1151[_acc]`. Checks (1)-(3) are unchanged and equally bitwise (the gfx1151 kernel must equal the gfx1201 kernel
//! and the K-E0 twin on every word); the probe object MUST be built for gfx1151 (`--offload-arch=gfx1151`; the gfx1151 source is the byte-identical-body
//! twin of the gfx1201 probe: `crates/hipfire-arch-qwen4/examples/oracle_kernels/qwen4_moe_sym_g1_probe.gfx1151.hip`, in this worktree, byte-identical to the
//! R9 copy, sha256 836a4e93cf0afb13dc8857765c8cd53d6b9fc9b31479e05156a81242b93239ee). Identity: the device arch must equal `--expect-arch`; gfx1151 has no ROCr UUID, so
//! `--expect-uuid` is OPTIONAL (exact case-insensitive match when given; the uuid is recorded verbatim, `uuid_authoritative: false`) and `--expect-bdf` is
//! REQUIRED there (`lspci -D -d 1002:` on hipx).
//! Real layers: `--samples-dir DIR` (repeatable / comma separated, one per pinned artifact; the extractor output of `tools/e8p_k1_extract_samples.py`, the
//! same directory the K1 harness `e8p_i4_moe_gemv_gfx1100_g1` consumes) adds the scenario set `real_*` (the rows/div sweep, expert `n % count` per
//! tile, and a mixed-expert scenario, over the four cold experts of layers 0 and 47 of the artifact, both projections), with the same F32 `_acc` and
//! BF16 checks against the K-E0 twin / device-probe epilogue. Every manifest field is verified strictly (`provisional: false`, extractor rc 0 / cpu, pinned
//! extractor sha256 54cdfed3e5314b6eb7685c2f35248e59224b3f332f04ea5d287fcb7a985cef0d, artifact sha256 in the pin table, qt 135/136, group 128, shape,
//! bare file name, size, sha256 of every tensor file); `--verify-artifact-range` re-hashes `[artifact_offset, +bytes)` of every tensor in the artifact
//! (it must be reachable), `--rehash-artifact` re-hashes the whole artifact. Pins: `c24-e8p`, `c24h6306-e8p` (the XTX `.hfq` exports) and
//! `c24h6306-d2-mq2-xts` = the Halo slab artifact hipx `/home/kaden/pm-wave/consumer24/models/c24h6306-d2/qwen3.8-flash-next.mq2-xts`
//! (sha256 d3d536c040f834699b5a261a5122b67d076461a9493eca16669803757928a949). On gfx1151 `--samples-dir` binding the slab pin is REQUIRED (the receipt's
//! `gate_eligible` is true only then); `--synthetic-only` runs the synthetic set alone and marks the receipt `gate_eligible: false`. Real scales that fail
//! `writer_guard` fail the run (`real_samples.writer_guard_violations`).
//! Halo commands (CPU steps on hipx; the device step through the fabric, class `correct`, label `halo`):
//!   python3 tools/e8p_k1_extract_samples.py --artifact /home/kaden/pm-wave/consumer24/models/c24h6306-d2/qwen3.8-flash-next.mq2-xts \
//!     --expect-sha256 d3d536c040f834699b5a261a5122b67d076461a9493eca16669803757928a949 --output-dir /abs/halo-e8p-k1-samples     (must not exist)
//!   hipcc --genco --offload-arch=gfx1151 -O3 --no-offload-compress -fuse-cuid=none -mcode-object-version=6 -ffp-contract=off \
//!     -o /abs/qwen4_moe_sym_g1_probe_gfx1151.co \
//!     crates/hipfire-arch-qwen4/examples/oracle_kernels/qwen4_moe_sym_g1_probe.gfx1151.hip      (run from the repo root)
//!   cargo build -p hipfire-runtime --release --features lab --example qwen4_moe_sym_e8p_g1_oracle
//!   /home/kaden/qcal/locks/fabric/fabric-run.sh --class correct --card halo --mem-gb N --owner NAME -- \
//!     target/release/examples/qwen4_moe_sym_e8p_g1_oracle --e8p /abs/qwen4_moe_iu4_sym_e8p_pm_gfx1151.hxaco \
//!     --probe-codeobject /abs/qwen4_moe_sym_g1_probe_gfx1151.co --samples-dir /abs/halo-e8p-k1-samples \
//!     --out /abs/r10.k2.g1.e8p-gfx1151.json --card halo --expect-arch gfx1151 --expect-bdf 0000:XX:00.0 [--expect-uuid U] [--seed N] \
//!     [--verify-artifact-range] [--rehash-artifact] [--negative-control 1|2] [--overwrite]
//!   Synthetic only (diagnostic): the same without `--samples-dir`, with `--synthetic-only`. Exit codes unchanged (0 exact, 3 mismatch, 1 error).

use std::cell::{Cell, OnceCell, RefCell};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use hip_bridge::{DeviceBuffer, Function, HipRuntime};
use hipfire_runtime::e8p_i4_reference as e8p;
use hipfire_runtime::mq_i4_reference::{self, A4Block, DecodeKind, EpilogueIntrinsics};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

type Result<T, E = String> = std::result::Result<T, E>;
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

const SCHEMA: &str = "qwen4-moe-sym-e8p-g1-oracle-v1";
const SCHEMA_GFX1151: &str = "qwen4-moe-sym-e8p-g1-oracle-gfx1151-v1";
const FABRIC_RUNNER: &str = "/home/kaden/qcal/locks/fabric/fabric-run.sh";
const HARNESS_SRC: &[u8] = include_bytes!("qwen4_moe_sym_e8p_g1_oracle.rs");
const REFERENCE_SRC: &[u8] = include_bytes!("../src/e8p_i4_reference.rs");
const USAGE: &str = "usage: qwen4_moe_sym_e8p_g1_oracle --e8p MODULE --probe-codeobject PROBE.co --out RECEIPT.json --card LABEL \
[--expect-arch gfx1201|gfx1151] [--expect-bdf BDF (required for gfx1151)] [--expect-uuid UUID (optional; gfx1151 has no ROCr UUID)] \
[--samples-dir DIR[,DIR2]]... [--verify-artifact-range] [--rehash-artifact] [--synthetic-only] \
[--seed N] [--negative-control 1|2] [--probe-kernarg-bytes N] [--overwrite]";
const BLK: usize = 72;
const SLAB_ALIGN: usize = 256;
const GUARD: usize = 256;
const FIRST_LIMIT: usize = 8;
const POISON_F32: u32 = 0xa5a5_a5a5;
const POISON_BF16: u16 = 0xa5a5;
/// Pinned real-sample artifacts (`sample.json: artifact_sha256`): the two XTX `.hfq` exports and the Halo slab `mq2-xts` (shipped d2 base).
static ARTIFACT_PINS: [(&str, &str); 3] = [
    (
        "c24-e8p",
        "5f280cff75e1ae29e45bbbdcbeca31bec4eb026fb105d5c3bcd7c5238fe281db",
    ),
    (
        "c24h6306-e8p",
        "8d30e9d7df324e16f7aa811c114109ba3ad238e630134cdd4a282a7dcc4abc9f",
    ),
    (
        "c24h6306-d2-mq2-xts",
        "d3d536c040f834699b5a261a5122b67d076461a9493eca16669803757928a949",
    ),
];
/// `sample.json: extractor_source_sha256` of `tools/e8p_k1_extract_samples.py`.
const EXTRACTOR_SHA256: &str = "54cdfed3e5314b6eb7685c2f35248e59224b3f332f04ea5d287fcb7a985cef0d";
const REAL_LAYERS: [u64; 2] = [0, 47];

/// The architectures served: gfx1201 (default) and gfx1151 (Strix Halo; the gfx1100 arm with only arch-forced deltas).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Target {
    Gfx1201,
    Gfx1151,
}
impl Target {
    fn of_name(arch: &str) -> Option<Self> {
        match arch {
            "gfx1201" => Some(Self::Gfx1201),
            "gfx1151" => Some(Self::Gfx1151),
            _ => None,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Gfx1201 => "gfx1201",
            Self::Gfx1151 => "gfx1151",
        }
    }
    fn schema(self) -> &'static str {
        match self {
            Self::Gfx1201 => SCHEMA,
            Self::Gfx1151 => SCHEMA_GFX1151,
        }
    }
    /// The sample pin a gate-eligible receipt must bind on this target (`None`: real samples are optional).
    fn required_pin(self) -> Option<&'static str> {
        match self {
            Self::Gfx1201 => None,
            Self::Gfx1151 => Some("c24h6306-d2-mq2-xts"),
        }
    }
}

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn fnv64(s: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn align(n: usize) -> usize {
    n.div_ceil(SLAB_ALIGN) * SLAB_ALIGN
}

fn u32s_le(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn i32s_le(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn le_u32s(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn up(hip: &HipRuntime, bytes: &[u8]) -> Result<DeviceBuffer> {
    let b = hip.malloc(bytes.len().max(8)).map_err(err)?;
    if !bytes.is_empty() {
        hip.memcpy_htod(&b, bytes).map_err(err)?;
    }
    Ok(b)
}

fn read_back(hip: &HipRuntime, b: &DeviceBuffer, n: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; n];
    if n > 0 {
        hip.memcpy_dtoh(&mut v, b).map_err(err)?;
    }
    Ok(v)
}

fn free_all(hip: &HipRuntime, bufs: Vec<DeviceBuffer>) -> Result<()> {
    for b in bufs {
        hip.free(b).map_err(err)?;
    }
    Ok(())
}

/// xorshift32 (the W3 G1 oracle's generator).
struct Rng(u32);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
}

/// splitmix64 (divide known-answer random pairs).
struct Rng64(u64);
impl Rng64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

// ---------------------------------------------------------------------------------------------------------------
// msgpack note scan (`.kernarg_segment_size` of a kernel), the scan of the real-layer K2 G1 harness
// ---------------------------------------------------------------------------------------------------------------

fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if from >= hay.len() {
        return None;
    }
    hay[from..]
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|p| p + from)
}
fn mp_uint(b: &[u8], at: usize) -> Option<u64> {
    let t = *b.get(at)?;
    match t {
        0x00..=0x7f => Some(t as u64),
        0xcc => Some(*b.get(at + 1)? as u64),
        0xcd => Some(u16::from_be_bytes(b.get(at + 1..at + 3)?.try_into().ok()?) as u64),
        0xce => Some(u32::from_be_bytes(b.get(at + 1..at + 5)?.try_into().ok()?) as u64),
        _ => None,
    }
}
fn mp_str(b: &[u8], at: usize) -> Option<&[u8]> {
    let t = *b.get(at)?;
    let (len, start) = match t {
        0xa0..=0xbf => ((t & 0x1f) as usize, at + 1),
        0xd9 => (*b.get(at + 1)? as usize, at + 2),
        _ => return None,
    };
    b.get(start..start + len)
}
fn metadata_kernarg_size(image: &[u8], symbol: &str) -> Option<usize> {
    let mut key = vec![0xa0 | 21u8];
    key.extend_from_slice(b".kernarg_segment_size");
    let mut name_key = vec![0xa5u8];
    name_key.extend_from_slice(b".name");
    let mut from = 0;
    while let Some(pos) = find(image, &key, from) {
        let vpos = pos + key.len();
        let next_key = find(image, &key, vpos).unwrap_or(image.len());
        if let (Some(val), Some(np)) = (mp_uint(image, vpos), find(image, &name_key, vpos)) {
            if np < next_key && mp_str(image, np + name_key.len()) == Some(symbol.as_bytes()) {
                return Some(val as usize);
            }
        }
        from = vpos;
    }
    None
}

// ---------------------------------------------------------------------------------------------------------------
// geometry
// ---------------------------------------------------------------------------------------------------------------

/// FN geometry: gate/up `[1280, 2560]` (H = `[P][640]`), down `[2560, 640]` (Y = `[P][2560]`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Proj {
    GateUp,
    Down,
}
impl Proj {
    const ALL: [Proj; 2] = [Proj::GateUp, Proj::Down];
    fn idx(self) -> usize {
        self as usize
    }
    fn m(self) -> usize {
        if self == Proj::Down {
            2560
        } else {
            1280
        }
    }
    fn k(self) -> usize {
        if self == Proj::Down {
            640
        } else {
            2560
        }
    }
    /// BF16 output columns.
    fn h(self) -> usize {
        if self == Proj::Down {
            2560
        } else {
            640
        }
    }
    fn groups(self) -> usize {
        self.k() / 128
    }
    fn threads(self) -> u32 {
        if self == Proj::Down {
            128
        } else {
            64
        }
    }
    fn kind(self) -> DecodeKind {
        if self == Proj::Down {
            DecodeKind::Down
        } else {
            DecodeKind::GateUp
        }
    }
    fn name(self) -> &'static str {
        if self == Proj::Down {
            "down"
        } else {
            "gate_up"
        }
    }
    fn symbol(self, target: Target) -> String {
        format!(
            "qwen4_moe_{}_iu4_sym_e8p_pm_{}",
            if self == Proj::Down {
                "down"
            } else {
                "gate_up_silu"
            },
            target.name()
        )
    }
}

// ---------------------------------------------------------------------------------------------------------------
// synthetic E8P experts
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Random,
    EveryWord,
    Extreme,
    Parity,
    Sentinel(u16),
    /// Index into the real experts loaded from `--samples-dir`.
    Real(u8),
}
impl Kind {
    fn label(self) -> String {
        match self {
            Kind::Random => "random".into(),
            Kind::EveryWord => "every_word".into(),
            Kind::Extreme => "extreme".into(),
            Kind::Parity => "parity".into(),
            Kind::Sentinel(f) => format!("sentinel_{f:04x}"),
            Kind::Real(i) => format!("real{i}"),
        }
    }
}

/// FP16 scale pool: normals and the guard boundary `0x0400`; the low entries (`0x0401`, `0x0403`, `0x0555`, `0x07ff`)
/// have an `d/4` that is NOT an exact FP16 (subnormal, mantissa lost): the decode must never form it.
const SCALE_POOL: [u16; 14] = [
    0x3000, 0x3800, 0x3c00, 0x4000, 0x2c00, 0x3555, 0x0400, 0x0401, 0x0403, 0x0555, 0x07ff, 0x1234,
    0x5640, 0x0800,
];
const SCALE_EXTREME: [u16; 5] = [0x7bff, 0x0400, 0x0401, 0x3c00, 0x0800];

struct Expert {
    label: String,
    words: Vec<u16>,
    scales: Vec<u16>,
    bytes: Vec<u8>,
    sha256: String,
    stats: Value,
    /// Decoded IU4 operands `o = (v-1)>>1` per flat weight (lazily from `unpack_word_ref`).
    o: OnceCell<Vec<i8>>,
}

fn make_words(kind: Kind, rng: &mut Rng, n: usize) -> Vec<u16> {
    let mut top: Vec<u8> = (0..=255u8).collect();
    top.sort_by_key(|&i| {
        std::cmp::Reverse(
            e8p::E8P_ABS2[i as usize]
                .iter()
                .map(|&x| u32::from(x))
                .sum::<u32>(),
        )
    });
    top.truncate(16);
    let odd_rows: Vec<u8> = (0..=255u8)
        .filter(|&i| e8p::abs_row_parity(i) == 1)
        .collect();
    let even_rows: Vec<u8> = (0..=255u8)
        .filter(|&i| e8p::abs_row_parity(i) == 0)
        .collect();
    (0..n)
        .map(|i| match kind {
            Kind::Random | Kind::Sentinel(_) => e8p::pack_word(
                rng.below(256) as u8,
                rng.below(128) as u8,
                rng.below(2) == 1,
            ),
            // every wire word once per 65536-word block; each block is a different bijection of the 16-bit space.
            Kind::EveryWord => {
                let blk = (i / 65536) as u32;
                (((i % 65536) as u32).wrapping_mul(2 * blk + 1) & 0xffff) as u16
            }
            Kind::Extreme => {
                let idx = top[i % top.len()];
                let sign7 = if (i / 16) % 2 == 0 { 0x7f } else { 0 };
                let shift = (i / 32) % 2 == 0;
                e8p::pack_word(idx, sign7, shift)
            }
            Kind::Parity => {
                let rows = if (i / 256) % 2 == 0 {
                    &odd_rows
                } else {
                    &even_rows
                };
                let idx = rows[(i * 7 + i / 256) % rows.len()];
                let sign7 = ((i * 0x35 + i / 256) & 0x7f) as u8;
                e8p::pack_word(idx, sign7, (i / 3) % 2 == 1)
            }
            Kind::Real(_) => {
                unreachable!("real experts are loaded from --samples-dir, never generated")
            }
        })
        .collect()
}

fn expert(kind: Kind, proj: Proj, seed: u32) -> Result<Expert> {
    if let Kind::Real(_) = kind {
        return Err("real experts are loaded from --samples-dir, never generated".into());
    }
    let (m, k) = (proj.m(), proj.k());
    let mut rng = Rng((seed ^ fnv64(&kind.label()) as u32) | 1);
    let words = make_words(kind, &mut rng, m * k / 8);
    let mut scales: Vec<u16> = match kind {
        Kind::Extreme => (0..m * k / 128)
            .map(|i| SCALE_EXTREME[(i + i / 7) % SCALE_EXTREME.len()])
            .collect(),
        _ => (0..m * k / 128)
            .map(|_| SCALE_POOL[rng.below(SCALE_POOL.len() as u32) as usize])
            .collect(),
    };
    if let Kind::Sentinel(first) = kind {
        scales[0] = first;
    }
    for (i, &d) in scales.iter().enumerate() {
        e8p::writer_guard(d).map_err(|e| format!("{} scale {i}: {e}", kind.label()))?;
    }
    Ok(finish_expert(
        format!("{}/{}", proj.name(), kind.label()),
        words,
        scales,
    ))
}

/// Wire bytes, statistics and sha256 of an expert from its decoded planes.
fn finish_expert(label: String, words: Vec<u16>, scales: Vec<u16>) -> Expert {
    let mut bytes = Vec::with_capacity(words.len() * 2 + scales.len() * 2);
    bytes.extend(words.iter().flat_map(|w| w.to_le_bytes()));
    bytes.extend(scales.iter().flat_map(|s| s.to_le_bytes()));
    let mut seen = vec![0u64; 1024];
    for &w in &words {
        seen[(w >> 6) as usize] |= 1 << (w & 63);
    }
    let distinct: u32 = seen.iter().map(|x| x.count_ones()).sum();
    let quarter_inexact = scales
        .iter()
        .filter(|&&d| {
            let q = e8p::scale_quarter_f32(d);
            half::f16::from_f32(q).to_f32() != q
        })
        .count();
    let stats = json!({
        "words": words.len(), "distinct_wire_words": distinct, "scales": scales.len(),
        "scale_min_bits": format!("{:#06x}", scales.iter().min().copied().unwrap_or(0)),
        "scales_at_guard_0400": scales.iter().filter(|&&d| d == e8p::GUARD_MIN_D_BITS).count(),
        "scales_quarter_not_fp16_exact": quarter_inexact,
        "shift_words": words.iter().filter(|&&w| w >> 15 == 1).count(),
        "odd_parity_rows": words.iter().filter(|&&w| e8p::abs_row_parity((w & 0xff) as u8) == 1).count(),
    });
    let sha256 = sha(&bytes);
    Expert {
        label,
        words,
        scales,
        bytes,
        sha256,
        stats,
        o: OnceCell::new(),
    }
}

// ---------------------------------------------------------------------------------------------------------------
// real experts (`--samples-dir`: the extractor output of `tools/e8p_k1_extract_samples.py`)
// ---------------------------------------------------------------------------------------------------------------

/// Exact wire size of one projection: `M*K/8` u16 words then `M*K/128` FP16 scales.
fn blob_bytes(proj: Proj) -> usize {
    proj.m() * proj.k() / 8 * 2 + proj.m() * proj.k() / 128 * 2
}

fn expert_from_wire(label: String, proj: Proj, raw: &[u8]) -> Result<Expert> {
    if raw.len() != blob_bytes(proj) {
        return Err(format!(
            "{label}: {} bytes != {}",
            raw.len(),
            blob_bytes(proj)
        ));
    }
    let nw = proj.m() * proj.k() / 8;
    let le16 = |b: &[u8]| -> Vec<u16> {
        b.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    };
    let (words, scales) = (le16(&raw[..nw * 2]), le16(&raw[nw * 2..]));
    Ok(finish_expert(label, words, scales))
}

#[derive(Clone)]
struct RealExpert {
    label: String,
    gate: Rc<Expert>,
    down: Rc<Expert>,
}
impl RealExpert {
    fn proj(&self, p: Proj) -> Rc<Expert> {
        match p {
            Proj::GateUp => self.gate.clone(),
            Proj::Down => self.down.clone(),
        }
    }
}

struct RealSet {
    tag: &'static str,
    experts: Vec<RealExpert>,
    meta: Value,
    guard_violations: u64,
}

fn file_sha256(path: &str) -> Result<(String, u64)> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let (mut h, mut n) = (Sha256::new(), 0u64);
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let k = f.read(&mut buf).map_err(|e| format!("{path}: {e}"))?;
        if k == 0 {
            break;
        }
        h.update(&buf[..k]);
        n += k as u64;
    }
    Ok((h.finalize().iter().map(|b| format!("{b:02x}")).collect(), n))
}

fn range_sha256(path: &str, offset: u64, len: usize) -> Result<String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    f.seek(SeekFrom::Start(offset))
        .map_err(|e| format!("{path}: seek {offset}: {e}"))?;
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf)
        .map_err(|e| format!("{path}: read {len} bytes at {offset}: {e}"))?;
    Ok(sha(&buf))
}

/// One extractor directory: four cold experts (two at layer 0, two at layer 47), both projections, strictly verified.
fn load_real(dir: &str, verify_range: bool, rehash: bool) -> Result<RealSet> {
    let mp = Path::new(dir).join("sample.json");
    let raw = std::fs::read(&mp).map_err(|e| format!("{}: {e}", mp.display()))?;
    let root: Value = serde_json::from_slice(&raw).map_err(|e| format!("{}: {e}", mp.display()))?;
    let text = |k: &str| -> Result<String> {
        root.get(k)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| format!("{dir}: sample.json lacks string `{k}`"))
    };
    let (art_path, art_sha) = (
        text("artifact_path")?,
        text("artifact_sha256")?.to_ascii_lowercase(),
    );
    if root.get("provisional").and_then(Value::as_bool) != Some(false) {
        return Err(format!(
            "{dir}: sample.json must carry `provisional: false`"
        ));
    }
    if root.get("rc").and_then(Value::as_i64) != Some(0)
        || root.get("device").and_then(Value::as_str) != Some("cpu")
    {
        return Err(format!(
            "{dir}: extractor binding needs rc 0 and device \"cpu\" (got rc {:?}, device {:?})",
            root.get("rc"),
            root.get("device")
        ));
    }
    let extractor_sha = text("extractor_source_sha256")?.to_ascii_lowercase();
    if extractor_sha != EXTRACTOR_SHA256 {
        return Err(format!(
            "{dir}: extractor_source_sha256 {extractor_sha} != pinned {EXTRACTOR_SHA256}"
        ));
    }
    if !root.get("argv").is_some_and(Value::is_array) {
        return Err(format!(
            "{dir}: sample.json must record the extractor `argv`"
        ));
    }
    let (tag, pin_sha) = ARTIFACT_PINS
        .iter()
        .find(|(_, s)| *s == art_sha)
        .copied()
        .ok_or_else(|| {
            format!("{dir}: artifact_sha256 {art_sha} is not one of the pinned artifacts")
        })?;
    // No silent fallback: the artifact is only touched when its verification was requested, and then it MUST be reachable.
    let art_size = if verify_range || rehash {
        Some(
            std::fs::metadata(&art_path)
                .map_err(|e| format!("{dir}: artifact {art_path}: {e}"))?
                .len(),
        )
    } else {
        None
    };
    let rehash_json = if rehash {
        let (h, n) = file_sha256(&art_path)?;
        if h != art_sha {
            return Err(format!(
                "{dir}: artifact {art_path} sha256 {h} != manifest {art_sha}"
            ));
        }
        json!({"performed": true, "sha256": h, "bytes": n, "equal_to_manifest": true})
    } else {
        json!({"performed": false})
    };
    let list = root
        .get("samples")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{dir}: sample.json has no `samples` array"))?;
    if list.len() != 4 {
        return Err(format!(
            "{dir}: {} samples; expected 4 (first/last cold expert of layers 0 and 47)",
            list.len()
        ));
    }
    let mut experts: Vec<RealExpert> = Vec::new();
    let mut recs: Vec<Value> = Vec::new();
    let mut seen: std::collections::BTreeSet<(u64, u64)> = std::collections::BTreeSet::new();
    let mut guard_violations = 0u64;
    for (i, s) in list.iter().enumerate() {
        let field = |k: &str| -> Result<u64> {
            s.get(k)
                .and_then(Value::as_u64)
                .ok_or_else(|| format!("samples[{i}]: missing integer `{k}`"))
        };
        let (layer, expert) = (field("layer")?, field("expert")?);
        if !REAL_LAYERS.contains(&layer) || expert >= 512 || !seen.insert((layer, expert)) {
            return Err(format!(
                "samples[{i}]: unexpected or duplicate (layer {layer}, expert {expert})"
            ));
        }
        if s.get("hot").and_then(Value::as_bool) != Some(false) {
            return Err(format!(
                "samples[{i}] L{layer}.E{expert}: E8P samples are cold experts (`hot: false`)"
            ));
        }
        let tensors = s
            .get("tensors")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("samples[{i}]: missing `tensors`"))?;
        if tensors.len() != 2 {
            return Err(format!(
                "samples[{i}] L{layer}.E{expert}: needs exactly gate_up_proj and down_proj"
            ));
        }
        let mut loaded: [Option<Expert>; 2] = [None, None];
        let mut trecs: Vec<Value> = Vec::new();
        for t in tensors {
            let kind_name = t.get("kind").and_then(Value::as_str).unwrap_or("");
            let (proj, qt) = match kind_name {
                "gate_up_proj" => (Proj::GateUp, 135u64),
                "down_proj" => (Proj::Down, 136u64),
                other => {
                    return Err(format!(
                        "samples[{i}] L{layer}.E{expert}: unknown tensor kind `{other}`"
                    ))
                }
            };
            let who = format!("samples[{i}] L{layer}.E{expert}.{kind_name}");
            if t.get("qt").and_then(Value::as_u64) != Some(qt) {
                return Err(format!("{who}: qt {:?} != {qt}", t.get("qt")));
            }
            if t.get("group").and_then(Value::as_u64) != Some(128) {
                return Err(format!("{who}: group {:?} != 128", t.get("group")));
            }
            let shape: Option<Vec<u64>> = t
                .get("shape")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_u64).collect());
            if shape.as_deref() != Some(&[proj.m() as u64, proj.k() as u64][..]) {
                return Err(format!(
                    "{who}: shape {:?} != [{}, {}]",
                    t.get("shape"),
                    proj.m(),
                    proj.k()
                ));
            }
            let file = t
                .get("file")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("{who}: missing `file`"))?;
            if file.contains('/') || file.contains("..") {
                return Err(format!(
                    "{who}: file `{file}` must be a bare name inside the sample dir"
                ));
            }
            let path = Path::new(dir).join(file);
            let bytes =
                std::fs::read(&path).map_err(|e| format!("{who}: {}: {e}", path.display()))?;
            if t.get("bytes").and_then(Value::as_u64) != Some(bytes.len() as u64) {
                return Err(format!(
                    "{who}: manifest bytes {:?} != file {}",
                    t.get("bytes"),
                    bytes.len()
                ));
            }
            let got = sha(&bytes);
            if t.get("sha256").and_then(Value::as_str) != Some(got.as_str()) {
                return Err(format!(
                    "{who}: sha256 {got} != manifest {:?}",
                    t.get("sha256")
                ));
            }
            let offset = t
                .get("artifact_offset")
                .and_then(Value::as_u64)
                .ok_or_else(|| format!("{who}: missing integer `artifact_offset`"))?;
            if verify_range {
                if art_size.is_none_or(|n| offset + bytes.len() as u64 > n) {
                    return Err(format!(
                        "{who}: artifact range [{offset}, +{}) is outside {art_path}",
                        bytes.len()
                    ));
                }
                let h = range_sha256(&art_path, offset, bytes.len())?;
                if h != got {
                    return Err(format!(
                        "{who}: artifact range sha256 {h} != tensor file sha256 {got}"
                    ));
                }
            }
            let e = expert_from_wire(
                format!("real:{tag}:L{layer}.E{expert}/{}", proj.name()),
                proj,
                &bytes,
            )?;
            guard_violations += e
                .scales
                .iter()
                .filter(|&&d| e8p::writer_guard(d).is_err())
                .count() as u64;
            trecs.push(json!({"kind": kind_name, "qt": qt, "file": file, "sha256": got, "bytes": bytes.len(), "artifact_offset": offset,
                "artifact_range_reread_here": verify_range}));
            if loaded[proj.idx()].replace(e).is_some() {
                return Err(format!("{who}: tensor given twice"));
            }
        }
        let (Some(gate), Some(down)) = (loaded[0].take(), loaded[1].take()) else {
            return Err(format!(
                "samples[{i}] L{layer}.E{expert}: needs both gate_up_proj and down_proj"
            ));
        };
        recs.push(json!({"layer": layer, "expert": expert, "tensors": trecs}));
        experts.push(RealExpert {
            label: format!("{tag}:L{layer}.E{expert}"),
            gate: Rc::new(gate),
            down: Rc::new(down),
        });
    }
    for layer in REAL_LAYERS {
        if seen.iter().filter(|(l, _)| *l == layer).count() != 2 {
            return Err(format!(
                "{dir}: layer {layer} needs exactly two cold experts"
            ));
        }
    }
    let meta = json!({"dir": dir, "artifact_tag": tag, "artifact_path": art_path, "artifact_sha256": art_sha, "artifact_sha256_pinned": pin_sha,
        "artifact_bytes": art_size, "artifact_full_rehash": rehash_json, "artifact_range_reread_here": verify_range,
        "sample_manifest_sha256": sha(&raw), "extractor_source_sha256": extractor_sha, "extractor_rc": root.get("rc"),
        "extractor_device": root.get("device"), "extractor_argv": root.get("argv"), "provisional": false,
        "writer_guard_violations": guard_violations, "experts": recs});
    Ok(RealSet {
        tag,
        experts,
        meta,
        guard_violations,
    })
}

/// Every `--samples-dir` (at most one per pin), flattened to one expert list in command-line order.
struct RealSets {
    experts: Vec<RealExpert>,
    sets: Vec<Value>,
    bound: Vec<&'static str>,
    guard_violations: u64,
    gate_eligible: bool,
}
impl RealSets {
    fn json(&self) -> Value {
        if self.sets.is_empty() {
            return Value::Null;
        }
        json!({"n_experts": self.experts.len(), "labels": self.experts.iter().map(|e| e.label.clone()).collect::<Vec<_>>(),
            "bound_pins": self.bound, "pinned_artifacts": ARTIFACT_PINS.iter().map(|(t, s)| json!({"tag": t, "sha256": s})).collect::<Vec<_>>(),
            "writer_guard_violations": self.guard_violations, "sets": self.sets})
    }
}

fn load_real_sets(args: &Args, target: Target) -> Result<RealSets> {
    let mut out = RealSets {
        experts: Vec::new(),
        sets: Vec::new(),
        bound: Vec::new(),
        guard_violations: 0,
        gate_eligible: true,
    };
    if args.synthetic_only && !args.samples.is_empty() {
        return Err("--synthetic-only and --samples-dir are exclusive".into());
    }
    if args.samples.len() > ARTIFACT_PINS.len() {
        return Err(format!(
            "at most {} sample dirs (one per pinned artifact)",
            ARTIFACT_PINS.len()
        ));
    }
    for dir in &args.samples {
        eprintln!("[real] loading E8P samples from {dir}");
        let s = load_real(dir, args.verify_range, args.rehash)?;
        if out.bound.contains(&s.tag) {
            return Err(format!("{dir}: artifact {} bound twice", s.tag));
        }
        out.bound.push(s.tag);
        out.guard_violations += s.guard_violations;
        out.sets.push(s.meta);
        out.experts.extend(s.experts);
    }
    if out.experts.len() > usize::from(u8::MAX) {
        return Err("too many real experts".into());
    }
    if let Some(pin) = target.required_pin() {
        if args.samples.is_empty() && !args.synthetic_only {
            return Err(format!("{} needs --samples-dir binding the `{pin}` artifact (or --synthetic-only for a diagnostic run)\n{USAGE}", target.name()));
        }
        out.gate_eligible = out.bound.contains(&pin);
    }
    Ok(out)
}

fn decode_o(words: &[u16]) -> Vec<i8> {
    let mut o = Vec::with_capacity(words.len() * 8);
    for &w in words {
        let (idx, sign7, shift) = e8p::unpack_word_fields(w);
        let nib = e8p::unpack_word_ref(idx, sign7, shift);
        for k in 0..8 {
            let b = ((nib >> (4 * k)) & 0xf) as u8;
            o.push(((b << 4) as i8) >> 4);
        }
    }
    o
}

/// One K-E0 twin row (all `M` F32 sums) for a token's `groups` A4 blocks.
fn twin_row(e: &Expert, proj: Proj, x: &[A4Block]) -> Vec<f32> {
    let (m, k, groups) = (proj.m(), proj.k(), proj.groups());
    let o = e.o.get_or_init(|| decode_o(&e.words));
    let q: Vec<[i8; 128]> = x
        .iter()
        .map(|b| {
            let mut q = [0i8; 128];
            for (j, v) in q.iter_mut().enumerate() {
                *v = b.element(j);
            }
            q
        })
        .collect();
    (0..m)
        .map(|r| {
            let mut sum = 0f32;
            for g in 0..groups {
                let w = &o[r * k + g * 128..r * k + (g + 1) * 128];
                let c: i32 = w
                    .iter()
                    .zip(q[g].iter())
                    .map(|(&a, &b)| i32::from(a) * i32::from(b))
                    .sum();
                let a = e8p::MAGIC.wrapping_add(c as u32);
                let folded = e8p::fold128(a, x[g].s);
                sum = e8p::epoch_sum(x[g].d, e.scales[r * groups + g], folded, sum);
            }
            sum
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------------------------
// activations
// ---------------------------------------------------------------------------------------------------------------

/// 72-byte A4 blocks for `rows` rows, epoch-major (`[g][row]`), plus the token-major copy for the twin.
fn make_x(rows: usize, groups: usize, seed: u32) -> (Vec<u8>, Vec<A4Block>) {
    let mut rng = Rng(seed | 1);
    let ds = [1.0f32, 0.5, 1.0 / 64.0, 1.0e-3, 3.0, 0.0, 65536.0, 1.0e-20];
    let mut tok: Vec<A4Block> = Vec::new();
    for r in 0..rows {
        for g in 0..groups {
            let mut qs = [0u8; 64];
            let mut s = 0i32;
            for j in 0..128 {
                let v: i8 = match (r + g) % 9 {
                    0 => -8,
                    1 => 7,
                    2 => {
                        if j % 2 == 0 {
                            -8
                        } else {
                            7
                        }
                    }
                    _ => rng.below(16) as i8 - 8,
                };
                qs[j / 2] |= ((v as u8) & 0xf) << (4 * (j % 2));
                s += i32::from(v);
            }
            tok.push(A4Block {
                d: ds[(r * 3 + g) % ds.len()],
                s,
                qs,
            });
        }
    }
    let mut epoch_major = vec![0u8; rows * groups * BLK];
    for g in 0..groups {
        for r in 0..rows {
            let b = &tok[r * groups + g];
            let o = (g * rows + r) * BLK;
            epoch_major[o..o + 4].copy_from_slice(&b.d.to_le_bytes());
            epoch_major[o + 4..o + 8].copy_from_slice(&b.s.to_le_bytes());
            epoch_major[o + 8..o + BLK].copy_from_slice(&b.qs);
        }
    }
    (epoch_major, tok)
}

// ---------------------------------------------------------------------------------------------------------------
// scenarios
// ---------------------------------------------------------------------------------------------------------------

type Tile = (i32, [i32; 16]);

fn tile(expert: i32, slots: &[i32]) -> Tile {
    let mut t = [-1i32; 16];
    t[..slots.len()].copy_from_slice(slots);
    (expert, t)
}
fn tiles_for(expert: i32, slots: Vec<i32>) -> Vec<Tile> {
    slots.chunks(16).map(|c| tile(expert, c)).collect()
}
fn sentinel_tile() -> Tile {
    (-1, [0; 16])
}

struct Scenario {
    name: String,
    div: usize,
    rows: usize,
    tiles: Vec<Tile>,
    kinds: Vec<Kind>,
}

/// Row-count sweep: tile expert `n % base.len()`, `x_row_div` 1 and 10, a trailing sentinel tile that must store nothing.
fn sweep(base: &[Kind], prefix: &str) -> Vec<Scenario> {
    let mut out = Vec::new();
    for (div, rows_per) in [(1usize, 20usize), (10, 4)] {
        for n in [1usize, 2, 7, 8, 15, 16, 17, 33] {
            let slots: Vec<i32> = (0..n)
                .map(|i| ((i * 7 + 3) % (rows_per * div)) as i32)
                .collect();
            let mut tiles = tiles_for((n % base.len()) as i32, slots);
            tiles.push(sentinel_tile());
            out.push(Scenario {
                name: format!("{prefix}rows{n}_div{div}"),
                div,
                rows: rows_per,
                tiles,
                kinds: base.to_vec(),
            });
        }
    }
    out
}

/// Mixed experts (`kinds[e]` is expert `e`, four of them), non-monotone expert and slot order, pad holes, mid-sequence and trailing sentinel tiles.
fn mixed(name: String, kinds: Vec<Kind>) -> Scenario {
    let mut tiles = tiles_for(2, vec![5, 3, 19, 0, 1, 2, 11, 12]);
    tiles.push(sentinel_tile());
    tiles.extend(tiles_for(0, (0..17).map(|i| (i * 3 % 20) as i32).collect()));
    tiles.extend(tiles_for(1, vec![7]));
    tiles.extend(tiles_for(3, vec![14, 6, 9]));
    tiles.push(tile(0, &[-1, 4, -1, 9]));
    tiles.push(sentinel_tile());
    Scenario {
        name,
        div: 1,
        rows: 20,
        tiles,
        kinds,
    }
}

fn scenarios() -> Vec<Scenario> {
    let base = vec![Kind::Random, Kind::EveryWord, Kind::Extreme, Kind::Parity];
    let mut out = sweep(&base, "");
    out.push(mixed(
        "mixed_experts_unsorted_pad".into(),
        vec![Kind::Random, Kind::Extreme, Kind::EveryWord, Kind::Parity],
    ));
    for (tag, first) in [("sentinel_ff", 0x3cffu16), ("sentinel_00", 0x3c00u16)] {
        out.push(Scenario {
            name: format!("{tag}_last_row_chunk"),
            div: 1,
            rows: 16,
            tiles: tiles_for(1, (0..16).collect()),
            kinds: vec![Kind::Sentinel(first), Kind::Sentinel(first), Kind::Random],
        });
    }
    out
}

/// The same sweep and mixed-expert geometry over the `count` (>= 4) real experts loaded from `--samples-dir`.
fn real_scenarios(count: usize) -> Vec<Scenario> {
    let base: Vec<Kind> = (0..count).map(|i| Kind::Real(i as u8)).collect();
    let mut out = sweep(&base, "real_");
    out.push(mixed(
        "real_mixed_experts_unsorted_pad".into(),
        vec![Kind::Real(2), Kind::Real(0), Kind::Real(3), Kind::Real(1)],
    ));
    out
}

fn all_scenarios(real: usize) -> Vec<Scenario> {
    let mut v = scenarios();
    if real > 0 {
        v.extend(real_scenarios(real));
    }
    v
}

/// One slab buffer: each expert's tensor at a 256-byte aligned offset, 0xFF guards before, between and after.
fn slab(tensors: &[&[u8]]) -> (Vec<u8>, Vec<usize>) {
    let mut buf = vec![0xffu8; GUARD];
    let mut offs = Vec::new();
    for t in tensors {
        offs.push(buf.len());
        buf.extend_from_slice(t);
        buf.resize(align(buf.len()) + GUARD, 0xff);
    }
    (buf, offs)
}

// ---------------------------------------------------------------------------------------------------------------
// device BF16 intrinsic hook (probe kernels) and the replayed epilogue DAG
// ---------------------------------------------------------------------------------------------------------------

struct Probe {
    exp2: Function,
    ldexp: Function,
    div: Function,
    ka: [usize; 3],
}

fn probe_launch(
    hip: &HipRuntime,
    f: &Function,
    ka: usize,
    bufs: &[&DeviceBuffer],
    n: usize,
) -> Result<()> {
    let off = bufs.len() * 8;
    if ka < off + 4 {
        return Err(format!(
            "probe kernarg size {ka} < explicit size {}",
            off + 4
        ));
    }
    let mut a = vec![0u8; ka];
    for (i, b) in bufs.iter().enumerate() {
        a[i * 8..i * 8 + 8].copy_from_slice(&(b.as_ptr() as u64).to_le_bytes());
    }
    a[off..off + 4].copy_from_slice(&(n as u32).to_le_bytes());
    // SAFETY: explicit args laid out per the probe kernels' (pointer.., i32 n) ABI; blob sized to the code object's
    // `.kernarg_segment_size`; every buffer outlives the synchronize below.
    unsafe {
        hip.launch_kernel_blob(
            f,
            [n.div_ceil(256) as u32, 1, 1],
            [256, 1, 1],
            0,
            None,
            &mut a,
        )
    }
    .map_err(err)?;
    hip.device_synchronize().map_err(err)
}

/// Bitwise equality, any two NaNs equal (the payload/sign of an invalid-operation NaN is not part of the contract).
fn same_f32_bits(a: u32, b: u32) -> bool {
    a == b || (f32::from_bits(a).is_nan() && f32::from_bits(b).is_nan())
}

/// Correctly rounded (RNE, denormals preserved) IEEE-754 quotient bits on the CPU.
fn ieee_div_bits(num: u32, den: u32) -> u32 {
    (f32::from_bits(num) / f32::from_bits(den)).to_bits()
}

struct Hook<'a> {
    hip: &'a HipRuntime,
    probe: Probe,
    div_checked: Cell<u64>,
    div_vs_fdiv_rn_mismatch: Cell<u64>,
    div_vs_ieee_mismatch: Cell<u64>,
    div_list: RefCell<Vec<Value>>,
    dag_passes: Cell<u64>,
    dag_calls: Cell<u64>,
}

struct Replay<'a> {
    prev: &'a [Vec<(u32, u32)>; 3],
    prev_ans: &'a [Vec<u32>; 3],
    cur: RefCell<[Vec<(u32, u32)>; 3]>,
    miss: Cell<bool>,
}
impl Replay<'_> {
    fn call(&self, k: usize, a: u32, b: u32) -> f32 {
        let mut cur = self.cur.borrow_mut();
        let pos = cur[k].len();
        cur[k].push((a, b));
        if pos < self.prev[k].len() && self.prev[k][pos] == (a, b) {
            f32::from_bits(self.prev_ans[k][pos])
        } else {
            self.miss.set(true);
            0.0
        }
    }
}
impl EpilogueIntrinsics for Replay<'_> {
    fn exp2(&self, x: f32) -> f32 {
        self.call(0, x.to_bits(), 0)
    }
    fn ldexp(&self, x: f32, e: i32) -> f32 {
        self.call(1, x.to_bits(), e as u32)
    }
    fn divide(&self, numerator: f32, denominator: f32) -> f32 {
        self.call(2, numerator.to_bits(), denominator.to_bits())
    }
}

impl Hook<'_> {
    fn eval_exp2(&self, x: &[u32]) -> Result<Vec<u32>> {
        if x.is_empty() {
            return Ok(vec![]);
        }
        let hip = self.hip;
        let (din, dout) = (up(hip, &u32s_le(x))?, hip.malloc(x.len() * 4).map_err(err)?);
        probe_launch(
            hip,
            &self.probe.exp2,
            self.probe.ka[0],
            &[&din, &dout],
            x.len(),
        )?;
        let r = le_u32s(&read_back(hip, &dout, x.len() * 4)?);
        free_all(hip, vec![din, dout])?;
        Ok(r)
    }

    fn eval_ldexp(&self, x: &[u32], e: &[u32]) -> Result<Vec<u32>> {
        if x.is_empty() {
            return Ok(vec![]);
        }
        let hip = self.hip;
        let (dx, de, dout) = (
            up(hip, &u32s_le(x))?,
            up(hip, &u32s_le(e))?,
            hip.malloc(x.len() * 4).map_err(err)?,
        );
        probe_launch(
            hip,
            &self.probe.ldexp,
            self.probe.ka[1],
            &[&dx, &de, &dout],
            x.len(),
        )?;
        let r = le_u32s(&read_back(hip, &dout, x.len() * 4)?);
        free_all(hip, vec![dx, de, dout])?;
        Ok(r)
    }

    /// `(explicit v_div_scale/rcp/fma/fmas/fixup sequence, hipcc __fdiv_rn)`.
    fn eval_div(&self, num: &[u32], den: &[u32]) -> Result<(Vec<u32>, Vec<u32>)> {
        if num.is_empty() {
            return Ok((vec![], vec![]));
        }
        let hip = self.hip;
        let n = num.len();
        let (dn, dd) = (up(hip, &u32s_le(num))?, up(hip, &u32s_le(den))?);
        let (d_seq, d_rn) = (
            hip.malloc(n * 4).map_err(err)?,
            hip.malloc(n * 4).map_err(err)?,
        );
        probe_launch(
            hip,
            &self.probe.div,
            self.probe.ka[2],
            &[&dn, &dd, &d_seq, &d_rn],
            n,
        )?;
        let r = (
            le_u32s(&read_back(hip, &d_seq, n * 4)?),
            le_u32s(&read_back(hip, &d_rn, n * 4)?),
        );
        free_all(hip, vec![dn, dd, d_seq, d_rn])?;
        Ok(r)
    }

    /// Known-answer test of the divide probe against CPU IEEE division: the explicit `v_div_scale/rcp/fma/fmas/fixup`
    /// sequence, hipcc's `__fdiv_rn` and the exact quotient must agree bit for bit (denominators in `[2^126, inf]`
    /// where `v_div_scale` rescales, denormal operands/results, zero/inf/NaN, plus 65536 random bit-pattern pairs).
    fn divide_known_answer(&self) -> Result<(Value, bool)> {
        let nums: [f32; 22] = [
            -87.5,
            87.5,
            -88.7,
            -88.72,
            -87.0,
            -80.0,
            -1.0,
            1.0,
            0.5,
            1e-30,
            -1e-30,
            1e-40,
            3.0e38,
            -3.0e38,
            f32::MIN_POSITIVE,
            0.0,
            -0.0,
            65536.0,
            1e-20,
            4.0,
            1.0e30,
            f32::MAX,
        ];
        let mut dens: Vec<u32> = (0..64u32).map(|i| 0x7e80_0000 + i * 0x0004_0000).collect();
        dens.extend([
            0x7f80_0000,
            0x7f7f_ffff,
            0x7f7f_fffe,
            0x7e7f_ffff,
            0x7e80_0001,
            1.0f32.to_bits(),
            1.0f32.to_bits() + 1,
            0x0000_0001,
            0x0080_0000,
            0x0000_7fff,
            3.0f32.to_bits(),
            1.0e30f32.to_bits(),
            65536.0f32.to_bits(),
            0.5f32.to_bits(),
            0x8000_0000,
            0x7fc0_0000,
        ]);
        let r = 1.0017680e38f32.to_bits(); // 1 + exp(87.5) rounded to f32 and its neighbours
        dens.extend([r - 1, r, r + 1]);
        let (mut num, mut den): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
        for n in nums {
            for d in &dens {
                num.push(n.to_bits());
                den.push(*d);
            }
        }
        let mut rng = Rng64(0xD1_71DE_0000_0001);
        for _ in 0..65536 {
            let w = rng.next();
            num.push(w as u32);
            den.push((w >> 32) as u32);
        }
        let (seq, rn) = self.eval_div(&num, &den)?;
        let (mut seq_ieee, mut rn_ieee, mut seq_rn) = (0u64, 0u64, 0u64);
        let mut first: Vec<Value> = Vec::new();
        for i in 0..num.len() {
            let ieee = ieee_div_bits(num[i], den[i]);
            let (a, b, c) = (
                !same_f32_bits(seq[i], ieee),
                !same_f32_bits(rn[i], ieee),
                !same_f32_bits(seq[i], rn[i]),
            );
            seq_ieee += a as u64;
            rn_ieee += b as u64;
            seq_rn += c as u64;
            if (a || b || c) && first.len() < 64 {
                first.push(json!({"num": format!("{:#010x}", num[i]), "den": format!("{:#010x}", den[i]), "probe_sequence": format!("{:#010x}", seq[i]), "hipcc_fdiv_rn": format!("{:#010x}", rn[i]), "cpu_ieee": format!("{ieee:#010x}")}));
            }
        }
        let ok = seq_ieee == 0 && rn_ieee == 0 && seq_rn == 0;
        Ok((
            json!({"cases": num.len(), "probe_sequence_vs_cpu_ieee_mismatches": seq_ieee, "hipcc_fdiv_rn_vs_cpu_ieee_mismatches": rn_ieee, "probe_sequence_vs_fdiv_rn_mismatches": seq_rn,
            "first_mismatches": first, "ok": ok}),
            ok,
        ))
    }

    /// The frozen BF16 epilogue DAG (`grouped_prefill_epilogue_with`) over `sums` with `v_exp_f32`, `v_ldexp_f32` and
    /// the IEEE divide taken from the device probes, replayed until a pass is answered entirely by device results.
    fn dag_epilogue(
        &self,
        kind: DecodeKind,
        m: usize,
        sums: &[f32],
        slot_ids: &[i32],
    ) -> Result<Vec<u16>> {
        let mut prev: [Vec<(u32, u32)>; 3] = Default::default();
        let mut prev_ans: [Vec<u32>; 3] = Default::default();
        let mut prev_rn: Vec<u32> = Vec::new();
        for _ in 0..8 {
            let rp = Replay {
                prev: &prev,
                prev_ans: &prev_ans,
                cur: RefCell::new(Default::default()),
                miss: Cell::new(false),
            };
            let out = mq_i4_reference::grouped_prefill_epilogue_with(kind, m, sums, slot_ids, &rp);
            let miss = rp.miss.get();
            let cur = rp.cur.into_inner();
            self.dag_passes.set(self.dag_passes.get() + 1);
            if !miss {
                self.dag_calls.set(
                    self.dag_calls.get() + (cur[0].len() + cur[1].len() + cur[2].len()) as u64,
                );
                // transcription self-check over exactly the divide requests the twin consumed
                for i in 0..prev_ans[2].len() {
                    self.div_checked.set(self.div_checked.get() + 1);
                    let (nu, de, seq) = (prev[2][i].0, prev[2][i].1, prev_ans[2][i]);
                    let ieee = ieee_div_bits(nu, de);
                    let ieee_bad = !same_f32_bits(seq, ieee);
                    let rn_bad = seq != prev_rn[i];
                    if ieee_bad {
                        self.div_vs_ieee_mismatch
                            .set(self.div_vs_ieee_mismatch.get() + 1);
                    }
                    if rn_bad {
                        self.div_vs_fdiv_rn_mismatch
                            .set(self.div_vs_fdiv_rn_mismatch.get() + 1);
                    }
                    if ieee_bad || rn_bad {
                        let mut l = self.div_list.borrow_mut();
                        if l.len() < 256 {
                            l.push(json!({"num": format!("{nu:#010x}"), "den": format!("{de:#010x}"), "probe_sequence": format!("{seq:#010x}"), "hipcc_fdiv_rn": format!("{:#010x}", prev_rn[i]), "cpu_ieee": format!("{ieee:#010x}")}));
                        }
                    }
                }
                return Ok(out);
            }
            let a0 = self.eval_exp2(&cur[0].iter().map(|p| p.0).collect::<Vec<_>>())?;
            let a1 = self.eval_ldexp(
                &cur[1].iter().map(|p| p.0).collect::<Vec<_>>(),
                &cur[1].iter().map(|p| p.1).collect::<Vec<_>>(),
            )?;
            let (a2, rn) = self.eval_div(
                &cur[2].iter().map(|p| p.0).collect::<Vec<_>>(),
                &cur[2].iter().map(|p| p.1).collect::<Vec<_>>(),
            )?;
            prev = cur;
            prev_ans = [a0, a1, a2];
            prev_rn = rn;
        }
        Err("epilogue DAG replay did not converge in 8 passes".into())
    }
}

// ---------------------------------------------------------------------------------------------------------------
// one scenario on one projection
// ---------------------------------------------------------------------------------------------------------------

struct Ctx<'a> {
    hip: &'a HipRuntime,
    hook: Hook<'a>,
    acc: [Function; 2],
    ship: [Function; 2],
    /// Kernarg blob size per projection: exactly the 64-byte ABI (60 bytes + `n_experts@60`), checked against the code object metadata.
    ka: [usize; 2],
    neg: u8,
    seed: u32,
    experts: RefCell<HashMap<(String, Proj), Rc<Expert>>>,
    target: Target,
    real: Vec<RealExpert>,
}

#[derive(Default)]
struct Counts {
    acc_checked: u64,
    acc_live: u64,
    acc_pad: u64,
    acc_poison: u64,
    acc_bad: u64,
    h_checked: u64,
    h_live: u64,
    h_pad: u64,
    h_poison: u64,
    h_bad: u64,
}

impl Ctx<'_> {
    fn expert(&self, kind: Kind, proj: Proj) -> Result<Rc<Expert>> {
        if let Kind::Real(i) = kind {
            return self
                .real
                .get(usize::from(i))
                .map(|r| r.proj(proj))
                .ok_or_else(|| format!("real expert {i} is not loaded"));
        }
        let key = (kind.label(), proj);
        if let Some(e) = self.experts.borrow().get(&key) {
            return Ok(e.clone());
        }
        let e = Rc::new(expert(kind, proj, self.seed.wrapping_mul(2654435761))?);
        self.experts.borrow_mut().insert(key, e.clone());
        Ok(e)
    }

    fn run_one(&self, sc: &Scenario, proj: Proj) -> Result<(Value, Counts)> {
        let hip = self.hip;
        let (m, k, h, groups, pi) = (proj.m(), proj.k(), proj.h(), proj.groups(), proj.idx());
        let experts: Vec<Rc<Expert>> = sc
            .kinds
            .iter()
            .map(|kd| self.expert(*kd, proj))
            .collect::<Result<_>>()?;
        let (xq, x_tok) = make_x(sc.rows, groups, self.seed ^ 0x51ed);
        let ntiles = sc.tiles.len();
        let m_total = ntiles * 16;
        let tile_ids: Vec<i32> = sc.tiles.iter().map(|t| t.0).collect();
        let sorted: Vec<i32> = sc.tiles.iter().flat_map(|t| t.1).collect();
        let (mut buf, offs) = slab(
            &experts
                .iter()
                .map(|e| e.bytes.as_slice())
                .collect::<Vec<_>>(),
        );
        if self.neg == 2 {
            buf[offs[0] + 1000] ^= 0x10; // device-side corruption only; the twin keeps the clean wire
        }
        let slab_dev = up(hip, &buf)?;
        let base = slab_dev.as_ptr() as u64;
        let ptr_table: Vec<u8> = offs
            .iter()
            .flat_map(|o| (base + *o as u64).to_le_bytes())
            .collect();
        let ptrs_dev = up(hip, &ptr_table)?;
        let tiles_dev = up(hip, &i32s_le(&tile_ids))?;
        let sorted_dev = up(hip, &i32s_le(&sorted))?;
        let x_dev = up(hip, &xq)?;
        let (acc_bytes, h_bytes) = (m_total * m * 4, m_total * h * 2);
        let (y_acc, y_h) = (
            hip.malloc(acc_bytes).map_err(err)?,
            hip.malloc(h_bytes).map_err(err)?,
        );
        hip.memset(&y_acc, 0xa5, acc_bytes).map_err(err)?;
        hip.memset(&y_h, 0xa5, h_bytes).map_err(err)?;
        let launch = |f: &Function, y: &DeviceBuffer| -> Result<()> {
            let mut a = vec![0u8; self.ka[pi]];
            let p = [
                ptrs_dev.as_ptr() as u64,
                tiles_dev.as_ptr() as u64,
                sorted_dev.as_ptr() as u64,
                x_dev.as_ptr() as u64,
                y.as_ptr() as u64,
            ];
            for (i, v) in p.iter().enumerate() {
                a[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
            }
            for (off, v) in [
                (40usize, m as u32),
                (44, k as u32),
                (48, sc.div as u32),
                (52, m_total as u32),
                (56, sc.rows as u32),
                (60, (ptrs_dev.size() / 8) as u32),
            ] {
                a[off..off + 4].copy_from_slice(&v.to_le_bytes());
            }
            // SAFETY: the shared PM 64-byte kernarg ABI (5 pointers, 6 u32 incl. n_experts@60 = the pointer table's actual entry
            // count); every buffer outlives the synchronize.
            unsafe {
                hip.launch_kernel_blob(
                    f,
                    [(m / 64) as u32, ntiles as u32, 1],
                    [proj.threads(), 1, 1],
                    0,
                    None,
                    &mut a,
                )
            }
            .map_err(err)?;
            hip.device_synchronize().map_err(err)
        };
        launch(&self.acc[pi], &y_acc)?;
        launch(&self.ship[pi], &y_h)?;
        let got_acc = read_back(hip, &y_acc, acc_bytes)?;
        let got_h = read_back(hip, &y_h, h_bytes)?;
        free_all(
            hip,
            vec![slab_dev, ptrs_dev, tiles_dev, sorted_dev, x_dev, y_acc, y_h],
        )?;

        // Twin F32 rows, one per (expert, x row) the slots reference.
        let mut dense: HashMap<(i32, usize), Vec<f32>> = HashMap::new();
        let mut sums = vec![0f32; m_total * m];
        let mut slot_ids = vec![-1i32; m_total];
        let mut c = Counts::default();
        let mut acc_first: Vec<Value> = Vec::new();
        for (ti, t) in sc.tiles.iter().enumerate() {
            for lane in 0..16 {
                let p = ti * 16 + lane;
                let live = t.0 >= 0 && t.1[lane] >= 0;
                if live {
                    let xr = t.1[lane] as usize / sc.div;
                    let row = dense.entry((t.0, xr)).or_insert_with(|| {
                        let e = &experts[t.0 as usize];
                        let mut r = twin_row(e, proj, &x_tok[xr * groups..(xr + 1) * groups]);
                        if self.neg == 1 {
                            r[0] = f32::from_bits(r[0].to_bits() ^ 1); // twin-side negative control
                        }
                        r
                    });
                    sums[p * m..(p + 1) * m].copy_from_slice(row);
                    slot_ids[p] = t.1[lane];
                }
                for r in 0..m {
                    let o = (p * m + r) * 4;
                    let bits = u32::from_le_bytes(got_acc[o..o + 4].try_into().unwrap());
                    let want = if t.0 < 0 {
                        c.acc_poison += 1;
                        POISON_F32
                    } else if live {
                        c.acc_live += 1;
                        sums[p * m + r].to_bits()
                    } else {
                        c.acc_pad += 1;
                        0
                    };
                    c.acc_checked += 1;
                    if bits != want {
                        c.acc_bad += 1;
                        if acc_first.len() < FIRST_LIMIT {
                            acc_first.push(json!({"position": p, "tile": ti, "lane": lane, "row": r, "device": format!("{bits:#010x}"), "k_e0": format!("{want:#010x}"),
                                "class": if t.0 < 0 { "sentinel_poison" } else if live { "live" } else { "pad" }}));
                        }
                    }
                }
            }
        }

        // BF16: the frozen epilogue DAG over the twin sums with device intrinsics; pad slots +0, sentinel tiles poison.
        let mut want_h = self.hook.dag_epilogue(proj.kind(), m, &sums, &slot_ids)?;
        for (ti, t) in sc.tiles.iter().enumerate() {
            if t.0 < 0 {
                for v in &mut want_h[ti * 16 * h..(ti + 1) * 16 * h] {
                    *v = POISON_BF16;
                }
            }
        }
        let mut h_first: Vec<Value> = Vec::new();
        for (i, want) in want_h.iter().enumerate() {
            let p = i / h;
            let (ti, lane) = (p / 16, p % 16);
            let t = &sc.tiles[ti];
            let class = if t.0 < 0 {
                "sentinel_poison"
            } else if t.1[lane] >= 0 {
                "live"
            } else {
                "pad"
            };
            match class {
                "sentinel_poison" => c.h_poison += 1,
                "live" => c.h_live += 1,
                _ => c.h_pad += 1,
            }
            c.h_checked += 1;
            let got = u16::from_le_bytes([got_h[2 * i], got_h[2 * i + 1]]);
            if got != *want {
                c.h_bad += 1;
                if h_first.len() < FIRST_LIMIT {
                    h_first.push(json!({"position": p, "tile": ti, "lane": lane, "col": i % h, "device": format!("{got:#06x}"), "reference": format!("{want:#06x}"), "class": class}));
                }
            }
        }
        let live_slots = slot_ids.iter().filter(|&&s| s >= 0).count();
        let report = json!({
            "scenario": sc.name, "proj": proj.name(), "symbol": proj.symbol(self.target), "x_row_div": sc.div, "x_rows": sc.rows,
            "tiles": ntiles, "live_slots": live_slots, "pad_slots": m_total - live_slots,
            "sentinel_tiles": sc.tiles.iter().filter(|t| t.0 < 0).count(),
            "experts": experts.iter().map(|e| json!({"label": e.label, "wire_sha256": e.sha256, "wire_bytes": e.bytes.len(), "stats": e.stats})).collect::<Vec<_>>(),
            "experts_used": dense.keys().map(|k| k.0).collect::<std::collections::BTreeSet<i32>>().into_iter().collect::<Vec<_>>(),
            "x_sha256": sha(&xq),
            "tile_table_sha256": sha(&[i32s_le(&tile_ids), i32s_le(&sorted)].concat()),
            "acc_outputs_checked": c.acc_checked, "acc_live_checked": c.acc_live, "acc_pad_checked": c.acc_pad, "acc_sentinel_poison_checked": c.acc_poison,
            "acc_mismatches": c.acc_bad, "acc_first": acc_first,
            "h_outputs_checked": c.h_checked, "h_live_checked": c.h_live, "h_pad_checked": c.h_pad, "h_sentinel_poison_checked": c.h_poison,
            "h_mismatches": c.h_bad, "h_first": h_first,
            "acc_sha256": sha(&got_acc), "h_sha256": sha(&got_h),
        });
        Ok((report, c))
    }
}

// ---------------------------------------------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------------------------------------------

#[derive(Default)]
struct Args {
    e8p: Option<PathBuf>,
    probe: Option<PathBuf>,
    out: Option<PathBuf>,
    card: Option<String>,
    expect_bdf: Option<String>,
    expect_arch: Option<String>,
    expect_uuid: Option<String>,
    samples: Vec<String>,
    verify_range: bool,
    rehash: bool,
    synthetic_only: bool,
    seed: u32,
    neg: u8,
    probe_ka: Option<usize>,
    overwrite: bool,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        seed: 1,
        ..Default::default()
    };
    let mut it = std::env::args().skip(1);
    while let Some(f) = it.next() {
        if matches!(
            f.as_str(),
            "--overwrite" | "--verify-artifact-range" | "--rehash-artifact" | "--synthetic-only"
        ) {
            match f.as_str() {
                "--overwrite" => a.overwrite = true,
                "--verify-artifact-range" => a.verify_range = true,
                "--rehash-artifact" => a.rehash = true,
                _ => a.synthetic_only = true,
            }
            continue;
        }
        let v = it
            .next()
            .ok_or_else(|| format!("{f} needs a value\n{USAGE}"))?;
        match f.as_str() {
            "--e8p" => a.e8p = Some(v.into()),
            "--probe-codeobject" => a.probe = Some(v.into()),
            "--out" => a.out = Some(v.into()),
            "--card" => a.card = Some(v),
            "--expect-bdf" => a.expect_bdf = Some(v.to_ascii_lowercase()),
            "--expect-arch" => a.expect_arch = Some(v),
            "--expect-uuid" => a.expect_uuid = Some(v),
            "--samples-dir" => a
                .samples
                .extend(v.split(',').filter(|s| !s.is_empty()).map(str::to_string)),
            "--seed" => a.seed = v.parse().map_err(err)?,
            "--negative-control" => a.neg = v.parse().map_err(err)?,
            "--probe-kernarg-bytes" => a.probe_ka = Some(v.parse().map_err(err)?),
            _ => return Err(format!("unknown flag {f}\n{USAGE}")),
        }
    }
    Ok(a)
}

fn env_or_null(k: &str) -> Value {
    std::env::var(k).map(Value::String).unwrap_or(Value::Null)
}

/// Returns the process exit code (0 exact, 3 mismatch); `Err` = exit 1.
fn run() -> Result<i32> {
    let args = parse_args()?;
    let (e8p_path, probe_path, out_path, card) = (
        args.e8p.clone().ok_or(USAGE)?,
        args.probe.clone().ok_or(USAGE)?,
        args.out.clone().ok_or(USAGE)?,
        args.card.clone().ok_or(USAGE)?,
    );
    for v in ["HSA_CU_MASK", "ROC_GLOBAL_CU_MASK"] {
        if std::env::var_os(v).is_some() {
            return Err(format!("refusing to run with {v} set"));
        }
    }
    if out_path.exists() && !args.overwrite {
        return Err(format!("{} exists (pass --overwrite)", out_path.display()));
    }
    let started = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let command: Vec<String> = std::env::args().collect();
    let target =
        Target::of_name(args.expect_arch.as_deref().unwrap_or("gfx1201")).ok_or_else(|| {
            format!(
                "--expect-arch {:?}: this oracle serves gfx1201 and gfx1151\n{USAGE}",
                args.expect_arch
            )
        })?;
    if target == Target::Gfx1151 && args.expect_bdf.is_none() {
        return Err(format!("gfx1151 has no ROCr UUID: --expect-bdf is required (--expect-uuid is optional)\n{USAGE}"));
    }
    let real = load_real_sets(&args, target)?;

    // K-E0 self-check: SWAR unpack == reference unpack over every wire word (frozen functions, no new arithmetic).
    let (lut, mask) = (e8p::build_lut(), e8p::build_mask());
    let mut swar_bad = 0u64;
    for w in 0..=0xffffu16 {
        let (idx, sign7, shift) = e8p::unpack_word_fields(w);
        if e8p::unpack_word_swar(&lut, &mask, idx, sign7, shift)
            != e8p::unpack_word_ref(idx, sign7, shift)
        {
            swar_bad += 1;
        }
    }

    let hip = HipRuntime::load().map_err(err)?;
    let id = hip.current_device().map_err(err)?;
    let (uuid, bdf, arch) = (
        hip.device_uuid(id).map_err(err)?,
        hip.device_pci_bus_id(id).map_err(err)?.to_ascii_lowercase(),
        hip.get_arch(id).map_err(err)?,
    );
    if arch != target.name() {
        return Err(format!("expected {}, got {arch}", target.name()));
    }
    if let Some(u) = &args.expect_uuid {
        if !uuid.eq_ignore_ascii_case(u) {
            return Err(format!("expected UUID {u}, got {uuid}"));
        }
    }
    if let Some(b) = &args.expect_bdf {
        if *b != bdf {
            return Err(format!("expected BDF {b}, got {bdf}"));
        }
    }
    let rd = |p: &Path| std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()));
    let (e8p_img, probe_img) = (rd(&e8p_path)?, rd(&probe_path)?);
    let (m_e8p, m_probe) = (
        hip.module_load_data(&e8p_img).map_err(err)?,
        hip.module_load_data(&probe_img).map_err(err)?,
    );
    let probe_ka = |sym: &str| -> Result<usize> {
        match (metadata_kernarg_size(&probe_img, sym), args.probe_ka) {
            (Some(m), Some(o)) if m != o => Err(format!(
                "{sym}: kernarg metadata {m} != --probe-kernarg-bytes {o}"
            )),
            (Some(m), _) => Ok(m),
            (None, Some(o)) => Ok(o),
            (None, None) => Err(format!(
                "{sym}: `.kernarg_segment_size` not found in {}; pass --probe-kernarg-bytes",
                probe_path.display()
            )),
        }
    };
    let ka = [
        probe_ka("g1_probe_exp2")?,
        probe_ka("g1_probe_ldexp")?,
        probe_ka("g1_probe_div")?,
    ];
    let probe = Probe {
        exp2: hip
            .module_get_function(&m_probe, "g1_probe_exp2")
            .map_err(err)?,
        ldexp: hip
            .module_get_function(&m_probe, "g1_probe_ldexp")
            .map_err(err)?,
        div: hip
            .module_get_function(&m_probe, "g1_probe_div")
            .map_err(err)?,
        ka,
    };
    let getf = |name: &str| {
        hip.module_get_function(&m_e8p, name)
            .map_err(|e| format!("{name}: {e}"))
    };
    let (sym_gu, sym_dn) = (Proj::GateUp.symbol(target), Proj::Down.symbol(target));
    let acc = [
        getf(&format!("{sym_gu}_acc"))?,
        getf(&format!("{sym_dn}_acc"))?,
    ];
    let ship = [getf(&sym_gu)?, getf(&sym_dn)?];
    let e8p_kernarg = |sym: &str| -> Result<usize> {
        match metadata_kernarg_size(&e8p_img, sym) {
            Some(64) => Ok(64),
            other => Err(format!(
                "{sym}: e8p kernarg segment {other:?}, expected the 64-byte ABI (n_experts@60)"
            )),
        }
    };
    let e8p_ka = [e8p_kernarg(&sym_gu)?, e8p_kernarg(&sym_dn)?];
    let hook = Hook {
        hip: &hip,
        probe,
        div_checked: Cell::new(0),
        div_vs_fdiv_rn_mismatch: Cell::new(0),
        div_vs_ieee_mismatch: Cell::new(0),
        div_list: RefCell::new(Vec::new()),
        dag_passes: Cell::new(0),
        dag_calls: Cell::new(0),
    };
    let (div_kat, div_kat_ok) = hook.divide_known_answer()?;
    let ctx = Ctx {
        hip: &hip,
        hook,
        acc,
        ship,
        ka: e8p_ka,
        neg: args.neg,
        seed: args.seed,
        experts: RefCell::new(HashMap::new()),
        target,
        real: real.experts.clone(),
    };

    let mut rows: Vec<Value> = Vec::new();
    let mut tot = Counts::default();
    let mut first_mismatch: Option<Value> = None;
    let scs = all_scenarios(ctx.real.len());
    for proj in Proj::ALL {
        for sc in &scs {
            let (mut r, c) = ctx.run_one(sc, proj)?;
            if first_mismatch.is_none() {
                let f = if c.acc_bad > 0 {
                    Some(("acc_f32", r["acc_first"][0].clone()))
                } else if c.h_bad > 0 {
                    Some(("h_bf16", r["h_first"][0].clone()))
                } else {
                    None
                };
                if let Some((check, detail)) = f {
                    first_mismatch = Some(
                        json!({"scenario": sc.name, "proj": proj.name(), "check": check, "detail": detail}),
                    );
                }
            }
            r["pass"] = json!(c.acc_bad == 0 && c.h_bad == 0);
            tot.acc_checked += c.acc_checked;
            tot.acc_live += c.acc_live;
            tot.acc_pad += c.acc_pad;
            tot.acc_poison += c.acc_poison;
            tot.acc_bad += c.acc_bad;
            tot.h_checked += c.h_checked;
            tot.h_live += c.h_live;
            tot.h_pad += c.h_pad;
            tot.h_poison += c.h_poison;
            tot.h_bad += c.h_bad;
            rows.push(r);
        }
    }
    let hk = &ctx.hook;
    let (div_rn_bad, div_ieee_bad) = (
        hk.div_vs_fdiv_rn_mismatch.get(),
        hk.div_vs_ieee_mismatch.get(),
    );
    let mismatch_total =
        tot.acc_bad + tot.h_bad + swar_bad + div_rn_bad + div_ieee_bad + u64::from(!div_kat_ok);
    let pass = mismatch_total == 0 && real.guard_violations == 0;
    let rc = if pass { 0 } else { 3 };
    let scenario_names: Vec<String> = scs.iter().map(|s| s.name.clone()).collect();
    let receipt = json!({
        "schema": target.schema(), "target_arch": target.name(), "started_unix": started, "rc": rc, "pass": pass,
        "command": command.join(" "), "argv": command,
        "fabric": {"runner": FABRIC_RUNNER, "under_fabric": std::env::var_os("FABRIC_JOB").is_some(), "FABRIC_JOB": env_or_null("FABRIC_JOB"), "FABRIC_CARDS": env_or_null("FABRIC_CARDS")},
        "device": {"uuid": uuid, "bdf": bdf, "arch": arch, "card_arg": card, "expect_bdf": args.expect_bdf, "expect_uuid": args.expect_uuid,
            "uuid_authoritative": target != Target::Gfx1151,
            "env": {"HIPFIRE_DEVICES": env_or_null("HIPFIRE_DEVICES"), "ROCR_VISIBLE_DEVICES": env_or_null("ROCR_VISIBLE_DEVICES"), "HIP_VISIBLE_DEVICES": env_or_null("HIP_VISIBLE_DEVICES")}},
        "inputs": {
            "e8p_module": {"path": e8p_path.display().to_string(), "bytes": e8p_img.len(), "sha256": sha(&e8p_img), "kernarg_bytes": {"gate_up": e8p_ka[0], "down": e8p_ka[1]}},
            "probe_codeobject": {"path": probe_path.display().to_string(), "bytes": probe_img.len(), "sha256": sha(&probe_img), "kernarg_bytes": {"exp2": ka[0], "ldexp": ka[1], "div": ka[2]}},
            "harness": {"source": "crates/hipfire-runtime/examples/qwen4_moe_sym_e8p_g1_oracle.rs", "sha256": sha(HARNESS_SRC), "embedded_at_build": true},
            "k_e0_reference": {"source": "crates/hipfire-runtime/src/e8p_i4_reference.rs", "sha256": sha(REFERENCE_SRC), "embedded_at_build": true, "books_sha256": e8p::E8P_BOOKS_SHA256},
        },
        "real_samples": real.json(),
        "gate_eligible": real.gate_eligible,
        "symbols": {"gate_up": sym_gu, "down": sym_dn, "acc_suffix": "_acc", "probe": ["g1_probe_exp2", "g1_probe_ldexp", "g1_probe_div"]},
        "seed": args.seed, "negative_control": args.neg,
        "geometry": {"gate_up": {"m": 1280, "k": 2560, "threads": 64, "h_cols": 640}, "down": {"m": 2560, "k": 640, "threads": 128, "h_cols": 2560}, "tile_slots": 16, "grid": "(M/64, tiles, 1)", "static_lds_bytes": 1536},
        "checks": {
            "acc_f32_exact": {"what": "every per-slot F32 accumulator == K-E0 twin (unpack_word_ref nibbles, MAGIC seed, fold128, epoch_sum), live/pad(+0)/sentinel-poison", "checked": tot.acc_checked, "live": tot.acc_live, "pad": tot.acc_pad, "sentinel_poison": tot.acc_poison, "mismatches": tot.acc_bad},
            "h_bf16_device_hook": {"what": "shipped BF16 == grouped_prefill_epilogue_with over twin sums with DEVICE v_exp_f32/v_ldexp_f32/IEEE-divide probes (not PortableEpilogue); down is round(sum)>>16", "checked": tot.h_checked, "live": tot.h_live, "pad": tot.h_pad, "sentinel_poison": tot.h_poison, "mismatches": tot.h_bad},
            "k_e0_unpack_swar_vs_ref": {"words": 65536, "mismatches": swar_bad},
            "divide_known_answer": div_kat,
            "divide_transcription_self_check": {"requests_checked": hk.div_checked.get(), "probe_sequence_vs_fdiv_rn_mismatches": div_rn_bad, "probe_sequence_vs_cpu_ieee_mismatches": div_ieee_bad, "mismatch_list": hk.div_list.borrow().clone()},
            "epilogue_dag_replay": {"passes": hk.dag_passes.get(), "final_pass_intrinsic_calls": hk.dag_calls.get()},
        },
        "scenario_names": scenario_names,
        "scenario_runs": rows.len(),
        "scenarios": rows,
        "mismatch_total": mismatch_total,
        "first_mismatch": first_mismatch,
    });
    if let Some(parent) = out_path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    std::fs::write(
        &out_path,
        serde_json::to_string_pretty(&receipt).map_err(err)?,
    )
    .map_err(|e| format!("{}: {e}", out_path.display()))?;
    println!(
        "e8p g1: {} scenario runs, acc {}/{} bad, h {}/{} bad, mismatch_total {mismatch_total}, pass={pass}, receipt {}",
        rows.len(), tot.acc_bad, tot.acc_checked, tot.h_bad, tot.h_checked, out_path.display()
    );
    Ok(rc)
}

fn main() {
    match run() {
        Ok(rc) => std::process::exit(rc),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
