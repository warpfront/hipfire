// SPDX-License-Identifier: Apache-2.0
//! Consumer24 K1 E8P: raw PM **gfx1100 / gfx1151** G1 exact oracle for the E8P-on-IU4 pointer-addressed MoE decode GEMV.
//! Never participates in runtime dispatch.
//!
//! Subject: the code object `e8p_i4_moe_gemv_pm_gfx1100` (`--expect-arch gfx1100`, the default) or `e8p_i4_moe_gemv_pm_gfx1151` (`--expect-arch
//! gfx1151`, Strix Halo) with the symbols `e8p_i4_moe_gemv_{r256_gate_up,r128_down}_pm_<arch>`, launched with the frozen 96-byte kernarg ABI (7 pointers
//! `table, ids, offs, slots, x, y0, y1`, then `m k u c xdiv xsrc xrs ys n_experts` as u32 and a zero u32; grid `(M, U, 1)`, block 32 = one
//! wave32). Every output word (all slots, all rows, padded strides, unwritten slots, trailing guard words) is
//! compared **bit for bit** (signed zeros and NaN payloads included, no tolerance, no NaN exception) with the full
//! original CPU reference `hipfire_runtime::e8p_i4_reference::decode_gemv_f32_lanepair` (R256 gate/up, R128 down).
//!
//! Weight layout per expert projection (identical to the gfx1201 K-E2 module): `M*K/8` little-endian `u16` wire
//! words (`idx | sign7 << 8 | shift << 15`, row-major) immediately followed by `M*K/128` FP16 scales (row-major).
//! Each projection of each expert is its own allocation with 256 bytes of `0xFF` before and after the tensor; after
//! the launches the whole allocation (guards + tensor) is re-read and must equal what was uploaded.
//!
//! Placement (the kernel only sees raw pointers in the per-expert table): `vram` (`hipMalloc` between random-size
//! fillers), `host_mapped` (**true** `hipHostMalloc(hipHostMallocMapped)` + `hipHostGetDevicePointer`; the pointer
//! attributes from `hipPointerGetAttributes` are recorded) and `split_vram_host` (alternating experts).
//!
//! Synthetic fixtures (scales are deliberately NOT passed through the writer guard; the CPU decode takes every
//! finite FP16): the six `coverage` experts together put every one of the 65536 wire words on every one of the 16
//! lane pairs of a K128 group in both projections (shift, all 128 sign7 values, all 256 abs rows) under plain,
//! non-zero scales; further experts use random / max-|v| / parity-balanced wire words, and scale classes FP16 max
//! 0x7bff, FP16 subnormals (0x0001, random, 0x03ff), smallest normal 0x0400, low normals 0x0401/3/5, 1.0, plain.
//! Cases cover C1/C2/C3/C5/C8, repeated experts across lanes, an expert with exactly 8 slots (`MAX_SLOTS`), a single
//! unique expert with unlisted descriptor slots, an expert with 9 slots (refused by the kernel: its slots keep the
//! sentinel), 80 unique experts, padded `x_row_stride` / `y_slot_stride` (pads keep their poison).
//!
//! Real samples: `--samples-dir DIR` (repeatable and/or comma separated) for BOTH pinned artifacts. Each dir is the
//! output of `tools/e8p_k1_extract_samples.py`: `sample.json` `{artifact_path, artifact_sha256, provisional: false,
//! samples: [{layer, expert, hot: false, tensors: [{kind: gate_up_proj|down_proj, qt: 135|136, shape, group: 128,
//! file, sha256, artifact_offset, bytes}]}]}` plus the raw packed files. Four cold experts per artifact (two at layer 0,
//! two at layer 47); every file's sha256/size/qt/group/shape is verified strictly. The ~100 GB artifacts are NOT needed
//! on the device host: the binding is the extractor manifest (`artifact_sha256`, `artifact_path`, per-tensor
//! `artifact_offset`, `extractor_source_sha256`, `rc: 0`, `device: "cpu"`, `argv`), whose extractor streamed the whole-artifact SHA
//! and re-read every sampled range and written file; all of it is preserved in the receipt. Only with the opt-in
//! `--verify-artifact-range` (artifact reachable, else error) is the range `[artifact_offset, +bytes)` re-hashed here. Real cases repeat expert ids
//! across slots (no 10-expert minimum), including counts of exactly 8 slots per expert.
//!
//! Negative controls (ALL required for both projections, any non-detection => rc 1):
//! * output / pad / tail corruption of the downloaded GPU buffers (exactly one word must be flagged);
//! * `wrong_lut_entry`, `wrong_sign_mask`, `swapped_lane_pair`: mutated CPU decode arithmetic (one LUT entry, one MASK
//!   entry, the two lanes of every pair keeping the other half). The un-mutated local transcription MUST equal the
//!   reference on every word first; the mutant MUST differ from the original reference (sensitivity count recorded)
//!   and from the GPU words;
//! * `wrong_scale_row`, `wrong_expert_pointer`: the INPUT is corrupted (scale rows rewritten in place in the placement
//!   memory / table entries of two experts swapped) and the kernels are relaunched: GPU MUST equal the full reference
//!   of the corrupted input, MUST differ from the original reference exactly in the corrupted experts' slots, and
//!   after restoring MUST equal the original reference again.
//!
//! Build:  cargo build -p hipfire-runtime --release --features lab --example e8p_i4_moe_gemv_gfx1100_g1
//! G1 (xtx; the parent grants the card):
//!   target/release/examples/e8p_i4_moe_gemv_gfx1100_g1 --codeobject /abs/e8p_i4_moe_gemv_pm_gfx1100.hxaco \
//!     --samples-dir /abs/c24-e8p-samples --samples-dir /abs/c24h6306-e8p-samples --out /abs/receipt.json \
//!     [--card xtx] [--device N] [--expect-uuid GPU-… --expect-bdf 0000:… --expect-arch gfx1100] \
//!     [--codeobject-sha256 HEX] [--fabric-class C --fabric-owner O --fabric-job J --fabric-card CARD --fabric-cmd STR] \
//!     [--only SUBSTR]… [--overwrite] [--verify-artifact-range] [--rehash-artifact] [--no-reclaim-arm]
//!   `--only` matches `<case>@<placement>`; it is a diagnostic filter: the receipt is marked `gate_eligible: false`.
//! One JSON receipt (stdout and `--out`); exit code 0 only if every comparison is exact, every control passed and
//! the coverage requirements hold.
//!
//! # gfx1151 (Strix Halo; `--expect-arch gfx1151`)
//! Same oracle, same arithmetic, same synthetic cases / real-sample cases / five negative controls / exit code. The gfx1151 module is the gfx1100 arm
//! with only arch-forced deltas, so its outputs must be bit-identical to the gfx1100 / gfx1201 kernels and therefore to the K-E0 twin
//! (`e8p_i4_reference::decode_gemv_f32_lanepair`); the harness gives no tolerance and no NaN exception there either. What the arch selects:
//! * module `e8p_i4_moe_gemv_pm_gfx1151`, symbols `e8p_i4_moe_gemv_{r256_gate_up,r128_down}_pm_gfx1151`, schema `e8p-i4-moe-gemv-gfx1151-g1-v1`,
//!   default card label `halo`. `--expect-arch gfx1100` (default) keeps every existing default (xtx uuid / bdf, card `xtx`, both `.hfq` pins).
//! * identity: the device arch must equal `--expect-arch`. gfx1151 has no ROCr UUID (the string `hipDeviceGetUuid` returns is recorded verbatim,
//!   `uuid_authoritative: false`), so `--expect-uuid` is OPTIONAL there (checked, exact match, only when given) and `--expect-bdf` is REQUIRED (the pin
//!   that names the card; `lspci -D -d 1002:` on hipx, the Halo is the only gfx1151 function).
//! * real samples: the Halo slab artifact is hipx `/home/kaden/pm-wave/consumer24/models/c24h6306-d2/qwen3.8-flash-next.mq2-xts` (47 038 936 600 B,
//!   sha256 d3d536c040f834699b5a261a5122b67d076461a9493eca16669803757928a949, 18 270 E8P cold experts), pin tag `c24h6306-d2-mq2-xts`. On gfx1151 ONLY
//!   that sample set is REQUIRED for a gate-eligible receipt (it supplies the real layers 0 and 47, four cold experts, both projections); the two
//!   `.hfq` sets `c24-e8p` / `c24h6306-e8p` are accepted as optional extra sets (at most one directory per pin). gfx1100 still accepts only those two
//!   and requires both.
//!
//! Halo, step 1 (CPU only, on hipx; `tools/e8p_k1_extract_samples.py`, whose sha256 54cdfed3e5314b6eb7685c2f35248e59224b3f332f04ea5d287fcb7a985cef0d
//! is pinned in this harness; the output directory must not exist):
//!   python3 tools/e8p_k1_extract_samples.py \
//!     --artifact /home/kaden/pm-wave/consumer24/models/c24h6306-d2/qwen3.8-flash-next.mq2-xts \
//!     --expect-sha256 d3d536c040f834699b5a261a5122b67d076461a9493eca16669803757928a949 --output-dir /abs/halo-e8p-k1-samples
//! Halo, step 2 (the full gate: synthetic + real, one receipt; `--card halo` is the fabric label, `HIPFIRE_DEVICES=gfx1151`):
//!   cargo build -p hipfire-runtime --release --example e8p_i4_moe_gemv_gfx1100_g1
//!   /home/kaden/qcal/locks/fabric/fabric-run.sh --class correct --card halo --mem-gb N --owner NAME -- \
//!     target/release/examples/e8p_i4_moe_gemv_gfx1100_g1 --codeobject /abs/e8p_i4_moe_gemv_pm_gfx1151.hxaco \
//!     --samples-dir /abs/halo-e8p-k1-samples --out /abs/r10.k1.g1.e8p-gfx1151.json \
//!     --expect-arch gfx1151 --expect-bdf 0000:XX:00.0 [--expect-uuid U] [--card halo] [--codeobject-sha256 HEX] \
//!     [--verify-artifact-range (the artifact is on hipx; re-hashes the four sampled ranges)] [--rehash-artifact (47 GB)] [--overwrite]
//!   The receipt carries `module`, `symbols_loaded` (both `_pm_gfx1151`), the artifact pin binding (`samples.bound`, `all_pinned_bound`) and
//!   `byte_exact`; the exit code is 0 only if `pass`. Host-mapped / split placements are exactness runs on the UMA APU (no latency/bandwidth claim).
use hip_bridge::{DeviceBuffer, Function, HipRuntime, MemoryType, HIP_HOST_MALLOC_MAPPED};
use hipfire_runtime::e8p_i4_reference as e8p;
use hipfire_runtime::mq_i4::MqI4Rotation;
use hipfire_runtime::mq_i4_reference::gemv_tree_f32_q;
use rayon::prelude::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::c_void;
use std::path::Path;
use std::rc::Rc;
use std::sync::LazyLock;
/// `extractor_source_sha256` both sample manifests carry (`tools/e8p_k1_extract_samples.py`).
const EXTRACTOR_SHA256: &str = "54cdfed3e5314b6eb7685c2f35248e59224b3f332f04ea5d287fcb7a985cef0d";

type Result<T, E = String> = std::result::Result<T, E>;
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

const SCHEMA: &str = "e8p-i4-moe-gemv-gfx1100-g1-v1";
const SCHEMA_GFX1151: &str = "e8p-i4-moe-gemv-gfx1151-g1-v1";
const MODULE: &str = "e8p_i4_moe_gemv_pm_gfx1100";
const MODULE_GFX1151: &str = "e8p_i4_moe_gemv_pm_gfx1151";
const BUILD_COMMAND: &str =
    "cargo build -p hipfire-runtime --release --features lab --example e8p_i4_moe_gemv_gfx1100_g1";
const HARNESS_SRC: &[u8] = include_bytes!("e8p_i4_moe_gemv_gfx1100_g1.rs");
const REFERENCE_SRC: &[u8] = include_bytes!("../src/e8p_i4_reference.rs");
const TREE_SRC: &[u8] = include_bytes!("../src/mq_i4_reference.rs");
const DEFAULT_UUID: &str = "GPU-43390a851e296ee5";
const DEFAULT_BDF: &str = "0000:67:00.0";
const DEFAULT_ARCH: &str = "gfx1100";
const DEFAULT_CARD: &str = "xtx";
const DEFAULT_CARD_GFX1151: &str = "halo";
/// The artifacts a G1 can bind (`sample.json: artifact_sha256`): the two XTX `.hfq` exports and the Halo slab `mq2-xts` (shipped d2 base).
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
const LINK_CAVEAT: &str = "The XTX endpoint is x16 but its upstream root port 0000:00:03.1 is PCIe 4.0 x4 \
(16 GT/s, ~7.9 GB/s theoretical): host-mapped runs are exactness runs only, no latency/bandwidth claim.";
const LINK_CAVEAT_GFX1151: &str = "gfx1151 is the Strix Halo APU: VRAM and host memory are one UMA pool, so host-mapped and split \
placements are exactness runs only, no latency/bandwidth claim.";

/// The gfx11 targets this oracle serves (`--expect-arch`). The gfx1151 module is the gfx1100 arm with only arch-forced deltas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Target {
    Gfx1100,
    Gfx1151,
}
impl Target {
    fn of_name(arch: &str) -> Option<Self> {
        match arch {
            "gfx1100" => Some(Self::Gfx1100),
            "gfx1151" => Some(Self::Gfx1151),
            _ => None,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Gfx1100 => "gfx1100",
            Self::Gfx1151 => "gfx1151",
        }
    }
    fn schema(self) -> &'static str {
        match self {
            Self::Gfx1100 => SCHEMA,
            Self::Gfx1151 => SCHEMA_GFX1151,
        }
    }
    fn module(self) -> &'static str {
        match self {
            Self::Gfx1100 => MODULE,
            Self::Gfx1151 => MODULE_GFX1151,
        }
    }
    fn default_card(self) -> &'static str {
        match self {
            Self::Gfx1100 => DEFAULT_CARD,
            Self::Gfx1151 => DEFAULT_CARD_GFX1151,
        }
    }
    fn link_caveat(self) -> &'static str {
        match self {
            Self::Gfx1100 => LINK_CAVEAT,
            Self::Gfx1151 => LINK_CAVEAT_GFX1151,
        }
    }
    /// Pins a sample set may bind on this target (at most one directory per pin).
    fn pins(self) -> &'static [(&'static str, &'static str)] {
        match self {
            Self::Gfx1100 => &ARTIFACT_PINS[..2],
            Self::Gfx1151 => &ARTIFACT_PINS[..],
        }
    }
    /// Pins a gate-eligible receipt must have bound.
    fn required_pins(self) -> &'static [(&'static str, &'static str)] {
        match self {
            Self::Gfx1100 => &ARTIFACT_PINS[..2],
            Self::Gfx1151 => &ARTIFACT_PINS[2..],
        }
    }
}
const KERNARG_BYTES: usize = 96;
const N_EXPERTS: usize = 512;
/// Expert ids the kernel must reject without a table load or a store: `-1`, the table extent, `INT_MAX`, `INT_MIN`.
const BAD_IDS: [u32; 4] = [u32::MAX, N_EXPERTS as u32, i32::MAX as u32, i32::MIN as u32];
const GATE_M: usize = 1280;
const GATE_K: usize = 2560;
const DOWN_M: usize = 2560;
const DOWN_K: usize = 640;
/// Most slots one expert workgroup serves; more => the workgroup returns without a store.
const MAX_SLOTS: usize = 8;
/// Guard bytes (`0xFF`) before and after every weight tensor.
const GUARD: usize = 256;
/// Trailing sentinel / NaN words after every Y / X buffer.
const TAIL_WORDS: usize = 256;
/// Unwritten output words: what `memset(0xa5)` leaves (a NaN, compared by bits).
const SENTINEL_BITS: u32 = 0xa5a5_a5a5;
const PAD_BITS: u32 = 0x7fc0_0000;
const REAL_LAYERS: [u64; 2] = [0, 47];
const NEGATIVES: [&str; 5] = [
    "wrong_lut_entry",
    "wrong_sign_mask",
    "wrong_scale_row",
    "wrong_expert_pointer",
    "swapped_lane_pair",
];

// ---------------------------------------------------------------------------------------
// Axes
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    GateUp,
    Down,
}
impl Kind {
    const ALL: [Kind; 2] = [Kind::GateUp, Kind::Down];
    fn suffix(self) -> &'static str {
        match self {
            Kind::GateUp => "r256_gate_up",
            Kind::Down => "r128_down",
        }
    }
    fn name(self) -> &'static str {
        match self {
            Kind::GateUp => "gate_up",
            Kind::Down => "down",
        }
    }
    fn idx(self) -> usize {
        self as usize
    }
    fn symbol(self, target: Target) -> String {
        format!("e8p_i4_moe_gemv_{}_pm_{}", self.suffix(), target.name())
    }
    fn m(self) -> usize {
        match self {
            Kind::GateUp => GATE_M,
            Kind::Down => DOWN_M,
        }
    }
    fn k(self) -> usize {
        match self {
            Kind::GateUp => GATE_K,
            Kind::Down => DOWN_K,
        }
    }
    /// Rows written to each of Y0/Y1 per slot (gate rows `>= M/2` go to Y1).
    fn out_rows(self) -> usize {
        match self {
            Kind::GateUp => GATE_M / 2,
            Kind::Down => DOWN_M,
        }
    }
    fn x_row_div(self) -> usize {
        match self {
            Kind::GateUp => 10,
            Kind::Down => 1,
        }
    }
    fn rotation(self) -> MqI4Rotation {
        match self {
            Kind::GateUp => MqI4Rotation::R256,
            Kind::Down => MqI4Rotation::R128,
        }
    }
    fn n_words(self) -> usize {
        self.m() * self.k() / 8
    }
    fn groups_per_row(self) -> usize {
        self.k() / 128
    }
    fn n_scales(self) -> usize {
        self.m() * self.k() / 128
    }
    fn blob_bytes(self) -> usize {
        self.n_words() * 2 + self.n_scales() * 2
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Placement {
    Vram,
    HostMapped,
    Split,
}
impl Placement {
    fn label(self) -> &'static str {
        match self {
            Placement::Vram => "vram",
            Placement::HostMapped => "host_mapped",
            Placement::Split => "split_vram_host",
        }
    }
    fn host_for(self, order_index: usize) -> bool {
        match self {
            Placement::Vram => false,
            Placement::HostMapped => true,
            Placement::Split => order_index % 2 == 0,
        }
    }
}

// ---------------------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------------------

struct Rng(u64);
fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(mix64(seed))
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix64(self.0)
    }
    fn u32(&mut self) -> u32 {
        (self.next() >> 32) as u32
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
fn name_seed(name: &str) -> u64 {
    let d = Sha256::digest(name.as_bytes());
    u64::from_le_bytes(d[..8].try_into().unwrap())
}
fn sha(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}
fn hex(b: &[u8]) -> String {
    b.iter().map(|v| format!("{v:02x}")).collect()
}
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u32_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u64_bytes(v: &[u64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn bits_hex(v: u32) -> String {
    format!("0x{v:08x}")
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
    Ok((hex(&h.finalize()), n))
}
fn range_sha256(path: &str, offset: u64, len: usize) -> Result<String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    f.seek(SeekFrom::Start(offset))
        .map_err(|e| format!("{path}: seek {offset}: {e}"))?;
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf)
        .map_err(|e| format!("{path}: read {len} bytes at {offset}: {e}"))?;
    Ok(hex(&sha(&buf)))
}
fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

// ---------------------------------------------------------------------------------------
// Tensors and experts
// ---------------------------------------------------------------------------------------

/// One projection of one expert: planes plus the exact wire image (`words` LE then `scales` LE).
struct Tensor {
    words: Vec<u16>,
    scales: Vec<u16>,
    bytes: Vec<u8>,
    sha: [u8; 32],
}
impl Tensor {
    fn from_planes(words: Vec<u16>, scales: Vec<u16>) -> Tensor {
        let mut bytes = u16_bytes(&words);
        bytes.extend_from_slice(&u16_bytes(&scales));
        let sha = sha(&bytes);
        Tensor {
            words,
            scales,
            bytes,
            sha,
        }
    }
    fn from_bytes(kind: Kind, bytes: Vec<u8>) -> Result<Tensor> {
        if bytes.len() != kind.blob_bytes() {
            return Err(format!(
                "{} tensor: {} bytes != {}",
                kind.name(),
                bytes.len(),
                kind.blob_bytes()
            ));
        }
        let split = kind.n_words() * 2;
        let le = |b: &[u8]| -> Vec<u16> {
            b.chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect()
        };
        let (words, scales) = (le(&bytes[..split]), le(&bytes[split..]));
        let sha = sha(&bytes);
        Ok(Tensor {
            words,
            scales,
            bytes,
            sha,
        })
    }
}

struct Expert {
    label: String,
    prov: Value,
    gate: Tensor,
    down: Tensor,
}
impl Expert {
    fn proj(&self, kind: Kind) -> &Tensor {
        match kind {
            Kind::GateUp => &self.gate,
            Kind::Down => &self.down,
        }
    }
}

struct RowTables {
    five: Vec<u8>,
    odd: Vec<u8>,
    even: Vec<u8>,
}
static ROW_TABLES: LazyLock<RowTables> = LazyLock::new(|| {
    let all = 0..=255u8;
    RowTables {
        five: all
            .clone()
            .filter(|&i| e8p::E8P_ABS2[i as usize].contains(&5))
            .collect(),
        odd: all
            .clone()
            .filter(|&i| e8p::abs_row_parity(i) == 1)
            .collect(),
        even: all.filter(|&i| e8p::abs_row_parity(i) == 0).collect(),
    }
});

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Wire {
    Random,
    /// Coverage expert `j` (see [`coverage_word`]).
    Coverage(usize),
    /// Max-|v| abs rows with all-negative / all-positive / mixed signs and both shifts.
    Extreme,
    /// Odd and even abs rows with `popcount(sign7)` of both parities.
    Parity,
    /// Row `r` takes the generator `r % 4` of Random / Coverage(7) / Extreme / Parity.
    Mixed,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScaleMode {
    /// Every group has a plain mid-range scale (2^-7 .. 2^5).
    Plain,
    /// Scale class by (row, group): ten classes including FP16 max and subnormals.
    Special,
    /// Eight classes rotating over every row and group.
    Edges,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Spec {
    wire: Wire,
    scale: ScaleMode,
}

/// Word of coverage-expert position `(p, u)`: for fixed lane-pair `p` every `u` in a run of 65536 consecutive values
/// yields every wire word exactly once (`u * odd + b` is a bijection of Z/65536).
fn coverage_word(p: usize, u: usize) -> u16 {
    let a = (mix64(0xc0de_0000 + p as u64) as u16) | 1;
    let b = (mix64(0xb0b0_0000 + p as u64) >> 16) as u16;
    (u as u16).wrapping_mul(a).wrapping_add(b)
}

fn gen_words(kind: Kind, wire: Wire, rng: &mut Rng) -> Vec<u16> {
    let n = kind.n_words();
    let t = &*ROW_TABLES;
    match wire {
        Wire::Random => (0..n).map(|_| rng.u32() as u16).collect(),
        Wire::Coverage(j) => {
            let per = n / 16;
            (0..n)
                .map(|i| coverage_word(i % 16, j * per + i / 16))
                .collect()
        }
        Wire::Extreme => {
            const SIGNS: [u8; 6] = [0x00, 0x7f, 0x2a, 0x55, 0x01, 0x40];
            (0..n)
                .map(|i| {
                    let idx = if i % 4 == 3 {
                        rng.below(256) as u8
                    } else {
                        t.five[rng.below(t.five.len())]
                    };
                    e8p::pack_word(idx, SIGNS[(i / 4) % 6], (i / 24) % 2 == 1)
                })
                .collect()
        }
        Wire::Parity => (0..n)
            .map(|i| {
                let rows = if i % 2 == 0 { &t.odd } else { &t.even };
                let idx = rows[rng.below(rows.len())];
                let mut s = (rng.u32() & 0x7f) as u8;
                if (s.count_ones() & 1) as usize != (i / 2) % 2 {
                    s ^= 1;
                }
                e8p::pack_word(idx, s, rng.below(2) == 1)
            })
            .collect(),
        Wire::Mixed => {
            let parts = [
                gen_words(kind, Wire::Random, rng),
                gen_words(kind, Wire::Coverage(7), rng),
                gen_words(kind, Wire::Extreme, rng),
                gen_words(kind, Wire::Parity, rng),
            ];
            let wpr = kind.k() / 8;
            let mut out = Vec::with_capacity(n);
            for r in 0..kind.m() {
                out.extend_from_slice(&parts[r % 4][r * wpr..(r + 1) * wpr]);
            }
            out
        }
    }
}

/// Scale bits of a class: 0 plain, 1 FP16 max, 2 random subnormal, 3 smallest normal, 4 smallest subnormal,
/// 5 one, 6 low normals (FP16 `d/4` inexact), 7 largest subnormal, anything else plain.
fn scale_class(class: usize, rng: &mut Rng) -> u16 {
    match class {
        1 => 0x7bff,
        2 => 1 + rng.below(0x3ff) as u16,
        3 => 0x0400,
        4 => 0x0001,
        5 => 0x3c00,
        6 => [0x0401u16, 0x0403, 0x0405][rng.below(3)],
        7 => 0x03ff,
        _ => (((8 + rng.below(13)) as u16) << 10) | (rng.u32() as u16 & 0x3ff),
    }
}

fn gen_scales(kind: Kind, mode: ScaleMode, salt: usize, rng: &mut Rng) -> Vec<u16> {
    let groups = kind.groups_per_row();
    let mut out = Vec::with_capacity(kind.n_scales());
    for r in 0..kind.m() {
        for g in 0..groups {
            let class = match mode {
                ScaleMode::Plain => 0,
                ScaleMode::Special => (r * 5 + g * 3 + salt) % 10,
                ScaleMode::Edges => (r + g + salt) % 8,
            };
            out.push(scale_class(class, rng));
        }
    }
    out
}

fn spec_code(s: Spec) -> u64 {
    let w = match s.wire {
        Wire::Random => 0,
        Wire::Coverage(j) => 16 + j as u64,
        Wire::Extreme => 1,
        Wire::Parity => 2,
        Wire::Mixed => 3,
    };
    w * 4 + s.scale as u64
}

fn gen_expert(id: u32, spec: Spec) -> Expert {
    let seed = mix64(((id as u64) << 24) ^ spec_code(spec));
    let mut tensors = Vec::new();
    for kind in Kind::ALL {
        let mut rng = Rng::new(seed.wrapping_add(1 + kind.idx() as u64));
        let words = gen_words(kind, spec.wire, &mut rng);
        let scales = gen_scales(kind, spec.scale, id as usize, &mut rng);
        tensors.push(Tensor::from_planes(words, scales));
    }
    let down = tensors.pop().unwrap();
    let gate = tensors.pop().unwrap();
    Expert {
        label: format!("synthetic:{:?}/{:?}", spec.wire, spec.scale),
        prov: json!({"wire": format!("{:?}", spec.wire), "scale": format!("{:?}", spec.scale), "id": id}),
        gate,
        down,
    }
}

fn palette(i: usize) -> Spec {
    let (wire, scale) = match i % 10 {
        0 => (Wire::Mixed, ScaleMode::Special),
        1 => (Wire::Random, ScaleMode::Plain),
        2 => (Wire::Extreme, ScaleMode::Special),
        3 => (Wire::Parity, ScaleMode::Plain),
        4 => (Wire::Random, ScaleMode::Special),
        5 => (Wire::Mixed, ScaleMode::Plain),
        6 => (Wire::Extreme, ScaleMode::Plain),
        7 => (Wire::Parity, ScaleMode::Special),
        8 => (Wire::Random, ScaleMode::Edges),
        _ => (Wire::Mixed, ScaleMode::Edges),
    };
    Spec { wire, scale }
}

fn gen_x_row(style: usize, n: usize, rng: &mut Rng) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let v = match style % 4 {
            0 => {
                let (s, e, m) = (
                    rng.u32() >> 31,
                    124 + rng.below(5) as u32,
                    rng.u32() & 0x7f_ffff,
                );
                f32::from_bits((s << 31) | (e << 23) | m)
            }
            1 => {
                let (s, e, m) = (
                    rng.u32() >> 31,
                    107 + rng.below(24) as u32,
                    rng.u32() & 0x7f_ffff,
                );
                f32::from_bits((s << 31) | (e << 23) | m)
            }
            2 => [16777216.0f32, 1.0, -16777216.0, 1.0][i % 4],
            _ => [0.0f32, -0.0, 1.0e-30, -1.0e-30, 1.0, -1.0, 0.5, -0.5][i % 8],
        };
        out.push(v);
    }
    out
}

// ---------------------------------------------------------------------------------------
// Cases and descriptors
// ---------------------------------------------------------------------------------------

const ALL_STYLES: &[usize] = &[0, 1, 2, 3];
/// Dense, never-zero activations (coverage observability).
const RANDOM_STYLES: &[usize] = &[0, 1];

#[derive(Clone)]
struct Case {
    name: String,
    origin: &'static str,
    c: usize,
    /// `route[lane*10 + rank]` = expert id.
    route: Vec<u32>,
    /// Slots present in the descriptor (unlisted slots are never written).
    listed: Vec<bool>,
    experts: BTreeMap<u32, Rc<Expert>>,
    x_gate: Vec<f32>,
    x_down: Vec<f32>,
    xrs_gate: usize,
    xrs_down: usize,
    ys_gate: usize,
    ys_down: usize,
}

impl Case {
    fn x(&self, kind: Kind) -> (&[f32], usize, usize) {
        match kind {
            Kind::GateUp => (&self.x_gate, self.xrs_gate, self.ys_gate),
            Kind::Down => (&self.x_down, self.xrs_down, self.ys_down),
        }
    }
    fn slots(&self) -> usize {
        10 * self.c
    }
    fn bucket(&self) -> Bucket {
        build_bucket(&self.route, &self.listed)
    }
    fn input_sha256(&self) -> String {
        let mut h = Sha256::new();
        h.update(self.name.as_bytes());
        h.update((self.c as u64).to_le_bytes());
        h.update(u32_bytes(&self.route));
        h.update(self.listed.iter().map(|&b| b as u8).collect::<Vec<u8>>());
        for (id, e) in &self.experts {
            h.update(id.to_le_bytes());
            h.update(e.label.as_bytes());
            h.update(e.gate.sha);
            h.update(e.down.sha);
        }
        for v in [self.xrs_gate, self.xrs_down, self.ys_gate, self.ys_down] {
            h.update((v as u64).to_le_bytes());
        }
        h.update(f32_bytes(&self.x_gate));
        h.update(f32_bytes(&self.x_down));
        hex(&h.finalize())
    }
}

/// Activations: per-lane (gate) / per-slot (down) style rotation; padded strides are poisoned with NaN so any read
/// outside a row's `K` words is visible.
#[allow(clippy::too_many_arguments)]
fn build_case(
    name: &str,
    origin: &'static str,
    c: usize,
    route: Vec<u32>,
    experts: BTreeMap<u32, Rc<Expert>>,
    listed: Option<Vec<bool>>,
    padded: bool,
    xshift: usize,
    styles: &[usize],
) -> Case {
    assert_eq!(route.len(), 10 * c, "{name}: route length");
    let listed = listed.unwrap_or_else(|| vec![true; route.len()]);
    assert_eq!(listed.len(), route.len());
    let (xrs_gate, ys_gate, xrs_down, ys_down) = if padded {
        (GATE_K + 64, GATE_M / 2 + 32, DOWN_K + 64, DOWN_M + 32)
    } else {
        (GATE_K, GATE_M / 2, DOWN_K, DOWN_M)
    };
    let mut rng = Rng::new(name_seed(name));
    let fill = |rows: usize, k: usize, xrs: usize, rng: &mut Rng, shift: usize| {
        let mut x = vec![f32::from_bits(PAD_BITS); rows * xrs];
        for r in 0..rows {
            let row = gen_x_row(styles[(r + shift) % styles.len()], k, rng);
            x[r * xrs..r * xrs + k].copy_from_slice(&row);
        }
        x
    };
    let x_gate = fill(c, GATE_K, xrs_gate, &mut rng, xshift);
    let x_down = fill(10 * c, DOWN_K, xrs_down, &mut rng, xshift + 1);
    Case {
        name: name.to_string(),
        origin,
        c,
        route,
        listed,
        experts,
        x_gate,
        x_down,
        xrs_gate,
        xrs_down,
        ys_gate,
        ys_down,
    }
}

#[derive(Clone, Default, Debug, PartialEq, Eq)]
struct Bucket {
    ids: Vec<u32>,
    offs: Vec<u32>,
    slots: Vec<u32>,
}

/// Ascending expert ids, slots in stable original order per expert, offsets spanning the listed slots.
fn build_bucket(route: &[u32], listed: &[bool]) -> Bucket {
    let ids: BTreeSet<u32> = route
        .iter()
        .zip(listed)
        .filter(|(_, &l)| l)
        .map(|(&e, _)| e)
        .collect();
    let mut b = Bucket::default();
    for id in ids {
        b.ids.push(id);
        b.offs.push(b.slots.len() as u32);
        for (s, (&e, &l)) in route.iter().zip(listed).enumerate() {
            if l && e == id {
                b.slots.push(s as u32);
            }
        }
    }
    b.offs.push(b.slots.len() as u32);
    b
}

fn pick_ids(rng: &mut Rng, n: usize, edges: bool) -> Vec<u32> {
    let mut set = BTreeSet::new();
    if edges {
        set.insert(0u32);
        set.insert(511u32);
    }
    while set.len() < n {
        set.insert(rng.below(N_EXPERTS) as u32);
    }
    set.into_iter().collect()
}

fn experts_for(ids: &[u32], spec_of: impl Fn(usize) -> Spec) -> BTreeMap<u32, Rc<Expert>> {
    ids.iter()
        .enumerate()
        .map(|(i, &id)| (id, Rc::new(gen_expert(id, spec_of(i)))))
        .collect()
}

/// Ten distinct experts per lane from `pool`.
fn lane_routes(rng: &mut Rng, pool: &[u32], c: usize) -> Vec<u32> {
    let mut route = Vec::with_capacity(10 * c);
    for _ in 0..c {
        let mut p = pool.to_vec();
        rng.shuffle(&mut p);
        route.extend_from_slice(&p[..10]);
    }
    route
}

const SYNTH_CASES: &[&str] = &[
    "c1_coverage_six",
    "c1_mixed_stress_x",
    "c1_scale_edges",
    "c1_n8_boundary_u3",
    "c1_u1_n8_unlisted_tail",
    "c1_n9_refused",
    "c2_shared_lanes",
    "c3_partial_lanes",
    "c5_union_repeated",
    "c8_union_repeated",
    "c8_unique_80",
    "c8_identical_lanes_n8",
    "c8_padded_strides",
];
/// Cases that additionally run from true host-mapped weights.
const HOST_MAPPED_CASES: &[&str] = &[
    "c1_coverage_six",
    "c1_scale_edges",
    "c1_n9_refused",
    "c3_partial_lanes",
    "c8_union_repeated",
    "c8_identical_lanes_n8",
    "c8_padded_strides",
];
/// Cases that additionally run with alternating VRAM / mapped-host experts inside one launch.
const SPLIT_CASES: &[&str] = &["c1_coverage_six", "c8_union_repeated"];
/// Cases carrying the output / pad / tail corruption controls (VRAM placement).
const CONTROL_CASES: &[&str] = &["c1_coverage_six", "c8_padded_strides"];
/// Cases carrying the mutated-CPU-decode controls (VRAM placement).
const MUTANT_CASES: &[&str] = &["c1_coverage_six"];
/// Cases carrying the corrupted-input relaunch controls (every placement of the case).
const TAMPER_CASES: &[&str] = &["c1_coverage_six"];

fn build_synthetic(name: &str) -> Case {
    let mut rng = Rng::new(name_seed(name) ^ 0x5eed);
    let origin = "synthetic";
    match name {
        "c1_coverage_six" => {
            let ids = pick_ids(&mut rng, 10, true);
            let experts = experts_for(&ids, |i| {
                if i < 6 {
                    Spec {
                        wire: Wire::Coverage(i),
                        scale: ScaleMode::Plain,
                    }
                } else {
                    palette([0, 2, 3, 4][i - 6])
                }
            });
            let mut route = ids.clone();
            rng.shuffle(&mut route);
            build_case(
                name,
                origin,
                1,
                route,
                experts,
                None,
                false,
                0,
                RANDOM_STYLES,
            )
        }
        "c1_mixed_stress_x" => {
            let ids = pick_ids(&mut rng, 10, true);
            let experts = experts_for(&ids, palette);
            let mut route = ids.clone();
            rng.shuffle(&mut route);
            build_case(name, origin, 1, route, experts, None, false, 1, ALL_STYLES)
        }
        "c1_scale_edges" => {
            let ids = pick_ids(&mut rng, 10, true);
            let experts = experts_for(&ids, |i| Spec {
                wire: [Wire::Random, Wire::Mixed, Wire::Extreme][i % 3],
                scale: ScaleMode::Edges,
            });
            let mut route = ids.clone();
            rng.shuffle(&mut route);
            build_case(name, origin, 1, route, experts, None, false, 2, ALL_STYLES)
        }
        "c1_n8_boundary_u3" => {
            let ids = pick_ids(&mut rng, 3, true);
            let experts = experts_for(&ids, palette);
            let mut route = vec![ids[0]; 8];
            route.push(ids[1]);
            route.push(ids[2]);
            rng.shuffle(&mut route);
            build_case(name, origin, 1, route, experts, None, false, 3, ALL_STYLES)
        }
        "c1_u1_n8_unlisted_tail" => {
            let ids = pick_ids(&mut rng, 3, false);
            let experts = experts_for(&ids[..1], |_| palette(0));
            let mut route = vec![ids[0]; 8];
            route.push(ids[1]);
            route.push(ids[2]);
            rng.shuffle(&mut route);
            let listed: Vec<bool> = route.iter().map(|&e| e == ids[0]).collect();
            build_case(
                name,
                origin,
                1,
                route,
                experts,
                Some(listed),
                false,
                0,
                ALL_STYLES,
            )
        }
        "c1_n9_refused" => {
            let ids = pick_ids(&mut rng, 2, true);
            let experts = experts_for(&ids, palette);
            let mut route = vec![ids[1]; 9];
            route.push(ids[0]);
            rng.shuffle(&mut route);
            build_case(name, origin, 1, route, experts, None, false, 1, ALL_STYLES)
        }
        "c2_shared_lanes" => {
            let ids = pick_ids(&mut rng, 10, true);
            let experts = experts_for(&ids, |i| palette(i + 3));
            let route = lane_routes(&mut rng, &ids, 2);
            build_case(name, origin, 2, route, experts, None, false, 2, ALL_STYLES)
        }
        "c3_partial_lanes" => {
            let ids = pick_ids(&mut rng, 12, false);
            let experts = experts_for(&ids, |i| palette(i + 1));
            let route = lane_routes(&mut rng, &ids, 3);
            build_case(name, origin, 3, route, experts, None, false, 0, ALL_STYLES)
        }
        "c5_union_repeated" => {
            let ids = pick_ids(&mut rng, 12, true);
            let experts = experts_for(&ids, |i| palette(i + 5));
            let route = lane_routes(&mut rng, &ids, 5);
            build_case(name, origin, 5, route, experts, None, false, 1, ALL_STYLES)
        }
        "c8_union_repeated" | "c8_padded_strides" => {
            let ids = pick_ids(&mut rng, 14, true);
            let experts = experts_for(&ids, |i| palette(i + 2));
            let route = lane_routes(&mut rng, &ids, 8);
            build_case(
                name,
                origin,
                8,
                route,
                experts,
                None,
                name == "c8_padded_strides",
                0,
                ALL_STYLES,
            )
        }
        "c8_unique_80" => {
            let ids = pick_ids(&mut rng, 80, true);
            let experts = experts_for(&ids, |i| palette(i + 7));
            let mut route = ids.clone();
            rng.shuffle(&mut route);
            build_case(name, origin, 8, route, experts, None, false, 3, ALL_STYLES)
        }
        "c8_identical_lanes_n8" => {
            let ids = pick_ids(&mut rng, 10, true);
            let experts = experts_for(&ids, |i| palette(i + 4));
            let mut lane = ids.clone();
            rng.shuffle(&mut lane);
            let route: Vec<u32> = (0..8).flat_map(|_| lane.clone()).collect();
            build_case(name, origin, 8, route, experts, None, false, 0, ALL_STYLES)
        }
        other => panic!("unknown synthetic case {other}"),
    }
}

// ---------------------------------------------------------------------------------------
// Real sample dirs (tools/e8p_k1_extract_samples.py output)
// ---------------------------------------------------------------------------------------

struct SampleSet {
    tag: &'static str,
    sha: &'static str,
    /// (global id = table index, expert).
    experts: Vec<(u32, Rc<Expert>)>,
    meta: Value,
    guard_violations: u64,
}

fn load_sample_set(
    dir: &str,
    first_id: u32,
    verify_artifact: bool,
    rehash: bool,
    pins: &'static [(&'static str, &'static str)],
) -> Result<SampleSet> {
    let manifest_path = Path::new(dir).join("sample.json");
    let bytes =
        std::fs::read(&manifest_path).map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let manifest_sha = hex(&sha(&bytes));
    let root: Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let need_str = |v: &Value, k: &str, who: &str| -> Result<String> {
        v.get(k)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| format!("{who}: missing string `{k}`"))
    };
    let art_sha = need_str(&root, "artifact_sha256", "sample.json")?.to_ascii_lowercase();
    let art_path = need_str(&root, "artifact_path", "sample.json")?;
    if root.get("provisional").and_then(Value::as_bool) != Some(false) {
        return Err(format!(
            "{dir}: sample.json must carry `provisional: false` (got {:?})",
            root.get("provisional")
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
    let extractor_sha =
        need_str(&root, "extractor_source_sha256", "sample.json")?.to_ascii_lowercase();
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
    let (tag, pin_sha) = pins
        .iter()
        .find(|(_, s)| *s == art_sha)
        .copied()
        .ok_or_else(|| {
            let allowed: Vec<String> = pins.iter().map(|(t, s)| format!("{t}={s}")).collect();
            format!("{dir}: artifact_sha256 {art_sha} is not one of the artifacts pinned for this arch ({})", allowed.join(", "))
        })?;
    // No silent fallback: the artifact is only touched when its verification was requested, and then it MUST be reachable.
    let art_size = if verify_artifact || rehash {
        Some(
            std::fs::metadata(&art_path)
                .map_err(|e| format!("{dir}: artifact {art_path}: {e}"))?
                .len(),
        )
    } else {
        None
    };
    let artifact_rehash = if rehash {
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
    let mut experts: Vec<(u32, Rc<Expert>)> = Vec::new();
    let mut recs = Vec::new();
    let mut lines = String::new();
    let mut guard_violations = 0u64;
    let mut seen: BTreeSet<(u64, u64)> = BTreeSet::new();
    for (i, s) in list.iter().enumerate() {
        let field_u64 = |k: &str| -> Result<u64> {
            s.get(k)
                .and_then(Value::as_u64)
                .ok_or_else(|| format!("samples[{i}]: missing integer `{k}`"))
        };
        let (layer, expert) = (field_u64("layer")?, field_u64("expert")?);
        if !REAL_LAYERS.contains(&layer)
            || expert >= N_EXPERTS as u64
            || !seen.insert((layer, expert))
        {
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
        let mut loaded: [Option<(Tensor, Value)>; 2] = [None, None];
        for t in tensors {
            let kind_name = t.get("kind").and_then(Value::as_str).unwrap_or("");
            let (kind, qt) = match kind_name {
                "gate_up_proj" => (Kind::GateUp, 135u64),
                "down_proj" => (Kind::Down, 136u64),
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
            if shape.as_deref() != Some(&[kind.m() as u64, kind.k() as u64][..]) {
                return Err(format!(
                    "{who}: shape {:?} != [{}, {}]",
                    t.get("shape"),
                    kind.m(),
                    kind.k()
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
            let raw =
                std::fs::read(&path).map_err(|e| format!("{who}: {}: {e}", path.display()))?;
            if raw.len() != kind.blob_bytes() {
                return Err(format!(
                    "{who}: {} bytes != {}",
                    raw.len(),
                    kind.blob_bytes()
                ));
            }
            if t.get("bytes").and_then(Value::as_u64) != Some(raw.len() as u64) {
                return Err(format!(
                    "{who}: manifest bytes {:?} != file {}",
                    t.get("bytes"),
                    raw.len()
                ));
            }
            let got = hex(&sha(&raw));
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
            let reread = if verify_artifact {
                if art_size.is_none_or(|n| offset + raw.len() as u64 > n) {
                    return Err(format!(
                        "{who}: artifact range [{offset}, +{}) is outside {art_path}",
                        raw.len()
                    ));
                }
                let h = range_sha256(&art_path, offset, raw.len())?;
                if h != got {
                    return Err(format!(
                        "{who}: artifact range sha256 {h} != tensor file sha256 {got}"
                    ));
                }
                true
            } else {
                false
            };
            let tensor = Tensor::from_bytes(kind, raw)?;
            guard_violations += tensor
                .scales
                .iter()
                .filter(|&&d| e8p::writer_guard(d).is_err())
                .count() as u64;
            let rec = json!({"kind": kind_name, "qt": qt, "group": 128, "shape": [kind.m(), kind.k()], "file": file,
                "sha256": got, "bytes": kind.blob_bytes(), "artifact_offset": offset,
                "artifact_range_reread_here": reread,
                "binding_verified_by": if reread { "extractor_manifest+local_artifact_range_reread" } else { "extractor_manifest" }});
            loaded[kind.idx()] = Some((tensor, rec));
        }
        let (Some((gate, grec)), Some((down, drec))) = (loaded[0].take(), loaded[1].take()) else {
            return Err(format!(
                "samples[{i}] L{layer}.E{expert}: needs both gate_up_proj and down_proj"
            ));
        };
        lines.push_str(&format!(
            "L{layer} E{expert} {} {}\n",
            hex(&gate.sha),
            hex(&down.sha)
        ));
        let id = first_id + experts.len() as u32;
        let prov = json!({"table_id": id, "layer": layer, "expert": expert, "artifact": tag,
            "artifact_sha256": pin_sha, "tensors": [grec, drec]});
        recs.push(prov.clone());
        experts.push((
            id,
            Rc::new(Expert {
                label: format!("{tag}:L{layer}.E{expert}"),
                prov,
                gate,
                down,
            }),
        ));
    }
    for layer in REAL_LAYERS {
        if seen.iter().filter(|(l, _)| *l == layer).count() != 2 {
            return Err(format!(
                "{dir}: layer {layer} needs exactly two cold experts"
            ));
        }
    }
    let meta = json!({
        "dir": dir, "artifact_tag": tag, "artifact_path": art_path, "artifact_sha256": art_sha,
        "artifact_bytes": art_size, "artifact_full_rehash": artifact_rehash,
        "binding": {"verified_by": "extractor manifest fields: the extractor streamed the whole-artifact sha256 and re-read each sampled range and written file",
            "artifact_sha256_pinned": pin_sha, "sample_files_sha256_strict_here": true,
            "artifact_present_here": art_size.is_some(), "artifact_range_reread_here": verify_artifact},
        "provisional": false,
        "sample_manifest_sha256": manifest_sha,
        "extractor_source_path": root.get("extractor_source_path"),
        "extractor_source_sha256": extractor_sha,
        "extractor_rc": root.get("rc"),
        "extractor_device": root.get("device"),
        "extractor_argv": root.get("argv"),
        "n_samples": list.len(),
        "writer_guard_violations": guard_violations,
        "selected_input_sha256": hex(&sha(lines.as_bytes())),
        "experts": recs,
    });
    Ok(SampleSet {
        tag,
        sha: pin_sha,
        experts,
        meta,
        guard_violations,
    })
}

fn route_from_counts(ids: &[u32], counts: &[usize], rng: &mut Rng) -> Vec<u32> {
    assert_eq!(ids.len(), counts.len());
    let mut route: Vec<u32> = ids
        .iter()
        .zip(counts)
        .flat_map(|(&id, &n)| vec![id; n])
        .collect();
    rng.shuffle(&mut route);
    route
}

/// Real-weight cases with repeated expert ids across slots; counts never exceed `MAX_SLOTS`.
fn sample_cases(sets: &[SampleSet]) -> Vec<Case> {
    let mut rng = Rng::new(0xe8b1_1100);
    let mut cases = Vec::new();
    let map_of = |sel: &[&SampleSet]| -> BTreeMap<u32, Rc<Expert>> {
        sel.iter()
            .flat_map(|s| s.experts.iter().map(|(id, e)| (*id, e.clone())))
            .collect()
    };
    for s in sets {
        let ids: Vec<u32> = s.experts.iter().map(|(id, _)| *id).collect();
        let r1 = route_from_counts(&ids, &[3, 3, 2, 2], &mut rng);
        cases.push(build_case(
            &format!("real_{}_c1_repeat", s.tag),
            "real_samples",
            1,
            r1,
            map_of(&[s]),
            None,
            false,
            0,
            ALL_STYLES,
        ));
        let r3 = route_from_counts(&ids, &[8, 8, 7, 7], &mut rng);
        cases.push(build_case(
            &format!("real_{}_c3_n8_boundary", s.tag),
            "real_samples",
            3,
            r3,
            map_of(&[s]),
            None,
            false,
            1,
            ALL_STYLES,
        ));
    }
    if sets.len() >= 2 {
        let refs: Vec<&SampleSet> = sets.iter().collect();
        let ids: Vec<u32> = sets
            .iter()
            .flat_map(|s| s.experts.iter().map(|(id, _)| *id))
            .collect();
        let counts: Vec<usize> = (0..ids.len()).map(|i| if i < 4 { 8 } else { 7 }).collect();
        let total: usize = counts.iter().sum();
        assert_eq!(total, 60, "union real case: 60 slots = C6");
        let route = route_from_counts(&ids, &counts, &mut rng);
        cases.push(build_case(
            "real_union_c6_repeat_padded",
            "real_samples",
            6,
            route,
            map_of(&refs),
            None,
            true,
            2,
            ALL_STYLES,
        ));
    }
    cases
}

// ---------------------------------------------------------------------------------------
// Device plumbing
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Buf {
    idx: usize,
    ptr: u64,
}
struct Dev<'a> {
    hip: &'a HipRuntime,
    bufs: Vec<DeviceBuffer>,
    hosts: Vec<*mut c_void>,
}
impl<'a> Dev<'a> {
    fn new(hip: &'a HipRuntime) -> Self {
        Dev {
            hip,
            bufs: Vec::new(),
            hosts: Vec::new(),
        }
    }
    fn alloc(&mut self, n: usize) -> Result<Buf> {
        let b = self.hip.malloc(n.max(4)).map_err(err)?;
        let ptr = b.as_ptr() as u64;
        self.bufs.push(b);
        Ok(Buf {
            idx: self.bufs.len() - 1,
            ptr,
        })
    }
    fn upload(&mut self, bytes: &[u8]) -> Result<Buf> {
        let b = self.alloc(bytes.len())?;
        self.hip
            .memcpy_htod(&self.bufs[b.idx], bytes)
            .map_err(err)?;
        Ok(b)
    }
    fn download(&self, b: Buf, n: usize) -> Result<Vec<u8>> {
        let mut out = vec![0u8; n];
        self.hip
            .memcpy_dtoh(&mut out, &self.bufs[b.idx])
            .map_err(err)?;
        Ok(out)
    }
    /// True `hipHostMalloc(hipHostMallocMapped)` + `hipHostGetDevicePointer`. Returns
    /// `(host pointer, device-visible pointer, pointer-attribute evidence)`.
    fn alloc_host(&mut self, n: usize) -> Result<(*mut u8, u64, Value)> {
        let host = self
            .hip
            .host_malloc(n.max(4096), HIP_HOST_MALLOC_MAPPED)
            .map_err(err)?;
        self.hosts.push(host);
        let dev = self.hip.host_get_device_pointer(host, 0).map_err(err)?;
        // SAFETY: non-owning view used only for hipPointerGetAttributes, then forgotten (never freed).
        let view = unsafe { DeviceBuffer::from_raw(dev, n) };
        let attr = self.hip.pointer_get_attributes(&view);
        std::mem::forget(view);
        let ev = match attr {
            Ok(a) => {
                let mt = MemoryType::from_raw(a.mem_type);
                let ok = matches!(
                    mt,
                    Some(MemoryType::Host) | Some(MemoryType::Managed) | Some(MemoryType::Unified)
                );
                json!({"host_ptr": format!("{:p}", host), "device_ptr": format!("{:p}", dev),
                    "alias_identical": host == dev, "mem_type": a.mem_type, "mem_type_name": format!("{mt:?}"),
                    "allocation_flags": a.allocation_flags, "is_managed": a.is_managed,
                    "flags_requested": HIP_HOST_MALLOC_MAPPED, "is_host_memory": ok, "ok": ok})
            }
            Err(e) => json!({"host_ptr": format!("{:p}", host), "device_ptr": format!("{:p}", dev),
                "ok": false, "error": e.to_string()}),
        };
        Ok((host as *mut u8, dev as u64, ev))
    }
    fn release(self) {
        let hip = self.hip;
        for b in self.bufs {
            let _ = hip.free(b);
        }
        for h in self.hosts {
            let _ = hip.host_free(h);
        }
    }
}

enum Loc {
    Vram(Buf),
    Host(*mut u8),
}
/// One tensor allocation: `GUARD` bytes of 0xFF, the tensor, `GUARD` bytes of 0xFF.
struct Blob {
    /// Device pointer of the first tensor byte (what the expert table holds).
    ptr: u64,
    len: usize,
    loc: Loc,
}
impl Blob {
    fn image_len(&self) -> usize {
        self.len + 2 * GUARD
    }
}
struct ExpertSlab {
    gate: Blob,
    down: Blob,
    host: bool,
}
impl ExpertSlab {
    fn blob(&self, kind: Kind) -> &Blob {
        match kind {
            Kind::GateUp => &self.gate,
            Kind::Down => &self.down,
        }
    }
}
struct Slabs {
    by_id: BTreeMap<u32, ExpertSlab>,
    host_slabs: usize,
    vram_slabs: usize,
    host_attr_failures: usize,
    evidence: Vec<Value>,
}

fn image_of(tensor_bytes: &[u8]) -> Vec<u8> {
    let mut v = vec![0xffu8; GUARD];
    v.extend_from_slice(tensor_bytes);
    v.extend(std::iter::repeat_n(0xffu8, GUARD));
    v
}

/// One allocation per (expert, projection), allocated in shuffled order; VRAM allocations sit between random-size
/// fillers so pointers are neither monotone nor uniformly strided.
fn upload_experts(dev: &mut Dev, case: &Case, placement: Placement) -> Result<Slabs> {
    let mut rng = Rng::new(name_seed(&case.name) ^ 0xa110c);
    let mut order: Vec<u32> = case.experts.keys().copied().collect();
    rng.shuffle(&mut order);
    let mut slabs = Slabs {
        by_id: BTreeMap::new(),
        host_slabs: 0,
        vram_slabs: 0,
        host_attr_failures: 0,
        evidence: Vec::new(),
    };
    for (i, id) in order.into_iter().enumerate() {
        let e = &case.experts[&id];
        let host = placement.host_for(i);
        let mut blobs = Vec::new();
        for kind in Kind::ALL {
            let t = e.proj(kind);
            let image = image_of(&t.bytes);
            if host {
                let (hp, dp, mut ev) = dev.alloc_host(image.len())?;
                // SAFETY: `hp` is a live mapped host allocation of at least `image.len()` bytes.
                unsafe { std::ptr::copy_nonoverlapping(image.as_ptr(), hp, image.len()) };
                if ev["ok"] != json!(true) {
                    slabs.host_attr_failures += 1;
                }
                if slabs.evidence.len() < 3 {
                    ev["expert"] = json!(id);
                    ev["projection"] = json!(kind.name());
                    slabs.evidence.push(ev);
                }
                blobs.push(Blob {
                    ptr: dp + GUARD as u64,
                    len: t.bytes.len(),
                    loc: Loc::Host(hp),
                });
            } else {
                if rng.below(3) != 0 {
                    dev.alloc(256 * (1 + rng.below(300)))?;
                }
                let b = dev.upload(&image)?;
                blobs.push(Blob {
                    ptr: b.ptr + GUARD as u64,
                    len: t.bytes.len(),
                    loc: Loc::Vram(b),
                });
            }
        }
        if host {
            slabs.host_slabs += 1;
        } else {
            slabs.vram_slabs += 1;
        }
        let down = blobs.pop().unwrap();
        let gate = blobs.pop().unwrap();
        slabs.by_id.insert(id, ExpertSlab { gate, down, host });
    }
    Ok(slabs)
}

/// Rewrite one tensor in place in its placement memory (VRAM copy of the whole image or direct host store).
fn write_blob(dev: &Dev, b: &Blob, tensor_bytes: &[u8]) -> Result<()> {
    if tensor_bytes.len() != b.len {
        return Err(format!(
            "blob length changed ({} != {})",
            tensor_bytes.len(),
            b.len
        ));
    }
    match b.loc {
        Loc::Vram(buf) => dev
            .hip
            .memcpy_htod(&dev.bufs[buf.idx], &image_of(tensor_bytes))
            .map_err(err),
        Loc::Host(p) => {
            // SAFETY: `p` is the live mapped host allocation of `image_len` bytes; no kernel is in flight
            // (every launch is followed by device_synchronize).
            unsafe {
                std::ptr::copy_nonoverlapping(
                    tensor_bytes.as_ptr(),
                    p.add(GUARD),
                    tensor_bytes.len(),
                )
            };
            Ok(())
        }
    }
}

fn read_image(dev: &Dev, b: &Blob) -> Result<Vec<u8>> {
    match b.loc {
        Loc::Vram(buf) => dev.download(buf, b.image_len()),
        // SAFETY: live mapped host allocation of `image_len` bytes, no kernel in flight.
        Loc::Host(p) => Ok(unsafe { std::slice::from_raw_parts(p, b.image_len()) }.to_vec()),
    }
}

#[derive(Default)]
struct GuardCheck {
    tensors: u64,
    guard_bytes: u64,
    guard_mismatch_bytes: u64,
    tensor_mismatch_bytes: u64,
}
/// Every tensor allocation must still be exactly guards + the original tensor.
fn verify_blobs(dev: &Dev, slabs: &Slabs, case: &Case) -> Result<GuardCheck> {
    let mut g = GuardCheck::default();
    for (id, s) in &slabs.by_id {
        for kind in Kind::ALL {
            let b = s.blob(kind);
            let want = image_of(&case.experts[id].proj(kind).bytes);
            let got = read_image(dev, b)?;
            g.tensors += 1;
            g.guard_bytes += 2 * GUARD as u64;
            for (i, (a, w)) in got.iter().zip(&want).enumerate() {
                if a != w {
                    if i < GUARD || i >= GUARD + b.len {
                        g.guard_mismatch_bytes += 1;
                    } else {
                        g.tensor_mismatch_bytes += 1;
                    }
                }
            }
        }
    }
    Ok(g)
}

#[allow(clippy::too_many_arguments)]
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
) -> [u8; KERNARG_BYTES] {
    let mut a = [0u8; KERNARG_BYTES];
    for (i, p) in ptrs.iter().enumerate() {
        a[i * 8..i * 8 + 8].copy_from_slice(&p.to_le_bytes());
    }
    for (off, v) in [
        (56, m),
        (60, k),
        (64, u),
        (68, c),
        (72, xdiv),
        (76, xsrc),
        (80, xrs),
        (84, ys),
        (88, N_EXPERTS),
    ] {
        a[off..off + 4].copy_from_slice(&(v as u32).to_le_bytes());
    }
    a
}

struct Ctx {
    hip: HipRuntime,
    funcs: Vec<(Kind, Function)>,
    target: Target,
}
impl Ctx {
    fn func(&self, kind: Kind) -> &Function {
        &self
            .funcs
            .iter()
            .find(|(k, _)| *k == kind)
            .expect("both symbols loaded")
            .1
    }
}

// ---------------------------------------------------------------------------------------
// CPU expected buffers (the full original reference)
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Lay {
    slots: usize,
    ys: usize,
    out: usize,
}
impl Lay {
    fn body(&self) -> usize {
        self.slots * self.ys
    }
}

#[derive(Clone)]
struct Expected {
    y0: Vec<f32>,
    y1: Option<Vec<f32>>,
    /// Slots whose rows the kernel writes.
    computed: Vec<bool>,
    /// Experts whose workgroups return without a store (`n == 0`, `n > MAX_SLOTS` or an id `>= n_experts`).
    refused: Vec<u32>,
}

type Eval<'a> = &'a (dyn Fn(&Tensor, Kind, &[f32]) -> Vec<f32> + Sync);

/// The ORIGINAL reference: the lane-pair GEMV twin of `e8p_i4_reference` on the whole tensor.
fn eval_reference(t: &Tensor, kind: Kind, xs: &[f32]) -> Vec<f32> {
    e8p::decode_gemv_f32_lanepair(kind.rotation(), kind.m(), kind.k(), &t.words, &t.scales, xs)
}

fn cpu_expected(case: &Case, kind: Kind, b: &Bucket, eval: Eval) -> Expected {
    let (x, xrs, ys) = case.x(kind);
    let slots = case.slots();
    let (m, k, out) = (kind.m(), kind.k(), kind.out_rows());
    let total = slots * ys + TAIL_WORDS;
    let sentinel = f32::from_bits(SENTINEL_BITS);
    let mut y0 = vec![sentinel; total];
    let mut y1 = if kind == Kind::GateUp {
        Some(vec![sentinel; total])
    } else {
        None
    };
    let mut computed = vec![false; slots];
    let mut refused = Vec::new();
    let mut work: Vec<(usize, &Tensor, &[f32])> = Vec::new();
    for (u, &id) in b.ids.iter().enumerate() {
        let sl = &b.slots[b.offs[u] as usize..b.offs[u + 1] as usize];
        // An id at or past `n_experts` (unsigned, so a negative id too) exits before the table is indexed.
        if sl.is_empty() || sl.len() > MAX_SLOTS || id >= N_EXPERTS as u32 {
            refused.push(id);
            continue;
        }
        let t = case.experts[&id].proj(kind);
        for &s in sl {
            let s = s as usize;
            computed[s] = true;
            work.push((s, t, &x[(s / kind.x_row_div()) * xrs..][..k]));
        }
    }
    let rows: Vec<(usize, Vec<f32>)> = work
        .par_iter()
        .map(|&(s, t, xs)| (s, eval(t, kind, xs)))
        .collect();
    for (s, vals) in rows {
        assert_eq!(vals.len(), m);
        let base = s * ys;
        for (r, v) in vals.into_iter().enumerate() {
            if r < out {
                y0[base + r] = v;
            } else {
                y1.as_mut().expect("gate has Y1")[base + r - out] = v;
            }
        }
    }
    Expected {
        y0,
        y1,
        computed,
        refused,
    }
}

// ---------------------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------------------

#[derive(Default, Clone)]
struct Cat {
    compared: u64,
    mismatches: u64,
}
#[derive(Default)]
struct Cmp {
    /// Words the kernel must write.
    value: Cat,
    /// In-row words of unwritten slots (must keep the 0xa5 sentinel).
    unwritten: Cat,
    /// Row padding words inside a slot (must keep the sentinel).
    pad: Cat,
    /// Trailing guard words after the last slot.
    tail: Cat,
    first: Vec<Value>,
    /// Mismatching body words per slot.
    slot_counts: Vec<u64>,
}
impl Cmp {
    fn total(&self) -> u64 {
        self.value.mismatches
            + self.unwritten.mismatches
            + self.pad.mismatches
            + self.tail.mismatches
    }
    fn compared(&self) -> u64 {
        self.value.compared + self.unwritten.compared + self.pad.compared + self.tail.compared
    }
    fn json(&self) -> Value {
        let c = |c: &Cat| json!({"compared": c.compared, "mismatches": c.mismatches});
        json!({"words_compared": self.compared(), "mismatch_count": self.total(), "computed_words": c(&self.value),
            "unwritten_slot_words": c(&self.unwritten), "pad_words": c(&self.pad), "tail_guard_words": c(&self.tail),
            "first_mismatches": self.first})
    }
}

fn compare_buf(part: &str, gpu: &[u8], exp: &[f32], lay: Lay, computed: &[bool], cmp: &mut Cmp) {
    assert_eq!(gpu.len(), exp.len() * 4, "{part}: GPU buffer length");
    let body = lay.body();
    for (i, (b, e)) in gpu.chunks_exact(4).zip(exp).enumerate() {
        let g = u32::from_le_bytes(b.try_into().unwrap());
        let (cat, slot) = if i >= body {
            (&mut cmp.tail, None)
        } else {
            let (slot, row) = (i / lay.ys, i % lay.ys);
            if row >= lay.out {
                (&mut cmp.pad, Some(slot))
            } else if computed[slot] {
                (&mut cmp.value, Some(slot))
            } else {
                (&mut cmp.unwritten, Some(slot))
            }
        };
        cat.compared += 1;
        if g == e.to_bits() {
            continue;
        }
        cat.mismatches += 1;
        if let Some(s) = slot {
            cmp.slot_counts[s] += 1;
        }
        if cmp.first.len() < 8 {
            cmp.first.push(
                json!({"part": part, "word_index": i, "slot": slot, "row": slot.map(|_| i % lay.ys),
                "gpu_bits": bits_hex(g), "expected_bits": bits_hex(e.to_bits()),
                "expected_unwritten": e.to_bits() == SENTINEL_BITS}),
            );
        }
    }
}

fn compare_all(lay: Lay, exp: &Expected, g0: &[u8], g1: Option<&[u8]>) -> Cmp {
    let mut cmp = Cmp {
        slot_counts: vec![0; lay.slots],
        ..Default::default()
    };
    compare_buf("y0", g0, &exp.y0, lay, &exp.computed, &mut cmp);
    if let (Some(g1), Some(e1)) = (g1, exp.y1.as_ref()) {
        compare_buf("y1", g1, e1, lay, &exp.computed, &mut cmp);
    }
    cmp
}

/// Words that differ between two expected buffers (same layout), with the number of slots touched.
fn expected_diff(a: &Expected, b: &Expected, lay: Lay) -> (u64, usize) {
    let mut words = 0u64;
    let mut slots = BTreeSet::new();
    let mut scan = |x: &[f32], y: &[f32]| {
        for (i, (p, q)) in x.iter().zip(y).enumerate() {
            if p.to_bits() != q.to_bits() {
                words += 1;
                if i < lay.body() {
                    slots.insert(i / lay.ys);
                }
            }
        }
    };
    scan(&a.y0, &b.y0);
    if let (Some(x), Some(y)) = (&a.y1, &b.y1) {
        scan(x, y);
    }
    (words, slots.len())
}

// ---------------------------------------------------------------------------------------
// One (case, kind) GPU run: prepare / launch / read, so controls can relaunch
// ---------------------------------------------------------------------------------------

struct Prep {
    kind: Kind,
    lay: Lay,
    total_words: usize,
    bucket: Bucket,
    y0: Buf,
    y1: Option<Buf>,
    args: [u8; KERNARG_BYTES],
    grid: [u32; 3],
}

fn table_for(slabs: &Slabs, kind: Kind) -> Vec<u64> {
    let mut table = vec![0u64; N_EXPERTS];
    for (&id, s) in &slabs.by_id {
        table[id as usize] = s.blob(kind).ptr;
    }
    table
}

fn prepare(dev: &mut Dev, case: &Case, kind: Kind, slabs: &Slabs) -> Result<Prep> {
    let slots = case.slots();
    let (m, k, out) = (kind.m(), kind.k(), kind.out_rows());
    let (x, xrs, ys) = case.x(kind);
    let x_src_rows = x.len() / xrs;
    let bucket = case.bucket();
    let mut xg = x.to_vec();
    xg.extend(std::iter::repeat_n(f32::from_bits(PAD_BITS), TAIL_WORDS));
    let xb = dev.upload(&f32_bytes(&xg))?;
    let tb = dev.upload(&u64_bytes(&table_for(slabs, kind)))?;
    let total_words = slots * ys + TAIL_WORDS;
    let y0 = dev.alloc(total_words * 4)?;
    let y1 = if kind == Kind::GateUp {
        Some(dev.alloc(total_words * 4)?)
    } else {
        None
    };
    let ib = dev.upload(&u32_bytes(&bucket.ids))?;
    let ob = dev.upload(&u32_bytes(&bucket.offs))?;
    let sb = dev.upload(&u32_bytes(&bucket.slots))?;
    let args = kernarg(
        [
            tb.ptr,
            ib.ptr,
            ob.ptr,
            sb.ptr,
            xb.ptr,
            y0.ptr,
            y1.map_or(0, |b| b.ptr),
        ],
        m,
        k,
        bucket.ids.len(),
        case.c,
        kind.x_row_div(),
        x_src_rows,
        xrs,
        ys,
    );
    let grid = [m as u32, bucket.ids.len() as u32, 1];
    Ok(Prep {
        kind,
        lay: Lay { slots, ys, out },
        total_words,
        bucket,
        y0,
        y1,
        args,
        grid,
    })
}

/// Poison the outputs with 0xa5, launch, synchronize.
fn launch(ctx: &Ctx, dev: &Dev, p: &Prep, args: &[u8; KERNARG_BYTES]) -> Result<()> {
    dev.hip
        .memset(&dev.bufs[p.y0.idx], 0xa5, p.total_words * 4)
        .map_err(err)?;
    if let Some(y1) = p.y1 {
        dev.hip
            .memset(&dev.bufs[y1.idx], 0xa5, p.total_words * 4)
            .map_err(err)?;
    }
    let mut a = *args;
    // SAFETY: exact 96-byte ABI; every pointer is a live allocation of `dev` until release.
    unsafe {
        ctx.hip
            .launch_kernel_blob(ctx.func(p.kind), p.grid, [32, 1, 1], 0, None, &mut a)
    }
    .map_err(err)?;
    ctx.hip.device_synchronize().map_err(err)
}

fn read_out(dev: &Dev, p: &Prep) -> Result<(Vec<u8>, Option<Vec<u8>>)> {
    let g0 = dev.download(p.y0, p.total_words * 4)?;
    let g1 = match p.y1 {
        Some(b) => Some(dev.download(b, p.total_words * 4)?),
        None => None,
    };
    Ok((g0, g1))
}

/// New table buffer + kernarg with pointer 0 replaced.
fn retable(dev: &mut Dev, p: &Prep, table: &[u64]) -> Result<[u8; KERNARG_BYTES]> {
    let tb = dev.upload(&u64_bytes(table))?;
    let mut a = p.args;
    a[0..8].copy_from_slice(&tb.ptr.to_le_bytes());
    Ok(a)
}

struct KindRun {
    prep: Prep,
    exp: Expected,
    g0: Vec<u8>,
    g1: Option<Vec<u8>>,
}

// ---------------------------------------------------------------------------------------
// Coverage accounting (what the compared experts actually contained)
// ---------------------------------------------------------------------------------------

struct Coverage {
    words_all: [Vec<u64>; 2],
    /// Words in groups with a non-zero scale (the weights influence the output).
    words_obs: [Vec<u64>; 2],
    pairs_obs: [Vec<u64>; 2],
    scale_groups: [BTreeMap<&'static str, u64>; 2],
}
fn scale_label(s: u16) -> &'static str {
    match s {
        0x7bff => "f16_max_0x7bff",
        0x0001 => "smallest_subnormal_0x0001",
        0x03ff => "largest_subnormal_0x03ff",
        0x0002..=0x03fe => "subnormal_other",
        0x0400 => "min_normal_0x0400",
        0x0401 | 0x0403 | 0x0405 => "low_normal_0x0401_03_05",
        0x3c00 => "one_0x3c00",
        _ => "plain_or_other",
    }
}
impl Coverage {
    fn new() -> Self {
        let w = || vec![0u64; 1024];
        let p = || vec![0u64; 16 * 1024];
        Coverage {
            words_all: [w(), w()],
            words_obs: [w(), w()],
            pairs_obs: [p(), p()],
            scale_groups: [BTreeMap::new(), BTreeMap::new()],
        }
    }
    fn set(v: &mut [u64], bit: usize) {
        v[bit / 64] |= 1u64 << (bit % 64);
    }
    fn update(&mut self, kind: Kind, t: &Tensor) {
        let ki = kind.idx();
        for (i, &w) in t.words.iter().enumerate() {
            let d = t.scales[i / 16];
            let p = i % 16;
            Self::set(&mut self.words_all[ki], w as usize);
            if d & 0x7fff != 0 {
                Self::set(&mut self.words_obs[ki], w as usize);
                Self::set(&mut self.pairs_obs[ki], p * 65536 + w as usize);
            }
        }
        for &s in &t.scales {
            *self.scale_groups[ki].entry(scale_label(s)).or_insert(0) += 1;
        }
    }
    fn count(v: &[u64]) -> u64 {
        v.iter().map(|w| w.count_ones() as u64).sum()
    }
    fn complete(&self) -> bool {
        Kind::ALL.iter().all(|k| {
            let i = k.idx();
            Self::count(&self.words_obs[i]) == 65536
                && Self::count(&self.pairs_obs[i]) == 16 * 65536
                && self.scale_groups[i].contains_key("f16_max_0x7bff")
                && (self.scale_groups[i].contains_key("smallest_subnormal_0x0001")
                    || self.scale_groups[i].contains_key("subnormal_other"))
        })
    }
    fn json(&self) -> Value {
        let mut m = serde_json::Map::new();
        for k in Kind::ALL {
            let i = k.idx();
            m.insert(
                k.name().to_string(),
                json!({"distinct_wire_words": Self::count(&self.words_all[i]), "required_distinct_wire_words": 65536,
                    "distinct_wire_words_under_nonzero_scale": Self::count(&self.words_obs[i]),
                    "distinct_lane_pair_word_under_nonzero_scale": Self::count(&self.pairs_obs[i]),
                    "required_lane_pair_word": 16 * 65536,
                    "scale_groups_by_class": self.scale_groups[i]}),
            );
        }
        Value::Object(m)
    }
}

// ---------------------------------------------------------------------------------------
// Negative controls
// ---------------------------------------------------------------------------------------

fn word_at(buf: &[u8], idx: usize) -> u32 {
    u32::from_le_bytes(buf[idx * 4..idx * 4 + 4].try_into().unwrap())
}

/// Output-corruption controls: a corrupted copy of the verified GPU buffers must be flagged exactly once.
fn control_output(run: &KindRun) -> Vec<Value> {
    let (p, exp) = (&run.prep, &run.exp);
    let lay = p.lay;
    let probe = |a: &[u8], b: Option<&[u8]>| -> u64 { compare_all(lay, exp, a, b).total() };
    let g1 = run.g1.as_deref();
    let mut res = Vec::new();
    let mut push = |name: &str, detail: &str, count: u64| {
        res.push(
            json!({"control": name, "kind": p.kind.name(), "detail": detail,
            "expected": "exactly 1 mismatching word", "mismatch_count": count, "pass": count == 1}),
        );
    };
    let last = lay.slots - 1;
    let mut t0 = run.g0.clone();
    t0[0] ^= 1;
    push("output_bit_flip", "y0 slot 0 row 0 bit 0", probe(&t0, g1));
    if let Some(g1) = g1 {
        let mut t1 = g1.to_vec();
        let idx = last * lay.ys + lay.out - 1;
        t1[idx * 4 + 3] ^= 0x80;
        push(
            "output_bit_flip",
            "y1 last slot last row sign bit",
            probe(&run.g0, Some(&t1)),
        );
    }
    if let Some(slot) = exp.computed.iter().position(|&c| c) {
        let mut tu = run.g0.clone();
        let idx = slot * lay.ys + lay.out - 1;
        tu[idx * 4..idx * 4 + 4].copy_from_slice(&SENTINEL_BITS.to_le_bytes());
        push(
            "output_unwritten_word",
            "y0 first computed slot last row replaced by the 0xa5 sentinel",
            probe(&tu, g1),
        );
    }
    if lay.ys > lay.out {
        let mut tp = run.g0.clone();
        let idx = last * lay.ys + lay.out;
        debug_assert_eq!(word_at(&run.g0, idx), SENTINEL_BITS);
        tp[idx * 4] ^= 1;
        push(
            "pad_bit_flip",
            "y0 last slot first pad word bit 0",
            probe(&tp, g1),
        );
    }
    let mut tt = run.g0.clone();
    tt[lay.body() * 4] ^= 1;
    push(
        "tail_guard_bit_flip",
        "y0 first trailing guard word bit 0",
        probe(&tt, g1),
    );
    let mut tl = run.g0.clone();
    let n = tl.len();
    tl[n - 1] ^= 0x80;
    push(
        "tail_guard_bit_flip",
        "y0 last trailing guard word sign bit",
        probe(&tl, g1),
    );
    res
}

struct Tables {
    lut: [u32; 256],
    mask: [u32; 128],
}
impl Tables {
    fn original() -> Self {
        Tables {
            lut: e8p::build_lut(),
            mask: e8p::build_mask(),
        }
    }
}

/// `decode_weights_q` with injectable producer tables and an optional swap of the two halves of every lane pair.
fn decode_q_with(words: &[u16], tabs: &Tables, swap: bool) -> Vec<i8> {
    let mut q = Vec::with_capacity(words.len() * 8);
    for c in 0..words.len() * 2 {
        let (idx, sign7, shift) = e8p::unpack_word_fields(words[c / 2]);
        let w = e8p::unpack_word_swar(&tabs.lut, &tabs.mask, idx, sign7, shift);
        let half = (c & 1) ^ (swap as usize);
        let base = 16 * half as u32;
        for l in 0..4u32 {
            let nib = ((w >> (base + 4 * l)) & 0xF) as i8;
            let o = (nib << 4) >> 4;
            q.push(2 * o + 1);
        }
    }
    q
}

fn eval_with(tabs: &Tables, swap: bool) -> impl Fn(&Tensor, Kind, &[f32]) -> Vec<f32> + Sync + '_ {
    move |t, kind, xs| {
        let q = decode_q_with(&t.words, tabs, swap);
        let d: Vec<f32> = t
            .scales
            .iter()
            .map(|&b| e8p::scale_quarter_f32(b))
            .collect();
        gemv_tree_f32_q(kind.rotation(), kind.m(), kind.k(), &q, &d, xs)
    }
}

const MUT_LUT_IDX: usize = 0x9d;
const MUT_LUT_XOR: u32 = 0x0000_2000;
const MUT_MASK_IDX: usize = 0x2d;
const MUT_MASK_XOR: u32 = 0x000f_0000;

/// Mutated-CPU-decode negatives against the verified GPU words of the whole case.
fn control_mutants(case: &Case, runs: &[KindRun]) -> Vec<Value> {
    let mut res = Vec::new();
    let orig = Tables::original();
    // Baseline: the local transcription with the original tables equals the reference on every word.
    let base_eval = eval_with(&orig, false);
    for r in runs {
        let exp_b = cpu_expected(case, r.prep.kind, &r.prep.bucket, &base_eval);
        let cmp = compare_all(r.prep.lay, &exp_b, &r.g0, r.g1.as_deref());
        let (d, _) = expected_diff(&exp_b, &r.exp, r.prep.lay);
        res.push(json!({"control": "mutant_transcription_baseline", "kind": r.prep.kind.name(),
            "baseline_vs_original_reference_words_differ": d, "baseline_vs_gpu_mismatches": cmp.total(),
            "words_compared": cmp.compared(), "expected": "0 (unmutated transcription == original reference)",
            "pass": d == 0 && cmp.total() == 0}));
    }
    let mut lut_m = Tables::original();
    lut_m.lut[MUT_LUT_IDX] ^= MUT_LUT_XOR;
    let mut mask_m = Tables::original();
    mask_m.mask[MUT_MASK_IDX] ^= MUT_MASK_XOR;
    let muts: [(&str, String, &Tables, bool); 3] = [
        (
            "wrong_lut_entry",
            format!("LUT[0x{MUT_LUT_IDX:02x}] ^= 0x{MUT_LUT_XOR:08x}"),
            &lut_m,
            false,
        ),
        (
            "wrong_sign_mask",
            format!("MASK[0x{MUT_MASK_IDX:02x}] ^= 0x{MUT_MASK_XOR:08x}"),
            &mask_m,
            false,
        ),
        (
            "swapped_lane_pair",
            "the two lanes of every pair keep the other half of the word".into(),
            &orig,
            true,
        ),
    ];
    for (name, detail, tabs, swap) in muts {
        let ev = eval_with(tabs, swap);
        for r in runs {
            let exp_m = cpu_expected(case, r.prep.kind, &r.prep.bucket, &ev);
            let (orig_diff, slots_hit) = expected_diff(&exp_m, &r.exp, r.prep.lay);
            let cmp = compare_all(r.prep.lay, &exp_m, &r.g0, r.g1.as_deref());
            let gpu_vs_orig = compare_all(r.prep.lay, &r.exp, &r.g0, r.g1.as_deref()).total();
            res.push(json!({"control": name, "method": "mutated_cpu_decode_arithmetic", "kind": r.prep.kind.name(),
                "mutation": detail, "words_compared": cmp.compared(),
                "mutant_vs_original_reference_words_differ": orig_diff, "slots_affected": slots_hit,
                "mutant_vs_gpu_words_differ": cmp.total(), "gpu_vs_original_reference_mismatches": gpu_vs_orig,
                "expected": "mutant differs from the original reference and from the GPU; GPU equals the reference",
                "pass": orig_diff > 0 && cmp.total() > 0 && gpu_vs_orig == 0}));
        }
    }
    res
}

/// Rewrite scale rows with the next row's scales (`row m-1` takes row 0): rows `{0, 1, M/2-1, M/2, M-1}` plus every
/// 31st. Returns the tampered tensor and the number of scale words that changed.
fn tamper_scale_rows(kind: Kind, t: &Tensor) -> Result<(Tensor, usize)> {
    let (m, groups) = (kind.m(), kind.groups_per_row());
    let mut rows: BTreeSet<usize> = [0, 1, m / 2 - 1, m / 2, m - 1].into_iter().collect();
    rows.extend((0..m).step_by(31));
    let mut scales = t.scales.clone();
    let mut changed = 0usize;
    for &r in &rows {
        let src = (r + 1) % m;
        for g in 0..groups {
            let v = t.scales[src * groups + g];
            if scales[r * groups + g] != v {
                changed += 1;
            }
            scales[r * groups + g] = v;
        }
    }
    if changed == 0 {
        return Err(format!("{}: scale-row tamper changed nothing", kind.name()));
    }
    Ok((Tensor::from_planes(t.words.clone(), scales), changed))
}

/// Corrupted-input relaunch: scale rows rewritten in place in the placement memory.
fn control_scale_row(
    ctx: &Ctx,
    dev: &Dev,
    case: &Case,
    placement: Placement,
    slabs: &Slabs,
    runs: &[KindRun],
) -> Result<Vec<Value>> {
    let bucket = &runs[0].prep.bucket;
    let mut ids = vec![bucket.ids[0]];
    if bucket.ids.len() > 1 {
        ids.push(*bucket.ids.last().unwrap());
    }
    let mut tcase = case.clone();
    let mut changed: BTreeMap<(u32, &'static str), usize> = BTreeMap::new();
    for &id in &ids {
        let e = &case.experts[&id];
        let (gate, cg) = tamper_scale_rows(Kind::GateUp, &e.gate)?;
        let (down, cd) = tamper_scale_rows(Kind::Down, &e.down)?;
        changed.insert((id, "gate_up"), cg);
        changed.insert((id, "down"), cd);
        let ne = Expert {
            label: format!("{}+scale_rows", e.label),
            prov: e.prov.clone(),
            gate,
            down,
        };
        tcase.experts.insert(id, Rc::new(ne));
    }
    for &id in &ids {
        for kind in Kind::ALL {
            write_blob(
                dev,
                slabs.by_id[&id].blob(kind),
                &tcase.experts[&id].proj(kind).bytes,
            )?;
        }
    }
    let mut res = Vec::new();
    for r in runs {
        let exp_t = cpu_expected(&tcase, r.prep.kind, &r.prep.bucket, &eval_reference);
        launch(ctx, dev, &r.prep, &r.prep.args)?;
        let (g0, g1) = read_out(dev, &r.prep)?;
        let vs_t = compare_all(r.prep.lay, &exp_t, &g0, g1.as_deref());
        let vs_o = compare_all(r.prep.lay, &r.exp, &g0, g1.as_deref());
        let (mut inside, mut outside) = (BTreeMap::new(), 0u64);
        for (slot, &n) in vs_o.slot_counts.iter().enumerate() {
            let id = case.route[slot];
            if ids.contains(&id) {
                *inside.entry(id).or_insert(0u64) += n;
            } else {
                outside += n;
            }
        }
        let all_detect = ids
            .iter()
            .all(|id| inside.get(id).copied().unwrap_or(0) > 0);
        let (orig_vs_corrupt, _) = expected_diff(&exp_t, &r.exp, r.prep.lay);
        let pass = vs_t.total() == 0
            && vs_o.total() > 0
            && outside == 0
            && all_detect
            && orig_vs_corrupt > 0;
        res.push(json!({"control": "wrong_scale_row", "method": "input_corruption_relaunch",
            "placement": placement.label(), "kind": r.prep.kind.name(),
            "corrupted_experts": ids.iter().map(|id| json!({"id": id, "host_mapped": slabs.by_id[id].host,
                "changed_scale_words_gate_up": changed[&(*id, "gate_up")],
                "changed_scale_words_down": changed[&(*id, "down")]})).collect::<Vec<_>>(),
            "words_compared": vs_t.compared(),
            "gpu_vs_reference_of_corrupted_input_mismatches": vs_t.total(),
            "gpu_vs_original_reference_mismatches": vs_o.total(),
            "corrupted_vs_original_reference_words_differ": orig_vs_corrupt,
            "mismatches_in_corrupted_experts_slots": inside.iter().map(|(k, v)| json!({"id": k, "words": v})).collect::<Vec<_>>(),
            "mismatches_outside_corrupted_slots": outside,
            "expected": "GPU == reference(corrupted input); GPU != original reference only in corrupted experts' slots",
            "pass": pass}));
    }
    for &id in &ids {
        for kind in Kind::ALL {
            write_blob(
                dev,
                slabs.by_id[&id].blob(kind),
                &case.experts[&id].proj(kind).bytes,
            )?;
        }
    }
    for r in runs {
        launch(ctx, dev, &r.prep, &r.prep.args)?;
        let (g0, g1) = read_out(dev, &r.prep)?;
        let back = compare_all(r.prep.lay, &r.exp, &g0, g1.as_deref());
        res.push(json!({"control": "wrong_scale_row_restore", "placement": placement.label(), "kind": r.prep.kind.name(),
            "mismatch_count": back.total(), "words_compared": back.compared(),
            "expected": "0 after restoring the original scales", "pass": back.total() == 0}));
    }
    Ok(res)
}

/// Corrupted-input relaunch: the table entries of two experts point at each other's tensors.
fn control_pointer(
    ctx: &Ctx,
    dev: &mut Dev,
    case: &Case,
    placement: Placement,
    slabs: &Slabs,
    runs: &[KindRun],
) -> Result<Vec<Value>> {
    let bucket = &runs[0].prep.bucket;
    if bucket.ids.len() < 2 {
        return Err("wrong_expert_pointer control needs two experts".into());
    }
    let (a, b) = (bucket.ids[0], *bucket.ids.last().unwrap());
    let mut tcase = case.clone();
    tcase.experts.insert(a, case.experts[&b].clone());
    tcase.experts.insert(b, case.experts[&a].clone());
    let mut res = Vec::new();
    for r in runs {
        let kind = r.prep.kind;
        let mut table = table_for(slabs, kind);
        table.swap(a as usize, b as usize);
        let args = retable(dev, &r.prep, &table)?;
        let exp_t = cpu_expected(&tcase, kind, &r.prep.bucket, &eval_reference);
        launch(ctx, dev, &r.prep, &args)?;
        let (g0, g1) = read_out(dev, &r.prep)?;
        let vs_t = compare_all(r.prep.lay, &exp_t, &g0, g1.as_deref());
        let vs_o = compare_all(r.prep.lay, &r.exp, &g0, g1.as_deref());
        let (mut inside, mut outside) = (BTreeMap::new(), 0u64);
        for (slot, &n) in vs_o.slot_counts.iter().enumerate() {
            let id = case.route[slot];
            if id == a || id == b {
                *inside.entry(id).or_insert(0u64) += n;
            } else {
                outside += n;
            }
        }
        let all_detect = [a, b]
            .iter()
            .all(|id| inside.get(id).copied().unwrap_or(0) > 0);
        let (orig_vs_swapped, _) = expected_diff(&exp_t, &r.exp, r.prep.lay);
        let pass = vs_t.total() == 0
            && vs_o.total() > 0
            && outside == 0
            && all_detect
            && orig_vs_swapped > 0;
        res.push(json!({"control": "wrong_expert_pointer", "method": "input_corruption_relaunch",
            "placement": placement.label(), "kind": kind.name(),
            "swapped_table_entries": [a, b], "words_compared": vs_t.compared(),
            "gpu_vs_reference_of_swapped_input_mismatches": vs_t.total(),
            "gpu_vs_original_reference_mismatches": vs_o.total(),
            "swapped_vs_original_reference_words_differ": orig_vs_swapped,
            "mismatches_in_swapped_experts_slots": inside.iter().map(|(k, v)| json!({"id": k, "words": v})).collect::<Vec<_>>(),
            "mismatches_outside_swapped_slots": outside,
            "expected": "GPU == reference(swapped input); GPU != original reference only in the two experts' slots",
            "pass": pass}));
        launch(ctx, dev, &r.prep, &r.prep.args)?;
        let (g0, g1) = read_out(dev, &r.prep)?;
        let back = compare_all(r.prep.lay, &r.exp, &g0, g1.as_deref());
        res.push(
            json!({"control": "wrong_expert_pointer_restore", "placement": placement.label(),
            "kind": kind.name(), "mismatch_count": back.total(), "words_compared": back.compared(),
            "expected": "0 with the original table", "pass": back.total() == 0}),
        );
    }
    Ok(res)
}

/// Signed-bad expert ids: the ids of the first four experts of the bucket are replaced (in the id buffer only) by
/// `-1`, `n_experts`, `INT_MAX` and `INT_MIN`. The kernel's one unsigned compare against `n_experts@88` exits before
/// the table is indexed, so those experts' slots keep the `0xa5` sentinel while every other expert is still computed.
fn control_bad_ids(
    ctx: &Ctx,
    dev: &mut Dev,
    case: &Case,
    placement: Placement,
    runs: &[KindRun],
) -> Result<Vec<Value>> {
    if runs[0].prep.bucket.ids.len() <= BAD_IDS.len() {
        return Err("bad_expert_ids control needs more experts than bad ids".into());
    }
    let mut res = Vec::new();
    for r in runs {
        let kind = r.prep.kind;
        let mut bad = r.prep.bucket.clone();
        bad.ids[..BAD_IDS.len()].copy_from_slice(&BAD_IDS);
        let exp_b = cpu_expected(case, kind, &bad, &eval_reference);
        let ib = dev.upload(&u32_bytes(&bad.ids))?;
        let mut args = r.prep.args;
        args[8..16].copy_from_slice(&ib.ptr.to_le_bytes());
        launch(ctx, dev, &r.prep, &args)?;
        let (g0, g1) = read_out(dev, &r.prep)?;
        let vs_b = compare_all(r.prep.lay, &exp_b, &g0, g1.as_deref());
        let (differs, _) = expected_diff(&exp_b, &r.exp, r.prep.lay);
        let pass = vs_b.total() == 0 && exp_b.refused == BAD_IDS && differs > 0;
        res.push(json!({"control": "bad_expert_ids", "method": "descriptor_relaunch",
            "placement": placement.label(), "kind": kind.name(),
            "bad_ids": BAD_IDS, "refused_experts": exp_b.refused, "words_compared": vs_b.compared(),
            "gpu_vs_reference_with_rejected_experts_mismatches": vs_b.total(),
            "rejected_vs_original_reference_words_differ": differs,
            "expected": "bad-id experts store nothing (slots keep the sentinel), every other expert equals the reference",
            "pass": pass}));
        launch(ctx, dev, &r.prep, &r.prep.args)?;
        let (g0, g1) = read_out(dev, &r.prep)?;
        let back = compare_all(r.prep.lay, &r.exp, &g0, g1.as_deref());
        res.push(json!({"control": "bad_expert_ids_restore", "placement": placement.label(), "kind": kind.name(),
            "mismatch_count": back.total(), "words_compared": back.compared(),
            "expected": "0 with the original id buffer", "pass": back.total() == 0}));
    }
    Ok(res)
}

// ---------------------------------------------------------------------------------------
// Case driver
// ---------------------------------------------------------------------------------------

#[derive(Default)]
struct Totals {
    cases: u64,
    synthetic: u64,
    real_samples: u64,
    host_mapped_cases: u64,
    split_cases: u64,
    launches: u64,
    words_compared: u64,
    mismatches: u64,
    guard_tensors: u64,
    guard_bytes: u64,
    guard_failures: u64,
    control_checks: u64,
    control_failures: u64,
    neg_seen: BTreeSet<(String, String)>,
}

struct RunOpts {
    controls: bool,
    mutants: bool,
    tamper: bool,
}

fn run_case_on(
    ctx: &Ctx,
    dev: &mut Dev,
    case: &Case,
    placement: Placement,
    exps: &mut [Option<Expected>; 2],
    t: &mut Totals,
    o: &RunOpts,
) -> Result<Value> {
    let slabs = upload_experts(dev, case, placement)?;
    let mut runs: Vec<KindRun> = Vec::new();
    let mut kinds_json = serde_json::Map::new();
    let mut controls: Vec<Value> = Vec::new();
    let mut mismatches = 0u64;
    let mut compared = 0u64;
    for kind in Kind::ALL {
        let prep = prepare(dev, case, kind, &slabs)?;
        let exp = exps[kind.idx()].take().expect("expected computed");
        launch(ctx, dev, &prep, &prep.args)?;
        let (g0, g1) = read_out(dev, &prep)?;
        let cmp = compare_all(prep.lay, &exp, &g0, g1.as_deref());
        mismatches += cmp.total();
        compared += cmp.compared();
        t.launches += 1;
        let mut kj = cmp.json();
        kj["symbol"] = json!(kind.symbol(ctx.target));
        kj["grid"] = json!(prep.grid);
        kj["unique_experts"] = json!(prep.bucket.ids.len());
        kj["listed_slots"] = json!(prep.bucket.slots.len());
        kj["refused_experts"] = json!(exp.refused);
        kj["slots_written"] = json!(exp.computed.iter().filter(|&&c| c).count());
        kinds_json.insert(kind.name().to_string(), kj);
        runs.push(KindRun { prep, exp, g0, g1 });
    }
    if o.controls {
        for r in &runs {
            controls.extend(control_output(r));
        }
    }
    if o.mutants {
        controls.extend(control_mutants(case, &runs));
    }
    if placement != Placement::Vram {
        controls.push(json!({"control": "host_mapped_pointer_attributes", "placement": placement.label(),
            "host_slabs": slabs.host_slabs, "attribute_failures": slabs.host_attr_failures,
            "evidence": slabs.evidence, "expected": "every mapped slab reports host-resident memory",
            "pass": slabs.host_slabs > 0 && slabs.host_attr_failures == 0}));
    }
    if o.tamper {
        controls.extend(control_scale_row(ctx, dev, case, placement, &slabs, &runs)?);
        controls.extend(control_pointer(ctx, dev, case, placement, &slabs, &runs)?);
        controls.extend(control_bad_ids(ctx, dev, case, placement, &runs)?);
    }
    let guards = verify_blobs(dev, &slabs, case)?;
    t.guard_tensors += guards.tensors;
    t.guard_bytes += guards.guard_bytes;
    if guards.guard_mismatch_bytes + guards.tensor_mismatch_bytes != 0 {
        t.guard_failures += 1;
    }
    for c in &controls {
        t.control_checks += 1;
        if c["pass"] != json!(true) {
            t.control_failures += 1;
        } else if let (Some(n), Some(k)) = (c["control"].as_str(), c["kind"].as_str()) {
            if NEGATIVES.contains(&n) {
                t.neg_seen.insert((n.to_string(), k.to_string()));
            }
        }
    }
    t.cases += 1;
    if case.origin == "synthetic" {
        t.synthetic += 1;
    } else {
        t.real_samples += 1;
    }
    match placement {
        Placement::HostMapped => t.host_mapped_cases += 1,
        Placement::Split => t.split_cases += 1,
        Placement::Vram => {}
    }
    t.mismatches += mismatches;
    t.words_compared += compared;
    let bucket = &runs[0].prep.bucket;
    let slot_counts: BTreeMap<String, usize> = bucket
        .ids
        .iter()
        .enumerate()
        .map(|(u, id)| {
            (
                id.to_string(),
                (bucket.offs[u + 1] - bucket.offs[u]) as usize,
            )
        })
        .collect();
    let experts: Vec<Value> = bucket
        .ids
        .iter()
        .map(|id| {
            let e = &case.experts[id];
            json!({"id": id, "label": e.label, "gate_sha256": hex(&e.gate.sha), "down_sha256": hex(&e.down.sha),
                "provenance": e.prov})
        })
        .collect();
    Ok(json!({
        "case": case.name, "placement": placement.label(), "origin": case.origin, "c": case.c, "slots": case.slots(),
        "listed_slots": bucket.slots.len(), "unique_experts": bucket.ids.len(), "slots_per_expert": slot_counts,
        "max_slots_per_expert": slot_counts.values().max(),
        "strides": {"x_row_stride_gate": case.xrs_gate, "x_row_stride_down": case.xrs_down,
            "y_slot_stride_gate": case.ys_gate, "y_slot_stride_down": case.ys_down},
        "weights": {"placement": placement.label(), "host_mapped_slabs": slabs.host_slabs,
            "vram_slabs": slabs.vram_slabs, "host_pointer_evidence": slabs.evidence},
        "guards": {"tensor_allocations_checked": guards.tensors, "guard_bytes_checked": guards.guard_bytes,
            "guard_mismatch_bytes": guards.guard_mismatch_bytes, "tensor_mismatch_bytes": guards.tensor_mismatch_bytes},
        "input_sha256": case.input_sha256(),
        "experts": experts,
        "kernels": kinds_json,
        "words_compared": compared,
        "mismatch_count": mismatches,
        "controls": controls,
    }))
}

fn run_case(
    ctx: &Ctx,
    case: &Case,
    placements: &[Placement],
    o: &RunOpts,
    t: &mut Totals,
    cov: &mut Coverage,
    rows: &mut Vec<Value>,
) -> Result<()> {
    let bucket = case.bucket();
    // The full original reference is placement independent: computed once, cloned per placement.
    let base: Vec<Expected> = Kind::ALL
        .iter()
        .map(|&kind| cpu_expected(case, kind, &bucket, &eval_reference))
        .collect();
    for (i, &placement) in placements.iter().enumerate() {
        let mut exps: [Option<Expected>; 2] = [Some(base[0].clone()), Some(base[1].clone())];
        eprintln!("[g1] {}@{}", case.name, placement.label());
        let mut dev = Dev::new(&ctx.hip);
        let oo = RunOpts {
            controls: o.controls && placement == Placement::Vram,
            mutants: o.mutants && placement == Placement::Vram,
            tamper: o.tamper,
        };
        let r = run_case_on(ctx, &mut dev, case, placement, &mut exps, t, &oo);
        dev.release();
        rows.push(r?);
        if i == 0 {
            let launched: BTreeSet<u32> = bucket
                .ids
                .iter()
                .enumerate()
                .filter(|(u, _)| {
                    let n = (bucket.offs[u + 1] - bucket.offs[*u]) as usize;
                    n >= 1 && n <= MAX_SLOTS
                })
                .map(|(_, &id)| id)
                .collect();
            for id in launched {
                for kind in Kind::ALL {
                    cov.update(kind, case.experts[&id].proj(kind));
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------
// Host facts
// ---------------------------------------------------------------------------------------

fn read_trim(p: &str) -> Option<String> {
    std::fs::read_to_string(p)
        .ok()
        .map(|s| s.trim().to_string())
}

/// PCIe link facts of the endpoint and every upstream bridge up to the root port (sysfs).
fn pcie_info(bdf: &str, caveat: &str) -> Value {
    let attr = |dev: &str, name: &str| read_trim(&format!("/sys/bus/pci/devices/{dev}/{name}"));
    let mut chain = Vec::new();
    if let Ok(real) = std::fs::canonicalize(format!("/sys/bus/pci/devices/{bdf}")) {
        for comp in real.components() {
            let s = comp.as_os_str().to_string_lossy().to_string();
            if s.len() == 12 && s.matches(':').count() == 2 && s.contains('.') {
                chain.push(json!({"bdf": s,
                    "current_link_speed": attr(&s, "current_link_speed"),
                    "current_link_width": attr(&s, "current_link_width"),
                    "max_link_speed": attr(&s, "max_link_speed"),
                    "max_link_width": attr(&s, "max_link_width")}));
            }
        }
    }
    let root = chain.first().cloned();
    json!({"endpoint": bdf, "root_port": root, "chain_root_to_endpoint": chain,
        "vram_total_sysfs": attr(bdf, "mem_info_vram_total"), "link_caveat": caveat})
}

// ---------------------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------------------

#[derive(Debug)]
struct Cfg {
    codeobject: String,
    codeobject_sha256: Option<String>,
    samples_dirs: Vec<String>,
    out: String,
    overwrite: bool,
    device: Option<i32>,
    expect_uuid: Option<String>,
    expect_bdf: Option<String>,
    expect_arch: String,
    target: Target,
    card: String,
    fabric_class: Option<String>,
    fabric_owner: Option<String>,
    fabric_job: Option<String>,
    fabric_card: Option<String>,
    fabric_cmd: Option<String>,
    only: Vec<String>,
    no_reclaim_arm: bool,
    verify_artifact: bool,
    rehash_artifact: bool,
}

const USAGE: &str = "usage: e8p_i4_moe_gemv_gfx1100_g1 --codeobject PATH --samples-dir DIR [--samples-dir DIR2 | \
DIR1,DIR2] --out RECEIPT.json [--card NAME] [--device N] [--expect-uuid U] [--expect-bdf B] [--expect-arch gfx1100|gfx1151] \
[--codeobject-sha256 HEX] [--fabric-class C] [--fabric-owner O] [--fabric-job J] [--fabric-card NAME] \
[--fabric-cmd STR] [--only SUBSTR]... [--overwrite] [--verify-artifact-range] [--rehash-artifact] [--no-reclaim-arm] \
(gfx1100 default: xtx uuid/bdf, card xtx, both .hfq sample dirs; gfx1151: --expect-bdf required, --expect-uuid optional (no ROCr UUID), \
card halo, the mq2-xts slab sample dir required) \
(fabric flags fall back to env FABRIC_CLASS / FABRIC_OWNER / FABRIC_JOB / FABRIC_CARD / FABRIC_CMD)";

fn parse_args(argv: &[String]) -> Result<Cfg> {
    let mut cfg = Cfg {
        codeobject: String::new(),
        codeobject_sha256: None,
        samples_dirs: Vec::new(),
        out: String::new(),
        overwrite: false,
        device: None,
        expect_uuid: None,
        expect_bdf: None,
        expect_arch: DEFAULT_ARCH.into(),
        target: Target::Gfx1100,
        card: env_nonempty("FABRIC_CARD").unwrap_or_default(),
        fabric_class: env_nonempty("FABRIC_CLASS"),
        fabric_owner: env_nonempty("FABRIC_OWNER"),
        fabric_job: env_nonempty("FABRIC_JOB"),
        fabric_card: env_nonempty("FABRIC_CARD"),
        fabric_cmd: env_nonempty("FABRIC_CMD"),
        only: Vec::new(),
        no_reclaim_arm: false,
        verify_artifact: false,
        rehash_artifact: false,
    };
    let mut i = 0;
    let value = |i: &mut usize| -> Result<String> {
        *i += 1;
        argv.get(*i)
            .cloned()
            .ok_or_else(|| format!("{} needs a value\n{USAGE}", argv[*i - 1]))
    };
    while i < argv.len() {
        match argv[i].as_str() {
            "--codeobject" => cfg.codeobject = value(&mut i)?,
            "--codeobject-sha256" => {
                cfg.codeobject_sha256 = Some(value(&mut i)?.to_ascii_lowercase())
            }
            "--samples-dir" => {
                let v = value(&mut i)?;
                cfg.samples_dirs
                    .extend(v.split(',').filter(|s| !s.is_empty()).map(str::to_string));
            }
            "--out" => cfg.out = value(&mut i)?,
            "--overwrite" => cfg.overwrite = true,
            "--device" => {
                cfg.device = Some(
                    value(&mut i)?
                        .parse::<i32>()
                        .map_err(|e| format!("--device: {e}"))?,
                )
            }
            "--expect-uuid" => cfg.expect_uuid = Some(value(&mut i)?),
            "--expect-bdf" => cfg.expect_bdf = Some(value(&mut i)?.to_ascii_lowercase()),
            "--expect-arch" => cfg.expect_arch = value(&mut i)?,
            "--card" => cfg.card = value(&mut i)?,
            "--fabric-class" => cfg.fabric_class = Some(value(&mut i)?),
            "--fabric-owner" => cfg.fabric_owner = Some(value(&mut i)?),
            "--fabric-job" => cfg.fabric_job = Some(value(&mut i)?),
            "--fabric-card" => cfg.fabric_card = Some(value(&mut i)?),
            "--fabric-cmd" => cfg.fabric_cmd = Some(value(&mut i)?),
            "--only" => cfg.only.push(value(&mut i)?),
            "--no-reclaim-arm" => cfg.no_reclaim_arm = true,
            "--verify-artifact-range" => cfg.verify_artifact = true,
            "--rehash-artifact" => cfg.rehash_artifact = true,
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
        i += 1;
    }
    cfg.target = Target::of_name(&cfg.expect_arch).ok_or_else(|| {
        format!(
            "--expect-arch {:?}: this oracle serves gfx1100 and gfx1151\n{USAGE}",
            cfg.expect_arch
        )
    })?;
    match cfg.target {
        Target::Gfx1100 => {
            cfg.expect_uuid.get_or_insert_with(|| DEFAULT_UUID.into());
            cfg.expect_bdf.get_or_insert_with(|| DEFAULT_BDF.into());
        }
        Target::Gfx1151 => {
            if cfg.expect_bdf.is_none() {
                return Err(format!("gfx1151 has no ROCr UUID: --expect-bdf is required (--expect-uuid is optional)\n{USAGE}"));
            }
        }
    }
    if cfg.card.is_empty() {
        cfg.card = cfg.target.default_card().into();
    }
    if cfg.codeobject.is_empty() || cfg.out.is_empty() {
        return Err(USAGE.into());
    }
    let required: Vec<&str> = cfg.target.required_pins().iter().map(|(t, _)| *t).collect();
    if cfg.samples_dirs.is_empty() && cfg.only.is_empty() {
        return Err(format!(
            "--samples-dir for the required pinned artifact(s) {} is required (a full G1 binds real samples)\n{USAGE}",
            required.join(", ")
        ));
    }
    if cfg.samples_dirs.len() > cfg.target.pins().len() {
        return Err(format!(
            "at most {} sample dirs (one per pinned artifact)",
            cfg.target.pins().len()
        ));
    }
    if !cfg.overwrite && Path::new(&cfg.out).exists() {
        return Err(format!("{} exists (use --overwrite)", cfg.out));
    }
    Ok(cfg)
}

fn selected(cfg: &Cfg, name: &str) -> bool {
    cfg.only.is_empty() || cfg.only.iter().any(|s| name.contains(s.as_str()))
}

fn placements_for_synthetic(name: &str) -> Vec<Placement> {
    let mut v = vec![Placement::Vram];
    if HOST_MAPPED_CASES.contains(&name) {
        v.push(Placement::HostMapped);
    }
    if SPLIT_CASES.contains(&name) {
        v.push(Placement::Split);
    }
    v
}

fn run(cfg: &Cfg, argv: &[String]) -> Result<(Value, bool)> {
    if !cfg.no_reclaim_arm {
        // Production Qwen4 host-mapped configuration; must precede the first HIP runtime call.
        hip_bridge::keep_host_memory_out_of_reclaim(&format!(
            "{} E8P G1 host-mapped weights",
            cfg.target.name()
        ));
    }
    let hip = HipRuntime::load().map_err(err)?;
    if let Some(d) = cfg.device {
        hip.set_device(d).map_err(err)?;
    }
    let id = hip.current_device().map_err(err)?;
    let uuid = hip.device_uuid(id).map_err(err)?;
    let bdf = hip.device_pci_bus_id(id).map_err(err)?.to_ascii_lowercase();
    let arch = hip.get_arch(id).map_err(err)?;
    let identity_match = cfg.expect_uuid.as_deref().is_none_or(|u| uuid == u)
        && cfg.expect_bdf.as_deref().is_none_or(|b| bdf == b)
        && arch == cfg.expect_arch;
    let device = json!({"index": id, "uuid": uuid, "bdf": bdf, "arch": arch, "card": cfg.card,
        "expected_uuid": cfg.expect_uuid, "expected_bdf": cfg.expect_bdf, "expected_arch": cfg.expect_arch,
        "uuid_authoritative": cfg.target == Target::Gfx1100, "uuid_checked": cfg.expect_uuid.is_some(),
        "identity_match": identity_match});
    let fabric = json!({"class": cfg.fabric_class, "owner": cfg.fabric_owner, "job": cfg.fabric_job,
        "card": cfg.fabric_card, "cmd": cfg.fabric_cmd});
    let exe = std::env::current_exe().ok();
    let exe_sha = exe
        .as_ref()
        .and_then(|p| file_sha256(&p.to_string_lossy()).ok())
        .map(|(h, _)| h);
    let command = json!({"argv": argv, "exe": exe.as_ref().map(|p| p.display().to_string()), "exe_sha256": exe_sha,
        "cwd": std::env::current_dir().ok().map(|p| p.display().to_string()), "build_command": BUILD_COMMAND,
        "env": {"HSA_USERPTR_FOR_PAGED_MEM": env_nonempty("HSA_USERPTR_FOR_PAGED_MEM"),
            "GPU_PINNED_MIN_XFER_SIZE": env_nonempty("GPU_PINNED_MIN_XFER_SIZE"),
            "reclaim_arm": !cfg.no_reclaim_arm}});
    let sources = json!({"harness_sha256": hex(&sha(HARNESS_SRC)), "e8p_i4_reference_sha256": hex(&sha(REFERENCE_SRC)),
        "mq_i4_reference_tree_sha256": hex(&sha(TREE_SRC))});
    let pcie = pcie_info(&bdf, cfg.target.link_caveat());
    if !identity_match {
        return Ok((
            json!({"schema": cfg.target.schema(), "command": command, "sources": sources, "device": device, "fabric": fabric,
                "pcie": pcie, "error": "E8P MoE GEMV device identity mismatch", "byte_exact": false,
                "pass": false, "rc": 1}),
            false,
        ));
    }
    let vram_before = hip.get_vram_info().ok();
    let image = std::fs::read(&cfg.codeobject).map_err(|e| format!("{}: {e}", cfg.codeobject))?;
    let codeobject_sha = hex(&sha(&image));
    if let Some(pin) = &cfg.codeobject_sha256 {
        if *pin != codeobject_sha {
            return Err(format!(
                "code object sha256 {codeobject_sha} != pinned {pin}"
            ));
        }
    }
    let module = hip.module_load_data(&image).map_err(err)?;
    let mut funcs = Vec::new();
    let mut loads = Vec::new();
    for kind in Kind::ALL {
        let name = kind.symbol(cfg.target);
        let f = hip
            .module_get_function(&module, &name)
            .map_err(|e| format!("symbol {name}: {e}"))?;
        loads.push(name);
        funcs.push((kind, f));
    }
    let ctx = Ctx {
        hip,
        funcs,
        target: cfg.target,
    };
    let mut totals = Totals::default();
    let mut cov = Coverage::new();
    let mut rows: Vec<Value> = Vec::new();

    for name in SYNTH_CASES {
        let placements: Vec<Placement> = placements_for_synthetic(name)
            .into_iter()
            .filter(|p| selected(cfg, &format!("{name}@{}", p.label())))
            .collect();
        if placements.is_empty() {
            continue;
        }
        let case = build_synthetic(name);
        let o = RunOpts {
            controls: CONTROL_CASES.contains(name),
            mutants: MUTANT_CASES.contains(name),
            tamper: TAMPER_CASES.contains(name),
        };
        run_case(
            &ctx,
            &case,
            &placements,
            &o,
            &mut totals,
            &mut cov,
            &mut rows,
        )?;
    }

    let mut samples_meta: Vec<Value> = Vec::new();
    let mut bound: BTreeSet<&'static str> = BTreeSet::new();
    let mut guard_violations = 0u64;
    if !cfg.samples_dirs.is_empty() {
        let mut sets: Vec<SampleSet> = Vec::new();
        for dir in &cfg.samples_dirs {
            eprintln!("[g1] loading E8P samples from {dir}");
            let first = 4 * sets.len() as u32;
            let s = load_sample_set(
                dir,
                first,
                cfg.verify_artifact,
                cfg.rehash_artifact,
                cfg.target.pins(),
            )?;
            if !bound.insert(s.sha) {
                return Err(format!("{dir}: artifact {} bound twice", s.tag));
            }
            guard_violations += s.guard_violations;
            samples_meta.push(s.meta.clone());
            sets.push(s);
        }
        for case in sample_cases(&sets) {
            let placements: Vec<Placement> =
                [Placement::Vram, Placement::HostMapped, Placement::Split]
                    .into_iter()
                    .filter(|p| selected(cfg, &format!("{}@{}", case.name, p.label())))
                    .collect();
            if placements.is_empty() {
                continue;
            }
            let o = RunOpts {
                controls: false,
                mutants: false,
                tamper: false,
            };
            run_case(
                &ctx,
                &case,
                &placements,
                &o,
                &mut totals,
                &mut cov,
                &mut rows,
            )?;
        }
    }

    let vram_after = ctx.hip.get_vram_info().ok();
    let full = cfg.only.is_empty();
    let all_pinned_bound = cfg
        .target
        .required_pins()
        .iter()
        .all(|(_, s)| bound.contains(s));
    let missing_negatives: Vec<String> = NEGATIVES
        .iter()
        .flat_map(|n| {
            Kind::ALL
                .iter()
                .map(move |k| (n.to_string(), k.name().to_string()))
        })
        .filter(|key| !totals.neg_seen.contains(key))
        .map(|(n, k)| format!("{n}@{k}"))
        .collect();
    let coverage_ok = !full
        || (totals.control_checks > 0
            && totals.host_mapped_cases > 0
            && totals.split_cases > 0
            && totals.real_samples > 0
            && all_pinned_bound
            && cov.complete()
            && missing_negatives.is_empty());
    let byte_exact = totals.mismatches == 0 && totals.guard_failures == 0 && !rows.is_empty();
    let controls_pass = totals.control_failures == 0;
    let samples_ok = guard_violations == 0;
    let pass = byte_exact && controls_pass && coverage_ok && samples_ok;
    let out = json!({
        "schema": cfg.target.schema(),
        "module": cfg.target.module(),
        "target_arch": cfg.target.name(),
        "command": command,
        "sources": sources,
        "config": {"samples_dirs": cfg.samples_dirs, "only": cfg.only, "out": cfg.out,
            "verify_artifact_range": cfg.verify_artifact, "rehash_artifact": cfg.rehash_artifact,
            "gate_eligible": full},
        "codeobject": {"path": cfg.codeobject, "sha256": codeobject_sha, "bytes": image.len(),
            "pinned_sha256": cfg.codeobject_sha256},
        "code_object_sha": codeobject_sha,
        "kernarg_bytes": KERNARG_BYTES,
        "symbols_loaded": loads,
        "device": device,
        "fabric": fabric,
        "pcie": pcie,
        "link_caveat": cfg.target.link_caveat(),
        "vram": {"before_free_total": vram_before, "after_free_total": vram_after},
        "case_counts": {"total": totals.cases, "synthetic": totals.synthetic, "real_samples": totals.real_samples,
            "host_mapped": totals.host_mapped_cases, "split_vram_host": totals.split_cases},
        "symbol_launches": totals.launches,
        "words_compared": totals.words_compared,
        "cases": rows,
        "mismatch_count": totals.mismatches,
        "guard_regions": {"tensor_allocations_checked": totals.guard_tensors,
            "guard_bytes_checked": totals.guard_bytes, "cases_with_guard_or_tensor_change": totals.guard_failures},
        "coverage": cov.json(),
        "coverage_complete": cov.complete(),
        "negative_controls": {"checks": totals.control_checks, "failures": totals.control_failures,
            "required": NEGATIVES, "missing": missing_negatives, "pass": controls_pass},
        "samples": {"pinned_artifacts": cfg.target.pins().iter().map(|(t, s)| json!({"tag": t, "sha256": s,
            "required": cfg.target.required_pins().iter().any(|(rt, _)| rt == t)})).collect::<Vec<_>>(),
            "bound": bound.iter().collect::<Vec<_>>(), "all_pinned_bound": all_pinned_bound,
            "writer_guard_violations": guard_violations, "sets": samples_meta},
        "coverage_ok": coverage_ok,
        "byte_exact": byte_exact,
        "pass": pass,
        "rc": if pass { 0 } else { 1 },
    });
    let _ = &module;
    Ok((out, pass))
}

fn write_receipt(path: &str, v: &Value) {
    if let Err(e) = std::fs::write(path, format!("{v}\n")) {
        eprintln!("cannot write receipt {path}: {e}");
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let cfg = match parse_args(&argv) {
        Ok(c) => c,
        Err(e) => {
            println!(
                "{}",
                json!({"schema": SCHEMA, "error": e, "byte_exact": false, "pass": false, "rc": 1})
            );
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    match run(&cfg, &argv) {
        Ok((v, pass)) => {
            println!("{v}");
            write_receipt(&cfg.out, &v);
            if !pass {
                eprintln!("E8P MoE GEMV {} G1 FAILED", cfg.target.name());
                std::process::exit(1);
            }
        }
        Err(e) => {
            let v = json!({"schema": cfg.target.schema(), "command": {"argv": argv, "build_command": BUILD_COMMAND},
                "error": e, "byte_exact": false, "pass": false, "rc": 1});
            println!("{v}");
            write_receipt(&cfg.out, &v);
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------------------
// Local known-answer tests (no GPU):
//   cargo test -p hipfire-runtime --release --features lab --example e8p_i4_moe_gemv_gfx1100_g1
// ---------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernarg_layout_is_the_frozen_96_bytes() {
        let a = kernarg([1, 2, 3, 4, 5, 6, 7], 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq!(a.len(), 96);
        for i in 0..7usize {
            assert_eq!(
                u64::from_le_bytes(a[i * 8..i * 8 + 8].try_into().unwrap()),
                i as u64 + 1
            );
        }
        for (j, off) in [56usize, 60, 64, 68, 72, 76, 80, 84].iter().enumerate() {
            assert_eq!(
                u32::from_le_bytes(a[*off..*off + 4].try_into().unwrap()),
                8 + j as u32
            );
        }
        assert_eq!(
            u32::from_le_bytes(a[88..92].try_into().unwrap()),
            N_EXPERTS as u32
        );
        assert_eq!(u32::from_le_bytes(a[92..96].try_into().unwrap()), 0);
    }

    #[test]
    fn blob_sizes_match_the_extractor() {
        assert_eq!(Kind::GateUp.blob_bytes(), 870_400);
        assert_eq!(Kind::Down.blob_bytes(), 435_200);
        assert_eq!(
            Kind::GateUp.symbol(Target::Gfx1100),
            "e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100"
        );
        assert_eq!(
            Kind::Down.symbol(Target::Gfx1100),
            "e8p_i4_moe_gemv_r128_down_pm_gfx1100"
        );
        assert_eq!(
            Kind::GateUp.symbol(Target::Gfx1151),
            "e8p_i4_moe_gemv_r256_gate_up_pm_gfx1151"
        );
        assert_eq!(
            Kind::Down.symbol(Target::Gfx1151),
            "e8p_i4_moe_gemv_r128_down_pm_gfx1151"
        );
    }

    #[test]
    fn coverage_experts_cover_every_word_on_every_lane_pair() {
        for kind in Kind::ALL {
            let mut v = vec![0u64; 16 * 1024];
            for j in 0..6 {
                let per = kind.n_words() / 16;
                for g in 0..per {
                    for p in 0..16 {
                        let w = coverage_word(p, j * per + g) as usize;
                        Coverage::set(&mut v, p * 65536 + w);
                    }
                }
            }
            assert_eq!(Coverage::count(&v), 16 * 65536, "{}", kind.name());
        }
    }

    #[test]
    fn local_decode_equals_reference_and_mutants_differ() {
        let mut rng = Rng::new(3);
        let words: Vec<u16> = (0..4096).map(|_| rng.u32() as u16).collect();
        let tabs = Tables::original();
        assert_eq!(
            decode_q_with(&words, &tabs, false),
            e8p::decode_weights_q(8, 4096 * 8 / 8, &words)
        );
        assert_ne!(
            decode_q_with(&words, &tabs, true),
            decode_q_with(&words, &tabs, false)
        );
        let mut lut = Tables::original();
        lut.lut[MUT_LUT_IDX] ^= MUT_LUT_XOR;
        assert_ne!(
            decode_q_with(&words, &lut, false),
            decode_q_with(&words, &tabs, false)
        );
        let mut mask = Tables::original();
        mask.mask[MUT_MASK_IDX] ^= MUT_MASK_XOR;
        assert_ne!(
            decode_q_with(&words, &mask, false),
            decode_q_with(&words, &tabs, false)
        );
    }

    #[test]
    fn every_wire_word_is_a_valid_word() {
        for w in 0..=u16::MAX {
            let (i, s, sh) = e8p::unpack_word_fields(w);
            assert_eq!(e8p::pack_word(i, s, sh), w);
        }
    }

    #[test]
    fn tamper_changes_scales_and_keeps_words() {
        let e = gen_expert(5, palette(1));
        for kind in Kind::ALL {
            let t = e.proj(kind);
            let (tt, changed) = tamper_scale_rows(kind, t).unwrap();
            assert!(changed > 0);
            assert_eq!(tt.words, t.words);
            assert_eq!(tt.bytes.len(), t.bytes.len());
        }
    }

    #[test]
    fn buckets_respect_listing() {
        let route = vec![7u32, 3, 7, 3, 9];
        let b = build_bucket(&route, &[true, true, true, false, true]);
        assert_eq!(b.ids, vec![3, 7, 9]);
        assert_eq!(b.offs, vec![0, 1, 3, 4]);
        assert_eq!(b.slots, vec![1, 0, 2, 4]);
    }

    #[test]
    fn scale_classes_include_max_and_subnormal() {
        let mut rng = Rng::new(1);
        assert_eq!(scale_class(1, &mut rng), 0x7bff);
        for _ in 0..64 {
            let s = scale_class(2, &mut rng);
            assert!((1..=0x3ff).contains(&s));
        }
    }

    #[test]
    fn targets_keep_the_gfx1100_defaults_and_add_gfx1151() {
        assert_eq!(Target::of_name("gfx1100"), Some(Target::Gfx1100));
        assert_eq!(Target::of_name("gfx1151"), Some(Target::Gfx1151));
        for other in ["gfx1201", "gfx1101", "GFX1151", ""] {
            assert_eq!(Target::of_name(other), None);
        }
        assert_eq!(Target::Gfx1100.module(), "e8p_i4_moe_gemv_pm_gfx1100");
        assert_eq!(Target::Gfx1151.module(), "e8p_i4_moe_gemv_pm_gfx1151");
        assert_eq!(Target::Gfx1100.schema(), "e8p-i4-moe-gemv-gfx1100-g1-v1");
        assert_eq!(Target::Gfx1151.schema(), "e8p-i4-moe-gemv-gfx1151-g1-v1");
        assert_eq!(Target::Gfx1100.default_card(), "xtx");
        assert_eq!(Target::Gfx1151.default_card(), "halo");
        assert_eq!(DEFAULT_ARCH, "gfx1100");
        let tags = |p: &[(&str, &str)]| p.iter().map(|(t, _)| t.to_string()).collect::<Vec<_>>();
        assert_eq!(tags(Target::Gfx1100.pins()), ["c24-e8p", "c24h6306-e8p"]);
        assert_eq!(
            tags(Target::Gfx1100.required_pins()),
            ["c24-e8p", "c24h6306-e8p"]
        );
        assert_eq!(
            tags(Target::Gfx1151.pins()),
            ["c24-e8p", "c24h6306-e8p", "c24h6306-d2-mq2-xts"]
        );
        assert_eq!(
            tags(Target::Gfx1151.required_pins()),
            ["c24h6306-d2-mq2-xts"]
        );
        for (_, sha) in ARTIFACT_PINS.iter() {
            assert_eq!(sha.len(), 64);
            assert!(sha
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()));
        }
    }
}
