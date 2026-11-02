// SPDX-License-Identifier: Apache-2.0
//! Consumer24 B2: raw PM K1 MoE decode-GEMV **gfx1100** G1 byte oracle + short timing runner.
//! Never participates in runtime dispatch.
//!
//! Loads the code object `mq_i4_moe_gemv_pm_gfx1100` and its four symbols
//! `mq_i4_moe_gemv_{mq2i4,mq4sym}_{r256_gate_up,r128_down}_pm_gfx1100`, launches them with the frozen
//! 96-byte kernarg ABI (R5 `subq4/r5-k1-moe-gemv`@e8a161801 plus the trailing `n_experts` and a zero word: 7 pointers `table, ids, offs, slots, x, y0, y1`
//! then `m k u c xdiv xsrc xrs ys n_experts` as u32 and a zero u32; block 32 = one wave32, grid `(M, U, 1)`) and compares every output
//! word bit-for-bit (signed zeros and NaN payloads included) with CPU oracles:
//!
//! * W2 (`mq2i4`, `MqI4Profile::MQ2_R256` / `MQ2_R128`, levels 3, zero-scale code 2):
//!   `mq_i4_reference::decode_moe_slots` (cross-checked against `decode_gemv_f32` on a bounded slot sample).
//! * MQ4 (`mq4sym`): `mq4_row_gate` / `mq4_row_down`, the audited legacy MQ4 arithmetic, preserved exactly
//!   from the gfx1201 oracle (`mq_i4_moe_gemv_byte_oracle.rs`; `mq_i4_reference` supports only W3/W2).
//!   `--old-kernel-parity` also runs the production HIP kernels (JIT through `rdna_compute::Gpu`, the existing
//!   gfx1100 MQ4 decode GEMVs) on all-MQ4 fixtures, bit-compares them with the oracle, and in `--timing` mode
//!   times them on the same shapes as the comparator.
//!
//! Weight placement (the kernel only sees raw pointers in the per-expert table, so placement is invisible to it):
//! * `vram`: `hipMalloc` slabs between random-size fillers.
//! * `host_mapped`: **true** `hipHostMalloc(hipHostMallocMapped)` + `hipHostGetDevicePointer` slabs through the
//!   `hip_bridge` API; the pointer attributes (`hipPointerGetAttributes`) are recorded as evidence.
//! * `split_vram_host`: alternating experts in VRAM / mapped host memory inside one launch.
//! Before `HipRuntime::load` the process arms `keep_host_memory_out_of_reclaim` (the production Qwen4 offload
//! configuration; `--no-reclaim-arm` disables) and the JSON records the switches.
//!
//! Negative controls (all REQUIRED to detect; any non-detection or oracle-transcription failure => rc 1):
//! * `output_bit_flip` / `output_unwritten_word` / `pad_bit_flip`: a downloaded GPU buffer is corrupted by one
//!   bit / replaced by the 0xa5 sentinel / one pad word is flipped; the comparator MUST report exactly one word.
//! * `wrong_arithmetic`: mutated CPU oracles (unfused mul+add, reversed contraction orientation, sequential lane
//!   fold, left-fold of the R256 chains, ignored MQ4 zero point, W2 evaluated with W3 levels) MUST differ from
//!   the GPU words; the un-mutated transcription MUST equal the oracle first, and each mutant MUST differ from
//!   the oracle (fixture sensitivity).
//! * `weight_corruption`: scales/codes of one W2 and one MQ4 expert are rewritten IN PLACE in the placement memory
//!   (VRAM, mapped host or split), the kernels are relaunched: GPU MUST equal the oracle of the corrupted weights
//!   bit-for-bit, MUST differ from the oracle of the original weights in exactly the corrupted experts' slots, and
//!   after restoring the weights it MUST equal the original oracle again (proves the kernels read the placement).
//!
//! Build (needs `--features lab`, the example's Cargo `required-features`):
//!   cargo build -p hipfire-runtime --release --features lab --example mq_i4_moe_gemv_gfx1100_g1
//! G1 (xtx; synthetic cases, then the C24Pack real sample dir):
//!   target/release/examples/mq_i4_moe_gemv_gfx1100_g1 --codeobject /abs/mq_i4_moe_gemv_pm_gfx1100.hxaco \
//!     [--samples-dir /home/kaden/qcal/release-0.4.3/consumer24/artifacts/c24-samples] [--old-kernel-parity] \
//!     [--device N] [--card xtx] [--expect-uuid GPU-… --expect-bdf 0000:… --expect-arch gfx1100] [--only SUBSTR]… \
//!     [--no-reclaim-arm]
//!   `--only` matches `<case>@<placement>` and is a diagnostic filter: the full gate coverage check is waived.
//!   `--mq4-only` (MQ4 regression bundle; mutually exclusive with `--only`, so full gate coverage is ALWAYS enforced):
//!   the W2 symbols are never looked up or loaded, every synthetic expert is MQ4 (`all_w2` cases are skipped), real
//!   samples use only the `hot` MQ4 experts (cold W2 samples are counted and skipped, never validated or launched),
//!   and the per-codec loops (launch, controls, weight corruption, timing) see MQ4 only.
//! Sample dir schema (C24Pack, mirrors `extract_samples.py`): `sample.json` =
//!   `{seed, artifact_path, artifact_sha256, [provisional, layers, layout_note], samples:[{layer, expert, hot,
//!   tensors:[{kind: gate_up_proj|down_proj, qt, shape, group, file, sha256, artifact_offset, bytes}]}]}` plus
//!   the raw per-tensor files. Cold experts are W2 (qt 133/134, group 128), hot experts MQ4 (qt 44 g256 / 53 g128).
//!   Every file's sha256/size/qt/group/shape is verified strictly. A manifest with null `artifact_sha256` or
//!   `provisional: true` is labelled provisional in the JSON (not a final artifact binding).
//! Timing (separate short fabric run after G1 passed; raw device-event samples, VRAM and host-mapped, optional
//!   old MQ4 HIP comparator):
//!   … --codeobject … --timing --reps 20 --warmup 3 [--launches 8] [--old-kernel-parity] --card xtx \
//!     [--fabric-owner OWNER --fabric-wrapper /path/fabric-run.sh --fabric-card NAME]
//! Prints one machine JSON line on stdout; exit code 1 on any mismatch, control failure or error.
use hip_bridge::{DeviceBuffer, Function, HipRuntime, MemoryType, HIP_HOST_MALLOC_MAPPED};
use hipfire_runtime::mq_i4::{self, MqI4Profile, MqI4Rotation, MqI4Tensor};
use hipfire_runtime::mq_i4_reference::{self, DecodeKind, ExpertTensor};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::c_void;
use std::path::Path;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};

type Result<T, E = String> = std::result::Result<T, E>;
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

const SCHEMA: &str = "mq-i4-moe-gemv-gfx1100-g1-v1";
const MODULE: &str = "mq_i4_moe_gemv_pm_gfx1100";
const BUILD_COMMAND: &str =
    "cargo build -p hipfire-runtime --release --features lab --example mq_i4_moe_gemv_gfx1100_g1";
const DEFAULT_UUID: &str = "GPU-43390a851e296ee5";
const DEFAULT_BDF: &str = "0000:67:00.0";
const DEFAULT_ARCH: &str = "gfx1100";
const DEFAULT_CARD: &str = "xtx";
const DEFAULT_SAMPLES_DIR: &str = "/home/kaden/qcal/release-0.4.3/consumer24/artifacts/c24-samples";
const LINK_CAVEAT: &str = "The XTX endpoint is x16 but its upstream root port 0000:00:03.1 is PCIe 4.0 x4 \
(16 GT/s, ~7.9 GB/s theoretical, ~25 GB/s on an x16 desktop slot): every host-mapped latency/bandwidth number \
carries this limitation and is pessimistic for an ordinary desktop. VRAM-resident numbers are unaffected.";
const KERNARG_BYTES: usize = 96;
const N_EXPERTS: usize = 512;
/// Expert ids the kernel must reject without a table load or a store: `-1`, the table extent, `INT_MAX`, `INT_MIN`.
const BAD_IDS: [u32; 4] = [u32::MAX, N_EXPERTS as u32, i32::MAX as u32, i32::MIN as u32];
const GATE_M: usize = 1280;
const GATE_K: usize = 2560;
const DOWN_M: usize = 2560;
const DOWN_K: usize = 640;
/// Unwritten output words: the byte pattern `memset(0xa5)` leaves (a NaN, compared by bits).
const SENTINEL_BITS: u32 = 0xa5a5_a5a5;
const PAD_BITS: u32 = 0x7fc0_0000;
/// `--mq4-only`: process-wide because codec choice reaches case construction, loading and the launch loops.
static MQ4_ONLY: AtomicBool = AtomicBool::new(false);
fn mq4_only() -> bool {
    MQ4_ONLY.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------------------
// Axes
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Codec {
    W2,
    Mq4,
}
impl Codec {
    const ALL: [Codec; 2] = [Codec::W2, Codec::Mq4];
    /// Codecs of this run: MQ4 alone under `--mq4-only`.
    fn active() -> Vec<Codec> {
        if mq4_only() {
            vec![Codec::Mq4]
        } else {
            Codec::ALL.to_vec()
        }
    }
    fn tag(self) -> &'static str {
        match self {
            Codec::W2 => "mq2i4",
            Codec::Mq4 => "mq4sym",
        }
    }
    fn label(self) -> &'static str {
        match self {
            Codec::W2 => "w2",
            Codec::Mq4 => "mq4",
        }
    }
    /// Codec name of the timing records.
    fn timing_name(self) -> &'static str {
        match self {
            Codec::W2 => "mq2i4",
            Codec::Mq4 => "mq4",
        }
    }
    fn idx(self) -> usize {
        self as usize
    }
}

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
    fn timing_profile(self) -> &'static str {
        match self {
            Kind::GateUp => "gate_up",
            Kind::Down => "down",
        }
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
    /// Rows written to each of Y0/Y1 per slot.
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
    fn decode_kind(self) -> DecodeKind {
        match self {
            Kind::GateUp => DecodeKind::GateUp,
            Kind::Down => DecodeKind::Down,
        }
    }
    fn w2_profile(self) -> MqI4Profile {
        match self {
            Kind::GateUp => MqI4Profile::MQ2_R256,
            Kind::Down => MqI4Profile::MQ2_R128,
        }
    }
    /// Bytes per output row of an MQ4 tensor.
    fn mq4_row_bytes(self) -> usize {
        match self {
            Kind::GateUp => (GATE_K / 256) * 136,
            Kind::Down => (DOWN_K / 128) * 68,
        }
    }
    /// Bytes of one expert's projection tensor.
    fn tensor_bytes(self, codec: Codec) -> usize {
        match codec {
            Codec::W2 => mq_i4::tensor_bytes(self.w2_profile(), self.m(), self.k())
                .expect("FN shapes are legal W2 shapes"),
            Codec::Mq4 => match self {
                Kind::GateUp => GATE_M * (GATE_K / 256) * 136,
                Kind::Down => DOWN_M * (DOWN_K / 128) * 68,
            },
        }
    }
}
fn symbol(codec: Codec, kind: Kind) -> String {
    format!(
        "mq_i4_moe_gemv_{}_{}_pm_gfx1100",
        codec.tag(),
        kind.suffix()
    )
}

/// Where the expert weight slabs live.
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
fn f16f(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}
/// Symmetric MQ4 zero: `f16(-8 * sc)` (exact for every finite `sc <= 0x6fff`).
fn sym_zero_bits(sc: u16) -> u16 {
    half::f16::from_f32(-8.0 * f16f(sc)).to_bits()
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
fn bits_hex(v: u32) -> String {
    format!("0x{v:08x}")
}
fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
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

// ---------------------------------------------------------------------------------------
// Oracle mutations (negative controls only; `Mut::None` is the exact audited arithmetic)
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mut {
    None,
    /// Separate multiply and add instead of fused multiply-add.
    Unfused,
    /// Contraction chain started from the other end.
    Orientation,
    /// Lanes summed left-to-right instead of the 16,8,4,2,1 shuffle tree.
    LaneFoldSeq,
    /// R256 four chains folded `((a0+a1)+a2)+a3` instead of `(a0+a1)+(a2+a3)`.
    ChainFoldLeft,
    /// MQ4: the stored zero point is ignored.
    ZeroPointIgnored,
    /// W2 evaluated with the W3 weight levels (`2c-7`).
    WrongLevels,
}
impl Mut {
    const ARITH: [Mut; 6] = [
        Mut::Unfused,
        Mut::Orientation,
        Mut::LaneFoldSeq,
        Mut::ChainFoldLeft,
        Mut::ZeroPointIgnored,
        Mut::WrongLevels,
    ];
    fn name(self) -> &'static str {
        match self {
            Mut::None => "none",
            Mut::Unfused => "unfused_mul_add",
            Mut::Orientation => "reversed_contraction_orientation",
            Mut::LaneFoldSeq => "sequential_lane_fold",
            Mut::ChainFoldLeft => "left_fold_r256_chains",
            Mut::ZeroPointIgnored => "ignored_zero_point",
            Mut::WrongLevels => "w3_levels_for_w2",
        }
    }
    fn applies(self, codec: Codec, kind: Kind) -> bool {
        match self {
            Mut::ChainFoldLeft => kind == Kind::GateUp,
            Mut::ZeroPointIgnored => codec == Codec::Mq4,
            Mut::WrongLevels => codec == Codec::W2,
            _ => true,
        }
    }
}

/// Wave32 shuffle-down tree 16,8,4,2,1; lane 0 is the result.
fn lane_tree(mut lanes: [f32; 32]) -> f32 {
    for off in [16usize, 8, 4, 2, 1] {
        let snap = lanes;
        for l in 0..32 - off {
            lanes[l] = snap[l] + snap[l + off];
        }
    }
    lanes[0]
}
fn fold_lanes(lanes: [f32; 32], m: Mut) -> f32 {
    if m == Mut::LaneFoldSeq {
        let mut s = lanes[0];
        for l in lanes.iter().skip(1) {
            s += *l;
        }
        s
    } else {
        lane_tree(lanes)
    }
}
fn fold_chains(acc: [f32; 4], m: Mut) -> f32 {
    if m == Mut::ChainFoldLeft {
        ((acc[0] + acc[1]) + acc[2]) + acc[3]
    } else {
        (acc[0] + acc[1]) + (acc[2] + acc[3])
    }
}

/// One lane's eight-element group dot: `d_i = fma(sc, q_i, zp)`; `RN(d1*x1)`, `fma(d0,x0)`,
/// then `fma(d_i, x_i)` for i = 2..8 (the "element 1 first" contraction of the legacy MQ4 kernels).
fn group_dot8(sc: f32, zp: f32, pk: u32, x: &[f32], m: Mut) -> f32 {
    let zp = if m == Mut::ZeroPointIgnored { 0.0 } else { zp };
    let d = |i: usize| {
        let q = ((pk >> (4 * i)) & 15) as f32;
        if m == Mut::Unfused {
            sc * q + zp
        } else {
            sc.mul_add(q, zp)
        }
    };
    match m {
        Mut::Orientation => {
            let mut s = d(0) * x[0];
            for i in 1..8 {
                s = d(i).mul_add(x[i], s);
            }
            s
        }
        Mut::Unfused => {
            let mut s = d(1) * x[1];
            s += d(0) * x[0];
            for i in 2..8 {
                s += d(i) * x[i];
            }
            s
        }
        _ => {
            let mut s = d(1) * x[1];
            s = d(0).mul_add(x[0], s);
            for i in 2..8 {
                s = d(i).mul_add(x[i], s);
            }
            s
        }
    }
}

/// MQ4G256V2 gate/up row: `row.len() / 136` groups of `[sc0 zp0 sc1 zp1 : f16][128 nibble bytes]`.
/// Lane `tid` reads header half `tid >= 16`, u32 `tid` of the payload (elements `tid*8..+8`).
fn mq4_row_gate_m(row: &[u8], x: &[f32], m: Mut) -> f32 {
    let groups = row.len() / 136;
    debug_assert_eq!(x.len(), groups * 256);
    let mut lanes = [0.0f32; 32];
    for (tid, lane) in lanes.iter_mut().enumerate() {
        let hoff = if tid < 16 { 0 } else { 4 };
        let mut acc = [0.0f32; 4];
        for g in 0..groups {
            let gp = &row[g * 136..(g + 1) * 136];
            let sc = f16f(u16::from_le_bytes([gp[hoff], gp[hoff + 1]]));
            let zp = f16f(u16::from_le_bytes([gp[hoff + 2], gp[hoff + 3]]));
            let pk = u32::from_le_bytes(gp[8 + tid * 4..12 + tid * 4].try_into().unwrap());
            let xb = g * 256 + tid * 8;
            acc[g & 3] += group_dot8(sc, zp, pk, &x[xb..xb + 8], m);
        }
        *lane = fold_chains(acc, m);
    }
    fold_lanes(lanes, m)
}
fn mq4_row_gate(row: &[u8], x: &[f32]) -> f32 {
    mq4_row_gate_m(row, x, Mut::None)
}

/// MQ4G128V2 down row: `row.len() / 68` groups of `[sc zp : f16][64 nibble bytes]`; lane `tid`
/// owns bytes `2*tid, 2*tid+1` (elements `tid*4..+4`), one accumulation chain.
fn mq4_row_down_m(row: &[u8], x: &[f32], m: Mut) -> f32 {
    let groups = row.len() / 68;
    debug_assert_eq!(x.len(), groups * 128);
    let mut lanes = [0.0f32; 32];
    for (tid, lane) in lanes.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for g in 0..groups {
            let gp = &row[g * 68..(g + 1) * 68];
            let sc = f16f(u16::from_le_bytes([gp[0], gp[1]]));
            let zp = f16f(u16::from_le_bytes([gp[2], gp[3]]));
            let zp = if m == Mut::ZeroPointIgnored { 0.0 } else { zp };
            let (b0, b1) = (gp[4 + tid * 2], gp[5 + tid * 2]);
            let q = [b0 & 15, b0 >> 4, b1 & 15, b1 >> 4];
            let d = |i: usize| {
                if m == Mut::Unfused {
                    sc * q[i] as f32 + zp
                } else {
                    sc.mul_add(q[i] as f32, zp)
                }
            };
            let xb = g * 128 + tid * 4;
            let s = match m {
                Mut::Orientation => {
                    let mut s = d(0) * x[xb];
                    for i in 1..4 {
                        s = d(i).mul_add(x[xb + i], s);
                    }
                    s
                }
                Mut::Unfused => {
                    let mut s = d(1) * x[xb + 1];
                    s += d(0) * x[xb];
                    s += d(2) * x[xb + 2];
                    s += d(3) * x[xb + 3];
                    s
                }
                _ => {
                    let mut s = d(1) * x[xb + 1];
                    s = d(0).mul_add(x[xb], s);
                    s = d(2).mul_add(x[xb + 2], s);
                    s = d(3).mul_add(x[xb + 3], s);
                    s
                }
            };
            acc += s;
        }
        *lane = acc;
    }
    fold_lanes(lanes, m)
}
fn mq4_row_down(row: &[u8], x: &[f32]) -> f32 {
    mq4_row_down_m(row, x, Mut::None)
}

/// Local transcription of the frozen F4 decode tree (`mq_i4_reference::tree_row`, private) on unpacked
/// codes/scales with optional mutations; `Mut::None` MUST equal the reference bit-for-bit (checked live).
fn ref_row_m(profile: MqI4Profile, codes: &[u8], scales: &[u16], x: &[f32], m: Mut) -> f32 {
    let levels: i32 = if m == Mut::WrongLevels {
        7
    } else {
        profile.codec.levels() as i32
    };
    let four = profile.rotation == MqI4Rotation::R256;
    let groups = scales.len();
    let mut lanes = [0.0f32; 32];
    for (lane, slot) in lanes.iter_mut().enumerate() {
        let mut acc = [0.0f32; 4];
        for g in 0..groups {
            let d = f16f(scales[g]);
            let j = g * 128 + 4 * lane;
            let w = |i: usize| d * ((2 * (codes[j + i] as i32) - levels) as f32);
            let p = match m {
                Mut::Orientation => {
                    let mut p = w(3) * x[j + 3];
                    for i in (0..3).rev() {
                        p = w(i).mul_add(x[j + i], p);
                    }
                    p
                }
                Mut::Unfused => {
                    let mut p = w(0) * x[j];
                    for i in 1..4 {
                        p += w(i) * x[j + i];
                    }
                    p
                }
                _ => {
                    let mut p = w(0) * x[j];
                    for i in 1..4 {
                        p = w(i).mul_add(x[j + i], p);
                    }
                    p
                }
            };
            acc[if four { g & 3 } else { 0 }] += p;
        }
        *slot = if four { fold_chains(acc, m) } else { acc[0] };
    }
    fold_lanes(lanes, m)
}

/// Bucketed MQ4 decode with the `decode_moe_slots` contract (slot `s` reads X row `s/div`, writes
/// `Y[s*ys + r]`; gate rows `>= m/2` go to Y1).
#[allow(clippy::too_many_arguments)]
fn mq4_decode_moe_slots(
    kind: Kind,
    experts: &[Option<&[u8]>],
    b: &Bucket,
    x: &[f32],
    xrs: usize,
    ys: usize,
    y0: &mut [f32],
    mut y1: Option<&mut [f32]>,
) {
    let (m, k, out) = (kind.m(), kind.k(), kind.out_rows());
    let row_bytes = kind.mq4_row_bytes();
    for (u, &id) in b.ids.iter().enumerate() {
        let bytes = experts[id as usize].expect("MQ4 expert present");
        for &slot in &b.slots[b.offs[u] as usize..b.offs[u + 1] as usize] {
            let slot = slot as usize;
            let xs = &x[(slot / kind.x_row_div()) * xrs..][..k];
            let base = slot * ys;
            for r in 0..m {
                let row = &bytes[r * row_bytes..(r + 1) * row_bytes];
                let v = match kind {
                    Kind::GateUp => mq4_row_gate(row, xs),
                    Kind::Down => mq4_row_down(row, xs),
                };
                if r < out {
                    y0[base + r] = v;
                } else {
                    y1.as_deref_mut().expect("gate has Y1")[base + r - out] = v;
                }
            }
        }
    }
}

/// Encode an MQ4 projection tensor (legacy 136 B / 68 B group layout, symmetric headers).
/// `codes` is one nibble per weight (`m*k`), `scales` one f16 per 128 columns (`m*k/128`).
fn encode_mq4(kind: Kind, m: usize, k: usize, codes: &[u8], scales: &[u16]) -> Vec<u8> {
    assert_eq!(codes.len(), m * k);
    assert_eq!(scales.len(), m * (k / 128));
    let pack = |dst: &mut [u8], base: usize| {
        for (j, byte) in dst.iter_mut().enumerate() {
            *byte = (codes[base + 2 * j] & 15) | ((codes[base + 2 * j + 1] & 15) << 4);
        }
    };
    match kind {
        Kind::GateUp => {
            let groups = k / 256;
            let mut out = vec![0u8; m * groups * 136];
            for r in 0..m {
                for g in 0..groups {
                    let off = (r * groups + g) * 136;
                    for half in 0..2 {
                        let sc = scales[r * (k / 128) + 2 * g + half];
                        out[off + 4 * half..off + 4 * half + 2].copy_from_slice(&sc.to_le_bytes());
                        out[off + 4 * half + 2..off + 4 * half + 4]
                            .copy_from_slice(&sym_zero_bits(sc).to_le_bytes());
                    }
                    pack(&mut out[off + 8..off + 136], r * k + g * 256);
                }
            }
            out
        }
        Kind::Down => {
            let groups = k / 128;
            let mut out = vec![0u8; m * groups * 68];
            for r in 0..m {
                for g in 0..groups {
                    let off = (r * groups + g) * 68;
                    let sc = scales[r * groups + g];
                    out[off..off + 2].copy_from_slice(&sc.to_le_bytes());
                    out[off + 2..off + 4].copy_from_slice(&sym_zero_bits(sc).to_le_bytes());
                    pack(&mut out[off + 4..off + 68], r * k + g * 128);
                }
            }
            out
        }
    }
}

// ---------------------------------------------------------------------------------------
// Synthetic expert + activation fixtures
// ---------------------------------------------------------------------------------------

struct Expert {
    codec: Codec,
    gate: Vec<u8>,
    down: Vec<u8>,
    gate_sha: [u8; 32],
    down_sha: [u8; 32],
}

/// Scale bits for a row/group class. W2 may use the full f16 max 0x7bff; MQ4 (symmetric zero `-8*sc`)
/// tops out at 0x6fff. Class 3 is the zero scale (codes forced to the zero-scale contract: W2 code 2).
fn class_scale(class: usize, codec: Codec, rng: &mut Rng) -> u16 {
    let max = match codec {
        Codec::W2 => 0x7bff,
        Codec::Mq4 => 0x6fff,
    };
    match class {
        1 => max,
        2 => 1 + rng.below(0x3ff) as u16, // f16 subnormal
        3 => 0,
        4 => 0x0400, // smallest normal
        5 => 0x0001, // smallest subnormal
        6 => 0x3c00,
        _ => (((8 + rng.below(13)) as u16) << 10) | (rng.u32() as u16 & 0x3ff),
    }
}

/// Deterministic expert tensor for `(id, codec, kind)`. Every non-plain expert rotates the code
/// so each code value lands on every in-group position across rows, and spreads zero, f16-max,
/// subnormal and smallest-normal scales over (row, group).
fn gen_tensor(codec: Codec, kind: Kind, id: u32, special: bool) -> Vec<u8> {
    let (m, k) = (kind.m(), kind.k());
    let groups = k / 128;
    let levels: usize = match codec {
        Codec::W2 => 3,
        Codec::Mq4 => 15,
    };
    let zero_code: u8 = match codec {
        Codec::W2 => 2,
        Codec::Mq4 => 0,
    };
    let mut rng = Rng::new(mix64(id as u64 * 8 + kind as u64 * 2 + codec as u64 + 1));
    let salt = id as usize;
    let mut codes = vec![0u8; m * k];
    let mut scales = vec![0u16; m * groups];
    for r in 0..m {
        for g in 0..groups {
            let class = if special {
                (r * 5 + g * 3 + salt) % 10
            } else {
                0
            };
            let sc = class_scale(class, codec, &mut rng);
            scales[r * groups + g] = sc;
            let base = r * k + g * 128;
            let rotate = special && (r + g) % 2 == 0;
            for j in 0..128 {
                codes[base + j] = if sc == 0 {
                    zero_code
                } else if rotate {
                    ((j + r + g + salt) % (levels + 1)) as u8
                } else {
                    ((rng.u32() as usize) & levels) as u8
                };
            }
        }
    }
    match codec {
        Codec::W2 => mq_i4::encode_tensor(kind.w2_profile(), m, k, &codes, &scales)
            .expect("synthetic W2 tensor is valid"),
        Codec::Mq4 => encode_mq4(kind, m, k, &codes, &scales),
    }
}

fn gen_expert(id: u32, codec: Codec) -> Expert {
    let special = id % 3 != 2;
    let gate = gen_tensor(codec, Kind::GateUp, id, special);
    let down = gen_tensor(codec, Kind::Down, id, special);
    Expert {
        codec,
        gate_sha: sha(&gate),
        down_sha: sha(&down),
        gate,
        down,
    }
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
// Cases and buckets
// ---------------------------------------------------------------------------------------

#[derive(Clone)]
struct Case {
    name: String,
    origin: &'static str,
    c: usize,
    /// `route[lane*10 + rank]` = expert id.
    route: Vec<u32>,
    experts: BTreeMap<u32, Rc<Expert>>,
    x_gate: Vec<f32>,
    x_down: Vec<f32>,
    xrs_gate: usize,
    xrs_down: usize,
    ys_gate: usize,
    ys_down: usize,
    old_parity: bool,
}

impl Case {
    fn x(&self, kind: Kind) -> (&[f32], usize, usize) {
        match kind {
            Kind::GateUp => (&self.x_gate, self.xrs_gate, self.ys_gate),
            Kind::Down => (&self.x_down, self.xrs_down, self.ys_down),
        }
    }
    fn tight(&self) -> bool {
        self.xrs_gate == GATE_K
            && self.xrs_down == DOWN_K
            && self.ys_gate == GATE_M / 2
            && self.ys_down == DOWN_M
    }
    fn codec_of_slot(&self, slot: usize) -> Codec {
        self.experts[&self.route[slot]].codec
    }
    fn input_sha256(&self) -> String {
        let mut h = Sha256::new();
        h.update(self.name.as_bytes());
        h.update((self.c as u64).to_le_bytes());
        h.update(u32_bytes(&self.route));
        for (id, e) in &self.experts {
            h.update(id.to_le_bytes());
            h.update(e.codec.tag().as_bytes());
            h.update(e.gate_sha);
            h.update(e.down_sha);
        }
        for v in [self.xrs_gate, self.xrs_down, self.ys_gate, self.ys_down] {
            h.update((v as u64).to_le_bytes());
        }
        h.update(f32_bytes(&self.x_gate));
        h.update(f32_bytes(&self.x_down));
        hex(&h.finalize())
    }
}

/// Activations: per-lane (gate) / per-slot (down) style rotation; padded strides are poisoned
/// with NaN so any read outside a row's `K` words is visible.
#[allow(clippy::too_many_arguments)]
fn build_case(
    name: &str,
    origin: &'static str,
    c: usize,
    route: Vec<u32>,
    experts: BTreeMap<u32, Rc<Expert>>,
    padded: bool,
    old_parity: bool,
    xshift: usize,
) -> Case {
    assert_eq!(route.len(), 10 * c);
    let (xrs_gate, ys_gate, xrs_down, ys_down) = if padded {
        (GATE_K + 64, GATE_M / 2 + 32, DOWN_K + 64, DOWN_M + 32)
    } else {
        (GATE_K, GATE_M / 2, DOWN_K, DOWN_M)
    };
    let mut rng = Rng::new(name_seed(name));
    let fill = |rows: usize, k: usize, xrs: usize, rng: &mut Rng, shift: usize| {
        let mut x = vec![f32::from_bits(PAD_BITS); rows * xrs];
        for r in 0..rows {
            let row = gen_x_row(r + shift, k, rng);
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
        experts,
        x_gate,
        x_down,
        xrs_gate,
        xrs_down,
        ys_gate,
        ys_down,
        old_parity,
    }
}

#[derive(Clone, Default, Debug, PartialEq, Eq)]
struct Bucket {
    ids: Vec<u32>,
    offs: Vec<u32>,
    slots: Vec<u32>,
}

impl Bucket {
    /// The bucket without its first `n` experts and their slots.
    fn without_first(&self, n: usize) -> Bucket {
        let cut = self.offs[n];
        Bucket {
            ids: self.ids[n..].to_vec(),
            offs: self.offs[n..].iter().map(|o| o - cut).collect(),
            slots: self.slots[cut as usize..].to_vec(),
        }
    }
}

/// Per-codec descriptor: ascending expert ids, slots in stable original order per expert,
/// offsets spanning only this bucket's slots.
fn build_bucket(route: &[u32], codec_of: &dyn Fn(u32) -> Codec, codec: Codec) -> Bucket {
    let ids: BTreeSet<u32> = route
        .iter()
        .copied()
        .filter(|&e| codec_of(e) == codec)
        .collect();
    let mut b = Bucket::default();
    for id in ids {
        b.ids.push(id);
        b.offs.push(b.slots.len() as u32);
        for (s, &e) in route.iter().enumerate() {
            if e == id {
                b.slots.push(s as u32);
            }
        }
    }
    b.offs.push(b.slots.len() as u32);
    b
}

fn buckets_of(case: &Case) -> [Bucket; 2] {
    let f = |e: u32| case.experts[&e].codec;
    [
        build_bucket(&case.route, &f, Codec::W2),
        build_bucket(&case.route, &f, Codec::Mq4),
    ]
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

fn experts_for(ids: &[u32], codec_of: impl Fn(usize, u32) -> Codec) -> BTreeMap<u32, Rc<Expert>> {
    ids.iter()
        .enumerate()
        .map(|(i, &id)| {
            let codec = if mq4_only() {
                Codec::Mq4
            } else {
                codec_of(i, id)
            };
            (id, Rc::new(gen_expert(id, codec)))
        })
        .collect()
}

const MIXED10: [Codec; 10] = [
    Codec::W2,
    Codec::Mq4,
    Codec::Mq4,
    Codec::W2,
    Codec::W2,
    Codec::Mq4,
    Codec::W2,
    Codec::Mq4,
    Codec::Mq4,
    Codec::W2,
];

fn lane_routes(rng: &mut Rng, pool: &[u32], c: usize) -> Vec<u32> {
    let mut route = Vec::with_capacity(10 * c);
    for _ in 0..c {
        let mut p = pool.to_vec();
        rng.shuffle(&mut p);
        route.extend_from_slice(&p[..10]);
    }
    route
}

/// Random codec per id with `w2_tenths`/10 W2 probability; both codecs forced present when mixed.
fn mixed_codecs(rng: &mut Rng, n: usize, w2_tenths: usize) -> Vec<Codec> {
    let mut codecs: Vec<Codec> = (0..n)
        .map(|_| {
            if rng.below(10) < w2_tenths {
                Codec::W2
            } else {
                Codec::Mq4
            }
        })
        .collect();
    if w2_tenths > 0 && w2_tenths < 10 {
        codecs[0] = Codec::W2;
        codecs[n - 1] = Codec::Mq4;
    }
    codecs
}

const SYNTH_CASES: &[&str] = &[
    "c1_mixed_interleaved",
    "c1_mixed_stress_x",
    "c1_all_w2",
    "c1_all_mq4",
    "c1_mixed_1w2_9mq4",
    "c1_mixed_9w2_1mq4",
    "c3_mixed_partial_lanes",
    "c8_union_repeated_mixed",
    "c8_unique_80_mixed",
    "c8_identical_lanes_mixed",
    "c8_union_all_w2",
    "c8_union_all_mq4",
    "c8_padded_strides_mixed",
];
/// Cases that additionally run from true host-mapped weights.
const HOST_MAPPED_CASES: &[&str] = &[
    "c1_mixed_interleaved",
    "c1_all_w2",
    "c1_all_mq4",
    "c3_mixed_partial_lanes",
    "c8_union_repeated_mixed",
    "c8_padded_strides_mixed",
];
/// Cases that additionally run with alternating VRAM / mapped-host experts inside one launch.
const SPLIT_CASES: &[&str] = &["c1_mixed_interleaved", "c8_union_repeated_mixed"];
/// Cases carrying the output / pad / wrong-arithmetic negative controls (VRAM placement).
const CONTROL_CASES: &[&str] = &["c1_mixed_interleaved", "c8_padded_strides_mixed"];
/// Cases carrying the in-place weight-corruption control (every placement of the case).
const TAMPER_CASES: &[&str] = &["c1_mixed_interleaved"];

fn build_synthetic(name: &str) -> Case {
    let mut rng = Rng::new(name_seed(name) ^ 0x5eed);
    match name {
        "c1_mixed_interleaved" | "c1_mixed_stress_x" => {
            let ids = pick_ids(&mut rng, 10, true);
            let experts = experts_for(&ids, |i, _| MIXED10[i]);
            let mut route = ids.clone();
            rng.shuffle(&mut route);
            let shift = if name == "c1_mixed_stress_x" { 1 } else { 0 };
            build_case(name, "synthetic", 1, route, experts, false, false, shift)
        }
        "c1_all_w2" | "c1_all_mq4" => {
            let codec = if name == "c1_all_w2" {
                Codec::W2
            } else {
                Codec::Mq4
            };
            let ids = pick_ids(&mut rng, 10, true);
            let experts = experts_for(&ids, |_, _| codec);
            let mut route = ids.clone();
            rng.shuffle(&mut route);
            build_case(
                name,
                "synthetic",
                1,
                route,
                experts,
                false,
                codec == Codec::Mq4,
                0,
            )
        }
        "c1_mixed_1w2_9mq4" | "c1_mixed_9w2_1mq4" => {
            let ids = pick_ids(&mut rng, 10, true);
            let single_w2 = name == "c1_mixed_1w2_9mq4";
            // Never an edge id, so the edge experts keep the majority codec.
            let lone = 1 + rng.below(8);
            let experts = experts_for(&ids, |i, _| {
                if (i == lone) == single_w2 {
                    Codec::W2
                } else {
                    Codec::Mq4
                }
            });
            let mut route = ids.clone();
            rng.shuffle(&mut route);
            build_case(name, "synthetic", 1, route, experts, false, false, 2)
        }
        "c3_mixed_partial_lanes" => {
            let ids = pick_ids(&mut rng, 12, false);
            let codecs = mixed_codecs(&mut rng, ids.len(), 5);
            let experts = experts_for(&ids, |i, _| codecs[i]);
            let route = lane_routes(&mut rng, &ids, 3);
            build_case(name, "synthetic", 3, route, experts, false, false, 0)
        }
        "c8_union_repeated_mixed"
        | "c8_union_all_w2"
        | "c8_union_all_mq4"
        | "c8_padded_strides_mixed" => {
            let ids = pick_ids(&mut rng, 14, true);
            let codecs: Vec<Codec> = match name {
                "c8_union_all_w2" => vec![Codec::W2; ids.len()],
                "c8_union_all_mq4" => vec![Codec::Mq4; ids.len()],
                _ => mixed_codecs(&mut rng, ids.len(), 5),
            };
            let experts = experts_for(&ids, |i, _| codecs[i]);
            let route = lane_routes(&mut rng, &ids, 8);
            let padded = name == "c8_padded_strides_mixed";
            build_case(
                name,
                "synthetic",
                8,
                route,
                experts,
                padded,
                name == "c8_union_all_mq4",
                0,
            )
        }
        "c8_unique_80_mixed" => {
            let ids = pick_ids(&mut rng, 80, true);
            let codecs = mixed_codecs(&mut rng, ids.len(), 5);
            let experts = experts_for(&ids, |i, _| codecs[i]);
            let mut route = ids.clone();
            rng.shuffle(&mut route);
            build_case(name, "synthetic", 8, route, experts, false, false, 3)
        }
        "c8_identical_lanes_mixed" => {
            let ids = pick_ids(&mut rng, 10, true);
            let experts = experts_for(&ids, |i, _| MIXED10[(i + 3) % 10]);
            let mut lane = ids.clone();
            rng.shuffle(&mut lane);
            let route: Vec<u32> = (0..8).flat_map(|_| lane.clone()).collect();
            build_case(name, "synthetic", 8, route, experts, false, false, 0)
        }
        other => panic!("unknown synthetic case {other}"),
    }
}

// ---------------------------------------------------------------------------------------
// Real c24 sample dir (C24Pack schema; cold = W2 qt133/134, hot = MQ4 qt44/53)
// ---------------------------------------------------------------------------------------

const SAMPLES_ORIGIN: &str = "c24_samples";

fn mq4_sym_violations(kind: Kind, bytes: &[u8]) -> u64 {
    let mut bad = 0u64;
    let (stride, hdrs): (usize, &[usize]) = match kind {
        Kind::GateUp => (136, &[0usize, 4][..]),
        Kind::Down => (68, &[0usize][..]),
    };
    for g in bytes.chunks_exact(stride) {
        for &h in hdrs {
            let sc = u16::from_le_bytes([g[h], g[h + 1]]);
            let zp = u16::from_le_bytes([g[h + 2], g[h + 3]]);
            let finite = sc & 0x7c00 != 0x7c00 && sc & 0x8000 == 0;
            // -8*sc must itself be a finite f16 and equal the stored zero (+-0 compare equal).
            if !finite || f16f(zp) != -8.0 * f16f(sc) || (-8.0 * f16f(sc)).abs() > 65504.0 {
                bad += 1;
            }
        }
    }
    bad
}

struct Samples {
    /// Key = sample ordinal (the kernel only indexes the pointer table by id; layer/expert are provenance).
    experts: BTreeMap<u32, Rc<Expert>>,
    meta: Value,
    mq4_violations: u64,
}

fn load_samples(dir: &str) -> Result<Samples> {
    let manifest_path = Path::new(dir).join("sample.json");
    let bytes =
        std::fs::read(&manifest_path).map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let manifest_sha = hex(&sha(&bytes));
    let root: Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let provisional = root
        .get("provisional")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        || root.get("artifact_sha256").is_none_or(Value::is_null);
    let list = root
        .get("samples")
        .and_then(Value::as_array)
        .ok_or_else(|| "sample.json: no `samples` array".to_string())?;
    if list.len() > N_EXPERTS {
        return Err(format!(
            "sample.json: {} samples exceed the {N_EXPERTS}-entry pointer table",
            list.len()
        ));
    }
    let mut experts = BTreeMap::new();
    let mut recs = Vec::new();
    let mut lines = String::new();
    let mut violations = 0u64;
    let mut skipped_cold = 0u64;
    for (i, s) in list.iter().enumerate() {
        let field_u64 = |k: &str| -> Result<u64> {
            s.get(k)
                .and_then(Value::as_u64)
                .ok_or_else(|| format!("samples[{i}]: missing integer `{k}`"))
        };
        let (layer, expert) = (field_u64("layer")?, field_u64("expert")?);
        let hot = s
            .get("hot")
            .and_then(Value::as_bool)
            .ok_or_else(|| format!("samples[{i}]: missing boolean `hot`"))?;
        if mq4_only() && !hot {
            skipped_cold += 1;
            continue;
        }
        let codec = if hot { Codec::Mq4 } else { Codec::W2 };
        let tensors = s
            .get("tensors")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("samples[{i}]: missing `tensors`"))?;
        let mut blobs: [Option<(Vec<u8>, String)>; 2] = [None, None];
        for t in tensors {
            let kind_name = t.get("kind").and_then(Value::as_str).unwrap_or("");
            let (kind, slot) = match kind_name {
                "gate_up_proj" => (Kind::GateUp, 0usize),
                "down_proj" => (Kind::Down, 1usize),
                other => {
                    return Err(format!(
                        "samples[{i}] L{layer}.E{expert}: unknown tensor kind `{other}`"
                    ))
                }
            };
            let who = format!("samples[{i}] L{layer}.E{expert}.{kind_name}");
            let want_qt: u64 = match (codec, kind) {
                (Codec::W2, Kind::GateUp) => 133,
                (Codec::W2, Kind::Down) => 134,
                (Codec::Mq4, Kind::GateUp) => 44,
                (Codec::Mq4, Kind::Down) => 53,
            };
            let want_group: u64 = match (codec, kind) {
                (Codec::W2, _) => 128,
                (Codec::Mq4, Kind::GateUp) => 256,
                (Codec::Mq4, Kind::Down) => 128,
            };
            if t.get("qt").and_then(Value::as_u64) != Some(want_qt) {
                return Err(format!(
                    "{who}: qt {:?} != {want_qt} (hot={hot})",
                    t.get("qt")
                ));
            }
            if t.get("group").and_then(Value::as_u64) != Some(want_group) {
                return Err(format!("{who}: group {:?} != {want_group}", t.get("group")));
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
            if raw.len() != kind.tensor_bytes(codec) {
                return Err(format!(
                    "{who}: {} bytes != {} for {}",
                    raw.len(),
                    kind.tensor_bytes(codec),
                    codec.label()
                ));
            }
            if let Some(b) = t.get("bytes").and_then(Value::as_u64) {
                if b != raw.len() as u64 {
                    return Err(format!("{who}: manifest bytes {b} != file {}", raw.len()));
                }
            }
            let got = hex(&sha(&raw));
            if t.get("sha256").and_then(Value::as_str) != Some(got.as_str()) {
                return Err(format!(
                    "{who}: sha256 {got} != manifest {:?}",
                    t.get("sha256")
                ));
            }
            blobs[slot] = Some((raw, got));
        }
        let (Some((gate, gs)), Some((down, ds))) = (blobs[0].take(), blobs[1].take()) else {
            return Err(format!(
                "samples[{i}] L{layer}.E{expert}: needs both gate_up_proj and down_proj"
            ));
        };
        match codec {
            Codec::W2 => {
                mq_i4::decode_tensor(MqI4Profile::MQ2_R256, GATE_M, GATE_K, &gate)
                    .map_err(|e| format!("samples[{i}] gate: {e}"))?;
                mq_i4::decode_tensor(MqI4Profile::MQ2_R128, DOWN_M, DOWN_K, &down)
                    .map_err(|e| format!("samples[{i}] down: {e}"))?;
            }
            Codec::Mq4 => {
                violations +=
                    mq4_sym_violations(Kind::GateUp, &gate) + mq4_sym_violations(Kind::Down, &down)
            }
        }
        lines.push_str(&format!("{} L{layer} E{expert} {gs} {ds}\n", codec.label()));
        recs.push(json!({"id": i, "layer": layer, "expert": expert, "hot": hot, "codec": codec.label(),
            "gate_bytes": gate.len(), "gate_sha256": gs, "down_bytes": down.len(), "down_sha256": ds,
            "artifact_offset": s.get("tensors").and_then(|t| t.get(0)).and_then(|t| t.get("artifact_offset")).cloned()}));
        let (gate_sha, down_sha) = (sha(&gate), sha(&down));
        experts.insert(
            i as u32,
            Rc::new(Expert {
                codec,
                gate,
                down,
                gate_sha,
                down_sha,
            }),
        );
    }
    let meta = json!({
        "dir": dir, "schema": "c24pack-sample-v1 (extract_samples.py compatible)",
        "sample_manifest_sha256": manifest_sha,
        "provisional": provisional,
        "provisional_note": if provisional { "manifest has null artifact_sha256/artifact_offset or provisional=true: raw tensors are verified strictly by sha256, but this is NOT a final artifact binding" } else { "final" },
        "artifact_path": root.get("artifact_path"), "artifact_sha256": root.get("artifact_sha256"),
        "seed": root.get("seed"), "layers": root.get("layers"), "layout_note": root.get("layout_note"),
        "n_samples": list.len(),
        "mq4_only": mq4_only(),
        "skipped_cold_w2_mq4_only": skipped_cold,
        "n_cold_w2": experts.values().filter(|e| e.codec == Codec::W2).count(),
        "n_hot_mq4": experts.values().filter(|e| e.codec == Codec::Mq4).count(),
        "mq4_sym_header_violations": violations,
        "selected_input_sha256": hex(&sha(lines.as_bytes())),
        "experts": recs,
    });
    Ok(Samples {
        experts,
        meta,
        mq4_violations: violations,
    })
}

fn spread(v: &[u32], n: usize) -> Vec<u32> {
    (0..n).map(|i| v[i * v.len() / n]).collect()
}

fn sample_cases(s: &Samples) -> Result<Vec<Case>> {
    let all: Vec<u32> = s.experts.keys().copied().collect();
    let by = |c: Codec| -> Vec<u32> {
        all.iter()
            .copied()
            .filter(|id| s.experts[id].codec == c)
            .collect()
    };
    let (w2, mq4) = (by(Codec::W2), by(Codec::Mq4));
    if all.len() < 10 || mq4.is_empty() || (!mq4_only() && w2.is_empty()) {
        return Err(format!(
            "sample dir needs >= 10 experts with both codecs (w2={}, mq4={}); a top-10 lane needs 10",
            w2.len(),
            mq4.len()
        ));
    }
    let mut rng = Rng::new(0x7ea1_1a7e);
    let sub = |ids: &[u32]| -> BTreeMap<u32, Rc<Expert>> {
        ids.iter().map(|id| (*id, s.experts[id].clone())).collect()
    };
    let mut cases = Vec::new();
    // C1 mixed: 10 experts, as balanced as the pool allows, route order shuffled.
    let mut ids: Vec<u32> = Vec::new();
    let (mut i2, mut i4) = (0usize, 0usize);
    while ids.len() < 10 {
        if (ids.len() % 2 == 0 && i2 < w2.len()) || i4 >= mq4.len() {
            ids.push(w2[i2]);
            i2 += 1;
        } else {
            ids.push(mq4[i4]);
            i4 += 1;
        }
    }
    let mut route1 = ids.clone();
    rng.shuffle(&mut route1);
    cases.push(build_case(
        "c24s_c1_mixed",
        SAMPLES_ORIGIN,
        1,
        route1,
        sub(&ids),
        false,
        false,
        0,
    ));
    let route8 = lane_routes(&mut rng, &all, 8);
    cases.push(build_case(
        "c24s_c8_union_mixed",
        SAMPLES_ORIGIN,
        8,
        route8,
        s.experts.clone(),
        false,
        false,
        1,
    ));
    for (codec, pool, name) in [
        (Codec::W2, &w2, "c24s_c1_all_w2"),
        (Codec::Mq4, &mq4, "c24s_c1_all_mq4"),
    ] {
        if pool.len() >= 10 {
            let mut pick = spread(pool, 10);
            rng.shuffle(&mut pick);
            cases.push(build_case(
                name,
                SAMPLES_ORIGIN,
                1,
                pick.clone(),
                sub(&pick),
                false,
                codec == Codec::Mq4,
                2,
            ));
        }
    }
    Ok(cases)
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
                // Host-resident memory must not be reported as device-local or unregistered.
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
    Vram(usize),
    Host(*mut u8),
}
struct Slab {
    gate_ptr: u64,
    down_ptr: u64,
    len: usize,
    loc: Loc,
}
struct Slabs {
    by_id: BTreeMap<u32, Slab>,
    host_slabs: usize,
    vram_slabs: usize,
    host_attr_failures: usize,
    /// Pointer-attribute evidence of the first host allocations.
    evidence: Vec<Value>,
}

/// One slab per expert (gate_up tensor then down tensor, as in the artifact), allocated in shuffled
/// order; VRAM slabs sit between random-size fillers so pointers are neither monotone nor uniformly
/// strided. Host slabs are true mapped host allocations.
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
        let len = e.gate.len() + e.down.len();
        let mut slab = Vec::with_capacity(len);
        slab.extend_from_slice(&e.gate);
        slab.extend_from_slice(&e.down);
        if placement.host_for(i) {
            let (hp, dp, mut ev) = dev.alloc_host(len)?;
            // SAFETY: `hp` is a live mapped host allocation of at least `len` bytes; `slab` is `len` bytes.
            unsafe { std::ptr::copy_nonoverlapping(slab.as_ptr(), hp, len) };
            if ev["ok"] != json!(true) {
                slabs.host_attr_failures += 1;
            }
            if slabs.evidence.len() < 3 {
                ev["expert"] = json!(id);
                slabs.evidence.push(ev);
            }
            slabs.host_slabs += 1;
            slabs.by_id.insert(
                id,
                Slab {
                    gate_ptr: dp,
                    down_ptr: dp + e.gate.len() as u64,
                    len,
                    loc: Loc::Host(hp),
                },
            );
        } else {
            if rng.below(3) != 0 {
                dev.alloc(256 * (1 + rng.below(300)))?;
            }
            let b = dev.upload(&slab)?;
            slabs.vram_slabs += 1;
            slabs.by_id.insert(
                id,
                Slab {
                    gate_ptr: b.ptr,
                    down_ptr: b.ptr + e.gate.len() as u64,
                    len,
                    loc: Loc::Vram(b.idx),
                },
            );
        }
    }
    Ok(slabs)
}

/// Rewrite one expert's slab in place in its placement memory (VRAM copy or direct host store).
fn overwrite_slab(dev: &Dev, slabs: &Slabs, id: u32, e: &Expert) -> Result<()> {
    let s = slabs
        .by_id
        .get(&id)
        .ok_or_else(|| format!("no slab for expert {id}"))?;
    let mut bytes = Vec::with_capacity(s.len);
    bytes.extend_from_slice(&e.gate);
    bytes.extend_from_slice(&e.down);
    if bytes.len() != s.len {
        return Err(format!(
            "expert {id}: slab length changed ({} != {})",
            bytes.len(),
            s.len
        ));
    }
    match s.loc {
        Loc::Vram(idx) => dev.hip.memcpy_htod(&dev.bufs[idx], &bytes).map_err(err),
        Loc::Host(p) => {
            // SAFETY: `p` is the live mapped host allocation of `len` bytes; no kernel is in flight
            // (every launch is followed by device_synchronize).
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len()) };
            Ok(())
        }
    }
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
    funcs: Vec<(Codec, Kind, Function)>,
    device: i32,
    old_parity: bool,
    gpu: Option<Gpu>,
}
impl Ctx {
    fn func(&self, codec: Codec, kind: Kind) -> &Function {
        &self
            .funcs
            .iter()
            .find(|(c, k, _)| *c == codec && *k == kind)
            .expect("all four symbols loaded")
            .2
    }
}

struct TimingCfg {
    reps: usize,
    warmup: usize,
    launches: usize,
}

// ---------------------------------------------------------------------------------------
// CPU expected buffers
// ---------------------------------------------------------------------------------------

struct Expected {
    y0: Vec<f32>,
    y1: Option<Vec<f32>>,
    twin_mismatches: u64,
    twin_slots_checked: usize,
}

fn proj_bytes(e: &Expert, kind: Kind) -> &[u8] {
    match kind {
        Kind::GateUp => &e.gate,
        Kind::Down => &e.down,
    }
}

fn cpu_expected(case: &Case, kind: Kind, buckets: &[Bucket; 2]) -> Expected {
    let (x, xrs, ys) = case.x(kind);
    let n = 10 * case.c * ys;
    let sentinel = f32::from_bits(SENTINEL_BITS);
    let mut y0 = vec![sentinel; n];
    let mut y1 = if kind == Kind::GateUp {
        Some(vec![sentinel; n])
    } else {
        None
    };
    let mut twin_mismatches = 0u64;
    let mut checked = 0usize;
    // W2: the mq_i4_reference twin.
    let b = &buckets[Codec::W2.idx()];
    if !b.ids.is_empty() {
        let mut exps: Vec<Option<ExpertTensor<'_>>> = (0..N_EXPERTS).map(|_| None).collect();
        for (&id, e) in &case.experts {
            if e.codec == Codec::W2 {
                exps[id as usize] = Some(ExpertTensor {
                    profile: kind.w2_profile(),
                    bytes: proj_bytes(e, kind),
                });
            }
        }
        mq_i4_reference::decode_moe_slots(
            kind.decode_kind(),
            kind.m(),
            kind.k(),
            &exps,
            &b.ids,
            &b.offs,
            &b.slots,
            case.c,
            x,
            xrs,
            ys,
            &mut y0,
            y1.as_deref_mut(),
        );
        // decode_moe_slots vs decode_gemv_f32 on a bounded slot sample (plumbing self-check).
        let mut sample: Vec<u32> = b.slots.clone();
        sample.sort_unstable();
        sample.truncate(12);
        for &slot in &sample {
            let slot = slot as usize;
            let e = &case.experts[&case.route[slot]];
            let xs = &x[(slot / kind.x_row_div()) * xrs..][..kind.k()];
            let row = mq_i4_reference::decode_gemv_f32(
                kind.w2_profile(),
                kind.m(),
                kind.k(),
                proj_bytes(e, kind),
                xs,
            );
            for (r, v) in row.iter().enumerate() {
                let got = if r < kind.out_rows() {
                    y0[slot * ys + r]
                } else {
                    y1.as_ref().unwrap()[slot * ys + r - kind.out_rows()]
                };
                if got.to_bits() != v.to_bits() {
                    twin_mismatches += 1;
                }
            }
        }
        checked = sample.len();
    }
    // MQ4: the transcribed legacy arithmetic.
    let b = &buckets[Codec::Mq4.idx()];
    if !b.ids.is_empty() {
        let mut exps: Vec<Option<&[u8]>> = vec![None; N_EXPERTS];
        for (&id, e) in &case.experts {
            if e.codec == Codec::Mq4 {
                exps[id as usize] = Some(proj_bytes(e, kind));
            }
        }
        mq4_decode_moe_slots(kind, &exps, b, x, xrs, ys, &mut y0, y1.as_deref_mut());
    }
    Expected {
        y0,
        y1,
        twin_mismatches,
        twin_slots_checked: checked,
    }
}

// ---------------------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------------------

#[derive(Default)]
struct Mm {
    count: u64,
    first: Vec<Value>,
}
impl Mm {
    fn push(&mut self, v: Value) {
        self.count += 1;
        if self.first.len() < 8 {
            self.first.push(v);
        }
    }
}

struct Cmp {
    per: [Mm; 2],
    stray: Mm,
    /// Mismatching words per slot (in-row and pad words alike).
    slot_counts: Vec<u64>,
}
impl Cmp {
    fn total(&self) -> u64 {
        self.per[0].count + self.per[1].count + self.stray.count
    }
}

#[allow(clippy::too_many_arguments)]
fn compare_buf(
    part: &str,
    gpu: &[u8],
    exp: &[f32],
    ys: usize,
    out_rows: usize,
    slot_codec: &[Codec],
    cmp: &mut Cmp,
) {
    for (i, (b, e)) in gpu.chunks_exact(4).zip(exp).enumerate() {
        let g = u32::from_le_bytes(b.try_into().unwrap());
        if g == e.to_bits() {
            continue;
        }
        let (slot, row) = (i / ys, i % ys);
        cmp.slot_counts[slot] += 1;
        let rec = json!({"part": part, "word_index": i, "slot": slot, "row": row,
            "gpu_bits": bits_hex(g), "oracle_bits": bits_hex(e.to_bits()),
            "oracle_unwritten": e.to_bits() == SENTINEL_BITS});
        if row < out_rows {
            cmp.per[slot_codec[slot].idx()].push(rec);
        } else {
            cmp.stray.push(rec);
        }
    }
}

fn compare_all(p: &Prep, case: &Case, exp: &Expected, g0: &[u8], g1: Option<&[u8]>) -> Cmp {
    let slot_codec: Vec<Codec> = (0..p.slots).map(|s| case.codec_of_slot(s)).collect();
    let mut cmp = Cmp {
        per: [Mm::default(), Mm::default()],
        stray: Mm::default(),
        slot_counts: vec![0; p.slots],
    };
    compare_buf("y0", g0, &exp.y0, p.ys, p.out, &slot_codec, &mut cmp);
    if let (Some(g1), Some(e1)) = (g1, exp.y1.as_ref()) {
        compare_buf("y1", g1, e1, p.ys, p.out, &slot_codec, &mut cmp);
    }
    cmp
}

// ---------------------------------------------------------------------------------------
// One (case, kind) GPU run, split into prepare / launch / read so controls can relaunch
// ---------------------------------------------------------------------------------------

struct Prep {
    kind: Kind,
    m: usize,
    slots: usize,
    ys: usize,
    out: usize,
    n_words: usize,
    buckets: [Bucket; 2],
    y0: Buf,
    y1: Option<Buf>,
    launches: Vec<(Codec, [u8; KERNARG_BYTES], [u32; 3])>,
}

fn prepare(dev: &mut Dev, case: &Case, kind: Kind, slabs: &Slabs) -> Result<Prep> {
    let c = case.c;
    let slots = 10 * c;
    let (m, k, out) = (kind.m(), kind.k(), kind.out_rows());
    let (x, xrs, ys) = case.x(kind);
    let x_src_rows = x.len() / xrs;
    let buckets = buckets_of(case);
    let xb = dev.upload(&f32_bytes(x))?;
    let mut table = vec![0u64; N_EXPERTS];
    for (&id, s) in &slabs.by_id {
        table[id as usize] = if kind == Kind::GateUp {
            s.gate_ptr
        } else {
            s.down_ptr
        };
    }
    let tb = dev.upload(&u64_bytes(&table))?;
    let n_words = slots * ys;
    let y0 = dev.alloc(n_words * 4)?;
    let y1 = if kind == Kind::GateUp {
        Some(dev.alloc(n_words * 4)?)
    } else {
        None
    };
    let mut launches = Vec::new();
    for codec in Codec::active() {
        let b = &buckets[codec.idx()];
        if b.ids.is_empty() {
            continue;
        }
        let ib = dev.upload(&u32_bytes(&b.ids))?;
        let ob = dev.upload(&u32_bytes(&b.offs))?;
        let sb = dev.upload(&u32_bytes(&b.slots))?;
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
            b.ids.len(),
            c,
            kind.x_row_div(),
            x_src_rows,
            xrs,
            ys,
        );
        launches.push((codec, args, [m as u32, b.ids.len() as u32, 1]));
    }
    Ok(Prep {
        kind,
        m,
        slots,
        ys,
        out,
        n_words,
        buckets,
        y0,
        y1,
        launches,
    })
}

/// Poison the outputs with 0xa5, launch every codec's kernel, synchronize.
fn launch(ctx: &Ctx, dev: &Dev, p: &Prep) -> Result<()> {
    launch_with(ctx, dev, p, &p.launches)
}

/// [`launch`] over an explicit launch list of the same prep (same output buffers).
fn launch_with(
    ctx: &Ctx,
    dev: &Dev,
    p: &Prep,
    launches: &[(Codec, [u8; KERNARG_BYTES], [u32; 3])],
) -> Result<()> {
    dev.hip
        .memset(&dev.bufs[p.y0.idx], 0xa5, p.n_words * 4)
        .map_err(err)?;
    if let Some(y1) = p.y1 {
        dev.hip
            .memset(&dev.bufs[y1.idx], 0xa5, p.n_words * 4)
            .map_err(err)?;
    }
    for (codec, args, grid) in launches {
        let mut a = *args;
        // SAFETY: exact 96-byte ABI; every pointer is a live allocation of `dev` until release.
        unsafe {
            ctx.hip
                .launch_kernel_blob(ctx.func(*codec, p.kind), *grid, [32, 1, 1], 0, None, &mut a)
        }
        .map_err(err)?;
    }
    ctx.hip.device_synchronize().map_err(err)
}

fn read_out(dev: &Dev, p: &Prep) -> Result<(Vec<u8>, Option<Vec<u8>>)> {
    let g0 = dev.download(p.y0, p.n_words * 4)?;
    let g1 = match p.y1 {
        Some(b) => Some(dev.download(b, p.n_words * 4)?),
        None => None,
    };
    Ok((g0, g1))
}

fn symbols_json(p: &Prep, cmp: &Cmp) -> Vec<Value> {
    let mut symbols = Vec::new();
    for codec in Codec::active() {
        let b = &p.buckets[codec.idx()];
        if b.ids.is_empty() {
            symbols.push(json!({"symbol": symbol(codec, p.kind), "codec": codec.label(), "profile": p.kind.suffix(),
                "launched": false, "slots": 0, "unique_experts": 0}));
        } else {
            symbols.push(json!({
                "symbol": symbol(codec, p.kind), "codec": codec.label(), "profile": p.kind.suffix(), "launched": true,
                "grid": [p.m, b.ids.len(), 1], "slots": b.slots.len(), "unique_experts": b.ids.len(),
                "mismatch_count": cmp.per[codec.idx()].count, "first_mismatches": cmp.per[codec.idx()].first,
            }));
        }
    }
    symbols.push(
        json!({"symbol": "stray_or_pad_words", "profile": p.kind.suffix(),
        "mismatch_count": cmp.stray.count, "first_mismatches": cmp.stray.first}),
    );
    symbols
}

type TimingRows = Vec<(Codec, Kind, usize, usize, Vec<f64>)>;

fn time_prep(ctx: &Ctx, p: &Prep, t: &TimingCfg) -> Result<TimingRows> {
    let mut rows = Vec::new();
    for (codec, args, grid) in &p.launches {
        let f = ctx.func(*codec, p.kind);
        let start = ctx.hip.event_create().map_err(err)?;
        let stop = ctx.hip.event_create().map_err(err)?;
        let mut samples = Vec::with_capacity(t.reps);
        for i in 0..t.warmup + t.reps {
            ctx.hip.event_record(&start, None).map_err(err)?;
            for _ in 0..t.launches {
                let mut a = *args;
                // SAFETY: as above; the launch reads the blob during the call.
                unsafe {
                    ctx.hip
                        .launch_kernel_blob(f, *grid, [32, 1, 1], 0, None, &mut a)
                }
                .map_err(err)?;
            }
            ctx.hip.event_record(&stop, None).map_err(err)?;
            ctx.hip.event_synchronize(&stop).map_err(err)?;
            let ms = ctx.hip.event_elapsed_ms(&start, &stop).map_err(err)?;
            if i >= t.warmup {
                samples.push(ms as f64 / 1000.0);
            }
        }
        let _ = ctx.hip.event_destroy(start);
        let _ = ctx.hip.event_destroy(stop);
        rows.push((*codec, p.kind, p.slots, grid[1] as usize, samples));
    }
    Ok(rows)
}

// ---------------------------------------------------------------------------------------
// Negative controls
// ---------------------------------------------------------------------------------------

fn word_at(buf: &[u8], idx: usize) -> u32 {
    u32::from_le_bytes(buf[idx * 4..idx * 4 + 4].try_into().unwrap())
}

/// Output-corruption controls: a corrupted copy of the verified GPU buffers must be flagged exactly once.
fn control_output(
    case: &Case,
    p: &Prep,
    exp: &Expected,
    g0: &[u8],
    g1: Option<&[u8]>,
) -> Vec<Value> {
    let probe = |a: &[u8], b: Option<&[u8]>| -> u64 { compare_all(p, case, exp, a, b).total() };
    let mut res = Vec::new();
    let mut push = |name: &str, detail: &str, count: u64| {
        res.push(
            json!({"control": name, "kind": p.kind.suffix(), "detail": detail,
            "expected": "exactly 1 mismatching word", "mismatch_count": count, "pass": count == 1}),
        );
    };
    let last = p.slots - 1;
    let mut t0 = g0.to_vec();
    t0[0] ^= 1;
    push("output_bit_flip", "y0 slot 0 row 0 bit 0", probe(&t0, g1));
    if let Some(g1) = g1 {
        let mut t1 = g1.to_vec();
        let idx = last * p.ys + p.out - 1;
        t1[idx * 4 + 3] ^= 0x80;
        push(
            "output_bit_flip",
            "y1 last slot last row sign bit",
            probe(g0, Some(&t1)),
        );
    }
    let mut tu = g0.to_vec();
    let idx = last * p.ys + p.out - 1;
    tu[idx * 4..idx * 4 + 4].copy_from_slice(&SENTINEL_BITS.to_le_bytes());
    push(
        "output_unwritten_word",
        "y0 last slot last row replaced by the 0xa5 sentinel",
        probe(&tu, g1),
    );
    if p.ys > p.out {
        let mut tp = g0.to_vec();
        let idx = last * p.ys + p.out;
        debug_assert_eq!(word_at(g0, idx), SENTINEL_BITS);
        tp[idx * 4] ^= 1;
        push(
            "pad_bit_flip",
            "y0 last slot first pad word bit 0",
            probe(&tp, g1),
        );
    }
    res
}

/// Wrong-arithmetic controls on a bounded row sample per codec.
fn control_arith(
    case: &Case,
    p: &Prep,
    exp: &Expected,
    g0: &[u8],
    g1: Option<&[u8]>,
) -> Vec<Value> {
    let kind = p.kind;
    let (x, xrs, ys) = case.x(kind);
    let (m, k, out) = (kind.m(), kind.k(), p.out);
    debug_assert_eq!(m, p.m);
    let rb = kind.mq4_row_bytes();
    let rows: Vec<usize> = match kind {
        Kind::GateUp => (0..16).chain(out..out + 16).collect(),
        Kind::Down => (0..32).collect(),
    };
    let word_of = |buf0: &[u8], buf1: Option<&[u8]>, slot: usize, row: usize| -> u32 {
        if row < out {
            word_at(buf0, slot * ys + row)
        } else {
            word_at(buf1.expect("gate has Y1"), slot * ys + row - out)
        }
    };
    let exp_word = |slot: usize, row: usize| -> u32 {
        if row < out {
            exp.y0[slot * ys + row].to_bits()
        } else {
            exp.y1.as_ref().expect("gate has Y1")[slot * ys + row - out].to_bits()
        }
    };
    let mut res = Vec::new();
    for codec in Codec::active() {
        let b = &p.buckets[codec.idx()];
        if b.ids.is_empty() {
            continue;
        }
        let mut sl = b.slots.clone();
        sl.sort_unstable();
        let step = sl.len().div_ceil(12).max(1);
        let picked: Vec<usize> = sl.iter().step_by(step).map(|&s| s as usize).collect();
        let mut decoded: BTreeMap<u32, MqI4Tensor> = BTreeMap::new();
        if codec == Codec::W2 {
            for &slot in &picked {
                let id = case.route[slot];
                if let std::collections::btree_map::Entry::Vacant(v) = decoded.entry(id) {
                    v.insert(
                        mq_i4::decode_tensor(
                            kind.w2_profile(),
                            m,
                            k,
                            proj_bytes(&case.experts[&id], kind),
                        )
                        .expect("validated expert tensor"),
                    );
                }
            }
        }
        let eval = |mu: Mut, slot: usize, r: usize| -> f32 {
            let id = case.route[slot];
            let xs = &x[(slot / kind.x_row_div()) * xrs..][..k];
            match codec {
                Codec::W2 => {
                    let t = &decoded[&id];
                    let groups = k / 128;
                    ref_row_m(
                        kind.w2_profile(),
                        &t.codes[r * k..(r + 1) * k],
                        &t.scale_bits[r * groups..(r + 1) * groups],
                        xs,
                        mu,
                    )
                }
                Codec::Mq4 => {
                    let bytes = proj_bytes(&case.experts[&id], kind);
                    let row = &bytes[r * rb..(r + 1) * rb];
                    match kind {
                        Kind::GateUp => mq4_row_gate_m(row, xs, mu),
                        Kind::Down => mq4_row_down_m(row, xs, mu),
                    }
                }
            }
        };
        let mut base_bad = 0u64;
        let mut checked = 0u64;
        for &slot in &picked {
            for &r in &rows {
                checked += 1;
                if eval(Mut::None, slot, r).to_bits() != exp_word(slot, r) {
                    base_bad += 1;
                }
            }
        }
        res.push(json!({"control": "oracle_transcription_baseline", "codec": codec.label(), "kind": kind.suffix(),
            "rows_checked": checked, "mismatch_count": base_bad, "expected": "0 (transcription == oracle)",
            "pass": base_bad == 0}));
        for mu in Mut::ARITH {
            if !mu.applies(codec, kind) {
                continue;
            }
            let (mut oracle_differs, mut gpu_differs) = (0u64, 0u64);
            for &slot in &picked {
                for &r in &rows {
                    let v = eval(mu, slot, r).to_bits();
                    if v != exp_word(slot, r) {
                        oracle_differs += 1;
                    }
                    if v != word_of(g0, g1, slot, r) {
                        gpu_differs += 1;
                    }
                }
            }
            res.push(json!({"control": "wrong_arithmetic", "mutation": mu.name(), "codec": codec.label(),
                "kind": kind.suffix(), "rows_checked": checked, "oracle_differs_rows": oracle_differs,
                "gpu_differs_rows": gpu_differs, "expected": "mutant differs from oracle and from GPU",
                "pass": base_bad == 0 && oracle_differs > 0 && gpu_differs > 0}));
        }
    }
    res
}

/// A finite non-negative f16 scale different from `sc` (zero becomes 1.0).
fn bump_scale(sc: u16) -> u16 {
    if sc == 0 {
        0x3c00
    } else if sc < 0x4000 {
        sc + 0x0400
    } else {
        sc - 0x0400
    }
}

fn tamper_tensor(codec: Codec, kind: Kind, bytes: &[u8]) -> Result<Vec<u8>> {
    let (m, k) = (kind.m(), kind.k());
    let groups = k / 128;
    match codec {
        Codec::W2 => {
            let mut t = mq_i4::decode_tensor(kind.w2_profile(), m, k, bytes).map_err(err)?;
            for r in (0..m).step_by(29) {
                let g = (r * 7) % groups;
                let si = r * groups + g;
                t.scale_bits[si] = bump_scale(t.scale_bits[si]);
                t.codes[r * k + g * 128 + 5] ^= 1;
            }
            mq_i4::encode_tensor(kind.w2_profile(), m, k, &t.codes, &t.scale_bits).map_err(err)
        }
        Codec::Mq4 => {
            let mut out = bytes.to_vec();
            for r in (0..m).step_by(29) {
                let g = (r * 7) % groups;
                let (hdr, payload) = match kind {
                    Kind::GateUp => {
                        let off = (r * (k / 256) + g / 2) * 136;
                        (off + 4 * (g % 2), off + 8 + (g % 2) * 64 + 2)
                    }
                    Kind::Down => {
                        let off = (r * groups + g) * 68;
                        (off, off + 4 + 2)
                    }
                };
                let sc = u16::from_le_bytes([out[hdr], out[hdr + 1]]);
                out[hdr..hdr + 2].copy_from_slice(&bump_scale(sc).to_le_bytes());
                out[payload] ^= 1;
            }
            Ok(out)
        }
    }
}

fn tamper_expert(e: &Expert) -> Result<Expert> {
    let gate = tamper_tensor(e.codec, Kind::GateUp, &e.gate)?;
    let down = tamper_tensor(e.codec, Kind::Down, &e.down)?;
    Ok(Expert {
        codec: e.codec,
        gate_sha: sha(&gate),
        down_sha: sha(&down),
        gate,
        down,
    })
}

struct KindRun {
    prep: Prep,
    exp: Expected,
}

/// In-place weight corruption in the placement memory, relaunch, restore, relaunch.
fn control_weight_corruption(
    ctx: &Ctx,
    dev: &Dev,
    case: &Case,
    placement: Placement,
    slabs: &Slabs,
    runs: &[KindRun],
) -> Result<Vec<Value>> {
    let mut ids: Vec<u32> = Vec::new();
    for codec in Codec::active() {
        if let Some(id) = case
            .experts
            .iter()
            .filter(|(_, e)| e.codec == codec)
            .map(|(id, _)| *id)
            .min_by_key(|id| (*id % 3 == 2) as u8)
        {
            ids.push(id);
        }
    }
    let mut tcase = case.clone();
    for &id in &ids {
        tcase
            .experts
            .insert(id, Rc::new(tamper_expert(&case.experts[&id])?));
    }
    let mut res = Vec::new();
    for &id in &ids {
        overwrite_slab(dev, slabs, id, &tcase.experts[&id])?;
    }
    for r in runs {
        let exp_t = cpu_expected(&tcase, r.prep.kind, &r.prep.buckets);
        launch(ctx, dev, &r.prep)?;
        let (g0, g1) = read_out(dev, &r.prep)?;
        let vs_tampered = compare_all(&r.prep, &tcase, &exp_t, &g0, g1.as_deref());
        let vs_orig = compare_all(&r.prep, case, &r.exp, &g0, g1.as_deref());
        let mut outside = 0u64;
        let mut inside = BTreeMap::new();
        for (slot, &n) in vs_orig.slot_counts.iter().enumerate() {
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
        let pass = vs_tampered.total() == 0 && vs_orig.total() > 0 && outside == 0 && all_detect;
        res.push(json!({"control": "weight_corruption", "placement": placement.label(), "kind": r.prep.kind.suffix(),
            "corrupted_experts": ids.iter().map(|id| json!({"id": id, "codec": case.experts[id].codec.label(),
                "host_mapped": matches!(slabs.by_id[id].loc, Loc::Host(_))})).collect::<Vec<_>>(),
            "gpu_vs_oracle_of_corrupted_mismatches": vs_tampered.total(),
            "gpu_vs_original_oracle_mismatches": vs_orig.total(),
            "mismatches_in_corrupted_experts_slots": inside.iter().map(|(k, v)| json!({"id": k, "words": v})).collect::<Vec<_>>(),
            "mismatches_outside_corrupted_slots": outside,
            "expected": "GPU == oracle(corrupted); GPU != oracle(original) only in corrupted experts' slots",
            "pass": pass}));
    }
    for &id in &ids {
        overwrite_slab(dev, slabs, id, &case.experts[&id])?;
    }
    for r in runs {
        launch(ctx, dev, &r.prep)?;
        let (g0, g1) = read_out(dev, &r.prep)?;
        let back = compare_all(&r.prep, case, &r.exp, &g0, g1.as_deref());
        res.push(json!({"control": "weight_restore", "placement": placement.label(), "kind": r.prep.kind.suffix(),
            "mismatch_count": back.total(), "expected": "0 after restoring the original weights",
            "pass": back.total() == 0}));
    }
    Ok(res)
}

/// Signed-bad expert ids: the first four experts (in launch order over the active codecs) get `-1`, `n_experts`,
/// `INT_MAX` and `INT_MIN` in their id buffer only. The kernel's one unsigned compare against `n_experts@88` exits
/// before the table is indexed, so those experts' slots keep the `0xa5` sentinel while every other expert is still
/// computed; the original id buffers are then relaunched and must reproduce the verified output.
fn control_bad_ids(
    ctx: &Ctx,
    dev: &mut Dev,
    case: &Case,
    placement: Placement,
    runs: &[KindRun],
) -> Result<Vec<Value>> {
    let mut res = Vec::new();
    for r in runs {
        let mut buckets = r.prep.buckets.clone();
        let mut launches = r.prep.launches.clone();
        let mut next = 0usize;
        for (codec, args, _) in launches.iter_mut() {
            let b = &r.prep.buckets[codec.idx()];
            let take = (BAD_IDS.len() - next).min(b.ids.len());
            if take == 0 {
                continue;
            }
            let mut ids = b.ids.clone();
            ids[..take].copy_from_slice(&BAD_IDS[next..next + take]);
            next += take;
            let ib = dev.upload(&u32_bytes(&ids))?;
            args[8..16].copy_from_slice(&ib.ptr.to_le_bytes());
            buckets[codec.idx()] = b.without_first(take);
        }
        if next != BAD_IDS.len() {
            return Err("bad_expert_ids control needs at least four experts".into());
        }
        let exp_b = cpu_expected(case, r.prep.kind, &buckets);
        launch_with(ctx, dev, &r.prep, &launches)?;
        let (g0, g1) = read_out(dev, &r.prep)?;
        let vs_b = compare_all(&r.prep, case, &exp_b, &g0, g1.as_deref());
        let vs_o = compare_all(&r.prep, case, &r.exp, &g0, g1.as_deref());
        res.push(json!({"control": "bad_expert_ids", "placement": placement.label(), "kind": r.prep.kind.suffix(),
            "bad_ids": BAD_IDS,
            "gpu_vs_oracle_with_rejected_experts_mismatches": vs_b.total(),
            "gpu_vs_original_oracle_mismatches": vs_o.total(),
            "expected": "bad-id experts store nothing (slots keep the sentinel), every other expert equals the oracle",
            "pass": vs_b.total() == 0 && vs_o.total() > 0}));
        launch(ctx, dev, &r.prep)?;
        let (g0, g1) = read_out(dev, &r.prep)?;
        let back = compare_all(&r.prep, case, &r.exp, &g0, g1.as_deref());
        res.push(json!({"control": "bad_expert_ids_restore", "placement": placement.label(), "kind": r.prep.kind.suffix(),
            "mismatch_count": back.total(), "expected": "0 with the original id buffers",
            "pass": back.total() == 0}));
    }
    Ok(res)
}

// ---------------------------------------------------------------------------------------
// Independent old-kernel parity / timing (production HIP kernels, JIT via rdna_compute::Gpu)
// ---------------------------------------------------------------------------------------

type OldSamples = Vec<(Kind, Vec<f64>)>;

fn old_kernel_run(
    gpu: &mut Gpu,
    case: &Case,
    placement: Placement,
    gate: &Expected,
    down: &Expected,
    timing: Option<&TimingCfg>,
) -> Result<(Value, OldSamples)> {
    let mut hosts: Vec<*mut c_void> = Vec::new();
    let r = old_kernel_inner(gpu, case, placement, gate, down, timing, &mut hosts);
    for h in hosts {
        let _ = gpu.hip.host_free(h);
    }
    r
}

#[allow(clippy::too_many_arguments)]
fn old_kernel_inner(
    gpu: &mut Gpu,
    case: &Case,
    placement: Placement,
    gate: &Expected,
    down: &Expected,
    timing: Option<&TimingCfg>,
    hosts: &mut Vec<*mut c_void>,
) -> Result<(Value, OldSamples)> {
    let c = case.c;
    fn up(gpu: &mut Gpu, bytes: &[u8]) -> Result<GpuTensor> {
        let t = gpu
            .alloc_tensor(&[bytes.len().max(4)], DType::Raw)
            .map_err(err)?;
        gpu.hip.memcpy_htod(&t.buf, bytes).map_err(err)?;
        Ok(t)
    }
    fn upf(gpu: &mut Gpu, v: &[f32]) -> Result<GpuTensor> {
        let t = gpu.alloc_tensor(&[v.len()], DType::F32).map_err(err)?;
        gpu.hip.memcpy_htod(&t.buf, &f32_bytes(v)).map_err(err)?;
        Ok(t)
    }
    fn blank(gpu: &mut Gpu, n: usize) -> Result<GpuTensor> {
        let t = gpu.alloc_tensor(&[n], DType::F32).map_err(err)?;
        gpu.hip.memset(&t.buf, 0xa5, n * 4).map_err(err)?;
        Ok(t)
    }
    fn read(gpu: &Gpu, t: &GpuTensor, n: usize) -> Result<Vec<u8>> {
        let mut out = vec![0u8; n * 4];
        gpu.hip.memcpy_dtoh(&mut out, &t.buf).map_err(err)?;
        Ok(out)
    }
    fn cmp(got: &[u8], exp: &[f32]) -> (u64, Vec<Value>) {
        let mut mm = Mm::default();
        for (i, (b, e)) in got.chunks_exact(4).zip(exp).enumerate() {
            let g = u32::from_le_bytes(b.try_into().unwrap());
            if g != e.to_bits() {
                mm.push(json!({"word_index": i, "old_kernel_bits": bits_hex(g), "oracle_bits": bits_hex(e.to_bits())}));
            }
        }
        (mm.count, mm.first)
    }
    let mut keep: Vec<GpuTensor> = Vec::new();
    let (mut gtab, mut dtab) = (vec![0u64; N_EXPERTS], vec![0u64; N_EXPERTS]);
    for (i, (&id, e)) in case.experts.iter().enumerate() {
        if placement.host_for(i) {
            let len = e.gate.len() + e.down.len();
            let h = gpu
                .hip
                .host_malloc(len.max(4096), HIP_HOST_MALLOC_MAPPED)
                .map_err(err)?;
            hosts.push(h);
            // SAFETY: `h` is a live mapped host allocation of at least `len` bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(e.gate.as_ptr(), h as *mut u8, e.gate.len());
                std::ptr::copy_nonoverlapping(
                    e.down.as_ptr(),
                    (h as *mut u8).add(e.gate.len()),
                    e.down.len(),
                );
            }
            let d = gpu.hip.host_get_device_pointer(h, 0).map_err(err)? as u64;
            gtab[id as usize] = d;
            dtab[id as usize] = d + e.gate.len() as u64;
        } else {
            let g = up(gpu, &e.gate)?;
            let d = up(gpu, &e.down)?;
            gtab[id as usize] = g.buf.as_ptr() as u64;
            dtab[id as usize] = d.buf.as_ptr() as u64;
            keep.push(g);
            keep.push(d);
        }
    }
    let gt = up(gpu, &u64_bytes(&gtab))?;
    let dt = up(gpu, &u64_bytes(&dtab))?;
    let topk: Vec<u8> = case
        .route
        .iter()
        .flat_map(|&e| (e as i32).to_le_bytes())
        .collect();
    let tk = up(gpu, &topk)?;
    let xg = upf(gpu, &case.x_gate)?;
    let xd = upf(gpu, &case.x_down)?;
    let (n_gate, n_down) = (c * 10 * GATE_M / 2, c * 10 * DOWN_M);
    let yg = blank(gpu, n_gate)?;
    let yu = blank(gpu, n_gate)?;
    let yd = blank(gpu, n_down)?;
    gpu.gemv_mq4g256v2_moe_gate_up_top10_indexed_batched(
        &gt, &tk, &xg, &yg, &yu, GATE_M, GATE_K, c,
    )
    .map_err(err)?;
    gpu.gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
        &dt, &tk, &xd, &yd, DOWN_M, DOWN_K, c, N_EXPERTS, 10,
    )
    .map_err(err)?;
    gpu.hip.device_synchronize().map_err(err)?;
    let (g0, g1, d0) = (
        read(gpu, &yg, n_gate)?,
        read(gpu, &yu, n_gate)?,
        read(gpu, &yd, n_down)?,
    );
    let rows = [
        ("gate_up_y0", cmp(&g0, &gate.y0)),
        (
            "gate_up_y1",
            cmp(&g1, gate.y1.as_ref().expect("gate has Y1")),
        ),
        ("down_y0", cmp(&d0, &down.y0)),
    ];
    let total: u64 = rows.iter().map(|(_, (n, _))| *n).sum();
    let detail: Vec<Value> = rows
        .iter()
        .map(|(n, (count, first))| json!({"part": n, "mismatch_count": count, "first_mismatches": first}))
        .collect();
    let mut samples: OldSamples = Vec::new();
    if let (Some(t), true) = (timing, total == 0) {
        let start = gpu.hip.event_create().map_err(err)?;
        let stop = gpu.hip.event_create().map_err(err)?;
        for kind in Kind::ALL {
            let mut s = Vec::with_capacity(t.reps);
            for i in 0..t.warmup + t.reps {
                gpu.hip.event_record(&start, None).map_err(err)?;
                for _ in 0..t.launches {
                    match kind {
                        Kind::GateUp => gpu
                            .gemv_mq4g256v2_moe_gate_up_top10_indexed_batched(
                                &gt, &tk, &xg, &yg, &yu, GATE_M, GATE_K, c,
                            )
                            .map_err(err)?,
                        Kind::Down => gpu
                            .gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
                                &dt, &tk, &xd, &yd, DOWN_M, DOWN_K, c, N_EXPERTS, 10,
                            )
                            .map_err(err)?,
                    }
                }
                gpu.hip.event_record(&stop, None).map_err(err)?;
                gpu.hip.event_synchronize(&stop).map_err(err)?;
                let ms = gpu.hip.event_elapsed_ms(&start, &stop).map_err(err)?;
                if i >= t.warmup {
                    s.push(ms as f64 / 1000.0);
                }
            }
            samples.push((kind, s));
        }
        let _ = gpu.hip.event_destroy(start);
        let _ = gpu.hip.event_destroy(stop);
    }
    for t in keep.into_iter().chain([gt, dt, tk, xg, xd, yg, yu, yd]) {
        gpu.free_tensor(t).map_err(err)?;
    }
    Ok((
        json!({"kernels": ["gemv_mq4g256v2_moe_gate_up_top10_indexed_batched",
            "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded"],
            "placement": placement.label(), "mismatch_count": total, "parts": detail,
            "timed": !samples.is_empty()}),
        samples,
    ))
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
    mismatches: u64,
    twin_mismatches: u64,
    old_kernel_mismatches: u64,
    old_kernel_cases: u64,
    control_checks: u64,
    control_failures: u64,
}

struct RunOpts<'a> {
    timing: Option<&'a TimingCfg>,
    /// Output / pad / wrong-arithmetic controls.
    controls: bool,
    /// In-place weight-corruption control.
    tamper: bool,
}

struct CaseOut {
    row: Value,
    timing: TimingRows,
    old_samples: OldSamples,
    expected: Vec<Expected>,
}

fn run_case_inner(
    ctx: &Ctx,
    dev: &mut Dev,
    case: &Case,
    placement: Placement,
    t: &mut Totals,
    o: &RunOpts,
) -> Result<CaseOut> {
    let slabs = upload_experts(dev, case, placement)?;
    let mut runs: Vec<KindRun> = Vec::new();
    let mut kind_syms: Vec<Vec<Value>> = Vec::new();
    let mut mismatches = 0u64;
    let mut twin = 0u64;
    let mut launches = 0u64;
    let mut controls: Vec<Value> = Vec::new();
    let mut timing_rows: TimingRows = Vec::new();
    for kind in Kind::ALL {
        let prep = prepare(dev, case, kind, &slabs)?;
        let exp = cpu_expected(case, kind, &prep.buckets);
        launch(ctx, dev, &prep)?;
        let (g0, g1) = read_out(dev, &prep)?;
        let cmp = compare_all(&prep, case, &exp, &g0, g1.as_deref());
        mismatches += cmp.total();
        twin += exp.twin_mismatches;
        launches += prep.launches.len() as u64;
        kind_syms.push(symbols_json(&prep, &cmp));
        if o.controls {
            controls.extend(control_output(case, &prep, &exp, &g0, g1.as_deref()));
            controls.extend(control_arith(case, &prep, &exp, &g0, g1.as_deref()));
        }
        if let Some(tc) = o.timing {
            if cmp.total() != 0 {
                return Err(format!(
                    "{} not byte-exact; refusing to time a wrong kernel",
                    kind.suffix()
                ));
            }
            timing_rows.extend(time_prep(ctx, &prep, tc)?);
        }
        runs.push(KindRun { prep, exp });
    }
    if placement != Placement::Vram {
        controls.push(json!({"control": "host_mapped_pointer_attributes", "placement": placement.label(),
            "host_slabs": slabs.host_slabs, "attribute_failures": slabs.host_attr_failures,
            "evidence": slabs.evidence, "expected": "every mapped slab reports host-resident memory",
            "pass": slabs.host_slabs > 0 && slabs.host_attr_failures == 0}));
    }
    if o.tamper {
        controls.extend(control_weight_corruption(
            ctx, dev, case, placement, &slabs, &runs,
        )?);
        controls.extend(control_bad_ids(ctx, dev, case, placement, &runs)?);
    }
    for c in &controls {
        t.control_checks += 1;
        if c["pass"] != json!(true) {
            t.control_failures += 1;
        }
    }
    let mut counts = BTreeMap::new();
    for codec in Codec::active() {
        counts.insert(
            codec.label(),
            case.route
                .iter()
                .filter(|e| case.experts[*e].codec == codec)
                .count(),
        );
    }
    let unique_by_codec: BTreeMap<&str, usize> = Codec::active()
        .iter()
        .map(|c| {
            (
                c.label(),
                case.experts.values().filter(|e| e.codec == *c).count(),
            )
        })
        .collect();
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
    t.launches += launches;
    t.mismatches += mismatches;
    t.twin_mismatches += twin;
    let symbols_down = kind_syms.pop().unwrap_or_default();
    let symbols_gate = kind_syms.pop().unwrap_or_default();
    let row = json!({
        "case": case.name, "placement": placement.label(), "origin": case.origin, "c": case.c, "slots": 10 * case.c,
        "unique_experts": case.experts.len(), "slots_by_codec": counts, "unique_experts_by_codec": unique_by_codec,
        "strides": {"x_row_stride_gate": case.xrs_gate, "x_row_stride_down": case.xrs_down,
            "y_slot_stride_gate": case.ys_gate, "y_slot_stride_down": case.ys_down},
        "weights": {"placement": placement.label(), "host_mapped_slabs": slabs.host_slabs,
            "vram_slabs": slabs.vram_slabs, "host_pointer_evidence": slabs.evidence},
        "input_sha256": case.input_sha256(),
        "gate_up": symbols_gate, "down": symbols_down,
        "mismatch_count": mismatches,
        "twin_selfcheck": {"slots_checked_gate": runs[0].exp.twin_slots_checked,
            "slots_checked_down": runs[1].exp.twin_slots_checked, "mismatch_count": twin},
        "controls": controls,
    });
    let expected = runs.into_iter().map(|r| r.exp).collect();
    Ok(CaseOut {
        row,
        timing: timing_rows,
        old_samples: Vec::new(),
        expected,
    })
}

fn run_case(
    ctx: &mut Ctx,
    case: &Case,
    placement: Placement,
    t: &mut Totals,
    o: &RunOpts,
) -> Result<(Value, TimingRows, OldSamples)> {
    let mut dev = Dev::new(&ctx.hip);
    let r = run_case_inner(&*ctx, &mut dev, case, placement, t, o);
    dev.release();
    let mut out = r?;
    if ctx.old_parity && case.old_parity && case.tight() {
        if ctx.gpu.is_none() {
            ctx.gpu = Some(Gpu::init_with_device(ctx.device).map_err(err)?);
        }
        let gpu = ctx.gpu.as_mut().expect("initialised above");
        let (r, samples) = old_kernel_run(
            gpu,
            case,
            placement,
            &out.expected[0],
            &out.expected[1],
            o.timing,
        )?;
        t.old_kernel_cases += 1;
        t.old_kernel_mismatches += r["mismatch_count"].as_u64().unwrap_or(1);
        out.row["old_kernel_parity"] = r;
        out.old_samples = samples;
    }
    Ok((out.row, out.timing, out.old_samples))
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
fn pcie_info(bdf: &str) -> Value {
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
        "vram_total_sysfs": attr(bdf, "mem_info_vram_total"), "link_caveat": LINK_CAVEAT})
}

// ---------------------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------------------

#[derive(Debug)]
struct Cfg {
    codeobject: String,
    samples_dir: Option<String>,
    old_parity: bool,
    timing: bool,
    reps: usize,
    warmup: usize,
    launches: usize,
    device: Option<i32>,
    expect_uuid: String,
    expect_bdf: String,
    expect_arch: String,
    card: String,
    fabric_card: Option<String>,
    fabric_owner: Option<String>,
    fabric_wrapper: Option<String>,
    only: Vec<String>,
    no_reclaim_arm: bool,
    mq4_only: bool,
}

const USAGE: &str = "usage: mq_i4_moe_gemv_gfx1100_g1 --codeobject PATH [--samples-dir DIR] [--old-kernel-parity] \
[--timing --reps N(>=20) --warmup N(>=3) [--launches N]] [--card NAME] [--fabric-owner OWNER] \
[--fabric-wrapper fabric-run.sh] [--fabric-card NAME] [--device N] [--expect-uuid U] [--expect-bdf B] \
[--expect-arch A] [--only SUBSTR]... [--no-reclaim-arm] [--mq4-only]  (card/fabric flags fall back to env FABRIC_CARD / \
FABRIC_OWNER / FABRIC_WRAPPER; --samples-dir default for reference: ";

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn parse_args(argv: &[String]) -> Result<Cfg> {
    let mut cfg = Cfg {
        codeobject: String::new(),
        samples_dir: None,
        old_parity: false,
        timing: false,
        reps: 20,
        warmup: 3,
        launches: 8,
        device: None,
        expect_uuid: DEFAULT_UUID.into(),
        expect_bdf: DEFAULT_BDF.into(),
        expect_arch: DEFAULT_ARCH.into(),
        card: env_nonempty("FABRIC_CARD").unwrap_or_else(|| DEFAULT_CARD.into()),
        fabric_card: None,
        fabric_owner: env_nonempty("FABRIC_OWNER"),
        fabric_wrapper: env_nonempty("FABRIC_WRAPPER"),
        only: Vec::new(),
        no_reclaim_arm: false,
        mq4_only: false,
    };
    let usage = || format!("{USAGE}{DEFAULT_SAMPLES_DIR})");
    let mut i = 0;
    let value = |i: &mut usize| -> Result<String> {
        *i += 1;
        argv.get(*i)
            .cloned()
            .ok_or_else(|| format!("{} needs a value\n{}", argv[*i - 1], usage()))
    };
    let number = |i: &mut usize| -> Result<usize> {
        let flag = argv[*i].clone();
        value(i)?
            .parse::<usize>()
            .map_err(|e| format!("{flag}: {e}"))
    };
    while i < argv.len() {
        match argv[i].as_str() {
            "--codeobject" => cfg.codeobject = value(&mut i)?,
            "--samples-dir" => cfg.samples_dir = Some(value(&mut i)?),
            "--old-kernel-parity" => cfg.old_parity = true,
            "--timing" => cfg.timing = true,
            "--reps" => cfg.reps = number(&mut i)?,
            "--warmup" => cfg.warmup = number(&mut i)?,
            "--launches" => cfg.launches = number(&mut i)?,
            "--device" => cfg.device = Some(number(&mut i)? as i32),
            "--expect-uuid" => cfg.expect_uuid = value(&mut i)?,
            "--expect-bdf" => cfg.expect_bdf = value(&mut i)?.to_ascii_lowercase(),
            "--expect-arch" => cfg.expect_arch = value(&mut i)?,
            "--card" => cfg.card = value(&mut i)?,
            "--fabric-card" => cfg.fabric_card = Some(value(&mut i)?),
            "--fabric-owner" => cfg.fabric_owner = Some(value(&mut i)?),
            "--fabric-wrapper" => cfg.fabric_wrapper = Some(value(&mut i)?),
            "--only" => cfg.only.push(value(&mut i)?),
            "--no-reclaim-arm" => cfg.no_reclaim_arm = true,
            "--mq4-only" => cfg.mq4_only = true,
            other => return Err(format!("unknown argument {other}\n{}", usage())),
        }
        i += 1;
    }
    if cfg.codeobject.is_empty() {
        return Err(usage());
    }
    if cfg.timing {
        if cfg.reps < 20 {
            return Err("--timing requires --reps >= 20".into());
        }
        if cfg.warmup < 3 {
            return Err("--timing requires --warmup >= 3".into());
        }
        if cfg.launches < 1 {
            return Err("--timing requires --launches >= 1".into());
        }
        if cfg.samples_dir.is_some() {
            return Err("--timing is a separate run: no --samples-dir".into());
        }
    }
    if cfg.mq4_only && !cfg.only.is_empty() {
        return Err("--mq4-only is the full-coverage MQ4 regression: --only (a diagnostic filter that waives coverage) is not allowed".into());
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

fn run() -> Result<(Value, bool)> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let cfg = parse_args(&argv)?;
    MQ4_ONLY.store(cfg.mq4_only, Ordering::Relaxed);
    if !cfg.no_reclaim_arm {
        // Production Qwen4 host-mapped configuration; must precede the first HIP runtime call.
        hip_bridge::keep_host_memory_out_of_reclaim("gfx1100 G1 host-mapped weights");
    }
    let hip = HipRuntime::load().map_err(err)?;
    if let Some(d) = cfg.device {
        hip.set_device(d).map_err(err)?;
    }
    let id = hip.current_device().map_err(err)?;
    let uuid = hip.device_uuid(id).map_err(err)?;
    let bdf = hip.device_pci_bus_id(id).map_err(err)?.to_ascii_lowercase();
    let arch = hip.get_arch(id).map_err(err)?;
    let identity_match =
        uuid == cfg.expect_uuid && bdf == cfg.expect_bdf && arch == cfg.expect_arch;
    let device = json!({"index": id, "uuid": uuid, "bdf": bdf, "arch": arch, "card": cfg.card,
        "expected_uuid": cfg.expect_uuid, "expected_bdf": cfg.expect_bdf, "expected_arch": cfg.expect_arch,
        "identity_match": identity_match});
    let exe = std::env::current_exe().ok();
    let exe_sha = exe
        .as_ref()
        .and_then(|p| file_sha256(&p.to_string_lossy()).ok())
        .map(|(h, _)| h);
    let command = json!({"argv": argv, "exe": exe.as_ref().map(|p| p.display().to_string()), "exe_sha256": exe_sha,
        "cwd": std::env::current_dir().ok().map(|p| p.display().to_string()), "build_command": BUILD_COMMAND,
        "env": {"FABRIC_CARD": env_nonempty("FABRIC_CARD"), "FABRIC_OWNER": env_nonempty("FABRIC_OWNER"),
            "FABRIC_WRAPPER": env_nonempty("FABRIC_WRAPPER"),
            "HSA_USERPTR_FOR_PAGED_MEM": env_nonempty("HSA_USERPTR_FOR_PAGED_MEM"),
            "GPU_PINNED_MIN_XFER_SIZE": env_nonempty("GPU_PINNED_MIN_XFER_SIZE"),
            "reclaim_arm": !cfg.no_reclaim_arm}});
    let pcie = pcie_info(&bdf);
    if !identity_match {
        return Ok((
            json!({"schema": SCHEMA, "command": command, "device": device, "pcie": pcie,
                "error": "MQ-I4 MoE GEMV device identity mismatch", "byte_exact": false, "pass": false, "rc": 1}),
            false,
        ));
    }
    let vram_before = hip.get_vram_info().ok();
    let image = std::fs::read(&cfg.codeobject).map_err(|e| format!("{}: {e}", cfg.codeobject))?;
    let codeobject_sha = hex(&sha(&image));
    let module = hip.module_load_data(&image).map_err(err)?;
    let mut funcs = Vec::new();
    let mut loads = Vec::new();
    for kind in Kind::ALL {
        for codec in Codec::active() {
            let name = symbol(codec, kind);
            let f = hip
                .module_get_function(&module, &name)
                .map_err(|e| format!("symbol {name}: {e}"))?;
            loads.push(name);
            funcs.push((codec, kind, f));
        }
    }
    let mut ctx = Ctx {
        hip,
        funcs,
        device: id,
        old_parity: cfg.old_parity,
        gpu: None,
    };
    let mut totals = Totals::default();
    let mut rows: Vec<Value> = Vec::new();
    let mut runs: Vec<Value> = Vec::new();
    let mut samples_meta = Value::Null;
    let mut mq4_violations = 0u64;
    let mut comparison = Value::Null;

    if cfg.timing {
        let tcfg = TimingCfg {
            reps: cfg.reps,
            warmup: cfg.warmup,
            launches: cfg.launches,
        };
        let fabric = json!({"class": "timing", "card": cfg.fabric_card.clone().unwrap_or_else(|| cfg.card.clone()),
            "owner": cfg.fabric_owner, "wrapper": cfg.fabric_wrapper});
        let device_rec = json!({"arch": arch, "card": cfg.card, "uuid": uuid, "bdf": bdf});
        // (profile, placement, codec_name) -> median seconds per launch, for the comparison block.
        let mut medians: BTreeMap<(String, String, String), f64> = BTreeMap::new();
        let mut record = |runs: &mut Vec<Value>,
                          codec_name: &str,
                          kind: Kind,
                          sym: String,
                          codec: Option<Codec>,
                          placement: Placement,
                          slots: usize,
                          unique: usize,
                          elapsed: Vec<f64>,
                          old: bool| {
            let rows_per_launch = kind.m() * slots;
            let weights_bytes = codec.map_or(Kind::tensor_bytes(kind, Codec::Mq4), |c| {
                Kind::tensor_bytes(kind, c)
            }) * unique;
            let med = median(&elapsed);
            let per_launch = med / tcfg.launches as f64;
            medians.insert(
                (
                    kind.timing_profile().to_string(),
                    placement.label().to_string(),
                    codec_name.to_string(),
                ),
                per_launch,
            );
            runs.push(json!({
                "codec": codec_name, "profile": kind.timing_profile(), "placement": placement.label(),
                "symbol": sym, "m": kind.m(), "k": kind.k(), "M": kind.m(), "K": kind.k(),
                "row_count": rows_per_launch, "c": 1, "slots": slots, "unique_experts": unique,
                "weights_bytes_per_launch": weights_bytes,
                "elapsed_s": elapsed, "reps": tcfg.reps, "warmup": tcfg.warmup,
                "launches_per_sample": tcfg.launches,
                "median_elapsed_s": med, "median_s_per_launch": per_launch,
                "median_us_per_row": per_launch / rows_per_launch as f64 * 1e6,
                "median_weight_gb_per_s": weights_bytes as f64 / per_launch / 1e9,
                "symbol_codeobject": if old { Value::Null } else { json!(MODULE) },
                "code_object_sha": if old { Value::Null } else { json!(codeobject_sha) },
                "kernel_source": if old { "existing gfx1100 HIP kernel via rdna_compute::Gpu (comparator)" } else { "PeaceMaker PM gfx1100 code object" },
                "timing_source": "device", "event_timing": "hipEventElapsedTime over launches_per_sample launches",
                "device": device_rec, "fabric": fabric,
                "link_caveat": LINK_CAVEAT,
                "verified_byte_exact_before_timing": true,
            }));
        };
        for codec in Codec::active() {
            // 10 distinct experts of one codec, C1: one launch covers `M * 10` rows per profile.
            let name = format!("timing_c1_all_{}", codec.label());
            let mut rng = Rng::new(name_seed(&name));
            let ids = pick_ids(&mut rng, 10, true);
            let experts = experts_for(&ids, |_, _| codec);
            let mut route = ids.clone();
            rng.shuffle(&mut route);
            let case = build_case(
                &name,
                "synthetic",
                1,
                route,
                experts,
                false,
                codec == Codec::Mq4,
                0,
            );
            for placement in [Placement::Vram, Placement::HostMapped] {
                eprintln!("[timing] {name}@{}", placement.label());
                let o = RunOpts {
                    timing: Some(&tcfg),
                    controls: false,
                    tamper: false,
                };
                let (row, samples, old_samples) =
                    run_case(&mut ctx, &case, placement, &mut totals, &o)?;
                rows.push(row);
                for (c, kind, slots, unique, elapsed) in samples {
                    record(
                        &mut runs,
                        c.timing_name(),
                        kind,
                        symbol(c, kind),
                        Some(c),
                        placement,
                        slots,
                        unique,
                        elapsed,
                        false,
                    );
                }
                for (kind, elapsed) in old_samples {
                    let sym = match kind {
                        Kind::GateUp => "gemv_mq4g256v2_moe_gate_up_top10_indexed_batched",
                        Kind::Down => "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded",
                    };
                    record(
                        &mut runs,
                        "mq4_old_hip",
                        kind,
                        sym.to_string(),
                        Some(Codec::Mq4),
                        placement,
                        10,
                        10,
                        elapsed,
                        true,
                    );
                }
            }
        }
        let mut cmp = Vec::new();
        for kind in Kind::ALL {
            for placement in [Placement::Vram, Placement::HostMapped] {
                let get = |codec: &str| {
                    medians
                        .get(&(
                            kind.timing_profile().to_string(),
                            placement.label().to_string(),
                            codec.to_string(),
                        ))
                        .copied()
                };
                let (w2, mq4, old) = (get("mq2i4"), get("mq4"), get("mq4_old_hip"));
                cmp.push(json!({"profile": kind.timing_profile(), "placement": placement.label(),
                    "pm_mq2i4_s_per_launch": w2, "pm_mq4sym_s_per_launch": mq4, "old_mq4_hip_s_per_launch": old,
                    "pm_mq4sym_over_old_mq4": match (mq4, old) { (Some(a), Some(b)) if b > 0.0 => json!(a / b), _ => Value::Null },
                    "pm_mq2i4_over_old_mq4": match (w2, old) { (Some(a), Some(b)) if b > 0.0 => json!(a / b), _ => Value::Null }}));
            }
        }
        comparison = Value::Array(cmp);
    } else {
        for name in SYNTH_CASES {
            if mq4_only() && name.contains("all_w2") {
                continue;
            }
            let placements: Vec<Placement> = placements_for_synthetic(name)
                .into_iter()
                .filter(|p| selected(&cfg, &format!("{name}@{}", p.label())))
                .collect();
            if placements.is_empty() {
                continue;
            }
            let case = build_synthetic(name);
            for placement in placements {
                eprintln!("[g1] {name}@{}", placement.label());
                let o = RunOpts {
                    timing: None,
                    controls: CONTROL_CASES.contains(name) && placement == Placement::Vram,
                    tamper: TAMPER_CASES.contains(name),
                };
                let (row, _, _) = run_case(&mut ctx, &case, placement, &mut totals, &o)?;
                rows.push(row);
            }
        }
        if let Some(dir) = &cfg.samples_dir {
            eprintln!("[g1] loading c24 samples from {dir}");
            let s = load_samples(dir)?;
            mq4_violations = s.mq4_violations;
            samples_meta = s.meta.clone();
            for case in sample_cases(&s)? {
                let mut placements = vec![Placement::Vram, Placement::HostMapped];
                if case.name == "c24s_c8_union_mixed" {
                    placements.push(Placement::Split);
                }
                for placement in placements {
                    if !selected(&cfg, &format!("{}@{}", case.name, placement.label())) {
                        continue;
                    }
                    eprintln!("[g1] {}@{}", case.name, placement.label());
                    let o = RunOpts {
                        timing: None,
                        controls: false,
                        tamper: false,
                    };
                    let (row, _, _) = run_case(&mut ctx, &case, placement, &mut totals, &o)?;
                    rows.push(row);
                }
            }
        }
    }

    let vram_after = ctx.hip.get_vram_info().ok();
    let full = cfg.only.is_empty();
    let coverage_ok = cfg.timing
        || !full
        || (totals.control_checks > 0
            && totals.host_mapped_cases > 0
            && totals.split_cases > 0
            && (cfg.samples_dir.is_none() || totals.real_samples > 0));
    let byte_exact = totals.mismatches == 0
        && totals.twin_mismatches == 0
        && totals.old_kernel_mismatches == 0
        && !rows.is_empty()
        && (!cfg.old_parity || totals.old_kernel_cases > 0);
    let controls_pass = totals.control_failures == 0;
    let pass = byte_exact && controls_pass && coverage_ok;
    let mut out = json!({
        "schema": SCHEMA,
        "mode": if cfg.timing { "timing" } else { "g1" },
        "module": MODULE,
        "command": command,
        "config": {"samples_dir": cfg.samples_dir, "old_kernel_parity": cfg.old_parity, "only": cfg.only,
            "mq4_only": cfg.mq4_only,
            "full_gate_coverage_required": full && !cfg.timing,
            "timing": cfg.timing, "reps": cfg.reps, "warmup": cfg.warmup, "launches_per_sample": cfg.launches,
            "default_samples_dir": DEFAULT_SAMPLES_DIR},
        "codeobject": {"path": cfg.codeobject, "sha256": codeobject_sha, "bytes": image.len()},
        "code_object_sha": codeobject_sha,
        "kernarg_bytes": KERNARG_BYTES,
        "symbols_loaded": loads,
        "device": device,
        "pcie": pcie,
        "link_caveat": LINK_CAVEAT,
        "vram": {"before_free_total": vram_before, "after_free_total": vram_after},
        "case_counts": {"total": totals.cases, "synthetic": totals.synthetic, "c24_samples": totals.real_samples,
            "host_mapped": totals.host_mapped_cases, "split_vram_host": totals.split_cases,
            "old_kernel_parity": totals.old_kernel_cases},
        "symbol_launches": totals.launches,
        "cases": rows,
        "mismatch_count": totals.mismatches,
        "twin_selfcheck_mismatch_count": totals.twin_mismatches,
        "old_kernel_mismatch_count": totals.old_kernel_mismatches,
        "negative_controls": {"checks": totals.control_checks, "failures": totals.control_failures,
            "pass": controls_pass},
        "mq4_sym_header_violations_samples": mq4_violations,
        "coverage_ok": coverage_ok,
        "byte_exact": byte_exact,
        "pass": pass,
        "rc": if pass { 0 } else { 1 },
    });
    if !samples_meta.is_null() {
        out["c24_samples"] = samples_meta;
    }
    if cfg.timing {
        out["runs"] = Value::Array(runs);
        out["timing_comparison"] = comparison;
    }
    Ok((out, pass))
}

fn main() {
    match run() {
        Ok((v, pass)) => {
            println!("{v}");
            if !pass {
                eprintln!("MQ-I4 MoE GEMV gfx1100 G1 FAILED");
                std::process::exit(1);
            }
        }
        Err(e) => {
            println!(
                "{}",
                json!({"schema": SCHEMA, "error": e, "byte_exact": false, "pass": false, "rc": 1})
            );
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------------------
// Local known-answer tests (no GPU):
//   cargo test -p hipfire-runtime --release --features lab --example mq_i4_moe_gemv_gfx1100_g1
// ---------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cyc(n: usize) -> Vec<u8> {
        (0..n).map(|j| (j % 16) as u8).collect()
    }

    #[test]
    fn mq4_gate_known_answer() {
        let (m, k) = (3usize, 256usize);
        let mut codes = cyc(m * k);
        codes[k..2 * k].fill(8); // row 1: d = 0 everywhere
        let mut scales = vec![0x3c00u16; m * 2];
        scales[2 * 2 + 1] = 0x4000; // row 2: half0 scale 1, half1 scale 2
        let bytes = encode_mq4(Kind::GateUp, m, k, &codes, &scales);
        let x = vec![1.0f32; k];
        let got: Vec<f32> = (0..m)
            .map(|r| mq4_row_gate(&bytes[r * 136..(r + 1) * 136], &x))
            .collect();
        assert_eq!(got, vec![-128.0, 0.0, -192.0]);
    }

    #[test]
    fn mq4_down_known_answer() {
        let (m, k) = (2usize, 640usize);
        let mut codes = cyc(m * k);
        codes[k..].fill(8);
        let scales = vec![0x3800u16; m * (k / 128)];
        let bytes = encode_mq4(Kind::Down, m, k, &codes, &scales);
        let x = vec![2.0f32; k];
        let rb = (k / 128) * 68;
        let got: Vec<f32> = (0..m)
            .map(|r| mq4_row_down(&bytes[r * rb..(r + 1) * rb], &x))
            .collect();
        assert_eq!(got, vec![-320.0, 0.0]);
    }

    #[test]
    fn lane_tree_order_is_the_shuffle_tree() {
        let mut lanes = [0.0f32; 32];
        lanes[0] = 16777216.0;
        lanes[1] = 1.0;
        lanes[2] = 1.0;
        lanes[3] = 1.0;
        assert_eq!(lane_tree(lanes), 16777218.0);
        assert_eq!(fold_lanes(lanes, Mut::LaneFoldSeq), 16777216.0);
    }

    /// The local W2 tree transcription equals `mq_i4_reference` bit-for-bit and every applicable
    /// mutation changes at least one row on full-mantissa data (control sensitivity).
    #[test]
    fn w2_transcription_matches_reference_and_mutants_differ() {
        for kind in Kind::ALL {
            let (m, k) = (48usize, kind.k());
            let profile = kind.w2_profile();
            let mut rng = Rng::new(5);
            let mut codes = vec![0u8; m * k];
            for c in codes.iter_mut() {
                *c = (rng.u32() & 3) as u8;
            }
            let groups = k / 128;
            let scales: Vec<u16> = (0..m * groups)
                .map(|_| 0x2800 + (rng.u32() as u16 & 0x3ff))
                .collect();
            let bytes = mq_i4::encode_tensor(profile, m, k, &codes, &scales).unwrap();
            let x = gen_x_row(0, k, &mut rng);
            let want = mq_i4_reference::decode_gemv_f32(profile, m, k, &bytes, &x);
            let row = |r: usize, mu: Mut| {
                ref_row_m(
                    profile,
                    &codes[r * k..(r + 1) * k],
                    &scales[r * groups..(r + 1) * groups],
                    &x,
                    mu,
                )
            };
            for (r, w) in want.iter().enumerate() {
                assert_eq!(
                    row(r, Mut::None).to_bits(),
                    w.to_bits(),
                    "{} row {r}",
                    kind.suffix()
                );
            }
            for mu in Mut::ARITH {
                if !mu.applies(Codec::W2, kind) {
                    continue;
                }
                let differing = (0..m)
                    .filter(|&r| row(r, mu).to_bits() != want[r].to_bits())
                    .count();
                assert!(
                    differing > 0,
                    "{} not sensitive on {}",
                    mu.name(),
                    kind.suffix()
                );
            }
        }
    }

    #[test]
    fn mq4_mutants_differ_on_random_data() {
        for kind in Kind::ALL {
            let (m, k) = (16usize, kind.k());
            let mut rng = Rng::new(9);
            let codes: Vec<u8> = (0..m * k).map(|_| (rng.u32() & 15) as u8).collect();
            let scales: Vec<u16> = (0..m * (k / 128))
                .map(|_| 0x2800 + (rng.u32() as u16 & 0x3ff))
                .collect();
            let bytes = encode_mq4(kind, m, k, &codes, &scales);
            let x = gen_x_row(0, k, &mut rng);
            let rb = kind.mq4_row_bytes();
            let ev = |r: usize, mu: Mut| match kind {
                Kind::GateUp => mq4_row_gate_m(&bytes[r * rb..(r + 1) * rb], &x, mu),
                Kind::Down => mq4_row_down_m(&bytes[r * rb..(r + 1) * rb], &x, mu),
            };
            for mu in Mut::ARITH {
                if !mu.applies(Codec::Mq4, kind) {
                    continue;
                }
                let differing = (0..m)
                    .filter(|&r| ev(r, mu).to_bits() != ev(r, Mut::None).to_bits())
                    .count();
                assert!(
                    differing > 0,
                    "{} not sensitive on {}",
                    mu.name(),
                    kind.suffix()
                );
            }
        }
    }

    #[test]
    fn kernarg_layout_is_the_frozen_96_bytes() {
        let a = kernarg([1, 2, 3, 4, 5, 6, 7], 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq!(a.len(), 96);
        assert_eq!(
            u32::from_le_bytes(a[88..92].try_into().unwrap()),
            N_EXPERTS as u32
        );
        assert_eq!(u32::from_le_bytes(a[92..96].try_into().unwrap()), 0);
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
    }

    #[test]
    fn synthetic_tensors_have_exact_sizes_and_valid_w2() {
        for kind in Kind::ALL {
            for codec in Codec::ALL {
                assert_eq!(
                    gen_tensor(codec, kind, 5, true).len(),
                    kind.tensor_bytes(codec)
                );
            }
            mq_i4::decode_tensor(
                kind.w2_profile(),
                kind.m(),
                kind.k(),
                &gen_tensor(Codec::W2, kind, 5, true),
            )
            .unwrap();
        }
        assert_eq!(Kind::GateUp.tensor_bytes(Codec::W2), 870_400);
        assert_eq!(Kind::Down.tensor_bytes(Codec::W2), 435_200);
        assert_eq!(Kind::GateUp.tensor_bytes(Codec::Mq4), 1_740_800);
        assert_eq!(Kind::Down.tensor_bytes(Codec::Mq4), 870_400);
    }

    #[test]
    fn tampered_experts_stay_valid_and_differ() {
        for codec in Codec::ALL {
            let e = gen_expert(3, codec);
            let t = tamper_expert(&e).unwrap();
            assert_ne!(t.gate, e.gate);
            assert_ne!(t.down, e.down);
            assert_eq!(t.gate.len(), e.gate.len());
            if codec == Codec::W2 {
                mq_i4::decode_tensor(MqI4Profile::MQ2_R256, GATE_M, GATE_K, &t.gate).unwrap();
                mq_i4::decode_tensor(MqI4Profile::MQ2_R128, DOWN_M, DOWN_K, &t.down).unwrap();
            }
        }
    }

    #[test]
    fn buckets_partition_slots() {
        let mut rng = Rng::new(11);
        let pool = pick_ids(&mut rng, 14, true);
        let codecs: BTreeMap<u32, Codec> = pool
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, MIXED10[i % 10]))
            .collect();
        let route = lane_routes(&mut rng, &pool, 8);
        let f = |e: u32| codecs[&e];
        let bs = [
            build_bucket(&route, &f, Codec::W2),
            build_bucket(&route, &f, Codec::Mq4),
        ];
        let mut seen = vec![false; route.len()];
        for b in &bs {
            assert_eq!(b.offs.len(), b.ids.len() + 1);
            for &s in &b.slots {
                assert!(!seen[s as usize]);
                seen[s as usize] = true;
            }
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn bump_scale_is_finite_nonnegative_and_different() {
        for sc in [0u16, 1, 0x03ff, 0x0400, 0x3bff, 0x4000, 0x6fff, 0x7bff] {
            let b = bump_scale(sc);
            assert_ne!(b, sc);
            assert!(b & 0x8000 == 0 && b & 0x7c00 != 0x7c00);
        }
    }
}
