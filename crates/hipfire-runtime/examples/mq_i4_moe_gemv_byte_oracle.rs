// SPDX-License-Identifier: Apache-2.0
//! Raw PM K1 MoE decode-GEMV G1 byte oracle; never participates in runtime dispatch.
//!
//! Loads the per-arch admitted narrow code object and its admitted narrow symbols and launches
//! them with the frozen 96-byte F4 kernarg ABI (DESIGN §4.2 plus `n_experts@88` and a zero
//! word at 92, block 32, grid `(M, U, 1)`) and compares every output word bit-for-bit
//! (signed zeros and NaN payloads included) with a CPU oracle:
//!
//! * W3 (`mq3i4`): `mq_i4_reference::decode_moe_slots` (cross-checked against
//!   `decode_gemv_f32` on a bounded slot sample).
//! * MQ4 (`mq4sym`): [`mq4_row_gate`]/[`mq4_row_down`], a transcription of the pinned legacy
//!   MQ4 `.hip` arithmetic (`mq_i4_reference` supports only W3/W2): per 8 (gate) / 4 (down)
//!   elements `d_i = fma(sc, float(q_i), zp)`, `s = RN(d1*x1)` then `fma(d0,x0,s)` then
//!   `fma(d_i,x_i,s)` ascending (the hipcc `-ffp-contract=fast` orientation audited on
//!   gfx1201 ISA for both `qwen4_gemv_mq4g256v2_moe_gate_up_k8_indexed_batched.hip`
//!   {k8 loop, k2560 rows8 helper} and `gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded.hip`),
//!   gate: `acc[g&3] += s` over G256 groups folded `(a0+a1)+(a2+a3)`, down: one chain over
//!   G128 groups, then the 16,8,4,2,1 wave shuffle tree. `--old-kernel-parity` additionally runs the
//!   production HIP kernels (JIT through `rdna_compute::Gpu`) on all-MQ4 fixtures and compares them
//!   bit-for-bit with this CPU oracle, so the transcription never rests on the ISA audit alone.
//! * E8P is NOT in this narrow arm: E8P slab experts use the planar `[words M*K/8 u16][scales
//!   M*K/128 F16]` wire of the sibling `e8p_i4_moe_gemv_oracle` and its frozen K-E0 lane-pair
//!   twin `hipfire_runtime::e8p_i4_reference::decode_gemv_f32_lanepair`; the narrow raw
//!   singleton below is never run on E8P bytes.
//!
//! MQ4 headers are symmetric (`zp = f16(-8*sc)`), so the legitimate finite maximum MQ4 scale is
//! f16 0x6fff (8188; `-8*sc` = -65504); W3 fixtures use the full f16 max 0x7bff (65504).
//! E8P scale plane entries pass `e8p_i4_reference::writer_guard` (finite, `>= 2^-14`).
//!
//! Build:
//!   cargo build -p hipfire-runtime --release --features lab,rdna-compute/lab --example mq_i4_moe_gemv_byte_oracle
//! G1 (expected card, fixtures first, then optional real layer):
//!   target/release/examples/mq_i4_moe_gemv_byte_oracle --codeobject /abs/ADMITTED-narrow.hxaco \
//!     [--artifact /abs/ctl3-hot.hfq --layer 0 [--real-experts 24]] [--old-kernel-parity] \
//!     [--device N] [--card NAME] [--expect-uuid GPU-… --expect-bdf 0000:… --expect-arch gfxARCH] [--wide|--wide-only|--wide-slab-only [--wide-c 1-64]] [--only SUBSTR]…
//!   Admitted narrow images: `mq_i4_moe_gemv_pm_gfx1201.hxaco` (symbols plain, as spelled by
//!   `rdna_compute::slab_moe::MQ_I4_MOE_GEMV_SYMBOLS`), `mq_i4_moe_gemv_pm_gfx1151.hxaco`
//!   (plain symbols), `kernels/mq_i4_moe_gemv_mq4only_pm_gfx1100.hxaco` (MQ4-only, suffixed
//!   symbols `mq_i4_moe_gemv_mq4sym_{r256_gate_up,r128_down}_pm_gfx1100`). gfx1151 W3 uses the
//!   same shared image as its slabs; gfx1100 has NO W3 narrow symbols (only MQ4).
//! Certified R1 records (real weights, no packed HFQM; cases labelled origin `real_records_layer`):
//!   … --records-dir …/r1/q/w3-gptq3/L0 --records-manifest …/r1/pack/w3-gptq3/L0/manifest.json \
//!     --alloc …/alloc.ctl3hot.json [--point ctl3-hot] --xts /models/….mq4 --layer0 \
//!     [--map …/ctl3-hot.map.json] [--records-manifest-sha256 HEX] [--xts-sha256 HEX] [--real-experts 24]
//!   Cold = W3 experts imported with `I4ExpertFinalCodeRecord::import_npz_file` (NPZ sha256 and encoded
//!   gate++down sha256 checked against the manifest); hot = MQ4-XTS per-expert slices of the XTS aggregate
//!   tensors read through `HfqFile`. Activations stay synthetic. Runs after the 13 synthetic cases.
//! Timing (separate fabric run after G1 passed; raw device-event samples, R6 takes the median):
//!   … --codeobject … --timing --reps 30 --warmup 5 [--launches 32] --card NAME \
//!     --fabric-owner OWNER --fabric-wrapper /path/fabric-run.sh [--fabric-card NAME]
//!   (`--card`/`--fabric-owner`/`--fabric-wrapper` fall back to env `FABRIC_CARD`/`FABRIC_OWNER`/`FABRIC_WRAPPER`.)
//! Prints one machine JSON line on stdout; exit code 1 on any mismatch or error.
//!
//! Whole-chain row tiles (`--wide`, `--wide-only`, `--wide-slab-only`, `--wide-c LIST`,
//! default R=1..64): the actual PM planner writes 482 words per eight-row tile, including
//! absolute slot IDs. Every descriptor word is checked independently against first-occurrence
//! CPU semantics before gate/up consumes it; the same immutable descriptors feed down after the
//! production fused BF16 gate/up round trips, SiLU multiply, BF16 activation round trip and
//! rotation. Production combine and its BF16 round trip are compared against sequential actual
//! runtime-selected singleton chains on the same device and embedded objects. Mixed W3/MQ4,
//! mixed E8P/MQ4, pure-bucket (`purew3_`, `puremq4_`, pure E8P-cold `e8p_` disjoint) and
//! all-MQ4 legacy indexed chains cover pool, identical, tile-disjoint and router-tie routes
//! at every R; invalid-ID prefixes (`invalid_`, `e8p_invalid_`) cover every admitted cold
//! codec, including pure-E8P-invalid disjoint.
//! The routed-only activation uses shared=None; this receipt does not claim shared-expert fusion
//! coverage.
//! A requested nonempty `--wide-c` subset can pass only after every admitted Cartesian case
//! at those rows passes; it is COMPONENT_PASS, not whole-G1 coverage. Full coverage still
//! requires every R=1..64 (3072 slab cells on gfx1201/gfx1151). The first failing wide case
//! stops the sweep and remains in the JSON with its byte/guard counterexample and missing cells.
//! gfx1151 explicitly selected with `--expect-arch gfx1151` requires actual BDF+arch, but
//! does not require ROCr UUID unless `--expect-uuid` is supplied. Discrete defaults retain
//! their UUID pin; local discrete runs must supply their authoritative UUID and BDF.
//! Integration: run the build above and local CPU test command below, then run the slab-only
//! sweep (`--wide-slab-only` with exact device identity and an independently pinned admitted
//! narrow --codeobject; `--wide-only --wide-c 1-64` plus `--only` filters remain available for
//! diagnostics). Repeat on each admitted architecture and with HIPFIRE_QWEN4_SLAB_K1_C1=0/1 on
//! gfx1151. Disjointness is per tile (80 experts); reuse across tiles keeps 640 slots legal in
//! 512 experts. Pure-bucket prefixes (`purew3_`, `puremq4_` plus second seeds) run pure W3
//! cold / pure MQ4 on every route axis (style no longer forces the codec for those prefixes);
//! `e8p_` disjoint is pure E8P-cold. Invalid prefixes (`invalid_`, `e8p_invalid_` plus second
//! seeds) cover every admitted cold codec, including pure-E8P-invalid disjoint. Pure W3 runs
//! where W3 is admitted and are enumerated NOT-APPLICABLE on gfx1100 (listed, not launched).
//! --artifact also accepts stock qt44/qt53 aggregate expert tensors. --layer 0..47 selects a
//! target layer; --layer 48 explicitly maps mtp.layers.0.mlp.experts.{gate_up_proj,down_proj}
//! and is labelled native-MTP-head diagnostic, never target-verify qualification.
use hip_bridge::{DeviceBuffer, Function, HipRuntime};
use hipfire_arch_qwen4::expert_slabs::{ExpertFormat, ExpertProj};
use hipfire_quantize::i4_records::{I4ExpertFinalCodeRecord, I4RecordShapes};
use hipfire_quantize::i4_slabs::{load_allocation, SlabGeometry};
use hipfire_runtime::e8p_i4_reference as e8p;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::mq_i4::{self, MqI4Profile, MqI4Rotation};
use hipfire_runtime::mq_i4_reference::{self, DecodeKind, ExpertTensor};
use rdna_compute::slab_moe::{self, SlabBuckets, SlabK1Module, SLAB_MAX_LANES as MAX_ROWS};
use rdna_compute::{qwen4_slab_codec_admitted, qwen4_slab_layer_cold_codec, DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::rc::Rc;

type Result<T, E = String> = std::result::Result<T, E>;
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

const SCHEMA: &str = "mq-i4-moe-gemv-byte-oracle-v1";
/// Module name of the device's admitted narrow code object (gfx1100 has MQ4-only symbols).
fn module_name(arch: &str) -> String {
    if arch == "gfx1100" {
        slab_moe::MQ_I4_MOE_GEMV_GFX1100_MODULE.to_string()
    } else {
        format!("mq_i4_moe_gemv_pm_{arch}")
    }
}
const BUILD_COMMAND: &str =
    "cargo build -p hipfire-runtime --release --features lab,rdna-compute/lab --example mq_i4_moe_gemv_byte_oracle";
const DEFAULT_UUID: &str = "GPU-6109a4cb5f833235";
const DEFAULT_BDF: &str = "0000:9a:00.0";
const DEFAULT_ARCH: &str = "gfx1201";
const KERNARG_BYTES: usize = 96;
const N_EXPERTS: usize = 512;
const GATE_M: usize = 1280;
const GATE_K: usize = 2560;
const DOWN_M: usize = 2560;
const DOWN_K: usize = 640;
/// Unwritten output words: the byte pattern `memset(0xa5)` leaves (a NaN, compared by bits).
const SENTINEL_BITS: u32 = 0xa5a5_a5a5;
const PAD_BITS: u32 = 0x7fc0_0000;

// ---------------------------------------------------------------------------------------
// Axes
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Codec {
    W3,
    Mq4,
    E8p,
}
impl Codec {
    /// Narrow raw-singleton codecs: the independent explicit --codeobject arm. E8P is slab-only
    /// and never enters the narrow raw singleton (no W3/MQ4-style raw symbol exists for it).
    const ALL: [Codec; 2] = [Codec::W3, Codec::Mq4];
    /// Every actually admitted slab codec: W3 (1), MQ4 (0), E8P (3). W2 has no variant and is
    /// never admitted on any arch.
    const SLAB_ALL: [Codec; 3] = [Codec::W3, Codec::Mq4, Codec::E8p];
    /// Slab route byte of this codec (the per-expert codec table vocabulary).
    fn route_byte(self) -> u8 {
        match self {
            Codec::W3 => 1,
            Codec::Mq4 => 0,
            Codec::E8p => 3,
        }
    }
    fn tag(self) -> &'static str {
        match self {
            Codec::W3 => "mq3i4",
            Codec::Mq4 => "mq4sym",
            Codec::E8p => "e8p",
        }
    }
    fn label(self) -> &'static str {
        match self {
            Codec::W3 => "w3",
            Codec::Mq4 => "mq4",
            Codec::E8p => "e8p",
        }
    }
    /// Codec name of the R6 timing records.
    fn timing_name(self) -> &'static str {
        match self {
            Codec::W3 => "mq3i4",
            Codec::Mq4 => "mq4",
            Codec::E8p => "e8p",
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
    /// Profile name of the R6 timing records.
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
    fn w3_profile(self) -> MqI4Profile {
        match self {
            Kind::GateUp => MqI4Profile::MQ3_R256,
            Kind::Down => MqI4Profile::MQ3_R128,
        }
    }
    /// Bytes of one expert's projection tensor. E8P is the planar wire `[words M*K/8 u16]
    /// [scales M*K/128 F16]` (gate/up 870400 B, down 435200 B).
    fn tensor_bytes(self, codec: Codec) -> usize {
        match codec {
            Codec::W3 => mq_i4::tensor_bytes(self.w3_profile(), self.m(), self.k())
                .expect("FN shapes are legal W3 shapes"),
            Codec::Mq4 => match self {
                Kind::GateUp => GATE_M * (GATE_K / 256) * 136,
                Kind::Down => DOWN_M * (DOWN_K / 128) * 68,
            },
            Codec::E8p => match self {
                Kind::GateUp => GATE_M * GATE_K / 4 + 2 * GATE_M * GATE_K / 128,
                Kind::Down => DOWN_M * DOWN_K / 4 + 2 * DOWN_M * DOWN_K / 128,
            },
        }
    }
    fn e8p_rotation(self) -> MqI4Rotation {
        match self {
            Kind::GateUp => MqI4Rotation::R256,
            Kind::Down => MqI4Rotation::R128,
        }
    }
}
/// Narrow independent raw-singleton symbol. E8P has no narrow arm: this panics on E8P rather
/// than substituting a W3/MQ4 raw kernel on E8P bytes.
fn symbol(codec: Codec, kind: Kind) -> String {
    match codec {
        Codec::W3 | Codec::Mq4 => format!("mq_i4_moe_gemv_{}_{}", codec.tag(), kind.suffix()),
        Codec::E8p => panic!("E8P has no narrow raw singleton symbol"),
    }
}

/// Sentinel words guarding both ends of every wide / reference output buffer.
const GUARD_WORDS: usize = 64;

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

// ---------------------------------------------------------------------------------------
// MQ4 exact CPU oracle (transcription of the pinned legacy MQ4 HIP kernels)
// ---------------------------------------------------------------------------------------

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

/// One lane's eight-element group dot: `d_i = fma(sc, q_i, zp)`; `RN(d1*x1)`, `fma(d0,x0)`,
/// then `fma(d_i, x_i)` for i = 2..8 (the "element 1 first" contraction hipcc picks on gfx1201).
fn group_dot8(sc: f32, zp: f32, pk: u32, x: &[f32]) -> f32 {
    let d = |i: usize| sc.mul_add(((pk >> (4 * i)) & 15) as f32, zp);
    let mut s = d(1) * x[1];
    s = d(0).mul_add(x[0], s);
    for i in 2..8 {
        s = d(i).mul_add(x[i], s);
    }
    s
}

/// MQ4G256V2 gate/up row: `row.len() / 136` groups of `[sc0 zp0 sc1 zp1 : f16][128 nibble bytes]`.
/// Lane `tid` reads header half `tid >= 16`, u32 `tid` of the payload (elements `tid*8..+8`).
fn mq4_row_gate(row: &[u8], x: &[f32]) -> f32 {
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
            acc[g & 3] += group_dot8(sc, zp, pk, &x[xb..xb + 8]);
        }
        *lane = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    }
    lane_tree(lanes)
}

/// MQ4G128V2 down row: `row.len() / 68` groups of `[sc zp : f16][64 nibble bytes]`; lane `tid`
/// owns bytes `2*tid, 2*tid+1` (elements `tid*4..+4`), one accumulation chain.
fn mq4_row_down(row: &[u8], x: &[f32]) -> f32 {
    let groups = row.len() / 68;
    debug_assert_eq!(x.len(), groups * 128);
    let mut lanes = [0.0f32; 32];
    for (tid, lane) in lanes.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for g in 0..groups {
            let gp = &row[g * 68..(g + 1) * 68];
            let sc = f16f(u16::from_le_bytes([gp[0], gp[1]]));
            let zp = f16f(u16::from_le_bytes([gp[2], gp[3]]));
            let (b0, b1) = (gp[4 + tid * 2], gp[5 + tid * 2]);
            let q = [b0 & 15, b0 >> 4, b1 & 15, b1 >> 4];
            let d = |i: usize| sc.mul_add(q[i] as f32, zp);
            let xb = g * 128 + tid * 4;
            let mut s = d(1) * x[xb + 1];
            s = d(0).mul_add(x[xb], s);
            s = d(2).mul_add(x[xb + 2], s);
            s = d(3).mul_add(x[xb + 3], s);
            acc += s;
        }
        *lane = acc;
    }
    lane_tree(lanes)
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
    let row_bytes = match kind {
        Kind::GateUp => (k / 256) * 136,
        Kind::Down => (k / 128) * 68,
    };
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

/// Scale bits for a row/group class. W3: f16 max 0x7bff is legal; MQ4 (symmetric zero `-8*sc`)
/// tops out at 0x6fff; E8P must pass `e8p_i4_reference::writer_guard` (finite, `>= 2^-14`),
/// so subnormal/zero E8P classes use the guarded edge vocabulary. Class 3 is the zero scale
/// (codes forced to the zero-scale contract; for E8P class 3 maps to the smallest normal
/// `0x0400` because the writer routes zero/subnormal E8P to MQ4).
fn class_scale(class: usize, codec: Codec, rng: &mut Rng) -> u16 {
    let max = match codec {
        Codec::W3 => 0x7bff,
        Codec::Mq4 => 0x6fff,
        Codec::E8p => 0x7bff,
    };
    match class {
        1 => max,
        2 if codec == Codec::E8p => 0x0400 + rng.below(0x100) as u16,
        2 => 1 + rng.below(0x3ff) as u16, // f16 subnormal
        3 if codec == Codec::E8p => 0x0400,
        3 => 0,
        4 => 0x0400, // smallest normal
        5 if codec == Codec::E8p => 0x0401,
        5 => 0x0001, // smallest subnormal
        6 => 0x3c00,
        _ => (((8 + rng.below(13)) as u16) << 10) | (rng.u32() as u16 & 0x3ff),
    }
}
/// Deterministic E8P planar wire `[words M*K/8 u16 LE][scales M*K/128 F16 LE]` for `(id, kind)`
/// using the sibling oracle's writer vocabulary: full word sweep styles by `id % 4` (uniform
/// random words, exhaustive sequential words, extreme all-5 words, parity-stride words), and
/// the guarded scale classes above; `special` spreads the writer edge scale vocabulary over
/// (row, group). `special == false` (chain fixtures) uses bounded ordinary scales and the
/// uniform random word style; every scale passes `e8p::writer_guard`.
fn gen_e8p_tensor(kind: Kind, id: u32, special: bool) -> Vec<u8> {
    let (m, k) = (kind.m(), kind.k());
    let (nw, ns, groups) = (m * k / 8, m * k / 128, k / 128);
    let wm = if special { (id % 4) as usize } else { 0 };
    let mut rng = Rng::new(mix64(
        id as u64 * 8 + kind as u64 * 2 + Codec::E8p as u64 + 1,
    ));
    let salt = id as usize;
    let hi: Vec<u8> = (0..=255u8)
        .filter(|&i| e8p::E8P_ABS2[i as usize].contains(&5))
        .collect();
    let mut out = Vec::with_capacity(kind.tensor_bytes(Codec::E8p));
    for i in 0..nw {
        let w: u16 = match wm {
            1 => {
                let (j, b) = ((i % 65536) as u32, (i / 65536) as u32 + id);
                ((j.wrapping_mul(40503)).wrapping_add(b.wrapping_mul(977)) & 0xffff) as u16
            }
            2 => {
                let idx = hi[i % hi.len()];
                let sign7 = if (i / hi.len()) % 2 == 0 { 0 } else { 127 };
                e8p::pack_word(idx, sign7, (i / (2 * hi.len())) % 2 == 1)
            }
            3 => {
                let idx = ((i * 7 + id as usize) % 256) as u8;
                let sign7 = ((i * 37 + (i >> 8)) & 0x7f) as u8;
                e8p::pack_word(idx, sign7, i & 1 == 1)
            }
            _ => rng.u32() as u16,
        };
        out.extend_from_slice(&w.to_le_bytes());
    }
    for s in 0..ns {
        let (r, g) = (s / groups, s % groups);
        let class = if special {
            (r * 5 + g * 3 + salt) % 10
        } else {
            0
        };
        let d = class_scale(class, Codec::E8p, &mut rng);
        debug_assert!(e8p::writer_guard(d).is_ok());
        out.extend_from_slice(&d.to_le_bytes());
    }
    assert_eq!(out.len(), kind.tensor_bytes(Codec::E8p));
    out
}

/// Deterministic expert tensor for `(id, codec, kind)`. Every non-plain W3/MQ4 expert rotates the
/// code so each code value lands on every in-group position across rows, and spreads zero,
/// f16-max, subnormal and smallest-normal scales over (row, group). E8P uses the planar wire
/// generator above (no W3/MQ4 code table).
fn gen_tensor(codec: Codec, kind: Kind, id: u32, special: bool) -> Vec<u8> {
    if codec == Codec::E8p {
        return gen_e8p_tensor(kind, id, special);
    }
    let (m, k) = (kind.m(), kind.k());
    let groups = k / 128;
    let levels: usize = match codec {
        Codec::W3 => 7,
        Codec::Mq4 => 15,
        Codec::E8p => unreachable!("E8P returns through gen_e8p_tensor"),
    };
    let zero_code: u8 = match codec {
        Codec::W3 => 4,
        Codec::Mq4 => 0,
        Codec::E8p => unreachable!("E8P returns through gen_e8p_tensor"),
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
        Codec::W3 => mq_i4::encode_tensor(kind.w3_profile(), m, k, &codes, &scales)
            .expect("synthetic W3 tensor is valid"),
        Codec::Mq4 => encode_mq4(kind, m, k, &codes, &scales),
        Codec::E8p => unreachable!("E8P returns through gen_e8p_tensor"),
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
    /// Whole-chain fixed-eight-row oracle, independent of the production width selector.
    wide: bool,
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
        wide: false,
    }
}

#[derive(Clone, Default, Debug, PartialEq, Eq)]
struct Bucket {
    ids: Vec<u32>,
    offs: Vec<u32>,
    slots: Vec<u32>,
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

/// Slab buckets (cold bucket first): W3 (codec byte 1) or E8P (byte 3) according to the case's
/// own cold codec, MQ4 (byte 0) second. The cold bucket is empty exactly when the layer holds
/// none of that codec.
fn buckets_of(case: &Case) -> [Bucket; 2] {
    let f = |e: u32| case.experts[&e].codec;
    let cold = match case_cold_codec(case).expect("mixed slab fixtures hold at most one cold codec")
    {
        0 => Bucket {
            ids: Vec::new(),
            offs: vec![0],
            slots: Vec::new(),
        },
        1 => build_bucket(&case.route, &f, Codec::W3),
        3 => build_bucket(&case.route, &f, Codec::E8p),
        codec => panic!("unknown slab cold codec {codec}"),
    };
    [cold, build_bucket(&case.route, &f, Codec::Mq4)]
}
/// Route-byte membership of `id` under layer cold codec `cold_codec`: cold codec = 1, MQ4 = 0.
/// `cold_codec == 0` means no cold codec (an MQ4-only layer): every expert is MQ4 membership.
fn membership_of(codec: Codec, cold_codec: u8) -> u8 {
    if cold_codec == 0 || codec == Codec::Mq4 {
        0
    } else if codec.route_byte() == cold_codec {
        1
    } else if codec == Codec::W3 || codec == Codec::E8p {
        panic!("layer mixes two cold codecs");
    } else {
        panic!("no membership for unknown codec");
    }
}
/// The single cold codec (route byte: 1 = W3, 3 = E8P, 0 = none) of a case's experts: a slab
/// layer holds at most one cold codec next to MQ4 (`qwen4_slab_layer_cold_codec`); a W3+E8P
/// mix is `Err`, never a silently picked bucket.
fn case_cold_codec(case: &Case) -> Result<u8> {
    let mut cold: Option<u8> = None;
    for e in case.experts.values() {
        if e.codec == Codec::Mq4 {
            continue;
        }
        match cold {
            None => cold = Some(e.codec.route_byte()),
            Some(first) if first != e.codec.route_byte() => {
                return Err(format!(
                    "case {} holds two cold codecs ({first} and {})",
                    case.name,
                    e.codec.route_byte()
                ))
            }
            Some(_) => {}
        }
    }
    Ok(cold.unwrap_or(0))
}

fn pick_ids(rng: &mut Rng, n: usize, edges: bool) -> Vec<u32> {
    assert!(
        n <= N_EXPERTS,
        "cannot pick {n} distinct IDs from {N_EXPERTS}"
    );
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
        .map(|(i, &id)| (id, Rc::new(gen_expert(id, codec_of(i, id)))))
        .collect()
}

const MIXED10: [Codec; 10] = [
    Codec::W3,
    Codec::Mq4,
    Codec::Mq4,
    Codec::W3,
    Codec::W3,
    Codec::Mq4,
    Codec::W3,
    Codec::Mq4,
    Codec::Mq4,
    Codec::W3,
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

/// Random codec per id with `cold_tenths`/10 cold probability against MQ4; both codecs forced
/// present when mixed. The cold codec is `cold`.
fn mixed_cold_codecs(rng: &mut Rng, n: usize, cold: Codec, cold_tenths: usize) -> Vec<Codec> {
    assert!(cold == Codec::W3 || cold == Codec::E8p);
    let mut codecs: Vec<Codec> = (0..n)
        .map(|_| {
            if rng.below(10) < cold_tenths {
                cold
            } else {
                Codec::Mq4
            }
        })
        .collect();
    if cold_tenths > 0 && cold_tenths < 10 {
        codecs[0] = cold;
        codecs[n - 1] = Codec::Mq4;
    }
    codecs
}
/// Random codec per id with `w3_tenths`/10 W3 probability; both codecs forced present when mixed.
fn mixed_codecs(rng: &mut Rng, n: usize, w3_tenths: usize) -> Vec<Codec> {
    mixed_cold_codecs(rng, n, Codec::W3, w3_tenths)
}

const SYNTH_CASES: &[&str] = &[
    "c1_mixed_interleaved",
    "c1_mixed_stress_x",
    "c1_all_w3",
    "c1_all_mq4",
    "c1_mixed_1w3_9mq4",
    "c1_mixed_9w3_1mq4",
    "c3_mixed_partial_lanes",
    "c8_union_repeated_mixed",
    "c8_unique_80_mixed",
    "c8_identical_lanes_mixed",
    "c8_union_all_w3",
    "c8_union_all_mq4",
    "c8_padded_strides_mixed",
];

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
        "c1_all_w3" | "c1_all_mq4" => {
            let codec = if name == "c1_all_w3" {
                Codec::W3
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
        "c1_mixed_1w3_9mq4" | "c1_mixed_9w3_1mq4" => {
            let ids = pick_ids(&mut rng, 10, true);
            let single_w3 = name == "c1_mixed_1w3_9mq4";
            // Never an edge id, so the edge experts keep the majority codec.
            let lone = 1 + rng.below(8);
            let experts = experts_for(&ids, |i, _| {
                if (i == lone) == single_w3 {
                    Codec::W3
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
        | "c8_union_all_w3"
        | "c8_union_all_mq4"
        | "c8_padded_strides_mixed" => {
            let ids = pick_ids(&mut rng, 14, true);
            let codecs: Vec<Codec> = match name {
                "c8_union_all_w3" => vec![Codec::W3; ids.len()],
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
// Real artifact (one HFQM layer, selected expert records only)
// ---------------------------------------------------------------------------------------

fn expert_entry(layer: usize, expert: usize, proj: &str) -> String {
    format!("model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}")
}

struct Real {
    experts: BTreeMap<u32, Rc<Expert>>,
    meta: Value,
    mq4_sym_violations: u64,
}

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

fn legacy_aggregate_name(layer: usize, projection: ExpertProj) -> String {
    if layer == 48 {
        format!("mtp.layers.0.mlp.experts.{}", projection.suffix())
    } else {
        SlabGeometry::aggregate_name(layer, projection)
    }
}

fn load_real(path: &str, layer: usize, want: usize) -> Result<Real> {
    let hfq = HfqFile::open_at_offset(std::path::Path::new(path), 0).map_err(err)?;
    if hfq
        .find_tensor_info(&legacy_aggregate_name(layer, ExpertProj::GateUp))
        .is_some()
    {
        return load_legacy_aggregate(&hfq, path, layer, want);
    }
    let mut avail: BTreeMap<Codec, Vec<u32>> = BTreeMap::new();
    let (mut w2, mut other) = (0usize, 0usize);
    for e in 0..N_EXPERTS {
        let (gn, dn) = (
            expert_entry(layer, e, "gate_up_proj"),
            expert_entry(layer, e, "down_proj"),
        );
        let (Some(g), Some(d)) = (hfq.find_tensor_info(&gn), hfq.find_tensor_info(&dn)) else {
            continue;
        };
        let codec = match (g.quant_type, d.quant_type) {
            (131, 132) => Codec::W3,
            (44, 53) => Codec::Mq4,
            (133, 134) => {
                w2 += 1;
                continue;
            }
            _ => {
                other += 1;
                continue;
            }
        };
        let shape_ok =
            g.shape == [GATE_M as u32, GATE_K as u32] && d.shape == [DOWN_M as u32, DOWN_K as u32];
        let size_ok = g.data_size == Kind::GateUp.tensor_bytes(codec)
            && d.data_size == Kind::Down.tensor_bytes(codec);
        if !shape_ok || !size_ok {
            return Err(format!(
                "layer {layer} expert {e}: unexpected shape/size for {} (gate {:?}/{} down {:?}/{})",
                codec.label(),
                g.shape,
                g.data_size,
                d.shape,
                d.data_size
            ));
        }
        avail.entry(codec).or_default().push(e as u32);
    }
    let counts = |c: Codec| avail.get(&c).map_or(0, |v| v.len());
    if counts(Codec::W3) == 0 || counts(Codec::Mq4) == 0 {
        return Err(format!(
            "layer {layer} of {path} has no mixed W3/MQ4 experts (w3={}, mq4={}, w2={w2}, other={other})",
            counts(Codec::W3),
            counts(Codec::Mq4)
        ));
    }
    let want = want.max(10);
    let half = want / 2;
    let (a, b) = (avail[&Codec::W3].clone(), avail[&Codec::Mq4].clone());
    let mut take_w3 = half.min(a.len());
    let mut take_mq4 = (want - half).min(b.len());
    if take_w3 + take_mq4 < want {
        // One side is short: top up from the other.
        take_w3 = (want - take_mq4).min(a.len());
        take_mq4 = (want - take_w3).min(b.len());
    }
    let spread =
        |v: &Vec<u32>, n: usize| -> Vec<u32> { (0..n).map(|i| v[i * v.len() / n]).collect() };
    let (sel_w3, sel_mq4) = (spread(&a, take_w3), spread(&b, take_mq4));
    if sel_w3.len() + sel_mq4.len() < 10 {
        return Err(format!(
            "layer {layer}: only {} usable experts; a top-10 lane needs 10",
            sel_w3.len() + sel_mq4.len()
        ));
    }
    let mut experts = BTreeMap::new();
    let mut lines = String::new();
    let mut recs = Vec::new();
    let mut violations = 0u64;
    for (codec, ids) in [(Codec::W3, &sel_w3), (Codec::Mq4, &sel_mq4)] {
        for &id in ids {
            let gn = expert_entry(layer, id as usize, "gate_up_proj");
            let dn = expert_entry(layer, id as usize, "down_proj");
            let (gi, gate) = hfq
                .tensor_data_vec(&gn)
                .ok_or_else(|| format!("short read of {gn}"))?;
            let (gq, goff, gsz) = (gi.quant_type, gi.data_offset, gi.data_size);
            let (di, down) = hfq
                .tensor_data_vec(&dn)
                .ok_or_else(|| format!("short read of {dn}"))?;
            let (dq, doff, dsz) = (di.quant_type, di.data_offset, di.data_size);
            match codec {
                Codec::W3 => {
                    mq_i4::decode_tensor(MqI4Profile::MQ3_R256, GATE_M, GATE_K, &gate)
                        .map_err(|e| format!("{gn}: {e}"))?;
                    mq_i4::decode_tensor(MqI4Profile::MQ3_R128, DOWN_M, DOWN_K, &down)
                        .map_err(|e| format!("{dn}: {e}"))?;
                }
                Codec::Mq4 => {
                    violations += mq4_sym_violations(Kind::GateUp, &gate)
                        + mq4_sym_violations(Kind::Down, &down)
                }
                // This loader selects only the legacy W3/MQ4 real-record arm above.
                // Whole-slab E8P fixtures use their own planar wire and CPU twin.
                Codec::E8p => return Err("narrow real-record loader does not admit E8P".into()),
            }
            let (gs, ds) = (sha(&gate), sha(&down));
            lines.push_str(&format!(
                "{gn} {goff} {gsz} {gq} {}\n{dn} {doff} {dsz} {dq} {}\n",
                hex(&gs),
                hex(&ds)
            ));
            recs.push(json!({
                "expert": id, "codec": codec.label(), "gate_quant_type": gq, "down_quant_type": dq,
                "gate_offset": goff, "gate_bytes": gsz, "gate_sha256": hex(&gs),
                "down_offset": doff, "down_bytes": dsz, "down_sha256": hex(&ds),
                "contiguous_gate_down": goff + gsz == doff,
            }));
            experts.insert(
                id,
                Rc::new(Expert {
                    codec,
                    gate_sha: gs,
                    down_sha: ds,
                    gate,
                    down,
                }),
            );
        }
    }
    let meta = json!({
        "path": path, "layer": layer, "file_len": hfq.file_len(), "arch_id": hfq.arch_id,
        "index_sha256": hex(&hfq.index_sha256().map_err(err)?),
        "mq4v2_symmetric_metadata": hfq.mq4v2_symmetric(),
        "layer_expert_formats": {"w3": counts(Codec::W3), "mq4": counts(Codec::Mq4), "w2": w2, "other": other},
        "selected": {"w3": sel_w3.len(), "mq4": sel_mq4.len()},
        "mq4_sym_header_violations": violations,
        "selected_input_sha256": hex(&sha(lines.as_bytes())),
        "experts": recs,
    });
    Ok(Real {
        experts,
        meta,
        mq4_sym_violations: violations,
    })
}

/// Stock qt44/qt53 aggregate representation, using the same checked slice law as R1 XTS
/// import below. Do not turn aggregate weights into a claimed slab production reference.
fn load_legacy_aggregate(hfq: &HfqFile, path: &str, layer: usize, want: usize) -> Result<Real> {
    let mut spans = BTreeMap::new();
    let mut aggregates = Vec::new();
    for (kind, projection, qt, group) in [
        (Kind::GateUp, ExpertProj::GateUp, 44u8, 256u32),
        (Kind::Down, ExpertProj::Down, 53u8, 128u32),
    ] {
        let name = legacy_aggregate_name(layer, projection);
        let info = hfq
            .find_tensor_info(&name)
            .ok_or_else(|| format!("{path}: missing aggregate {name}"))?;
        let per = kind.tensor_bytes(Codec::Mq4);
        let shape = vec![N_EXPERTS as u32, kind.m() as u32, kind.k() as u32];
        if info.quant_type != qt
            || info.shape != shape
            || info.group_size != group
            || info.data_size != N_EXPERTS * per
        {
            return Err(format!("{name}: expected qt{qt}, shape {shape:?}, group {group}, {} bytes; got qt{}, {:?}, group {}, {} bytes",
                N_EXPERTS * per, info.quant_type, info.shape, info.group_size, info.data_size));
        }
        spans.insert(kind.suffix(), (info.data_offset, per));
        aggregates.push(
            json!({"name": name, "quant_type": qt, "shape": shape, "group_size": group,
            "data_offset": info.data_offset, "data_size": info.data_size, "per_expert_bytes": per}),
        );
    }
    let ids = spread(
        &(0..N_EXPERTS as u32).collect::<Vec<_>>(),
        want.clamp(10, N_EXPERTS),
    );
    let mut experts = BTreeMap::new();
    let mut records = Vec::new();
    for id in ids {
        let slice = |kind: Kind| -> Result<(Vec<u8>, usize)> {
            let (base, per) = spans[kind.suffix()];
            let offset = base + id as usize * per;
            let bytes = hfq.data_range(offset, per).ok_or_else(|| {
                format!(
                    "{path}: {} expert {id} range {offset}+{per} is unavailable",
                    kind.suffix()
                )
            })?;
            Ok((bytes.to_vec(), offset))
        };
        let (gate, gate_offset) = slice(Kind::GateUp)?;
        let (down, down_offset) = slice(Kind::Down)?;
        let violations =
            mq4_sym_violations(Kind::GateUp, &gate) + mq4_sym_violations(Kind::Down, &down);
        if violations != 0 {
            return Err(format!("{path}: layer {layer} expert {id} has {violations} non-symmetric/nonfinite MQ4 headers"));
        }
        let (gate_sha, down_sha) = (sha(&gate), sha(&down));
        records.push(
            json!({"expert": id, "codec": "mq4", "gate_source_offset": gate_offset,
            "down_source_offset": down_offset, "gate_bytes": gate.len(), "down_bytes": down.len(),
            "gate_sha256": hex(&gate_sha), "down_sha256": hex(&down_sha)}),
        );
        experts.insert(
            id,
            Rc::new(Expert {
                codec: Codec::Mq4,
                gate,
                down,
                gate_sha,
                down_sha,
            }),
        );
    }
    let meta = json!({"path": path, "layer": layer, "representation": "legacy_indexed_aggregate",
        "file_len": hfq.file_len(), "arch_id": hfq.arch_id,
        "index_sha256": hex(&hfq.index_sha256().map_err(err)?),
        "execution_domain": if layer == 48 { "native_mtp_head_diagnostic" } else { "target_verify_layer" },
        "aggregates": aggregates, "experts": records, "mq4_sym_header_violations": 0});
    Ok(Real {
        experts,
        meta,
        mq4_sym_violations: 0,
    })
}

fn real_cases(real: &Real, origin: &'static str, prefix: &str, wide: bool) -> Vec<Case> {
    let all: Vec<u32> = real.experts.keys().copied().collect();
    let by = |c: Codec| -> Vec<u32> {
        all.iter()
            .copied()
            .filter(|id| real.experts[id].codec == c)
            .collect()
    };
    let (w3, mq4) = (by(Codec::W3), by(Codec::Mq4));
    let legacy = real.meta["representation"] == "legacy_indexed_aggregate";
    let origin = if legacy && real.meta["execution_domain"] == "native_mtp_head_diagnostic" {
        "native_mtp_head_diagnostic"
    } else {
        origin
    };
    let mut rng = Rng::new(0x7ea1_1a7e);
    // C1: 10 experts, as balanced as the pool allows, route order shuffled.
    let mut ids: Vec<u32> = Vec::new();
    let (mut i3, mut i4) = (0usize, 0usize);
    while ids.len() < 10 {
        if (ids.len() % 2 == 0 && i3 < w3.len()) || i4 >= mq4.len() {
            ids.push(w3[i3]);
            i3 += 1;
        } else {
            ids.push(mq4[i4]);
            i4 += 1;
        }
    }
    let mut route1 = ids.clone();
    rng.shuffle(&mut route1);
    let sub: BTreeMap<u32, Rc<Expert>> = ids
        .iter()
        .map(|id| (*id, real.experts[id].clone()))
        .collect();
    let c1 = build_case(
        &format!("{prefix}_c1_mixed"),
        origin,
        1,
        route1,
        sub,
        false,
        false,
        0,
    );
    let route8 = lane_routes(&mut rng, &all, 8);
    let c8 = build_case(
        &format!("{prefix}_c8_union_mixed"),
        origin,
        8,
        route8,
        real.experts.clone(),
        false,
        false,
        1,
    );
    let mut cases = vec![c1, c8];
    if wide {
        for (i, c) in (1usize..=64).enumerate() {
            let route = lane_routes(&mut rng, &all, c);
            let mut case = build_case(
                &if legacy {
                    format!("{prefix}_legacy_c{c}_union_mq4")
                } else {
                    format!("{prefix}_c{c}_union_mixed")
                },
                origin,
                c,
                route,
                real.experts.clone(),
                c % 2 == 1,
                false,
                2 + i,
            );
            case.wide = true;
            cases.push(case);
        }
    }
    cases
}

// ---------------------------------------------------------------------------------------
// Certified R1 records + XTS aggregate (real weights, no packed HFQM)
// ---------------------------------------------------------------------------------------

/// Origin label of the certified-records cases; deliberately not "packed HFQM".
const RECORDS_ORIGIN: &str = "real_records_layer";

/// sha256 and length of a whole file, streamed.
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

/// `layers["<layer>"]` of a `ctl3-hot.map.json`: `"w3"` / `"xts_mq4"` (/`"w2"`) per expert.
fn hot_map_formats(root: &Value, layer: usize) -> Result<Vec<ExpertFormat>> {
    let row = root
        .get("layers")
        .and_then(|l| l.get(layer.to_string()))
        .and_then(Value::as_array)
        .ok_or_else(|| format!("map has no layers[\"{layer}\"] array"))?;
    if row.len() != N_EXPERTS {
        return Err(format!(
            "map layers[\"{layer}\"] has {} experts, expected {N_EXPERTS}",
            row.len()
        ));
    }
    row.iter()
        .enumerate()
        .map(|(e, v)| match v.as_str() {
            Some("w3") => Ok(ExpertFormat::Mq3I4),
            Some("w2") => Ok(ExpertFormat::Mq2I4),
            Some("xts_mq4") => Ok(ExpertFormat::Mq4Xts),
            other => Err(format!(
                "map layers[\"{layer}\"][{e}] = {other:?} is not w3/w2/xts_mq4"
            )),
        })
        .collect()
}

/// How many W3 (cold) and MQ4-XTS (hot) experts to take for `want` total, half each, topping a short
/// side up from the other.
fn select_counts(w3_len: usize, mq4_len: usize, want: usize) -> (usize, usize) {
    let want = want.max(10);
    let half = want / 2;
    let mut take_w3 = half.min(w3_len);
    let mut take_mq4 = (want - half).min(mq4_len);
    if take_w3 + take_mq4 < want {
        take_w3 = (want - take_mq4).min(w3_len);
        take_mq4 = (want - take_w3).min(mq4_len);
    }
    (take_w3, take_mq4)
}

fn spread(v: &[u32], n: usize) -> Vec<u32> {
    (0..n).map(|i| v[i * v.len() / n]).collect()
}

/// Strict real weights: cold experts are the certified R1 NPZ records imported through
/// `I4ExpertFinalCodeRecord::import_npz_file` (S2 encode + round trip), hot experts are the
/// per-expert gate/down slices of the XTS aggregate tensors, read through `HfqFile` + the source entry span.
fn load_records(cfg: &Cfg) -> Result<Real> {
    let layer = cfg.layer;
    let (dir, manifest_path, alloc_path, xts_path) = match (
        &cfg.records_dir,
        &cfg.records_manifest,
        &cfg.alloc,
        &cfg.xts,
    ) {
        (Some(d), Some(m), Some(a), Some(x)) => (d.as_str(), m.as_str(), a.as_str(), x.as_str()),
        _ => {
            return Err(
                "certified-records mode needs --records-dir, --records-manifest, --alloc and --xts"
                    .into(),
            )
        }
    };
    // Manifest identity.
    let manifest_bytes =
        std::fs::read(manifest_path).map_err(|e| format!("{manifest_path}: {e}"))?;
    let manifest_sha = hex(&sha(&manifest_bytes));
    if let Some(want) = &cfg.records_manifest_sha256 {
        if *want != manifest_sha {
            return Err(format!("manifest sha256 {manifest_sha} != expected {want}"));
        }
    }
    let manifest: Value =
        serde_json::from_slice(&manifest_bytes).map_err(|e| format!("{manifest_path}: {e}"))?;
    let want_expert_bytes =
        Kind::GateUp.tensor_bytes(Codec::W3) + Kind::Down.tensor_bytes(Codec::W3);
    let header_ok = manifest.get("layer").and_then(Value::as_u64) == Some(layer as u64)
        && manifest.get("bits").and_then(Value::as_u64) == Some(3)
        && manifest.get("grid").and_then(Value::as_str) == Some("uniform")
        && manifest.get("expert_bytes").and_then(Value::as_u64) == Some(want_expert_bytes as u64);
    if !header_ok {
        return Err(format!("{manifest_path}: header is not layer {layer} / bits 3 / grid uniform / expert_bytes {want_expert_bytes}"));
    }
    // Allocation (+ optional hot map cross-check).
    let alloc =
        load_allocation(Path::new(alloc_path), &cfg.alloc_point, SlabGeometry::FN).map_err(err)?;
    let formats: Vec<ExpertFormat> =
        alloc.formats[layer * N_EXPERTS..(layer + 1) * N_EXPERTS].to_vec();
    let mut map_meta = Value::Null;
    if let Some(map_path) = &cfg.map {
        let bytes = std::fs::read(map_path).map_err(|e| format!("{map_path}: {e}"))?;
        let root: Value = serde_json::from_slice(&bytes).map_err(|e| format!("{map_path}: {e}"))?;
        let from_map = hot_map_formats(&root, layer)?;
        if let Some(e) = (0..N_EXPERTS).find(|&e| from_map[e] != formats[e]) {
            return Err(format!("{map_path} disagrees with allocation point {} at layer {layer} expert {e}: {:?} vs {:?}", alloc.point, from_map[e], formats[e]));
        }
        map_meta = json!({"path": map_path, "sha256": hex(&sha(&bytes)), "arm": root.get("arm"), "cross_checked_against_alloc": true});
    }
    let (mut cold, mut hot, mut w2) = (Vec::new(), Vec::new(), 0usize);
    for (e, f) in formats.iter().enumerate() {
        match f {
            ExpertFormat::Mq3I4 => cold.push(e as u32),
            ExpertFormat::Mq4Xts => hot.push(e as u32),
            ExpertFormat::Mq2I4 => w2 += 1,
            ExpertFormat::E8pI4 => {
                return Err(
                    "certified W3/MQ4 records arm does not admit E8P; use e8p_i4_moe_gemv_oracle for E8P records".into(),
                )
            }
        }
    }
    if cold.is_empty() || hot.is_empty() {
        return Err(format!(
            "layer {layer} allocation has no mixed W3/MQ4 experts (w3={}, mq4={}, w2={w2})",
            cold.len(),
            hot.len()
        ));
    }
    let (take_w3, take_mq4) = select_counts(cold.len(), hot.len(), cfg.real_experts);
    let (sel_w3, sel_mq4) = (spread(&cold, take_w3), spread(&hot, take_mq4));
    if sel_w3.len() + sel_mq4.len() < 10 {
        return Err(format!(
            "layer {layer}: only {} usable experts; a top-10 lane needs 10",
            sel_w3.len() + sel_mq4.len()
        ));
    }
    // XTS source: identity, aggregate layout, then only the selected spans.
    let (xts_sha, xts_len) = file_sha256(xts_path)?;
    if let Some(want) = &cfg.xts_sha256 {
        if *want != xts_sha {
            return Err(format!("{xts_path}: sha256 {xts_sha} != expected {want}"));
        }
    }
    let hfq = HfqFile::open_at_offset(Path::new(xts_path), 0).map_err(err)?;
    let mut aggregates = Vec::new();
    let mut agg_span: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
    for (kind, proj, qt, group) in [
        (Kind::GateUp, ExpertProj::GateUp, 44u8, 256u32),
        (Kind::Down, ExpertProj::Down, 53u8, 128u32),
    ] {
        let name = SlabGeometry::aggregate_name(layer, proj);
        let info = hfq
            .find_tensor_info(&name)
            .ok_or_else(|| format!("{xts_path}: missing aggregate `{name}`"))?;
        let per = kind.tensor_bytes(Codec::Mq4);
        let want_shape = vec![N_EXPERTS as u32, kind.m() as u32, kind.k() as u32];
        if info.quant_type != qt
            || info.shape != want_shape
            || info.group_size != group
            || info.data_size != N_EXPERTS * per
        {
            return Err(format!(
                "aggregate `{name}` is qt{} shape {:?} group {} {} B; expected qt{qt} shape {want_shape:?} group {group} {} B",
                info.quant_type, info.shape, info.group_size, info.data_size, N_EXPERTS * per
            ));
        }
        aggregates.push(
            json!({"name": name, "quant_type": qt, "shape": want_shape, "group_size": group,
            "data_offset": info.data_offset, "data_size": info.data_size, "per_expert_bytes": per}),
        );
        agg_span.insert(kind.suffix(), (info.data_offset, per));
    }
    let xts_slice = |kind: Kind, e: u32| -> Result<(Vec<u8>, usize)> {
        let (base, per) = agg_span[kind.suffix()];
        let off = base + e as usize * per;
        let s = hfq.data_range(off, per).ok_or_else(|| {
            format!(
                "{xts_path}: cannot map {} bytes at {off} (mmap dropped or range past EOF)",
                per
            )
        })?;
        Ok((s.to_vec(), off))
    };
    let shapes = I4RecordShapes {
        gate_up: [GATE_M, GATE_K],
        down: [DOWN_M, DOWN_K],
    };
    let manifest_entry = |e: u32| manifest.get("experts").and_then(|x| x.get(format!("E{e}")));
    let mut experts = BTreeMap::new();
    let mut lines = String::new();
    let mut recs = Vec::new();
    let mut violations = 0u64;
    for (codec, ids) in [(Codec::W3, &sel_w3), (Codec::Mq4, &sel_mq4)] {
        for &id in ids {
            let (gate, down, rec_meta) = match codec {
                Codec::W3 => {
                    let npz = Path::new(dir).join(format!("E{id}.npz"));
                    let rec = I4ExpertFinalCodeRecord::import_npz_file(
                        &npz,
                        layer,
                        id as usize,
                        ExpertFormat::Mq3I4,
                        shapes,
                    )
                    .map_err(|e| format!("{}: {e}", npz.display()))?;
                    let entry = manifest_entry(id)
                        .ok_or_else(|| format!("{manifest_path}: no experts.E{id}"))?;
                    if entry.get("npz_sha256").and_then(Value::as_str)
                        != Some(rec.source_npz_sha256.as_str())
                    {
                        return Err(format!(
                            "{}: npz sha256 {} != manifest {:?}",
                            npz.display(),
                            rec.source_npz_sha256,
                            entry.get("npz_sha256")
                        ));
                    }
                    let mut pack = rec.gate_up.bytes.clone();
                    pack.extend_from_slice(&rec.down.bytes);
                    let pack_sha = hex(&sha(&pack));
                    if entry.get("sha256").and_then(Value::as_str) != Some(pack_sha.as_str()) {
                        return Err(format!(
                            "{}: encoded gate++down sha256 {pack_sha} != manifest {:?}",
                            npz.display(),
                            entry.get("sha256")
                        ));
                    }
                    mq_i4::decode_tensor(MqI4Profile::MQ3_R256, GATE_M, GATE_K, &rec.gate_up.bytes)
                        .map_err(err)?;
                    mq_i4::decode_tensor(MqI4Profile::MQ3_R128, DOWN_M, DOWN_K, &rec.down.bytes)
                        .map_err(err)?;
                    let meta = json!({"source": "r1_npz_record", "npz_path": npz.display().to_string(),
                        "npz_sha256": rec.source_npz_sha256, "manifest_npz_sha256_match": true,
                        "manifest_pack_sha256": pack_sha, "manifest_pack_sha256_match": true,
                        "scale_rule": rec.scale_rule, "grid": rec.grid});
                    (rec.gate_up.bytes, rec.down.bytes, meta)
                }
                Codec::Mq4 => {
                    let (gate, goff) = xts_slice(Kind::GateUp, id)?;
                    let (down, doff) = xts_slice(Kind::Down, id)?;
                    violations += mq4_sym_violations(Kind::GateUp, &gate)
                        + mq4_sym_violations(Kind::Down, &down);
                    let meta = json!({"source": "xts_aggregate_slice", "gate_source_offset": goff, "down_source_offset": doff});
                    (gate, down, meta)
                }
                Codec::E8p => return Err("certified W3/MQ4 records arm does not admit E8P".into()),
            };
            let (gs, ds) = (sha(&gate), sha(&down));
            lines.push_str(&format!(
                "{} {id} {} {}\n",
                codec.label(),
                hex(&gs),
                hex(&ds)
            ));
            let mut rec_json = json!({"expert": id, "codec": codec.label(), "gate_bytes": gate.len(), "gate_sha256": hex(&gs),
                "down_bytes": down.len(), "down_sha256": hex(&ds)});
            rec_json["provenance"] = rec_meta;
            recs.push(rec_json);
            experts.insert(
                id,
                Rc::new(Expert {
                    codec,
                    gate_sha: gs,
                    down_sha: ds,
                    gate,
                    down,
                }),
            );
        }
    }
    let meta = json!({
        "origin": RECORDS_ORIGIN, "packed_hfqm": false, "layer": layer,
        "records_dir": dir,
        "manifest": {"path": manifest_path, "sha256": manifest_sha, "expected_sha256": cfg.records_manifest_sha256,
            "script_sha256": manifest.get("script_sha256"), "input_receipts": manifest.get("input_receipts"),
            "expert_bytes": want_expert_bytes},
        "alloc": {"path": alloc_path, "sha256": alloc.file_sha256, "point": alloc.point,
            "layer_formats": {"w3": cold.len(), "mq4_xts": hot.len(), "w2": w2}},
        "map": map_meta,
        "xts": {"path": xts_path, "sha256": xts_sha, "expected_sha256": cfg.xts_sha256, "bytes": xts_len,
            "file_len": hfq.file_len(), "index_sha256": hex(&hfq.index_sha256().map_err(err)?),
            "aggregates": aggregates},
        "selected": {"w3_cold": sel_w3.len(), "mq4_hot": sel_mq4.len()},
        "mq4_sym_header_violations": violations,
        "selected_input_sha256": hex(&sha(lines.as_bytes())),
        "experts": recs,
    });
    Ok(Real {
        experts,
        meta,
        mq4_sym_violations: violations,
    })
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
}
impl<'a> Dev<'a> {
    fn new(hip: &'a HipRuntime) -> Self {
        Dev {
            hip,
            bufs: Vec::new(),
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
    fn fill(&mut self, n: usize, value: i32) -> Result<Buf> {
        let b = self.alloc(n)?;
        self.hip
            .memset(&self.bufs[b.idx], value, n.max(4))
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
    fn release(self) {
        let hip = self.hip;
        for b in self.bufs {
            let _ = hip.free(b);
        }
    }
}

/// One slab per expert (gate_up tensor then down tensor, as in the artifact), allocated in
/// shuffled order between random-size fillers so pointers are neither monotone nor uniformly
/// strided. Returns `id -> (gate_ptr, down_ptr)`.
fn upload_experts(dev: &mut Dev, case: &Case) -> Result<BTreeMap<u32, (u64, u64)>> {
    let mut rng = Rng::new(name_seed(&case.name) ^ 0xa110c);
    let mut order: Vec<u32> = case.experts.keys().copied().collect();
    rng.shuffle(&mut order);
    let mut out = BTreeMap::new();
    for id in order {
        if rng.below(3) != 0 {
            dev.alloc(256 * (1 + rng.below(300)))?;
        }
        let e = &case.experts[&id];
        let mut slab = Vec::with_capacity(e.gate.len() + e.down.len());
        slab.extend_from_slice(&e.gate);
        slab.extend_from_slice(&e.down);
        let b = dev.upload(&slab)?;
        out.insert(id, (b.ptr, b.ptr + e.gate.len() as u64));
    }
    Ok(out)
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
    funcs: Vec<(Codec, Kind, Function, String)>,
    codeobject_sha: String,
    device_object_sha256: String,
    device: i32,
    old_parity: bool,
    gpu: Option<Gpu>,
}
impl Ctx {
    fn loaded(&self, codec: Codec, kind: Kind) -> &(Codec, Kind, Function, String) {
        self.funcs
            .iter()
            .find(|(c, k, _, _)| *c == codec && *k == kind)
            .unwrap_or_else(|| {
                panic!(
                    "narrow symbol for {} {} is not admitted on this arch",
                    codec.label(),
                    kind.suffix()
                )
            })
    }
    fn func(&self, codec: Codec, kind: Kind) -> &Function {
        &self.loaded(codec, kind).2
    }
    fn loaded_symbol(&self, codec: Codec, kind: Kind) -> &str {
        &self.loaded(codec, kind).3
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
    cpu_expected_x(case, kind, buckets, x, xrs, ys)
}

fn cpu_expected_x(
    case: &Case,
    kind: Kind,
    buckets: &[Bucket; 2],
    x: &[f32],
    xrs: usize,
    ys: usize,
) -> Expected {
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
    // The cold bucket is at index 0 and holds the layer's single cold codec (W3 or E8P);
    // index 1 is MQ4. On an E8P-cold layer the cold bucket is E8P, so the W3 arm below is
    // empty by construction and the E8P twin fills the cold slots.
    let cold_is_e8p = case_cold_codec(case)
        .map(|c| c == Codec::E8p.route_byte())
        .unwrap_or(false);
    let cold = &buckets[0];
    if !cold_is_e8p && !cold.ids.is_empty() {
        let mut exps: Vec<Option<ExpertTensor<'_>>> = (0..N_EXPERTS).map(|_| None).collect();
        for (&id, e) in &case.experts {
            if e.codec == Codec::W3 {
                exps[id as usize] = Some(ExpertTensor {
                    profile: kind.w3_profile(),
                    bytes: proj_bytes(e, kind),
                });
            }
        }
        mq_i4_reference::decode_moe_slots(
            kind.decode_kind(),
            kind.m(),
            kind.k(),
            &exps,
            &cold.ids,
            &cold.offs,
            &cold.slots,
            case.c,
            x,
            xrs,
            ys,
            &mut y0,
            y1.as_deref_mut(),
        );
        // decode_moe_slots vs decode_gemv_f32 on a bounded slot sample (plumbing self-check).
        let mut sample: Vec<u32> = cold.slots.clone();
        sample.sort_unstable();
        sample.truncate(12);
        for &slot in &sample {
            let slot = slot as usize;
            let e = &case.experts[&case.route[slot]];
            let xs = &x[(slot / kind.x_row_div()) * xrs..][..kind.k()];
            let row = mq_i4_reference::decode_gemv_f32(
                kind.w3_profile(),
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
    // E8P cold bucket: the actual frozen K-E0 lane-pair CPU twin. The narrow W3/MQ4 raw
    // singleton is never run on E8P bytes.
    if cold_is_e8p && !cold.ids.is_empty() {
        for &slot in &cold.slots {
            let slot = slot as usize;
            let id = case.route[slot];
            let e = &case.experts[&id];
            debug_assert_eq!(e.codec, Codec::E8p);
            let bytes = proj_bytes(e, kind);
            let (m, k) = (kind.m(), kind.k());
            let (nw, ns) = (m * k / 8, m * k / 128);
            let (words, scales) = (&bytes[..nw * 2], &bytes[nw * 2..nw * 2 + ns * 2]);
            let words: Vec<u16> = words
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            let scales: Vec<u16> = scales
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            let xs = &x[(slot / kind.x_row_div()) * xrs..][..kind.k()];
            let row = e8p::decode_gemv_f32_lanepair(kind.e8p_rotation(), m, k, &words, &scales, xs);
            let base = slot * ys;
            for (r, v) in row.iter().enumerate() {
                if r < kind.out_rows() {
                    y0[base + r] = *v;
                } else {
                    y1.as_deref_mut().expect("gate has Y1")[base + r - kind.out_rows()] = *v;
                }
            }
        }
        checked = checked.max(cold.slots.len());
    }
    // MQ4: the transcribed legacy arithmetic.
    let b = &buckets[1];
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

fn compare_buf(
    part: &str,
    gpu: &[u8],
    exp: &[f32],
    ys: usize,
    out_rows: usize,
    slot_codec: &[Codec],
    per: &mut [Mm; 2],
    stray: &mut Mm,
) {
    // Narrow attribution: W3=0, MQ4=1 (E8P never reaches this narrow arm).
    let narrow_idx = |codec: Codec| -> usize {
        match codec {
            Codec::W3 => 0,
            Codec::Mq4 => 1,
            Codec::E8p => panic!("E8P has no narrow arm"),
        }
    };
    for (i, (b, e)) in gpu.chunks_exact(4).zip(exp).enumerate() {
        let g = u32::from_le_bytes(b.try_into().unwrap());
        if g == e.to_bits() {
            continue;
        }
        let (slot, row) = (i / ys, i % ys);
        let rec = json!({"part": part, "word_index": i, "slot": slot, "row": row,
            "gpu_bits": bits_hex(g), "oracle_bits": bits_hex(e.to_bits()),
            "oracle_unwritten": e.to_bits() == SENTINEL_BITS});
        if row < out_rows {
            per[narrow_idx(slot_codec[slot])].push(rec);
        } else {
            stray.push(rec);
        }
    }
}

// ---------------------------------------------------------------------------------------
// One (case, kind) GPU run
// ---------------------------------------------------------------------------------------

struct KindOut {
    symbols: Vec<Value>,
    mismatches: u64,
    twin_mismatches: u64,
    launches: u64,
    expected: Expected,
    timing: Vec<(Codec, Kind, usize, usize, Vec<f64>)>,
}

fn run_kind(
    ctx: &Ctx,
    dev: &mut Dev,
    case: &Case,
    kind: Kind,
    slabs: &BTreeMap<u32, (u64, u64)>,
    timing: Option<&TimingCfg>,
) -> Result<KindOut> {
    let c = case.c;
    let slots = 10 * c;
    let (m, k, out) = (kind.m(), kind.k(), kind.out_rows());
    let (x, xrs, ys) = case.x(kind);
    let x_src_rows = x.len() / xrs;
    let buckets = buckets_of(case);
    let expected = cpu_expected(case, kind, &buckets);

    let xb = dev.upload(&f32_bytes(x))?;
    let mut table = vec![0u64; N_EXPERTS];
    for (&id, &(g, d)) in slabs {
        table[id as usize] = if kind == Kind::GateUp { g } else { d };
    }
    let tb = dev.upload(&u64_bytes(&table))?;
    let n_words = slots * ys;
    let y0 = dev.fill(n_words * 4, 0xa5)?;
    let y1 = if kind == Kind::GateUp {
        Some(dev.fill(n_words * 4, 0xa5)?)
    } else {
        None
    };

    // The narrow independent arm only covers the admitted narrow codecs (W3/MQ4). `buckets_of`
    // returns slab order (cold at 0, MQ4 at 1), so resolve narrow buckets explicitly and keep
    // the historical launch order W3 then MQ4.
    let empty_bucket = Bucket::default();
    let narrow_of = |codec: Codec| -> &Bucket {
        match codec {
            Codec::W3
                if case_cold_codec(case)
                    .map(|c| c == Codec::W3.route_byte())
                    .unwrap_or(false) =>
            {
                &buckets[0]
            }
            Codec::W3 => &empty_bucket,
            Codec::Mq4 => &buckets[1],
            Codec::E8p => panic!("E8P has no narrow arm"),
        }
    };
    let mut symbols = Vec::new();
    let mut prepared: Vec<(Codec, [u8; KERNARG_BYTES], [u32; 3])> = Vec::new();
    for codec in Codec::ALL {
        let b: &Bucket = narrow_of(codec);
        if b.ids.is_empty() {
            symbols.push(json!({"symbol": symbol(codec, kind), "codec": codec.label(), "profile": kind.suffix(),
                "launched": false, "slots": 0, "unique_experts": 0}));
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
        prepared.push((codec, args, [m as u32, b.ids.len() as u32, 1]));
    }
    for (codec, args, grid) in &prepared {
        let mut a = *args;
        // SAFETY: exact 96-byte ABI; every pointer is a live allocation of `dev` until release.
        unsafe {
            ctx.hip
                .launch_kernel_blob(ctx.func(*codec, kind), *grid, [32, 1, 1], 0, None, &mut a)
        }
        .map_err(err)?;
    }
    ctx.hip.device_synchronize().map_err(err)?;

    let slot_codec: Vec<Codec> = (0..slots).map(|s| case.codec_of_slot(s)).collect();
    // Narrow comparison attributions live in `compare_buf` (W3=0, MQ4=1).
    let mut per = [Mm::default(), Mm::default()];
    let mut stray = Mm::default();
    let g0 = dev.download(y0, n_words * 4)?;
    compare_buf(
        "y0",
        &g0,
        &expected.y0,
        ys,
        out,
        &slot_codec,
        &mut per,
        &mut stray,
    );
    if let (Some(yb), Some(e1)) = (y1, expected.y1.as_ref()) {
        let g1 = dev.download(yb, n_words * 4)?;
        compare_buf("y1", &g1, e1, ys, out, &slot_codec, &mut per, &mut stray);
    }
    let mut mismatches = stray.count;
    for (narrow_i, codec) in Codec::ALL.iter().enumerate() {
        let b = narrow_of(*codec);
        if b.ids.is_empty() {
            continue;
        }
        mismatches += per[narrow_i].count;
        symbols.push(json!({
            "symbol": ctx.loaded_symbol(*codec, kind), "codec": codec.label(), "profile": kind.suffix(), "launched": true,
            "grid": [m, b.ids.len(), 1], "slots": b.slots.len(), "unique_experts": b.ids.len(),
            "mismatch_count": per[narrow_i].count, "first_mismatches": per[narrow_i].first,
        }));
    }
    symbols.push(
        json!({"symbol": "stray_or_pad_words", "profile": kind.suffix(),
        "mismatch_count": stray.count, "first_mismatches": stray.first}),
    );

    let mut timing_rows = Vec::new();
    if let Some(t) = timing {
        if mismatches != 0 {
            return Err(format!(
                "{} not byte-exact; refusing to time a wrong kernel",
                kind.suffix()
            ));
        }
        for (codec, args, grid) in &prepared {
            let f = ctx.func(*codec, kind);
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
            timing_rows.push((*codec, kind, slots, grid[1] as usize, samples));
        }
    }
    Ok(KindOut {
        symbols,
        mismatches,
        twin_mismatches: expected.twin_mismatches,
        launches: prepared.len() as u64,
        expected,
        timing: timing_rows,
    })
}

// ---------------------------------------------------------------------------------------
// Independent old-kernel parity (production HIP kernels, JIT via rdna_compute::Gpu)
// ---------------------------------------------------------------------------------------

fn old_kernel_parity(
    gpu: &mut Gpu,
    case: &Case,
    gate: &Expected,
    down: &Expected,
) -> Result<Value> {
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
    for (&id, e) in &case.experts {
        let g = up(gpu, &e.gate)?;
        let d = up(gpu, &e.down)?;
        gtab[id as usize] = g.buf.as_ptr() as u64;
        dtab[id as usize] = d.buf.as_ptr() as u64;
        keep.push(g);
        keep.push(d);
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
    let yg = blank(gpu, c * 10 * GATE_M / 2)?;
    let yu = blank(gpu, c * 10 * GATE_M / 2)?;
    let yd = blank(gpu, c * 10 * DOWN_M)?;
    gpu.gemv_mq4g256v2_moe_gate_up_top10_indexed_batched(
        &gt, &tk, &xg, &yg, &yu, GATE_M, GATE_K, c,
    )
    .map_err(err)?;
    gpu.gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
        &dt, &tk, &xd, &yd, DOWN_M, DOWN_K, c, N_EXPERTS, 10,
    )
    .map_err(err)?;
    gpu.hip.device_synchronize().map_err(err)?;
    let (n_gate, n_down) = (c * 10 * GATE_M / 2, c * 10 * DOWN_M);
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
    for t in keep.into_iter().chain([gt, dt, tk, xg, xd, yg, yu, yd]) {
        gpu.free_tensor(t).map_err(err)?;
    }
    Ok(
        json!({"kernels": ["gemv_mq4g256v2_moe_gate_up_top10_indexed_batched",
        "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded"],
        "mismatch_count": total, "parts": detail}),
    )
}

// ---------------------------------------------------------------------------------------
// Case driver
// ---------------------------------------------------------------------------------------

#[derive(Default)]
struct Totals {
    cases: u64,
    synthetic: u64,
    real: u64,
    real_records: u64,
    launches: u64,
    mismatches: u64,
    twin_mismatches: u64,
    old_kernel_mismatches: u64,
    old_kernel_cases: u64,
    wide_cases: u64,
    wide_launches: u64,
    /// Wide words that differ from the CPU oracle (also counted in `mismatches`).
    wide_cpu_mismatches: u64,
    /// Whole-chain/router words differing from the sequential actual runtime singleton.
    wide_singleton_mismatches: u64,
    /// Wide projection words differing from the independent explicitly loaded raw singleton.
    wide_reference_mismatches: u64,
    /// Guard-band / sentinel violations around the wide and reference outputs.
    wide_guard_violations: u64,
}

type TimingSamples = Vec<(Codec, Kind, usize, usize, Vec<f64>)>;

fn run_case(
    ctx: &mut Ctx,
    case: &Case,
    t: &mut Totals,
    timing: Option<&TimingCfg>,
) -> Result<(Value, TimingSamples)> {
    if case.wide {
        return run_case_wide(ctx, case, t);
    }
    let mut dev = Dev::new(&ctx.hip);
    let slabs = upload_experts(&mut dev, case);
    let gate = match &slabs {
        Ok(s) => run_kind(ctx, &mut dev, case, Kind::GateUp, s, timing),
        Err(e) => Err(e.clone()),
    };
    let down = match (&slabs, &gate) {
        (Ok(s), Ok(_)) => run_kind(ctx, &mut dev, case, Kind::Down, s, timing),
        _ => Err("skipped after earlier failure".to_string()),
    };
    dev.release();
    slabs?;
    let (gate, down) = (gate?, down?);
    let mut counts = BTreeMap::new();
    for codec in Codec::ALL {
        counts.insert(
            codec.label(),
            case.route
                .iter()
                .filter(|e| case.experts[*e].codec == codec)
                .count(),
        );
    }
    let unique_by_codec: BTreeMap<&str, usize> = Codec::ALL
        .iter()
        .map(|c| {
            (
                c.label(),
                case.experts.values().filter(|e| e.codec == *c).count(),
            )
        })
        .collect();
    t.cases += 1;
    match case.origin {
        "synthetic" => t.synthetic += 1,
        RECORDS_ORIGIN => {
            t.real += 1;
            t.real_records += 1;
        }
        _ => t.real += 1,
    }
    t.launches += gate.launches + down.launches;
    t.mismatches += gate.mismatches + down.mismatches;
    t.twin_mismatches += gate.twin_mismatches + down.twin_mismatches;
    let mut row = json!({
        "case": case.name, "origin": case.origin, "c": case.c, "slots": 10 * case.c,
        "unique_experts": case.experts.len(), "slots_by_codec": counts, "unique_experts_by_codec": unique_by_codec,
        "strides": {"x_row_stride_gate": case.xrs_gate, "x_row_stride_down": case.xrs_down,
            "y_slot_stride_gate": case.ys_gate, "y_slot_stride_down": case.ys_down},
        "input_sha256": case.input_sha256(),
        "gate_up": gate.symbols, "down": down.symbols,
        "mismatch_count": gate.mismatches + down.mismatches,
        "twin_selfcheck": {"slots_checked_gate": gate.expected.twin_slots_checked,
            "slots_checked_down": down.expected.twin_slots_checked,
            "mismatch_count": gate.twin_mismatches + down.twin_mismatches},
        "pass": gate.mismatches + down.mismatches + gate.twin_mismatches + down.twin_mismatches == 0,
    });
    if ctx.old_parity && case.old_parity && case.tight() {
        if ctx.gpu.is_none() {
            ctx.gpu = Some(Gpu::init_with_device(ctx.device).map_err(err)?);
        }
        let gpu = ctx.gpu.as_mut().expect("initialised above");
        let r = old_kernel_parity(gpu, case, &gate.expected, &down.expected)?;
        t.old_kernel_cases += 1;
        t.old_kernel_mismatches += r["mismatch_count"].as_u64().unwrap_or(1);
        if r["mismatch_count"].as_u64() != Some(0) {
            row["pass"] = json!(false);
        }
        row["old_kernel_parity"] = r;
    }
    let mut samples = gate.timing;
    samples.extend(down.timing);
    Ok((row, samples))
}

// ---------------------------------------------------------------------------------------
// Whole-chain fixed-eight-row PM planner and K1 oracle, every R=1..64
// ---------------------------------------------------------------------------------------

/// Whole-slab case names: per R, four valid representation prefixes (plain and second seed;
/// `e8p_` mixed E8P/MQ4 and pure E8P-cold via disjoint plus second seed), two pure-bucket
/// prefixes (`purew3_`, `puremq4_` plus second seeds; ident/ties pin pure W3/pure MQ4 while
/// pool mixes and disjoint is pure E8P-cold, preserving each route axis), two legacy indexed
/// prefixes (excluded from the slab-only cohort), and four invalid-ID prefixes (`invalid_`,
/// `e8p_invalid_` plus second seeds, covering every admitted cold codec). Invalid cases keep
/// the cold codec of their representation and freeze invalid top-k IDs at launch.
fn wide_case_names(cs: &[usize]) -> Vec<String> {
    let mut names = Vec::new();
    for &c in cs {
        for prefix in [
            "",
            "seed1_",
            "e8p_",
            "e8p_seed1_",
            "purew3_",
            "purew3_seed1_",
            "puremq4_",
            "puremq4_seed1_",
            "legacy_",
            "legacy_seed1_",
            "invalid_",
            "invalid_seed1_",
            "e8p_invalid_",
            "e8p_invalid_seed1_",
        ] {
            for style in ["pool", "ident", "disjoint", "ties"] {
                names.push(format!("w{c}_{prefix}{style}"));
            }
        }
    }
    names
}
/// The `--wide-slab-only` cohort: every wide case except the legacy indexed prefixes. This is
/// the complete whole-slab G1 sweep; legacy remains available through `--wide`/`--wide-only`.
fn slab_only_case_names(cs: &[usize]) -> Vec<String> {
    wide_case_names(cs)
        .into_iter()
        .filter(|n| !n.contains("_legacy_"))
        .collect()
}
/// Route bytes a wide case name admits WITHOUT building any expert tensor: pure name->codec
/// map over the 14 known prefixes (legacy and pure-MQ4 are always all-MQ4 `&[0]`). Style never
/// changes the codec set: pure prefixes are pure on every style; `e8p_`-disjoint and
/// `e8p_invalid_`-disjoint are pure E8P-cold (byte 3), all other e8p/invalid cases are mixed
/// cold+MQ4. Returns the admitted route bytes (`&[0]` MQ4-only, `&[0,1]` W3-cold, `&[0,3]`
/// E8P-cold, `&[1]` pure W3-cold, `&[3]` pure E8P-cold).
fn wide_case_route_bytes(name: &str) -> &'static [u8] {
    if name.contains("_legacy_") || name.contains("puremq4_") {
        &[0]
    } else if name.contains("purew3_") {
        &[1]
    } else if name.contains("e8p_invalid_") || name.contains("_e8p_") {
        if name.ends_with("_disjoint") {
            &[3]
        } else {
            &[0, 3]
        }
    } else {
        &[0, 1]
    }
}
/// `true` when every route byte of a wide case name is slab-admitted on `arch`. Pure name
/// lookup: no expert tensor is synthesized. Legacy and pure-MQ4 are admitted everywhere a
/// slab exists; W3-cold needs arch W3 admission; E8P-cold needs arch E8P admission.
fn wide_case_admitted_on(arch: &str, name: &str) -> bool {
    wide_case_route_bytes(name)
        .iter()
        .all(|&b| qwen4_slab_codec_admitted(arch, b))
}
/// Invalid-ID slot positions for R rows: the union of the ragged tail (last absolute slot),
/// every tile-boundary slot inside `10*R` (79/80, 159/160, 319/320, 639), and a fixed 1/13
/// stride (mirroring the planner-semantics CPU test).
fn invalid_positions(rows: usize) -> Vec<usize> {
    let mut pos: Vec<usize> = Vec::new();
    let mut push = |s: usize| {
        if s < 10 * rows && !pos.contains(&s) {
            pos.push(s);
        }
    };
    for s in (0..10 * rows).step_by(13) {
        push(s);
    }
    push(10 * rows - 1);
    for s in [79usize, 80, 159, 160, 319, 320, 639] {
        push(s);
    }
    pos.sort_unstable();
    pos
}
/// Signed invalid top-k IDs (`-1/512/MIN/MAX`) written over a valid route at the invalid
/// positions above. The production launch flow and the independent raw reference both consume
/// these frozen IDs directly; the router is never asked to produce them.
fn invalid_topk_for(route: &[u32], rows: usize) -> Vec<i32> {
    const BAD: [i32; 4] = [-1, 512, i32::MIN, i32::MAX];
    let mut topk: Vec<i32> = route.iter().map(|&id| id as i32).collect();
    for (k, &s) in invalid_positions(rows).iter().enumerate() {
        topk[s] = BAD[k % BAD.len()];
    }
    topk
}

/// Legal lane unions. Disjointness is per physical tile; expert IDs are reused across tiles.
fn wide_plan(rng: &mut Rng, c: usize, style: &str) -> (Vec<u32>, Vec<u32>) {
    match style {
        "pool" => {
            let ids = pick_ids(rng, (12 + c / 2).min(64), true);
            let route = lane_routes(rng, &ids, c);
            (ids, route)
        }
        "ident" => {
            let ids = pick_ids(rng, 10, true);
            let mut lane = ids.clone();
            rng.shuffle(&mut lane);
            let route = (0..c).flat_map(|_| lane.clone()).collect();
            (ids, route)
        }
        "disjoint" => {
            let ids = pick_ids(rng, (10 * c).min(80), true);
            let mut tile = ids.clone();
            rng.shuffle(&mut tile);
            let route = (0..10 * c).map(|s| tile[s % 80]).collect();
            (ids, route)
        }
        "ties" => {
            let ids: Vec<u32> = (0..10).collect();
            let route = (0..c).flat_map(|_| ids.iter().copied()).collect();
            (ids, route)
        }
        other => panic!("unknown wide route style {other}"),
    }
}

fn build_wide_synthetic(name: &str) -> Case {
    let (head, style) = name.split_once('_').expect("wide case name w<C>_<style>");
    let c: usize = head.trim_start_matches('w').parse().expect("wide case C");
    let style = style.rsplit('_').next().expect("route style");
    let legacy = name.contains("_legacy_");
    let invalid = name.contains("_invalid_");
    let e8p = name.contains("_e8p_");
    let purew3 = name.contains("purew3_");
    let puremq4 = name.contains("puremq4_");
    let mut rng = Rng::new(name_seed(name) ^ 0x71de);
    let (ids, route) = wide_plan(&mut rng, c, style);
    let codecs = if legacy {
        vec![Codec::Mq4; ids.len()]
    } else if purew3 {
        // Pure W3 cold bucket (cold 1, empty MQ4 bucket) on every route style; invalidity
        // stays on the frozen top-k IDs, never on the experts.
        vec![Codec::W3; ids.len()]
    } else if puremq4 {
        // Pure MQ4 (cold 0, empty cold bucket) on every route style.
        vec![Codec::Mq4; ids.len()]
    } else if e8p {
        // Mixed E8P/MQ4 pool/ident/ties; disjoint is the pure E8P cold bucket (every expert
        // E8P); invalid E8P prefixes keep E8P-cold experts and carry invalidity on the
        // frozen top-k IDs.
        if invalid {
            if style == "disjoint" {
                vec![Codec::E8p; ids.len()]
            } else {
                mixed_cold_codecs(&mut rng, ids.len(), Codec::E8p, 5)
            }
        } else if style == "disjoint" {
            vec![Codec::E8p; ids.len()]
        } else {
            mixed_cold_codecs(&mut rng, ids.len(), Codec::E8p, 5)
        }
    } else if invalid {
        // Invalid cases keep a mixed W3/MQ4 cold codec against real experts; the frozen
        // invalid top-k IDs below (not the experts) carry the invalidity.
        mixed_codecs(&mut rng, ids.len(), 5)
    } else {
        mixed_codecs(&mut rng, ids.len(), 5)
    };
    // Narrow fixtures retain exponent extremes; chain fixtures use bounded ordinary scales.
    let experts = ids
        .iter()
        .enumerate()
        .map(|(i, &id)| {
            let codec = codecs[i];
            let gate = gen_tensor(codec, Kind::GateUp, id, false);
            let down = gen_tensor(codec, Kind::Down, id, false);
            (
                id,
                Rc::new(Expert {
                    codec,
                    gate_sha: sha(&gate),
                    down_sha: sha(&down),
                    gate,
                    down,
                }),
            )
        })
        .collect();
    let padded = style == "pool" && c % 2 == 1;
    let mut case = build_case(
        name,
        "synthetic_wide",
        c,
        route,
        experts,
        padded,
        false,
        c % 4,
    );
    case.wide = true;
    case
}

/// The production fixed-capacity descriptor of one codec bucket: live experts first, then `n == 0` tail experts up to the
/// capacity `S = 10*C` (`ids` zero filled, every tail offset the bucket's slot count, `slot_ids` zero filled to `S`).
#[cfg(test)]
fn pad_descriptor(b: &Bucket, s: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let (mut ids, mut offs, mut slots) = (b.ids.clone(), b.offs.clone(), b.slots.clone());
    ids.resize(s, 0);
    offs.resize(s + 1, *b.offs.last().expect("offsets are never empty"));
    slots.resize(s, 0);
    (ids, offs, slots)
}

/// Words of `buf` outside `[guard, guard + n_words*4)` that are not the sentinel byte.
fn guard_violations(buf: &[u8], n_words: usize) -> u64 {
    let g = GUARD_WORDS * 4;
    let bad = |s: &[u8]| s.iter().filter(|&&b| b != 0xa5).count() as u64;
    bad(&buf[..g]) + bad(&buf[g + n_words * 4..])
}

fn words_as_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_bits(u32::from_le_bytes(b.try_into().unwrap())))
        .collect()
}

const TILE_SLOTS: usize = 80;
// Derive the frozen 482-word/1928-byte tile extent from the production checked helper.
const TILE_WORDS: usize = match rdna_compute::scratch::qwen4_slab_rowtile_words_checked(8) {
    Some(words) => words,
    None => panic!("fixed eight-row slab descriptor extent overflows"),
};
/// Host specification of the PM ABI, not execution/emulation of the emitted PM instructions.
/// Bucket 0 holds codec byte 1; every other byte (0 = MQ4, 2, 255, ...) is bucket 1: the
/// production planner's exact `codec != 1` law. `membership_of` only ever produces 0/1 bits
/// (1 = the layer's cold codec, 0 = MQ4), so this stays exact for every fixture.
/// Membership (`codec` table) is cold = 1, MQ4 = 0 (never the raw codec byte); a 512-expert
/// membership table holds that bit per routed expert. `uniform` is the same vocabulary when
/// the whole row set is one bucket.
fn cpu_rowtile_descriptors(
    topk: &[i32],
    codec: &[u8],
    rows: usize,
    uniform: Option<usize>,
) -> Result<Vec<u32>> {
    if !(1..=MAX_ROWS).contains(&rows)
        || topk.len() != rows * 10
        || codec.len() != N_EXPERTS
        || uniform.is_some_and(|b| b > 1)
    {
        return Err("invalid rowtile planner geometry/membership extent".into());
    }
    let mut result = vec![0; rows.div_ceil(8) * TILE_WORDS];
    for (tile, descriptor) in result.chunks_exact_mut(TILE_WORDS).enumerate() {
        let start = tile * TILE_SLOTS;
        let end = (start + TILE_SLOTS).min(topk.len());
        for bucket in 0..2 {
            let mut first = Vec::new();
            for &id in &topk[start..end] {
                // Signed invalid IDs never index membership.
                if id < 0 || id as usize >= N_EXPERTS {
                    continue;
                }
                let b = uniform.unwrap_or(usize::from(codec[id as usize] != 1));
                if b == bucket && !first.contains(&id) {
                    first.push(id);
                }
            }
            let mut count = 0;
            for (e, &id) in first.iter().enumerate() {
                descriptor[bucket * 80 + e] = id as u32;
                descriptor[160 + bucket * 81 + e] = count;
                for (s, &routed) in topk.iter().enumerate().take(end).skip(start) {
                    if routed == id {
                        descriptor[322 + bucket * 80 + count as usize] = s as u32;
                        count += 1;
                    }
                }
            }
            descriptor[160 + bucket * 81 + first.len()..160 + bucket * 81 + 81].fill(count);
        }
    }
    Ok(result)
}

/// Owning allocation with both guards and poisoned inactive capacity. Never free the views.
struct Guarded {
    owner: GpuTensor,
    live: GpuTensor,
    capacity: usize,
}
impl Guarded {
    fn new(gpu: &mut Gpu, words: usize, capacity: usize) -> Result<Self> {
        let owner = gpu
            .alloc_tensor(&[capacity + 2 * GUARD_WORDS], DType::F32)
            .map_err(err)?;
        gpu.hip
            .memset(&owner.buf, 0xa5, (capacity + 2 * GUARD_WORDS) * 4)
            .map_err(err)?;
        let live = owner.sub_offset(GUARD_WORDS, words);
        Ok(Self {
            owner,
            live,
            capacity,
        })
    }
    fn read(&self, gpu: &Gpu) -> Result<Vec<u8>> {
        tensor_bytes(
            gpu,
            &self.owner.sub_offset(0, self.capacity + 2 * GUARD_WORDS),
        )
    }
}
fn tensor_bytes(gpu: &Gpu, tensor: &GpuTensor) -> Result<Vec<u8>> {
    let mut bytes = vec![0; tensor.buf.size()];
    gpu.hip.memcpy_dtoh(&mut bytes, &tensor.buf).map_err(err)?;
    Ok(bytes)
}
struct Frozen {
    name: String,
    allocation: Guarded,
    expected: Vec<u8>,
}
fn freeze(gpu: &mut Gpu, name: impl Into<String>, bytes: &[u8]) -> Result<Frozen> {
    if bytes.len() % 4 != 0 {
        return Err("immutable operand is not word aligned".into());
    }
    let allocation = Guarded::new(gpu, bytes.len() / 4, bytes.len() / 4)?;
    gpu.hip
        .memcpy_htod(&allocation.live.buf, bytes)
        .map_err(err)?;
    let mut expected = vec![0xa5; bytes.len() + GUARD_WORDS * 8];
    expected[GUARD_WORDS * 4..GUARD_WORDS * 4 + bytes.len()].copy_from_slice(bytes);
    Ok(Frozen {
        name: name.into(),
        allocation,
        expected,
    })
}
fn compare_words(label: &str, got: &[u8], expected: &[u8], row_words: usize) -> Value {
    let mut mm = Mm::default();
    if got.len() != expected.len() {
        mm.push(json!({"part": label, "got_bytes": got.len(), "expected_bytes": expected.len()}));
    }
    for (i, (g, e)) in got
        .chunks_exact(4)
        .zip(expected.chunks_exact(4))
        .enumerate()
    {
        if g != e {
            if mm.first.len() == 8 {
                mm.count += 1;
                continue;
            }
            let live = if label.starts_with("PM_descriptor") || label.starts_with("descriptor_") {
                Some(i)
            } else {
                i.checked_sub(GUARD_WORDS)
            };
            let slot_stride = if label.starts_with("down_unweighted") {
                Some(DOWN_M)
            } else if label.starts_with("gate_raw")
                || label.starts_with("up_raw")
                || label.starts_with("activation_rotated")
            {
                Some(DOWN_K)
            } else {
                None
            };
            mm.push(
                json!({"part": label, "word": i, "row": live.map(|w| w / row_words.max(1)),
                "row_word": live.map(|w| w % row_words.max(1)),
                "slot": live.zip(slot_stride).map(|(w, stride)| w / stride),
                "rank": live.zip(slot_stride).map(|(w, stride)| (w / stride) % 10),
                "output": live.zip(slot_stride).map(|(w, stride)| w % stride),
                "got": bits_hex(u32::from_le_bytes(g.try_into().unwrap())),
                "expected": bits_hex(u32::from_le_bytes(e.try_into().unwrap()))}),
            );
        }
    }
    json!({"part": label, "bytes": got.len(), "mismatch_count": mm.count, "first_mismatches": mm.first})
}
fn poison_check(label: &str, bytes: &[u8], live_words: usize) -> Value {
    let mut mm = Mm::default();
    for (word, chunk) in bytes.chunks_exact(4).enumerate() {
        if (GUARD_WORDS..GUARD_WORDS + live_words).contains(&word) {
            continue;
        }
        let got = u32::from_le_bytes(chunk.try_into().unwrap());
        if got != SENTINEL_BITS {
            if mm.first.len() == 8 {
                mm.count += 1;
                continue;
            }
            mm.push(json!({"part": label, "allocation_word": word, "allocation_byte": word * 4,
                "region": if word < GUARD_WORDS { "leading_guard" } else { "inactive_rows_or_trailing_guard" },
                "got": bits_hex(got), "expected": bits_hex(SENTINEL_BITS)}));
        }
    }
    json!({"part": label, "allocation_bytes": bytes.len(), "live_bytes": live_words * 4,
        "mismatch_count": mm.count, "first_mismatches": mm.first})
}
/// Actual loaded identity: module, container SHA and per-arch raw device SHA of the admitted
/// object (`lab_kernel_device_identity`). No container-only or expected-pin substitution.
fn identity(gpu: &Gpu, symbol: &str, arm: &str) -> Result<Value> {
    let (module, container_sha256, device_sha256) =
        gpu.lab_kernel_device_identity(symbol).map_err(err)?;
    Ok(json!({"arm": arm, "symbol": symbol, "module": module,
        "container_sha256": container_sha256, "object_sha256": container_sha256,
        "device_object_sha256": device_sha256, "device_arch": gpu.arch}))
}
fn stage_row_words(stage: &str) -> usize {
    if stage.starts_with("combine") {
        DOWN_M
    } else if stage == "down_unweighted" {
        10 * DOWN_M
    } else {
        10 * DOWN_K
    }
}
struct ChainBuffers {
    gate: Guarded,
    up: Guarded,
    rotated: Guarded,
    down: Guarded,
    target: Guarded,
}
impl ChainBuffers {
    fn new(gpu: &mut Gpu, rows: usize, residual: &[f32]) -> Result<Self> {
        let target = Guarded::new(gpu, rows * DOWN_M, MAX_ROWS * DOWN_M)?;
        gpu.hip
            .memcpy_htod(&target.live.buf, &f32_bytes(residual))
            .map_err(err)?;
        Ok(Self {
            gate: Guarded::new(gpu, rows * 10 * DOWN_K, MAX_ROWS * 10 * DOWN_K)?,
            up: Guarded::new(gpu, rows * 10 * DOWN_K, MAX_ROWS * 10 * DOWN_K)?,
            rotated: Guarded::new(gpu, rows * 10 * DOWN_K, MAX_ROWS * 10 * DOWN_K)?,
            down: Guarded::new(gpu, rows * 10 * DOWN_M, MAX_ROWS * 10 * DOWN_M)?,
            target,
        })
    }
    fn free(self, gpu: &mut Gpu) -> Result<()> {
        for b in [self.gate, self.up, self.rotated, self.down, self.target] {
            gpu.free_tensor(b.owner).map_err(err)?;
        }
        Ok(())
    }
}
struct ChainRun {
    snapshots: BTreeMap<&'static str, Vec<u8>>,
    identities: Vec<Value>,
    calls: usize,
    descriptors: Option<Vec<u8>>,
    guards: u64,
    guard_parts: Vec<Value>,
    immutable_parts: Vec<Value>,
}
fn snapshot(
    gpu: &Gpu,
    snapshots: &mut BTreeMap<&'static str, Vec<u8>>,
    label: &'static str,
    tensor: &Guarded,
    first: usize,
    rows: usize,
) -> Result<()> {
    let row_words = stage_row_words(label);
    let bytes = snapshots
        .entry(label)
        .or_insert_with(|| vec![0xa5; (tensor.capacity + 2 * GUARD_WORDS) * 4]);
    let got = tensor_bytes(
        gpu,
        &tensor.live.sub_offset(first * row_words, rows * row_words),
    )?;
    let offset = (GUARD_WORDS + first * row_words) * 4;
    bytes[offset..offset + got.len()].copy_from_slice(&got);
    Ok(())
}

/// Production slab_route K1/legacy entrypoints; reference executes sequential actual AR rows.
#[allow(clippy::too_many_arguments)]
fn chain_run(
    gpu: &mut Gpu,
    arch: &str,
    case: &Case,
    gate_ptrs: &GpuTensor,
    down_ptrs: &GpuTensor,
    codec: &GpuTensor,
    topk: &GpuTensor,
    weights: &GpuTensor,
    x: &GpuTensor,
    buffers: &ChainBuffers,
    tiled: bool,
    legacy: bool,
) -> Result<ChainRun> {
    let rows = case.c;
    let cold_codec = if legacy { 0 } else { case_cold_codec(case)? };
    // Production `slab_route::layer`: a per-expert codec table exists only for mixed layers
    // (MQ4 plus one cold codec); an MQ4-only layer (cold codec 0) is `Uniform(1)`, the MQ4
    // bucket. The runtime refuses a table without an admitted cold codec.
    let buckets = if cold_codec == 0 {
        SlabBuckets::Uniform(1)
    } else {
        SlabBuckets::Table(codec)
    };
    let uniform_bucket = (cold_codec == 0).then_some(1usize);
    // C1 serves only MQ4-only and W3 cold layers (source: `k1_rows8_cold_codec`); E8P-cold K1
    // is always the production K1 entrypoint, never C1.
    let actual_c1 = gpu.slab_k1_c1() && !legacy && slab_moe::k1_rows8_cold_codec(cold_codec);
    let arm = if tiled {
        "candidate"
    } else {
        "sequential_actual_singleton"
    };
    let mut result = ChainRun {
        snapshots: BTreeMap::new(),
        identities: Vec::new(),
        calls: 0,
        descriptors: None,
        guards: 0,
        guard_parts: Vec::new(),
        immutable_parts: Vec::new(),
    };
    let mut descriptor_before = None;
    if tiled && !legacy {
        gpu.reserve_qwen4_slab(0).map_err(err)?;
        let descriptor = gpu
            .qwen4_slab_union_descriptor_view(MAX_ROWS, true)
            .map_err(err)?;
        gpu.hip
            .memset(&descriptor.buf, 0xa5, descriptor.buf.size())
            .map_err(err)?;
        gpu.qwen4_slab_union_plan_rowtiles(topk, buckets, rows)
            .map_err(err)?;
        result.calls += 1;
        let (planner, _) =
            slab_moe::slab_rowtile_planner(arch).ok_or("missing architecture PM planner")?;
        result.identities.push(identity(gpu, planner, arm)?);
        let got = tensor_bytes(gpu, &descriptor)?;
        let membership = tensor_bytes(gpu, codec)?;
        let ids = tensor_bytes(gpu, topk)?;
        let signed: Vec<i32> = ids
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let expected = cpu_rowtile_descriptors(&signed, &membership, rows, uniform_bucket)?;
        let mut expected_bytes = vec![0xa5; descriptor.buf.size()];
        expected_bytes[..expected.len() * 4].copy_from_slice(&u32_bytes(&expected));
        let check = compare_words(
            "PM_descriptor_before_gate",
            &got,
            &expected_bytes,
            TILE_WORDS,
        );
        if check["mismatch_count"].as_u64().unwrap_or(1) != 0 {
            return Err(json!({"error": "PM planner differs from CPU first-occurrence semantics",
                "R": rows, "objects": result.identities, "descriptor_extent_bytes": descriptor.buf.size(),
                "active_descriptor_bytes": expected.len() * 4, "mismatch": check}).to_string());
        }
        descriptor_before = Some(got);
    }
    let step = if tiled { rows } else { 1 };
    for first in (0..rows).step_by(step) {
        let count = if tiled { rows } else { 1 };
        let tk = topk.sub_offset(first * 10, count * 10);
        let tw = weights.sub_offset(first * 10, count * 10);
        let input = x.sub_offset(first * GATE_K, count * GATE_K);
        let gate = buffers
            .gate
            .live
            .sub_offset(first * 10 * DOWN_K, count * 10 * DOWN_K);
        let up = buffers
            .up
            .live
            .sub_offset(first * 10 * DOWN_K, count * 10 * DOWN_K);
        let rotated = buffers
            .rotated
            .live
            .sub_offset(first * 10 * DOWN_K, count * 10 * DOWN_K);
        let down = buffers
            .down
            .live
            .sub_offset(first * 10 * DOWN_M, count * 10 * DOWN_M);
        let target = buffers
            .target
            .live
            .sub_offset(first * DOWN_M, count * DOWN_M);
        if legacy {
            gpu.gemv_mq4g256v2_moe_gate_up_top10_indexed_batched(
                gate_ptrs, &tk, &input, &gate, &up, GATE_M, GATE_K, count,
            )
            .map_err(err)?;
            result.identities.push(identity(
                gpu,
                "gemv_mq4g256v2_moe_gate_up_k2560_rows8_indexed_batched",
                arm,
            )?);
        } else if !tiled && actual_c1 {
            gpu.mq_i4_moe_gemv_slab_c1(
                false,
                buckets,
                gate_ptrs,
                &tk,
                &input,
                &gate,
                Some(&up),
                cold_codec,
            )
            .map_err(err)?;
            let metadata =
                slab_moe::slab_c1_k1_metadata(arch).ok_or("missing actual C1 metadata")?;
            result
                .identities
                .push(identity(gpu, metadata.symbols[0], arm)?);
        } else {
            if !tiled {
                gpu.qwen4_slab_union_plan(&tk, buckets, 1).map_err(err)?;
                result
                    .identities
                    .push(identity(gpu, "qwen4_slab_union_plan", arm)?);
                result.calls += 1;
            }
            gpu.mq_i4_moe_gemv_slab_with(
                false,
                buckets,
                gate_ptrs,
                &input,
                &gate,
                Some(&up),
                count,
                cold_codec,
                tiled.then_some(SlabK1Module::RowTiles),
            )
            .map_err(err)?;
            // Identities record ONLY actually launched buckets: cold bucket when the layer
            // holds its cold codec (cold 0 = NO cold bucket), MQ4 bucket when it holds MQ4
            // experts. MQ4-only never pretends a W3 inventory launch for the empty bucket.
            let launched_cold = case
                .experts
                .values()
                .any(|e| e.codec.route_byte() == cold_codec)
                && cold_codec != 0;
            let launched_mq4 = case.experts.values().any(|e| e.codec == Codec::Mq4);
            assert!(
                launched_cold || launched_mq4,
                "slab launch covered no bucket"
            );
            let module_for = |tiled: bool| {
                if tiled {
                    SlabK1Module::RowTiles
                } else {
                    SlabK1Module::K1
                }
            };
            if launched_cold {
                let cold_meta =
                    slab_moe::slab_cold_k1_metadata(arch, cold_codec).unwrap_or_else(|| {
                        panic!("missing admitted cold metadata for codec {cold_codec}")
                    });
                let picked = slab_moe::slab_k1_pick(
                    &cold_meta.k1,
                    module_for(tiled),
                    0,
                    false,
                    GATE_M,
                    count * 10,
                );
                result.identities.push(identity(gpu, picked.symbol, arm)?);
            }
            if launched_mq4 {
                let arch_meta =
                    slab_moe::slab_arch_kernels(arch).ok_or("missing slab architecture")?;
                let picked = slab_moe::slab_k1_pick(
                    &arch_meta.mq4_k1,
                    module_for(tiled),
                    0,
                    false,
                    GATE_M,
                    count * 10,
                );
                result.identities.push(identity(gpu, picked.symbol, arm)?);
            }
        }
        result.calls += 1;
        snapshot(
            gpu,
            &mut result.snapshots,
            "gate_raw",
            &buffers.gate,
            first,
            count,
        )?;
        snapshot(
            gpu,
            &mut result.snapshots,
            "up_raw",
            &buffers.up,
            first,
            count,
        )?;
        // Production AR fused BF16 gate/up round trips, SiLU, BF16 activation round trip, FWHT.
        // Routed-only branch: no shared-expert operands. Do not replace with host/split arithmetic.
        gpu.silu_mul_bf16_rt_rotate_x_mq_128_v2(&gate, &up, &rotated, DOWN_K, count * 10)
            .map_err(err)?;
        result
            .identities
            .push(identity(gpu, "mq_rotate_x_128_v2_silu_bf16", arm)?);
        result.calls += 1;
        snapshot(
            gpu,
            &mut result.snapshots,
            "activation_rotated",
            &buffers.rotated,
            first,
            count,
        )?;
        if legacy {
            gpu.gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
                down_ptrs, &tk, &rotated, &down, DOWN_M, DOWN_K, count, N_EXPERTS, 10,
            )
            .map_err(err)?;
            result.identities.push(identity(
                gpu,
                "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded",
                arm,
            )?);
        } else if !tiled && actual_c1 {
            gpu.mq_i4_moe_gemv_slab_c1(
                true, buckets, down_ptrs, &tk, &rotated, &down, None, cold_codec,
            )
            .map_err(err)?;
            let metadata =
                slab_moe::slab_c1_k1_metadata(arch).ok_or("missing actual C1 metadata")?;
            result
                .identities
                .push(identity(gpu, metadata.symbols[1], arm)?);
        } else {
            // Same descriptors as gate/up; no intervening planner launch.
            gpu.mq_i4_moe_gemv_slab_with(
                true,
                buckets,
                down_ptrs,
                &rotated,
                &down,
                None,
                count,
                cold_codec,
                tiled.then_some(SlabK1Module::RowTiles),
            )
            .map_err(err)?;
            let launched_cold = case
                .experts
                .values()
                .any(|e| e.codec.route_byte() == cold_codec)
                && cold_codec != 0;
            let launched_mq4 = case.experts.values().any(|e| e.codec == Codec::Mq4);
            assert!(
                launched_cold || launched_mq4,
                "slab launch covered no bucket"
            );
            let module_for = |tiled: bool| {
                if tiled {
                    SlabK1Module::RowTiles
                } else {
                    SlabK1Module::K1
                }
            };
            if launched_cold {
                let cold_meta =
                    slab_moe::slab_cold_k1_metadata(arch, cold_codec).unwrap_or_else(|| {
                        panic!("missing admitted cold metadata for codec {cold_codec}")
                    });
                let picked = slab_moe::slab_k1_pick(
                    &cold_meta.k1,
                    module_for(tiled),
                    0,
                    true,
                    DOWN_M,
                    count * 10,
                );
                result.identities.push(identity(gpu, picked.symbol, arm)?);
            }
            if launched_mq4 {
                let arch_meta =
                    slab_moe::slab_arch_kernels(arch).ok_or("missing slab architecture")?;
                let picked = slab_moe::slab_k1_pick(
                    &arch_meta.mq4_k1,
                    module_for(tiled),
                    0,
                    true,
                    DOWN_M,
                    count * 10,
                );
                result.identities.push(identity(gpu, picked.symbol, arm)?);
            }
        }
        result.calls += 1;
        snapshot(
            gpu,
            &mut result.snapshots,
            "down_unweighted",
            &buffers.down,
            first,
            count,
        )?;
        gpu.moe_down_combine_top10_batched(&down, &tk, &tw, &target, DOWN_M, count, N_EXPERTS)
            .map_err(err)?;
        result
            .identities
            .push(identity(gpu, "moe_down_combine_top10_batched", arm)?);
        snapshot(
            gpu,
            &mut result.snapshots,
            "combine_raw",
            &buffers.target,
            first,
            count,
        )?;
        gpu.bf16_round_trip_f32(&target).map_err(err)?;
        result
            .identities
            .push(identity(gpu, "bf16_round_trip_f32", arm)?);
        result.calls += 2;
        snapshot(
            gpu,
            &mut result.snapshots,
            "combine_bf16",
            &buffers.target,
            first,
            count,
        )?;
    }
    gpu.hip.device_synchronize().map_err(err)?;
    for (label, b) in [
        ("gate", &buffers.gate),
        ("up", &buffers.up),
        ("rotated", &buffers.rotated),
        ("down", &buffers.down),
        ("target", &buffers.target),
    ] {
        let bytes = b.read(gpu)?;
        let check = poison_check(label, &bytes, b.live.numel());
        result.guards += check["mismatch_count"].as_u64().unwrap_or(1);
        result.guard_parts.push(check);
        let stage = match label {
            "gate" => "gate_raw",
            "up" => "up_raw",
            "rotated" => "activation_rotated",
            "down" => "down_unweighted",
            _ => "combine_bf16",
        };
        result.immutable_parts.push(compare_words(
            &format!("{stage}_consumer_input_immutable"),
            &bytes,
            &result.snapshots[stage],
            stage_row_words(stage),
        ));
    }
    if let Some(before) = descriptor_before {
        let view = gpu
            .qwen4_slab_union_descriptor_view(MAX_ROWS, true)
            .map_err(err)?;
        let after = tensor_bytes(gpu, &view)?;
        if before != after {
            return Err(
                json!({"error": "descriptors mutated during chain", "R": rows,
                "objects": result.identities,
                "mismatch": compare_words("descriptor_after_chain", &after, &before, TILE_WORDS)})
                .to_string(),
            );
        }
        result.descriptors = Some(after);
    }
    result.identities.sort_by_key(Value::to_string);
    result.identities.dedup();
    Ok(result)
}

fn run_case_wide(
    ctx: &mut Ctx,
    case: &Case,
    totals: &mut Totals,
) -> Result<(Value, TimingSamples)> {
    let mut gpu = match ctx.gpu.take() {
        Some(gpu) => gpu,
        None => Gpu::init_with_device(ctx.device).map_err(err)?,
    };
    let result = run_whole_chain(ctx, &mut gpu, case, totals).map_err(|error| {
        json!({"error": error, "case": case.name, "R": case.c, "device": ctx.device,
            "strict_refusal": true, "pass": false})
        .to_string()
    });
    ctx.gpu = Some(gpu);
    result
}

fn run_whole_chain(
    ctx: &Ctx,
    gpu: &mut Gpu,
    case: &Case,
    totals: &mut Totals,
) -> Result<(Value, TimingSamples)> {
    let rows = case.c;
    let arch = gpu.hip.get_arch(ctx.device).map_err(err)?;
    let legacy = case.name.contains("_legacy_");
    let invalid = case.name.contains("_invalid_");
    let cold_codec = if legacy { 0 } else { case_cold_codec(case)? };
    let actual_c1 = gpu.slab_k1_c1() && !legacy && slab_moe::k1_rows8_cold_codec(cold_codec);
    // Source admission is explicit: the whole sweep covers exactly the slab-admitted codecs
    // (MQ4 everywhere, W3 except gfx1100, E8P everywhere); W2 has no Codec variant and is
    // never admitted. Source-non-applicable cells (e.g. W3 on gfx1100) are enumerated as
    // NOT-APPLICABLE applicability listings before any weight upload or GPU launch; they are
    // not launches, not proof, and not skipped bodies.
    let route_bytes: Vec<u8> = case
        .experts
        .values()
        .map(|e| e.codec.route_byte())
        .collect();
    if !legacy
        && route_bytes
            .iter()
            .any(|&b| !qwen4_slab_codec_admitted(&arch, b))
    {
        return Ok((
            json!({"case": case.name, "origin": case.origin, "wide": true, "R": rows,
            "arch": arch, "cold_codec": cold_codec, "codec_bytes": route_bytes,
            "status": "NOT-APPLICABLE",
            "reason": "case holds a codec the source admission matrix refuses on this arch",
            "admission": "rdna_compute::slab_admission (W2 refused everywhere; W3 refused on gfx1100)",
            "launched": false, "proof": false,
            "applicability": "listed before any freeze/upload/launch; excluded from the admitted-cohort denominator"}),
            Vec::new(),
        ));
    }
    if !legacy && qwen4_slab_layer_cold_codec(&arch, &route_bytes).map(|c| c) != Ok(cold_codec) {
        return Err(format!(
            "case {} cold codec {cold_codec} disagrees with source admission",
            case.name
        ));
    }
    let mut frozen = Vec::new();
    let (mut gate_table, mut down_table) = (vec![0u64; N_EXPERTS], vec![0u64; N_EXPERTS]);
    let mut membership = vec![0u8; N_EXPERTS];
    for (&id, expert) in &case.experts {
        if legacy && expert.codec != Codec::Mq4 {
            return Err("legacy chain requires all MQ4 experts".into());
        }
        for (projection, bytes, table) in [
            ("gate", &expert.gate, &mut gate_table),
            ("down", &expert.down, &mut down_table),
        ] {
            let operand = freeze(gpu, format!("expert_{id}_{projection}"), bytes)?;
            table[id as usize] = operand.allocation.live.buf.as_ptr() as u64;
            frozen.push(operand);
        }
        membership[id as usize] = if legacy {
            0
        } else {
            membership_of(expert.codec, cold_codec)
        };
    }
    // Every entry names a real guarded weight allocation held in `frozen` through both
    // chains, raw singleton launches, synchronization and input-immutability readback.
    let gate_ptrs = freeze(gpu, "gate_pointer_table", &u64_bytes(&gate_table))?;
    let down_ptrs = freeze(gpu, "down_pointer_table", &u64_bytes(&down_table))?;
    let codec = freeze(gpu, "cold_membership_table", &membership)?;
    // Pack only live K words for the production contiguous ABI; the original padded fixtures
    // and their digest/stride stay in the narrow arm. Never pass padded rows to this API.
    let input: Vec<f32> = case
        .x_gate
        .chunks(case.xrs_gate)
        .flat_map(|row| row[..GATE_K].iter().copied())
        .collect();
    let x = freeze(gpu, "gate_input_contiguous", &f32_bytes(&input))?;
    let mut logits = vec![-32.0f32; rows * N_EXPERTS];
    let ties = case.name.ends_with("_ties");
    for row in 0..rows {
        if ties {
            logits[row * N_EXPERTS..(row + 1) * N_EXPERTS].fill(0.0);
        } else {
            for rank in 0..10 {
                logits[row * N_EXPERTS + case.route[row * 10 + rank] as usize] = (16 - rank) as f32;
            }
        }
    }
    let router_input = freeze(gpu, "router_logits", &f32_bytes(&logits))?;
    let topk = Guarded::new(gpu, rows * 10, MAX_ROWS * 10)?;
    let weights = Guarded::new(gpu, rows * 10, MAX_ROWS * 10)?;
    let single_ids = Guarded::new(gpu, rows * 10, MAX_ROWS * 10)?;
    let single_weights = Guarded::new(gpu, rows * 10, MAX_ROWS * 10)?;
    // Valid cases run the REAL planner path: the router produces top-k from logits, so an
    // invalid ID can never flow through activation/down/combine. Invalid cases therefore
    // bypass the router and launch BOTH arms on the same explicit routed-ID frozen inputs
    // (`invalid_topk_for`); valid cases keep the router-vs-fixture agreement check below.
    let mut router_objects = Vec::new();
    let (topk_before, weights_before, singleton_ids_before, singleton_weights_before) = if invalid {
        let frozen_topk = invalid_topk_for(&case.route, rows);
        let frozen_topk_bytes =
            u32_bytes(&frozen_topk.iter().map(|&v| v as u32).collect::<Vec<_>>());
        let frozen_weights = vec![0.5f32; rows * 10];
        let frozen_weights_bytes = f32_bytes(&frozen_weights);
        gpu.hip
            .memcpy_htod(&topk.live.buf, &frozen_topk_bytes)
            .map_err(err)?;
        gpu.hip
            .memcpy_htod(&weights.live.buf, &frozen_weights_bytes)
            .map_err(err)?;
        gpu.hip
            .memcpy_htod(&single_ids.live.buf, &frozen_topk_bytes)
            .map_err(err)?;
        gpu.hip
            .memcpy_htod(&single_weights.live.buf, &frozen_weights_bytes)
            .map_err(err)?;
        let ids = topk.read(gpu)?;
        let w = weights.read(gpu)?;
        (ids.clone(), w.clone(), ids, w)
    } else {
        gpu.moe_router_softmax_top10_f32(
            &router_input.allocation.live,
            &topk.live,
            &weights.live,
            rows,
            true,
            true,
            0,
            0,
        )
        .map_err(err)?;
        let router_symbol = if rows > 1 && gpu.flags.qwen4_router_fast_enabled() {
            "moe_router_softmax_top10_f32_fast"
        } else {
            "moe_router_softmax_top10_f32"
        };
        router_objects.push(identity(gpu, router_symbol, "candidate_router")?);
        for row in 0..rows {
            gpu.moe_router_softmax_top10_f32(
                &router_input
                    .allocation
                    .live
                    .sub_offset(row * N_EXPERTS, N_EXPERTS),
                &single_ids.live.sub_offset(row * 10, 10),
                &single_weights.live.sub_offset(row * 10, 10),
                1,
                true,
                true,
                0,
                row,
            )
            .map_err(err)?;
        }
        router_objects.push(identity(
            gpu,
            "moe_router_softmax_top10_f32",
            "singleton_router",
        )?);
        (
            topk.read(gpu)?,
            weights.read(gpu)?,
            single_ids.read(gpu)?,
            single_weights.read(gpu)?,
        )
    };
    let residual: Vec<f32> = input.iter().map(|v| *v * 0.125).collect();
    let candidate = ChainBuffers::new(gpu, rows, &residual)?;
    let reference = ChainBuffers::new(gpu, rows, &residual)?;
    let mut candidate_run = chain_run(
        gpu,
        &arch,
        case,
        &gate_ptrs.allocation.live,
        &down_ptrs.allocation.live,
        &codec.allocation.live,
        &topk.live,
        &weights.live,
        &x.allocation.live,
        &candidate,
        true,
        legacy,
    )?;
    let mut reference_run = chain_run(
        gpu,
        &arch,
        case,
        &gate_ptrs.allocation.live,
        &down_ptrs.allocation.live,
        &codec.allocation.live,
        &single_ids.live,
        &single_weights.live,
        &x.allocation.live,
        &reference,
        false,
        legacy,
    )?;
    let mut comparisons = if invalid {
        // Both arms share the same explicit frozen invalid-ID inputs; there is no router to
        // agree with. Identity of the frozen inputs is checked by bytes below.
        vec![
            compare_words(
                "frozen_invalid_ids",
                &topk_before,
                &singleton_ids_before,
                10,
            ),
            compare_words(
                "frozen_invalid_weights",
                &weights_before,
                &singleton_weights_before,
                10,
            ),
        ]
    } else {
        vec![
            compare_words("router_ids", &topk_before, &singleton_ids_before, 10),
            compare_words(
                "router_weights",
                &weights_before,
                &singleton_weights_before,
                10,
            ),
        ]
    };
    if !invalid {
        let routed_bytes = tensor_bytes(gpu, &topk.live)?;
        if routed_bytes != u32_bytes(&case.route) {
            return Err(json!({"error": "router did not reach intended fixture route", "R": rows,
                "router_objects": router_objects, "expected_ids": case.route,
                "actual_ids": routed_bytes.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>()}).to_string());
        }
    } else {
        let frozen = invalid_topk_for(&case.route, rows);
        let got: Vec<i32> = tensor_bytes(gpu, &topk.live)?
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        if got != frozen {
            return Err(
                json!({"error": "frozen invalid-ID inputs disagree", "R": rows,
                "expected_ids": frozen, "actual_ids": got})
                .to_string(),
            );
        }
    }
    // All stage words, including raw combine before BF16 and every inactive/guard word,
    // are compared against the actually executed singleton chain, never a copied candidate.
    for (&stage, bytes) in &candidate_run.snapshots {
        let expected = reference_run
            .snapshots
            .get(stage)
            .ok_or_else(|| format!("singleton chain did not produce stage {stage}"))?;
        comparisons.push(compare_words(
            &format!("{stage}_vs_actual_runtime_singleton"),
            bytes,
            expected,
            stage_row_words(stage),
        ));
    }
    let singleton_mismatches: u64 = comparisons
        .iter()
        .map(|part| part["mismatch_count"].as_u64().unwrap_or(1))
        .sum();
    comparisons.append(&mut candidate_run.immutable_parts);
    comparisons.append(&mut reference_run.immutable_parts);
    let buckets = buckets_of(case);
    let gate_cpu = cpu_expected_x(case, Kind::GateUp, &buckets, &input, GATE_K, DOWN_K);
    let rotated = words_as_f32(
        &candidate_run.snapshots["activation_rotated"]
            [GUARD_WORDS * 4..(GUARD_WORDS + rows * 10 * DOWN_K) * 4],
    );
    // CPU down consumes the actual production activation/rotation, never synthetic down inputs.
    let down_cpu = cpu_expected_x(case, Kind::Down, &buckets, &rotated, DOWN_K, DOWN_M);
    let mut cpu_mismatches = 0;
    // Invalid slots have no CPU projection: the production K1 skips invalid expert IDs without
    // any load or store (a fault guard, not a routing rule), so those output words stay at the
    // poison pattern. Compare every raw output word exactly (valid slots against the CPU DAG,
    // invalid slots against poison), both arms from identical initial buffers, no approximate math.
    let poison_word = f32::from_bits(SENTINEL_BITS);
    if invalid {
        let bad: BTreeSet<usize> = invalid_positions(rows).into_iter().collect();
        for (stage, live, stride) in [
            ("gate_raw", rows * 10 * DOWN_K, DOWN_K),
            ("up_raw", rows * 10 * DOWN_K, DOWN_K),
            ("down_unweighted", rows * 10 * DOWN_M, DOWN_M),
        ] {
            let words = live / stride.max(1);
            let dag: &[f32] = match stage {
                "gate_raw" => &gate_cpu.y0,
                "up_raw" => gate_cpu.y1.as_ref().ok_or("missing CPU up")?,
                _ => &down_cpu.y0,
            };
            let mut flat: Vec<f32> = Vec::with_capacity(live);
            for slot in 0..words {
                for &v in &dag[slot * stride..(slot + 1) * stride] {
                    flat.push(if bad.contains(&slot) { poison_word } else { v });
                }
            }
            let mut bytes = vec![0xa5; candidate_run.snapshots[stage].len()];
            bytes[GUARD_WORDS * 4..GUARD_WORDS * 4 + flat.len() * 4]
                .copy_from_slice(&f32_bytes(&flat));
            let check = compare_words(
                &format!("{stage}_vs_cpu_DAG_invalid_poison"),
                &candidate_run.snapshots[stage],
                &bytes,
                stage_row_words(stage),
            );
            cpu_mismatches += check["mismatch_count"].as_u64().unwrap_or(1);
            comparisons.push(check);
        }
    } else {
        for (stage, expected) in [
            ("gate_raw", &gate_cpu.y0),
            ("up_raw", gate_cpu.y1.as_ref().ok_or("missing CPU up")?),
            ("down_unweighted", &down_cpu.y0),
        ] {
            let mut bytes = vec![0xa5; candidate_run.snapshots[stage].len()];
            bytes[GUARD_WORDS * 4..GUARD_WORDS * 4 + expected.len() * 4]
                .copy_from_slice(&f32_bytes(expected));
            let check = compare_words(
                &format!("{stage}_vs_cpu_DAG"),
                &candidate_run.snapshots[stage],
                &bytes,
                stage_row_words(stage),
            );
            cpu_mismatches += check["mismatch_count"].as_u64().unwrap_or(1);
            comparisons.push(check);
        }
    }
    // Preserve the independent raw singleton object arm for the W3/MQ4 narrow projections. It
    // cannot replace actual AR above. E8P is slab-only: the narrow raw singleton is never run
    // on E8P bytes (no fake W3/MQ4 reference), and invalid slots have no CPU raw projection.
    let mut raw_objects = Vec::new();
    let mut raw_mismatches = 0;
    let has_e8p = case.experts.values().any(|e| e.codec == Codec::E8p);
    if !has_e8p && !invalid {
        let device_object_sha256 = ctx.device_object_sha256.clone();
        for kind in Kind::ALL {
            let raw = raw_chain_projection(
                ctx,
                gpu,
                case,
                kind,
                if kind == Kind::GateUp {
                    &input
                } else {
                    &rotated
                },
                if kind == Kind::GateUp {
                    &gate_table
                } else {
                    &down_table
                },
            )?;
            for (part, expected) in raw {
                let mut padded = vec![0xa5; candidate_run.snapshots[part].len()];
                padded[GUARD_WORDS * 4..GUARD_WORDS * 4 + expected.len()]
                    .copy_from_slice(&expected);
                let check = compare_words(
                    &format!("{part}_vs_independent_raw_singleton"),
                    &candidate_run.snapshots[part],
                    &padded,
                    stage_row_words(part),
                );
                raw_mismatches += check["mismatch_count"].as_u64().unwrap_or(1);
                comparisons.push(check);
            }
            for codec in Codec::ALL {
                if case.experts.values().any(|e| e.codec == codec) {
                    raw_objects.push(json!({"arm": "independent_raw_singleton",
                        "module": "explicit_--codeobject", "symbol": ctx.loaded_symbol(codec, kind),
                        "container_sha256": ctx.codeobject_sha, "object_sha256": ctx.codeobject_sha,
                        "device_object_sha256": device_object_sha256}));
                }
            }
        }
    } else {
        raw_objects.push(json!({"arm": "independent_raw_singleton",
            "status": if has_e8p { "NOT-APPLICABLE-E8P-SLAB-ONLY" } else { "NOT-APPLICABLE-INVALID-SLOTS" },
            "reason": if has_e8p { "narrow raw singleton is never run on E8P bytes" }
                else { "invalid slots poison outputs; no CPU raw projection" }}));
    }
    for (name, tensor, before) in [
        ("routed_ids_immutable", &topk, &topk_before),
        ("route_weights_immutable", &weights, &weights_before),
        (
            "singleton_ids_immutable",
            &single_ids,
            &singleton_ids_before,
        ),
        (
            "singleton_weights_immutable",
            &single_weights,
            &singleton_weights_before,
        ),
    ] {
        comparisons.push(compare_words(name, &tensor.read(gpu)?, before, 10));
    }
    frozen.extend([gate_ptrs, down_ptrs, codec, x, router_input]);
    for operand in &frozen {
        comparisons.push(compare_words(
            &format!("immutable_{}", operand.name),
            &operand.allocation.read(gpu)?,
            &operand.expected,
            operand.allocation.capacity,
        ));
    }
    let mut router_guards = 0;
    let mut router_guard_parts = Vec::new();
    for tensor in [&topk, &weights, &single_ids, &single_weights] {
        let check = poison_check("router_output", &tensor.read(gpu)?, rows * 10);
        router_guards += check["mismatch_count"].as_u64().unwrap_or(1);
        router_guard_parts.push(check);
    }
    let guards = candidate_run.guards + reference_run.guards + router_guards;
    let mismatches: u64 = comparisons
        .iter()
        .map(|p| p["mismatch_count"].as_u64().unwrap_or(1))
        .sum();
    let prefix_evidence: Vec<Value> = (1..=rows)
        .map(|keep| {
            let equal = candidate_run.snapshots.iter().all(|(&stage, bytes)| {
                let range = GUARD_WORDS * 4..(GUARD_WORDS + keep * stage_row_words(stage)) * 4;
                bytes[range.clone()] == reference_run.snapshots[stage][range]
            });
            json!({"keep": keep, "candidate_rows_reached": rows, "singleton_rows_reached": rows,
            "last_absolute_slot": keep * 10 - 1, "all_stage_prefix_bytes_equal": equal})
        })
        .collect();
    let boundaries: Vec<usize> = (1..=rows.div_ceil(8))
        .flat_map(|tile| [tile * 80 - 1, tile * 80])
        .filter(|&slot| slot < rows * 10)
        .collect();
    let nonfinite: BTreeMap<&str, usize> = candidate_run
        .snapshots
        .iter()
        .map(|(&stage, bytes)| {
            let words = rows * stage_row_words(stage);
            (
                stage,
                bytes[GUARD_WORDS * 4..(GUARD_WORDS + words) * 4]
                    .chunks_exact(4)
                    .filter(|b| {
                        !f32::from_bits(u32::from_le_bytes((*b).try_into().unwrap())).is_finite()
                    })
                    .count(),
            )
        })
        .collect();
    totals.cases += 1;
    totals.wide_cases += 1;
    totals.launches +=
        (candidate_run.calls + reference_run.calls + if invalid { 0 } else { rows + 1 }) as u64;
    totals.wide_launches += candidate_run.calls as u64;
    totals.wide_cpu_mismatches += cpu_mismatches;
    totals.mismatches += mismatches;
    totals.wide_singleton_mismatches += singleton_mismatches;
    totals.wide_reference_mismatches += raw_mismatches;
    totals.wide_guard_violations += guards;
    totals.twin_mismatches += gate_cpu.twin_mismatches + down_cpu.twin_mismatches;
    if case.origin == "synthetic_wide" {
        totals.synthetic += 1;
    } else {
        totals.real += 1;
        if case.origin == RECORDS_ORIGIN {
            totals.real_records += 1;
        }
    }
    let receipt = json!({"case": case.name, "origin": case.origin, "wide": true, "R": rows,
        "arch": arch, "cold_codec": cold_codec, "codec_bytes": route_bytes,
        "invalid_ids": invalid, "invalid_slots": if invalid { invalid_positions(rows) } else { Vec::<usize>::new() },
        "route": if legacy { "legacy_indexed_MQ4" } else { "slab_PM_t8" },
        "execution_domain": if case.origin == "native_mtp_head_diagnostic" { "native_mtp_head_diagnostic" }
            else { "target_verify_component" },
        "reference_mode": if legacy { "actual_legacy_singleton" }
            else if actual_c1 { "actual_runtime_C1" } else { "actual_runtime_K1" },
        "reference_note": if legacy { "legacy indexed singleton (MQ4-only)" }
            else if actual_c1 { "C1 serves only cold 0/1 layers; this layer is cold 0/1" }
            else { "E8P-cold or C1-disabled layer: production K1 entrypoint on both arms" },
        "input_sha256": case.input_sha256(), "source_x_stride": case.xrs_gate, "production_x_stride": GATE_K,
        "objects": [candidate_run.identities, reference_run.identities, router_objects, raw_objects],
        "descriptor": {"words_per_tile": TILE_WORDS, "active_bytes": if legacy {0} else {rows.div_ceil(8) * TILE_WORDS * 4},
            "reserved_bytes": if legacy {0} else {8 * TILE_WORDS * 4}, "external_guards_available": false,
            "all_words_and_unused_descriptors_checked": !legacy,
            "immutable_through_down": candidate_run.descriptors.is_some(),
            "candidate_evidence_words": if legacy {0} else {rows.div_ceil(8) * TILE_WORDS * 4} / 4,
            "guard_formula": "descriptor has no external guards; internal unused words are 0x00000000-checked exactly"},
        "routes": {"guard_formula": "4 guarded route buffers (topk/weights/singleton pair), live words 10*R, capacity 640, guards 64 words each",
            "route_guard_bytes": 4 * (640 + 2 * 64) * 4, "live_route_words": rows * 10},
        "resources": {
            "chain_capacity_words_per_arm": 64 * (3 * 10 * 640 + 10 * 2560 + 2560),
            "chain_capacity_bytes_per_arm": 64 * (3 * 10 * 640 + 10 * 2560 + 2560) * 4 + 5 * 2 * 64 * 4,
            "chain_capacity_bytes_both_arms": 2 * (64 * (3 * 10 * 640 + 10 * 2560 + 2560) * 4 + 5 * 2 * 64 * 4),
            "chain_capacity_guard_words_per_buffer": 64,
            "descriptor_reserved_bytes": 8 * 482 * 4,
            "descriptor_formula": "reserved = 8 tiles * 482 words * 4 B; active = ceil(R/8) * 482 * 4 B",
            "gate_input_live_words": 10 * rows * 256 / 10 * 10 / 1,
            "router_logits_words": rows * 512,
            "expert_weight_bytes_formula": "W3 gate_up+down 1689600+844800; MQ4 1740800+870400; E8P 870400+435200; +2 projection guards of 512 B per expert",
            "shapes": "fixed slab [1280,2560] gate/up and [2560,640] down; expert_pair bytes as above"},
        "admission": {"source": "rdna_compute::slab_admission", "W2": "refused everywhere (no Codec variant)",
            "gfx1201": "MQ4+W3+E8P", "gfx1100": "MQ4+E8P (W3 NOT-ADMITTED)", "gfx1151": "MQ4+W3+E8P",
            "no_dispatch_claim": "this receipt names only the direct rowtile planner/launch entrypoints and slab planner/C1/K1 metadata; it does not claim a private dispatch-selection wrapper was invoked"},
        "reached_absolute_slot_boundaries": boundaries, "prefix_evidence": prefix_evidence,
        "domain": if invalid { "invalid slots poison outputs by contract; raw bytes compared exactly on both arms from identical initial buffers, no approximate math" }
            else { "finite operands; raw overflow bytes counted; no all-NaN or CPU-emulated-object claim" },
        "activation": "production fused BF16-round-trip SiLU multiply + BF16-round-trip + rotation, routed-only (shared=None)",
        "nonfinite_live_intermediates": nonfinite,
        "mismatch_count": mismatches, "guard_violation_count": guards, "parts": comparisons,
        "guard_parts": [candidate_run.guard_parts, reference_run.guard_parts, router_guard_parts],
        "pass": mismatches == 0 && guards == 0 && gate_cpu.twin_mismatches + down_cpu.twin_mismatches == 0});
    candidate.free(gpu)?;
    reference.free(gpu)?;
    for operand in frozen {
        gpu.free_tensor(operand.allocation.owner).map_err(err)?;
    }
    for b in [topk, weights, single_ids, single_weights] {
        gpu.free_tensor(b.owner).map_err(err)?;
    }
    Ok((receipt, Vec::new()))
}

fn raw_chain_projection(
    ctx: &Ctx,
    gpu: &Gpu,
    case: &Case,
    kind: Kind,
    input: &[f32],
    table: &[u64],
) -> Result<Vec<(&'static str, Vec<u8>)>> {
    let mut dev = Dev::new(&ctx.hip);
    let result = (|| {
        let x = dev.upload(&f32_bytes(input))?;
        let pointers = dev.upload(&u64_bytes(table))?;
        let words = case.c * 10 * kind.out_rows();
        let bytes = (words + 2 * GUARD_WORDS) * 4;
        let y0 = dev.fill(bytes, 0xa5)?;
        let y1 = if kind == Kind::GateUp {
            Some(dev.fill(bytes, 0xa5)?)
        } else {
            None
        };
        for row in 0..case.c {
            // Slab-order buckets: cold bucket first, MQ4 second. The narrow arm only covers
            // W3/MQ4, so resolve explicitly and skip an empty cold bucket exactly like `run_kind`.
            let cold = case_cold_codec(case)?;
            let narrow: Vec<Codec> = [Codec::W3, Codec::Mq4]
                .into_iter()
                .filter(|&c| {
                    if c == Codec::W3 {
                        cold == Codec::W3.route_byte()
                    } else {
                        true
                    }
                })
                .collect();
            for codec in narrow {
                let b = build_bucket(
                    &case.route[row * 10..(row + 1) * 10],
                    &|id| case.experts[&id].codec,
                    codec,
                );
                if b.ids.is_empty() {
                    continue;
                }
                let ids = dev.upload(&u32_bytes(&b.ids))?;
                let offsets = dev.upload(&u32_bytes(&b.offs))?;
                let slots = dev.upload(&u32_bytes(&b.slots))?;
                let offset = ((GUARD_WORDS + row * 10 * kind.out_rows()) * 4) as u64;
                let x_rows = if kind == Kind::GateUp { 1 } else { 10 };
                let mut args = kernarg(
                    [
                        pointers.ptr,
                        ids.ptr,
                        offsets.ptr,
                        slots.ptr,
                        x.ptr + (row * x_rows * kind.k() * 4) as u64,
                        y0.ptr + offset,
                        y1.map_or(0, |b| b.ptr + offset),
                    ],
                    kind.m(),
                    kind.k(),
                    b.ids.len(),
                    1,
                    kind.x_row_div(),
                    x_rows,
                    kind.k(),
                    kind.out_rows(),
                );
                // SAFETY: frozen ABI; every allocation stays alive through synchronize.
                unsafe {
                    ctx.hip.launch_kernel_blob(
                        ctx.func(codec, kind),
                        [kind.m() as u32, b.ids.len() as u32, 1],
                        [32, 1, 1],
                        0,
                        None,
                        &mut args,
                    )
                }
                .map_err(err)?;
            }
        }
        gpu.hip.device_synchronize().map_err(err)?;
        let mut outputs = Vec::new();
        for (name, buffer) in [
            (
                if kind == Kind::GateUp {
                    "gate_raw"
                } else {
                    "down_unweighted"
                },
                Some(y0),
            ),
            ("up_raw", y1),
        ] {
            if let Some(buffer) = buffer {
                let got = dev.download(buffer, bytes)?;
                let guard_check = poison_check(name, &got, words);
                if guard_check["mismatch_count"].as_u64().unwrap_or(1) != 0 {
                    return Err(json!({"error": "independent raw singleton overwrote guards",
                        "symbols": Codec::ALL.iter().filter(|&&codec| case.experts.values().any(|e| e.codec == codec))
                            .map(|&codec| ctx.loaded_symbol(codec, kind)).collect::<Vec<_>>(),
                        "container_sha256": ctx.codeobject_sha, "object_sha256": ctx.codeobject_sha,
                        "mismatch": guard_check}).to_string());
                }
                outputs.push((
                    name,
                    got[GUARD_WORDS * 4..(GUARD_WORDS + words) * 4].to_vec(),
                ));
            }
        }
        Ok(outputs)
    })();
    dev.release();
    result
}

// ---------------------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------------------

#[derive(Debug)]
struct Cfg {
    codeobject: String,
    artifact: Option<String>,
    layer: usize,
    real_experts: usize,
    old_parity: bool,
    timing: bool,
    reps: usize,
    warmup: usize,
    launches: usize,
    device: Option<i32>,
    expect_uuid: Option<String>,
    expect_bdf: String,
    expect_arch: String,
    card: Option<String>,
    fabric_card: Option<String>,
    fabric_owner: Option<String>,
    fabric_wrapper: Option<String>,
    only: Vec<String>,
    /// Directory holding the layer's `E{e}.npz` certified R1 records (e.g. `.../q/w3-gptq3/L0`).
    records_dir: Option<String>,
    /// The layer's pack `manifest.json` (npz_sha256 / sha256 per expert).
    records_manifest: Option<String>,
    records_manifest_sha256: Option<String>,
    /// R1 allocation JSON and its point (`ctl3-hot`): per-(layer, expert) W3 / MQ4-XTS map.
    alloc: Option<String>,
    alloc_point: String,
    /// Optional `ctl3-hot.map.json`, cross-checked against the allocation.
    map: Option<String>,
    /// MQ4-XTS source HFQM holding the aggregate expert tensors of the hot experts.
    xts: Option<String>,
    xts_sha256: Option<String>,
    layer0: bool,
    /// Include the whole-chain fixed-eight-row sweep, R1..64 by default.
    wide: bool,
    /// Run only the wide cases (no narrow synthetic / real cases).
    wide_only: bool,
    /// Run only the slab cohort (whole-chain minus the legacy indexed prefixes). Implies
    /// `--wide-only`; the complete whole-slab G1 sweep.
    wide_slab_only: bool,
    /// Lane counts of the wide synthetic sweep.
    wide_c: Vec<usize>,
}

const USAGE: &str = "usage: mq_i4_moe_gemv_byte_oracle --codeobject PATH [--artifact HFQM --layer N [--real-experts N]] \
[--records-dir DIR --records-manifest FILE --alloc FILE [--point NAME] --xts HFQ [--layer0|--layer N] [--map FILE] \
[--records-manifest-sha256 HEX] [--xts-sha256 HEX] [--real-experts N]] \
[--old-kernel-parity] [--timing --reps N(>=20) --warmup N(>=3) [--launches N] --card NAME --fabric-owner OWNER \
--fabric-wrapper fabric-run.sh [--fabric-card NAME]] [--device N] [--expect-uuid U] [--expect-bdf B] [--expect-arch A] \
[--wide|--wide-only|--wide-slab-only [--wide-c LIST]] [--only SUBSTR]...  (card/fabric flags fall back to env FABRIC_CARD / FABRIC_OWNER / FABRIC_WRAPPER; \
--wide adds whole-chain t8 cases; --wide-slab-only is the complete whole-slab G1 sweep (legacy excluded, still available via --wide/--wide-only); --wide-c lists/ranges R within 1..=64 (default 1-64))";

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn parse_args(argv: &[String]) -> Result<Cfg> {
    let mut cfg = Cfg {
        codeobject: String::new(),
        artifact: None,
        layer: 0,
        real_experts: 24,
        old_parity: false,
        timing: false,
        reps: 30,
        warmup: 5,
        launches: 32,
        device: None,
        expect_uuid: Some(DEFAULT_UUID.into()),
        expect_bdf: DEFAULT_BDF.into(),
        expect_arch: DEFAULT_ARCH.into(),
        card: env_nonempty("FABRIC_CARD"),
        fabric_card: None,
        fabric_owner: env_nonempty("FABRIC_OWNER"),
        fabric_wrapper: env_nonempty("FABRIC_WRAPPER"),
        only: Vec::new(),
        records_dir: None,
        records_manifest: None,
        records_manifest_sha256: None,
        alloc: None,
        alloc_point: "ctl3-hot".into(),
        map: None,
        xts: None,
        xts_sha256: None,
        layer0: false,
        wide: false,
        wide_only: false,
        wide_slab_only: false,
        wide_c: (1..=MAX_ROWS).collect(),
    };
    let mut i = 0;
    let mut layer_explicit: Option<usize> = None;
    let mut uuid_explicit = false;
    let mut wide_c_explicit = false;
    let value = |i: &mut usize| -> Result<String> {
        *i += 1;
        argv.get(*i)
            .cloned()
            .ok_or_else(|| format!("{} needs a value\n{USAGE}", argv[*i - 1]))
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
            "--artifact" => cfg.artifact = Some(value(&mut i)?),
            "--layer" => {
                let l = number(&mut i)?;
                layer_explicit = Some(l);
                cfg.layer = l;
            }
            "--real-experts" => cfg.real_experts = number(&mut i)?,
            "--old-kernel-parity" => cfg.old_parity = true,
            "--timing" => cfg.timing = true,
            "--reps" => cfg.reps = number(&mut i)?,
            "--warmup" => cfg.warmup = number(&mut i)?,
            "--launches" => cfg.launches = number(&mut i)?,
            "--device" => cfg.device = Some(number(&mut i)? as i32),
            "--expect-uuid" => {
                cfg.expect_uuid = Some(value(&mut i)?);
                uuid_explicit = true;
            }
            "--expect-bdf" => cfg.expect_bdf = value(&mut i)?.to_ascii_lowercase(),
            "--expect-arch" => cfg.expect_arch = value(&mut i)?,
            "--card" => cfg.card = Some(value(&mut i)?),
            "--fabric-card" => cfg.fabric_card = Some(value(&mut i)?),
            "--fabric-owner" => cfg.fabric_owner = Some(value(&mut i)?),
            "--fabric-wrapper" => cfg.fabric_wrapper = Some(value(&mut i)?),
            "--only" => cfg.only.push(value(&mut i)?),
            "--records-dir" => cfg.records_dir = Some(value(&mut i)?),
            "--records-manifest" => cfg.records_manifest = Some(value(&mut i)?),
            "--records-manifest-sha256" => {
                cfg.records_manifest_sha256 = Some(value(&mut i)?.to_ascii_lowercase())
            }
            "--alloc" => cfg.alloc = Some(value(&mut i)?),
            "--point" => cfg.alloc_point = value(&mut i)?,
            "--map" => cfg.map = Some(value(&mut i)?),
            "--xts" => cfg.xts = Some(value(&mut i)?),
            "--xts-sha256" => cfg.xts_sha256 = Some(value(&mut i)?.to_ascii_lowercase()),
            "--layer0" => cfg.layer0 = true,
            "--wide" => cfg.wide = true,
            "--wide-only" => {
                cfg.wide = true;
                cfg.wide_only = true;
            }
            "--wide-slab-only" => {
                cfg.wide = true;
                cfg.wide_only = true;
                cfg.wide_slab_only = true;
            }
            "--wide-c" => {
                cfg.wide_c = parse_c_list(&value(&mut i)?)?;
                wide_c_explicit = true;
            }
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
        i += 1;
    }
    if cfg.codeobject.is_empty() {
        return Err(USAGE.into());
    }
    // Halo has no authoritative ROCr UUID on every installation. Only the explicitly
    // selected Halo architecture may omit it; BDF and architecture remain mandatory.
    if cfg.expect_arch == "gfx1151" && !uuid_explicit {
        cfg.expect_uuid = None;
    }
    if cfg.expect_uuid.as_ref().is_some_and(|uuid| uuid.is_empty())
        || cfg.expect_bdf.is_empty()
        || cfg.expect_arch.is_empty()
    {
        return Err("device identity pins must not be empty".into());
    }
    if wide_c_explicit && !cfg.wide {
        return Err("--wide-c requires --wide, --wide-only or --wide-slab-only".into());
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
        if cfg.artifact.is_some() || cfg.old_parity {
            return Err("--timing is a separate run: no --artifact / --old-kernel-parity".into());
        }
        for (what, v) in [
            ("--card", &cfg.card),
            ("--fabric-owner", &cfg.fabric_owner),
            ("--fabric-wrapper", &cfg.fabric_wrapper),
        ] {
            if v.is_none() {
                return Err(format!("--timing requires {what} (or its env fallback): fabric metadata is part of the record"));
            }
        }
    }
    if cfg.layer0 {
        if layer_explicit.is_some_and(|l| l != 0) {
            return Err("--layer0 conflicts with --layer N (N != 0)".into());
        }
        cfg.layer = 0;
    }
    let records_flags = [
        ("--records-dir", cfg.records_dir.is_some()),
        ("--records-manifest", cfg.records_manifest.is_some()),
        ("--alloc", cfg.alloc.is_some()),
        ("--xts", cfg.xts.is_some()),
    ];
    let any_records = records_flags.iter().any(|(_, set)| *set);
    if any_records {
        if let Some((missing, _)) = records_flags.iter().find(|(_, set)| !*set) {
            return Err(format!("certified-records mode needs --records-dir, --records-manifest, --alloc and --xts together; {missing} is missing"));
        }
        if cfg.timing {
            return Err("--timing is a separate run: no certified-records flags".into());
        }
    } else if cfg.layer0
        || cfg.map.is_some()
        || cfg.xts_sha256.is_some()
        || cfg.records_manifest_sha256.is_some()
    {
        return Err("--layer0/--map/--xts-sha256/--records-manifest-sha256 only apply to certified-records mode".into());
    }
    if cfg.wide && cfg.timing {
        return Err("--wide is a correctness run: no --timing".into());
    }
    Ok(cfg)
}

/// `--wide-c` list: comma separated lane counts or inclusive `A-B` ranges within 1..=64.
fn parse_c_list(s: &str) -> Result<Vec<usize>> {
    let mut out = BTreeSet::new();
    for part in s.split(',') {
        let (lo, hi) = match part.split_once('-') {
            Some((a, b)) => (
                a.parse::<usize>()
                    .map_err(|e| format!("--wide-c {part}: {e}"))?,
                b.parse::<usize>()
                    .map_err(|e| format!("--wide-c {part}: {e}"))?,
            ),
            None => {
                let v = part
                    .parse::<usize>()
                    .map_err(|e| format!("--wide-c {part}: {e}"))?;
                (v, v)
            }
        };
        if lo < 1 || hi > 64 || lo > hi {
            return Err(format!(
                "--wide-c {part}: lane counts must satisfy 1 <= A <= B <= 64"
            ));
        }
        out.extend(lo..=hi);
    }
    if out.is_empty() {
        return Err("--wide-c needs at least one lane count".into());
    }
    Ok(out.into_iter().collect())
}

fn selected(cfg: &Cfg, name: &str) -> bool {
    cfg.only.is_empty() || cfg.only.iter().any(|s| name.contains(s.as_str()))
}

fn run() -> Result<(Value, bool)> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let cfg = parse_args(&argv)?;
    let hip = HipRuntime::load().map_err(err)?;
    if let Some(d) = cfg.device {
        hip.set_device(d).map_err(err)?;
    }
    let id = hip.current_device().map_err(err)?;
    let uuid = cfg
        .expect_uuid
        .as_ref()
        .map(|_| hip.device_uuid(id).map_err(err))
        .transpose()?;
    let bdf = hip.device_pci_bus_id(id).map_err(err)?.to_ascii_lowercase();
    let arch = hip.get_arch(id).map_err(err)?;
    let identity_match =
        uuid == cfg.expect_uuid && bdf == cfg.expect_bdf && arch == cfg.expect_arch;
    let device = json!({"index": id, "uuid": uuid, "bdf": bdf, "arch": arch, "card": cfg.card,
        "expected_uuid": cfg.expect_uuid, "expected_bdf": cfg.expect_bdf, "expected_arch": cfg.expect_arch,
        "uuid_required": cfg.expect_uuid.is_some(),
        "identity_contract": "actual HIP PCI BDF + architecture, plus UUID when pinned",
        "identity_match": identity_match});
    let command = json!({"argv": argv, "build_command": BUILD_COMMAND});
    if !identity_match {
        return Ok((
            json!({"schema": SCHEMA, "command": command, "device": device,
                "error": "MQ-I4 MoE GEMV device identity mismatch", "byte_exact": false, "pass": false, "rc": 1}),
            false,
        ));
    }
    let image = std::fs::read(&cfg.codeobject).map_err(|e| format!("{}: {e}", cfg.codeobject))?;
    let codeobject_sha = hex(&sha(&image));
    let device_object_sha256 = rdna_compute::lab_extract_device_bytes(&image, &arch)
        .map(|device| hex(&sha(device)))
        .ok_or_else(|| format!("explicit --codeobject has no decodable {arch} device image; refusing container-only identity fallback"))?;
    let module = hip.module_load_data(&image).map_err(err)?;
    // Admitted narrow symbols only: gfx1201/gfx1151 load the plain W3/MQ4 symbols spelled by
    // `slab_moe::MQ_I4_MOE_GEMV_SYMBOLS`; gfx1100 loads the MQ4-only suffixed symbols
    // (`MQ_I4_MOE_GEMV_GFX1100_MQ4_SYMBOLS`) and strict-refuses W3. E8P is slab-only and is
    // never loaded here. No fake Function fallback: a missing admitted symbol is an error.
    let narrow: Vec<(Codec, Kind, String)> = match arch.as_str() {
        "gfx1100" => Kind::ALL
            .iter()
            .flat_map(|&kind| {
                [Codec::Mq4].into_iter().map(move |codec| {
                    let name = match (codec, kind) {
                        (Codec::Mq4, Kind::GateUp) => {
                            "mq_i4_moe_gemv_mq4sym_r256_gate_up_pm_gfx1100".to_string()
                        }
                        (Codec::Mq4, Kind::Down) => {
                            "mq_i4_moe_gemv_mq4sym_r128_down_pm_gfx1100".to_string()
                        }
                        _ => unreachable!("gfx1100 admits only MQ4 narrow symbols"),
                    };
                    (codec, kind, name)
                })
            })
            .collect(),
        _ => Kind::ALL
            .iter()
            .flat_map(|&kind| {
                Codec::ALL
                    .iter()
                    .map(move |&codec| (codec, kind, symbol(codec, kind)))
            })
            .collect(),
    };
    if arch == "gfx1100" {
        for codec in [Codec::W3] {
            assert!(
                !qwen4_slab_codec_admitted(&arch, codec.route_byte()),
                "gfx1100 must not admit W3"
            );
        }
    }
    let mut funcs = Vec::new();
    let mut loads = Vec::new();
    for (codec, kind, name) in &narrow {
        let f = hip
            .module_get_function(&module, name)
            .map_err(|e| format!("admitted narrow symbol {name}: {e}"))?;
        loads.push(name.clone());
        funcs.push((*codec, *kind, f, name.clone()));
    }
    let mut ctx = Ctx {
        hip,
        funcs,
        codeobject_sha: codeobject_sha.clone(),
        device_object_sha256: device_object_sha256.clone(),
        device: id,
        old_parity: cfg.old_parity,
        gpu: None,
    };
    let mut totals = Totals::default();
    let mut rows: Vec<Value> = Vec::new();
    let mut runs: Vec<Value> = Vec::new();
    let mut artifact_meta = Value::Null;
    let mut mq4_violations = 0u64;
    let mut records_meta = Value::Null;
    let mut records_violations = 0u64;

    if cfg.timing {
        let tcfg = TimingCfg {
            reps: cfg.reps,
            warmup: cfg.warmup,
            launches: cfg.launches,
        };
        let card = cfg.card.clone().unwrap_or_default();
        let fabric = json!({"class": "timing", "card": cfg.fabric_card.clone().unwrap_or_else(|| card.clone()),
            "owner": cfg.fabric_owner, "wrapper": cfg.fabric_wrapper});
        for codec in Codec::ALL {
            // 10 distinct experts of one codec, C1: one launch covers `M * 10` rows per profile.
            let name = format!("timing_c1_all_{}", codec.label());
            let mut rng = Rng::new(name_seed(&name));
            let ids = pick_ids(&mut rng, 10, true);
            let experts = experts_for(&ids, |_, _| codec);
            let mut route = ids.clone();
            rng.shuffle(&mut route);
            let case = build_case(&name, "synthetic", 1, route, experts, false, false, 0);
            eprintln!("[timing] {name}");
            let (row, samples) = run_case(&mut ctx, &case, &mut totals, Some(&tcfg))?;
            rows.push(row);
            for (codec, kind, slots, unique, elapsed) in samples {
                runs.push(json!({
                    "codec": codec.timing_name(), "profile": kind.timing_profile(),
                    "symbol": ctx.loaded_symbol(codec, kind), "m": kind.m(), "k": kind.k(), "M": kind.m(), "K": kind.k(),
                    "row_count": kind.m() * 10, "c": 1, "slots": slots, "unique_experts": unique,
                    "elapsed_s": elapsed, "reps": tcfg.reps, "warmup": tcfg.warmup,
                    "launches_per_sample": tcfg.launches,
                    "symbol_codeobject": module_name(&arch), "code_object_sha": ctx.codeobject_sha,
                    "timing_source": "device", "event_timing": "hipEventElapsedTime over launches_per_sample launches",
                    "device": {"arch": arch, "card": card, "uuid": uuid, "bdf": bdf},
                    "fabric": fabric,
                    "verified_byte_exact_before_timing": true,
                }));
            }
        }
    } else {
        'correctness: {
            // Existing narrow diagnostics are unchanged on admitted arches. gfx1100 (MQ4-only
            // narrow image) strict-refuses narrow W3 cases unless wide-only: a narrow W3 load
            // there is a hard error, not a skip.
            if arch == "gfx1100" && !cfg.wide_only && !cfg.wide_slab_only {
                for name in SYNTH_CASES.iter() {
                    let case = build_synthetic(name);
                    if case.experts.values().any(|e| e.codec == Codec::W3) {
                        return Err(format!(
                        "narrow case {name} holds W3, which gfx1100 does not admit (use --wide-only/--wide-slab-only)"
                    ));
                    }
                }
            }
            for name in SYNTH_CASES
                .iter()
                .filter(|n| !cfg.wide_only && !cfg.wide_slab_only && selected(&cfg, n))
            {
                eprintln!("[g1] {name}");
                let case = build_synthetic(name);
                let (row, _) = run_case(&mut ctx, &case, &mut totals, None)?;
                rows.push(row);
            }
            if cfg.wide {
                // `--wide-slab-only` enumerates exactly the slab cohort (no legacy); the broad
                // `--wide`/`--wide-only` sweep keeps the legacy prefixes too.
                let sweep: Vec<String> = if cfg.wide_slab_only {
                    slab_only_case_names(&cfg.wide_c)
                } else {
                    wide_case_names(&cfg.wide_c)
                };
                for name in sweep.iter().filter(|n| selected(&cfg, n)) {
                    eprintln!("[g1] {name}");
                    let case = build_wide_synthetic(name);
                    let row = match run_case(&mut ctx, &case, &mut totals, None) {
                        Ok((row, _)) => row,
                        Err(error) => {
                            // Preserve the actual failing receipt, including planner/source-identity
                            // refusals, alongside earlier rows rather than discarding them via `?`.
                            let detail = serde_json::from_str::<Value>(&error)
                                .unwrap_or_else(|_| json!({"error": error}));
                            json!({"case": name, "origin": case.origin, "wide": true, "R": case.c,
                            "failure": detail, "pass": false, "rc": 1})
                        }
                    };
                    let failed = row.get("pass").and_then(Value::as_bool) == Some(false);
                    rows.push(row);
                    if failed {
                        break 'correctness;
                    }
                }
            }
            if let Some(path) = &cfg.artifact {
                eprintln!("[g1] loading layer {} of {path}", cfg.layer);
                let real = load_real(path, cfg.layer, cfg.real_experts)?;
                mq4_violations = real.mq4_sym_violations;
                artifact_meta = real.meta.clone();
                for case in real_cases(&real, "real", "real", cfg.wide)
                    .iter()
                    .filter(|c| selected(&cfg, &c.name) && (!cfg.wide_only || c.wide))
                {
                    eprintln!("[g1] {}", case.name);
                    let (row, _) = run_case(&mut ctx, case, &mut totals, None)?;
                    rows.push(row);
                }
            }
            if cfg.records_dir.is_some() {
                eprintln!("[g1] loading certified L{} records", cfg.layer);
                let real = load_records(&cfg)?;
                records_violations = real.mq4_sym_violations;
                records_meta = real.meta.clone();
                for case in real_cases(&real, RECORDS_ORIGIN, "real_records", cfg.wide)
                    .iter()
                    .filter(|c| selected(&cfg, &c.name) && (!cfg.wide_only || c.wide))
                {
                    eprintln!("[g1] {}", case.name);
                    let (row, _) = run_case(&mut ctx, case, &mut totals, None)?;
                    rows.push(row);
                }
            }
        }
    }

    let byte_exact = totals.mismatches == 0
        && totals.twin_mismatches == 0
        && totals.old_kernel_mismatches == 0
        && totals.wide_cpu_mismatches == 0
        && totals.wide_singleton_mismatches == 0
        && totals.wide_reference_mismatches == 0
        && totals.wide_guard_violations == 0
        && rows.iter().all(|row| {
            row.get("pass").and_then(Value::as_bool) == Some(true)
                || row.get("status").and_then(Value::as_str) == Some("NOT-APPLICABLE")
        })
        && !rows.is_empty()
        && (!cfg.wide || totals.wide_cases > 0)
        && (!cfg.old_parity || totals.old_kernel_cases > 0);
    // An invocation proves only its requested rows, and only after every admitted case at
    // those rows passes exactly once. Filters cannot turn missing Cartesian cells into proof.
    // The full-range flags remain separate: component success never substitutes for G1.
    let required_wide: Vec<String> = if cfg.wide_slab_only {
        slab_only_case_names(&cfg.wide_c)
    } else {
        wide_case_names(&cfg.wide_c)
    };
    // Pure name->codec lookup: no expert tensor is synthesized for the denominator.
    let admitted_required: Vec<&String> = required_wide
        .iter()
        .filter(|name| wide_case_admitted_on(&arch, name))
        .collect();
    let not_applicable: Vec<String> = rows
        .iter()
        .filter_map(|row| {
            (row.get("status").and_then(Value::as_str) == Some("NOT-APPLICABLE"))
                .then(|| row["case"].as_str().unwrap_or("").to_string())
        })
        .collect();
    let missing_cells: Vec<&str> = if cfg.wide {
        admitted_required
            .iter()
            .filter(|name| {
                !rows
                    .iter()
                    .any(|row| row["case"] == name.as_str() && row["pass"].as_bool() == Some(true))
            })
            .map(|s| s.as_str())
            .collect()
    } else {
        Vec::new()
    };
    let reached_wide = rows
        .iter()
        .filter(|row| row["origin"] == "synthetic_wide" && row["pass"].as_bool() == Some(true))
        .count();
    let requested_complete = cfg.wide
        && !admitted_required.is_empty()
        && missing_cells.is_empty()
        && reached_wide == admitted_required.len()
        && rows
            .iter()
            .filter(|row| row["origin"] == "synthetic_wide" && row["status"] != "NOT-APPLICABLE")
            .count()
            == admitted_required.len();
    let full_range = cfg.wide_c == (1..=MAX_ROWS).collect::<Vec<_>>();
    let complete_g1 = byte_exact && requested_complete && full_range && cfg.only.is_empty();
    let complete_slab_g1 = complete_g1
        && (cfg.wide_slab_only || {
            // The broad sweep also completes the slab cohort when every admitted slab cell
            // (NOT-APPLICABLE names excluded from the denominator) is present.
            slab_only_case_names(&(1..=64).collect::<Vec<_>>())
                .iter()
                .filter(|name| wide_case_admitted_on(&arch, name))
                .all(|name| rows.iter().any(|row| row["case"] == name.as_str()))
        });
    let pass = byte_exact && (!cfg.wide || requested_complete);
    let mut out = json!({
        "schema": SCHEMA,
        "mode": if cfg.timing { "timing" } else { "g1" },
        "module": module_name(&arch),
        "command": command,
        "config": {"layer": cfg.layer, "real_experts": cfg.real_experts, "old_kernel_parity": cfg.old_parity,
            "only": cfg.only, "timing": cfg.timing, "reps": cfg.reps, "warmup": cfg.warmup,
            "launches_per_sample": cfg.launches, "wide": cfg.wide, "wide_only": cfg.wide_only,
            "wide_slab_only": cfg.wide_slab_only, "wide_c": cfg.wide_c},
        "codeobject": {"path": cfg.codeobject, "sha256": codeobject_sha, "bytes": image.len(),
            "device_object_sha256": device_object_sha256,
            "device_object_note": "required per-arch raw device SHA via lab_extract_device_bytes; startup refuses when the helper has no decodable entry"},
        "code_object_sha": codeobject_sha,
        "device_object_sha256": device_object_sha256,
        "kernarg_bytes": KERNARG_BYTES,
        "symbols_loaded": loads,
        "symbols_admitted": {"gfx1201/gfx1151": "plain W3/MQ4", "gfx1100": "MQ4-only suffixed _pm_gfx1100; W3 refused"},
        "device": device,
        "case_counts": {"total": totals.cases, "synthetic": totals.synthetic, "real": totals.real,
            "real_records_layer": totals.real_records, "old_kernel_parity": totals.old_kernel_cases, "wide": totals.wide_cases,
            "wide_sweep_cells": required_wide.len(), "wide_admitted_cells": admitted_required.len(),
            "requested_wide_passed_cells": reached_wide,
            "wide_not_applicable": not_applicable.len(),
            "complete_slab_cells_per_R": 48, "complete_broad_cells_per_R": 56,
            "slab_only_cohort_total": 64 * 48, "broad_cohort_total": 64 * 56},
        "wide_symbol_launches": totals.wide_launches,
        "wide_cpu_mismatch_count": totals.wide_cpu_mismatches,
        "wide_vs_singleton_mismatch_count": totals.wide_singleton_mismatches,
        "wide_vs_independent_raw_singleton_mismatch_count": totals.wide_reference_mismatches,
        "wide_guard_violation_count": totals.wide_guard_violations,
        "symbol_launches": totals.launches,
        "cases": rows,
        "mismatch_count": totals.mismatches,
        "twin_selfcheck_mismatch_count": totals.twin_mismatches,
        "old_kernel_mismatch_count": totals.old_kernel_mismatches,
        "mq4_sym_header_violations_real": mq4_violations,
        "mq4_sym_header_violations_records": records_violations,
        "byte_exact": byte_exact,
        "whole_chain_g1_complete": complete_g1,
        "whole_slab_g1_complete": complete_slab_g1,
        "requested_rows": if cfg.wide { cfg.wide_c.clone() } else { Vec::new() },
        "requested_rows_complete": byte_exact && requested_complete,
        "full_range_requested": cfg.wide && full_range,
        "coverage_status": if complete_slab_g1 { "WHOLE_SLAB_G1_PASS" }
            else if pass && cfg.wide { "COMPONENT_PASS_GLOBAL_G1_INCOMPLETE" }
            else if cfg.wide { "INCOMPLETE" } else { "NARROW_COMPONENT" },
        "first_failed_case": rows.iter().find(|row| row["pass"].as_bool() == Some(false))
            .map(|row| row["case"].clone()),
        "missing_whole_chain_cells": missing_cells,
        "not_applicable_cells": not_applicable,
        "descriptor_scope": "internal descriptor view has no external guard allocation (false); unused 64-row capacity across eight physical descriptor tiles is poisoned and byte-compared, all active bytes compared, immutability through down checked by load/store compare. No production API or hot path added; no external-descriptor guard is proven or claimed",
        "cpu_claim": "host descriptor semantics and arithmetic transcription; not emitted PM instruction execution",
        "pass": pass,
        "rc": if pass { 0 } else { 1 },
    });
    if !artifact_meta.is_null() {
        out["artifact"] = artifact_meta;
    }
    if !records_meta.is_null() {
        out["real_records_layer"] = records_meta;
    }
    if cfg.timing {
        out["runs"] = Value::Array(runs);
    }
    Ok((out, pass))
}

fn main() {
    match run() {
        Ok((v, pass)) => {
            println!("{v}");
            if !pass {
                eprintln!("MQ-I4 MoE GEMV byte oracle FAILED");
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
//   cargo test -p hipfire-runtime --release --features lab,rdna-compute/lab --example mq_i4_moe_gemv_byte_oracle
// ---------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cyc(n: usize) -> Vec<u8> {
        (0..n).map(|j| (j % 16) as u8).collect()
    }

    #[test]
    fn sym_zero_is_exact_at_extremes() {
        assert_eq!(sym_zero_bits(0x6fff), 0xfbff); // -65504
        assert_eq!(sym_zero_bits(0x0001), 0x8008); // -2^-21
        assert_eq!(f16f(sym_zero_bits(0x3c00)), -8.0);
        assert_eq!(f16f(0x6fff), 8188.0);
    }

    /// Exactly representable data: every partial sum is an integer, so rounding cannot hide a
    /// header-half, nibble-order or lane-ownership error. Row 2's second half uses scale 2.
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
        // Each 16-code cycle sums to -8; 128 columns = 8 cycles = -64.
        assert_eq!(got, vec![-128.0, 0.0, -192.0]);
    }

    #[test]
    fn mq4_down_known_answer() {
        let (m, k) = (2usize, 640usize);
        let mut codes = cyc(m * k);
        codes[k..].fill(8);
        let scales = vec![0x3800u16; m * (k / 128)]; // 0.5, zp = -4
        let bytes = encode_mq4(Kind::Down, m, k, &codes, &scales);
        let x = vec![2.0f32; k];
        let rb = (k / 128) * 68;
        let got: Vec<f32> = (0..m)
            .map(|r| mq4_row_down(&bytes[r * rb..(r + 1) * rb], &x))
            .collect();
        assert_eq!(got, vec![-320.0, 0.0]); // 5 groups * 8 cycles * (-8)
    }

    /// The oracle takes the audited gfx1201 contraction ("element 1 first"); the opposite
    /// orientation must differ on full-mantissa data or a pass would prove nothing.
    #[test]
    fn mq4_orientation_is_element1_first_and_sensitive() {
        let k = 640usize;
        let mut rng = Rng::new(7);
        let mut codes = vec![0u8; 64 * k];
        for c in codes.iter_mut() {
            *c = (rng.u32() & 15) as u8;
        }
        let scales: Vec<u16> = (0..64 * 5)
            .map(|_| 0x2800 + (rng.u32() as u16 & 0x3ff))
            .collect();
        let bytes = encode_mq4(Kind::Down, 64, k, &codes, &scales);
        let x = gen_x_row(0, k, &mut rng);
        let rb = 5 * 68;
        let alt = |row: &[u8]| -> f32 {
            let mut lanes = [0.0f32; 32];
            for (tid, lane) in lanes.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for g in 0..5 {
                    let gp = &row[g * 68..];
                    let sc = f16f(u16::from_le_bytes([gp[0], gp[1]]));
                    let zp = f16f(u16::from_le_bytes([gp[2], gp[3]]));
                    let (b0, b1) = (gp[4 + tid * 2], gp[5 + tid * 2]);
                    let q = [b0 & 15, b0 >> 4, b1 & 15, b1 >> 4];
                    let d = |i: usize| sc.mul_add(q[i] as f32, zp);
                    let xb = g * 128 + tid * 4;
                    let mut s = d(0) * x[xb];
                    for i in 1..4 {
                        s = d(i).mul_add(x[xb + i], s);
                    }
                    acc += s;
                }
                *lane = acc;
            }
            lane_tree(lanes)
        };
        let differing = (0..64)
            .filter(|&r| {
                let row = &bytes[r * rb..(r + 1) * rb];
                mq4_row_down(row, &x).to_bits() != alt(row).to_bits()
            })
            .count();
        assert!(differing > 0, "fixture is not contraction-sensitive");
    }

    /// 2^24 + 1 + 1 + 1: the 16,8,4,2,1 tree folds lane1+lane3 first (1+1 = 2, exact) so the
    /// result is 2^24 + 2; a left-to-right sum rounds every +1 away and stays 2^24.
    #[test]
    fn lane_tree_order_is_the_shuffle_tree() {
        let mut lanes = [0.0f32; 32];
        lanes[0] = 16777216.0;
        lanes[1] = 1.0;
        lanes[2] = 1.0;
        lanes[3] = 1.0;
        // step 2: l0 = 2^24 + l2 (=1) rounds to 2^24 (ties-to-even); l1 = 1 + l3 = 2; step 1: 2^24 + 2.
        assert_eq!(lane_tree(lanes), 16777218.0);
    }

    fn check_partition(route: &[u32], codecs: &BTreeMap<u32, Codec>) {
        let f = |e: u32| codecs[&e];
        // Slab order: cold bucket first, MQ4 second. This mixed W3/MQ4 helper still uses
        // the historical two-bucket partition (cold W3 at 0, MQ4 at 1).
        let bs = [
            build_bucket(route, &f, Codec::W3),
            build_bucket(route, &f, Codec::Mq4),
        ];
        let mut seen = vec![false; route.len()];
        for (ci, b) in bs.iter().enumerate() {
            let want = [Codec::W3, Codec::Mq4][ci];
            assert_eq!(b.offs.len(), b.ids.len() + 1);
            assert_eq!(b.offs[0], 0);
            assert_eq!(*b.offs.last().unwrap() as usize, b.slots.len());
            assert!(
                b.ids.windows(2).all(|w| w[0] < w[1]),
                "ids ascending and unique"
            );
            for (u, &id) in b.ids.iter().enumerate() {
                let seg = &b.slots[b.offs[u] as usize..b.offs[u + 1] as usize];
                assert!(!seg.is_empty());
                assert!(
                    seg.windows(2).all(|w| w[0] < w[1]),
                    "stable original slot order"
                );
                for &s in seg {
                    assert_eq!(route[s as usize], id);
                    assert_eq!(codecs[&id], want);
                    assert!(!seen[s as usize], "slot appears once");
                    seen[s as usize] = true;
                }
            }
        }
        assert!(seen.iter().all(|&s| s), "buckets partition all 10C slots");
    }

    #[test]
    fn buckets_partition_slots_for_union_and_unique_routes() {
        let mut rng = Rng::new(11);
        let pool = pick_ids(&mut rng, 14, true);
        let codecs: BTreeMap<u32, Codec> = pool
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, MIXED10[i % 10]))
            .collect();
        check_partition(&lane_routes(&mut rng, &pool, 8), &codecs); // repeated across lanes
        let uniq = pick_ids(&mut rng, 80, true);
        let codecs: BTreeMap<u32, Codec> = uniq
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, MIXED10[(i * 7) % 10]))
            .collect();
        let mut route = uniq.clone();
        rng.shuffle(&mut route);
        check_partition(&route, &codecs); // 80 unique ids
    }

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
    fn synthetic_tensors_have_exact_sizes_and_valid_w3() {
        for kind in Kind::ALL {
            for codec in Codec::SLAB_ALL {
                assert_eq!(
                    gen_tensor(codec, kind, 5, true).len(),
                    kind.tensor_bytes(codec)
                );
            }
            mq_i4::decode_tensor(
                kind.w3_profile(),
                kind.m(),
                kind.k(),
                &gen_tensor(Codec::W3, kind, 5, true),
            )
            .unwrap();
        }
        assert_eq!(Kind::GateUp.tensor_bytes(Codec::E8p), 870_400);
        assert_eq!(Kind::Down.tensor_bytes(Codec::E8p), 435_200);
    }

    #[test]
    fn mq4_expert_headers_are_symmetric_and_finite() {
        for kind in Kind::ALL {
            let t = gen_tensor(Codec::Mq4, kind, 9, true);
            assert_eq!(mq4_sym_violations(kind, &t), 0);
        }
    }

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    const RECORDS: [&str; 8] = [
        "--records-dir",
        "/r/q/w3-gptq3/L0",
        "--records-manifest",
        "/r/m.json",
        "--alloc",
        "/r/a.json",
        "--xts",
        "/x.mq4",
    ];

    fn with(extra: &[&str]) -> Vec<String> {
        let mut v = vec!["--codeobject", "/co.hxaco"];
        v.extend_from_slice(&RECORDS);
        v.extend_from_slice(extra);
        args(&v)
    }

    #[test]
    fn records_flags_parse_and_default_to_layer0_point_ctl3_hot() {
        let c = parse_args(&with(&[
            "--layer0",
            "--map",
            "/r/map.json",
            "--xts-sha256",
            "AB12",
            "--records-manifest-sha256",
            "CD34",
        ]))
        .unwrap();
        assert_eq!(c.layer, 0);
        assert!(c.layer0);
        assert_eq!(c.alloc_point, "ctl3-hot");
        assert_eq!(c.records_dir.as_deref(), Some("/r/q/w3-gptq3/L0"));
        assert_eq!(c.xts_sha256.as_deref(), Some("ab12"));
        assert_eq!(c.records_manifest_sha256.as_deref(), Some("cd34"));
        assert_eq!(c.map.as_deref(), Some("/r/map.json"));
        assert_eq!(c.real_experts, 24);
        let c = parse_args(&with(&["--point", "other", "--layer", "0"])).unwrap();
        assert_eq!(c.alloc_point, "other");
    }

    #[test]
    fn records_flags_are_all_or_nothing_and_exclusive_of_timing() {
        let partial = args(&[
            "--codeobject",
            "/co",
            "--records-dir",
            "/d",
            "--alloc",
            "/a",
            "--xts",
            "/x",
        ]);
        assert!(parse_args(&partial)
            .unwrap_err()
            .contains("--records-manifest is missing"));
        assert!(parse_args(&args(&["--codeobject", "/co", "--layer0"])).is_err());
        assert!(parse_args(&with(&["--layer0", "--layer", "3"])).is_err());
        let timing = with(&[
            "--timing",
            "--card",
            "c",
            "--fabric-owner",
            "o",
            "--fabric-wrapper",
            "w",
        ]);
        assert!(parse_args(&timing).is_err());
        // The old flags are unchanged: --artifact alone still parses.
        assert!(parse_args(&args(&[
            "--codeobject",
            "/co",
            "--artifact",
            "/a.hfq",
            "--layer",
            "5"
        ]))
        .is_ok());
    }

    #[test]
    fn hot_map_labels_parse_strictly() {
        let mut row: Vec<Value> = (0..N_EXPERTS)
            .map(|e| json!(if e % 3 == 0 { "xts_mq4" } else { "w3" }))
            .collect();
        let root = json!({"arm": "ctl3-hot", "layers": {"0": row.clone()}});
        let f = hot_map_formats(&root, 0).unwrap();
        assert_eq!(f[0], ExpertFormat::Mq4Xts);
        assert_eq!(f[1], ExpertFormat::Mq3I4);
        assert!(hot_map_formats(&root, 1).is_err());
        row[7] = json!("mq4");
        assert!(hot_map_formats(&json!({"layers": {"0": row.clone()}}), 0).is_err());
        row.pop();
        assert!(hot_map_formats(&json!({"layers": {"0": row}}), 0).is_err());
    }

    /// The R1 allocation shape (`profile_indices[48][512]`, 0=MQ2 1=MQ3 2=MQ4-XTS) through S2's loader,
    /// cross-checked with a map row in the other vocabulary.
    #[test]
    fn allocation_layer_slice_matches_map_vocabulary() {
        let rows: Vec<Vec<u64>> = (0..48)
            .map(|l| {
                (0..N_EXPERTS)
                    .map(|e| if (e + l) % 3 == 0 { 2 } else { 1 })
                    .collect()
            })
            .collect();
        let root = json!({"schema": "ctl3-hot-allocation-v1", "profiles": ["mq2-i4", "mq3-i4", "mq4-xts"],
            "allocations": {"ctl3-hot": {"feasible": true, "profile_indices": rows}}});
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("alloc.json");
        std::fs::write(&p, root.to_string()).unwrap();
        let a = load_allocation(&p, "ctl3-hot", SlabGeometry::FN).unwrap();
        let l2 = &a.formats[2 * N_EXPERTS..3 * N_EXPERTS];
        let map_row: Vec<Value> = (0..N_EXPERTS)
            .map(|e| json!(if (e + 2) % 3 == 0 { "xts_mq4" } else { "w3" }))
            .collect();
        assert_eq!(
            l2,
            &hot_map_formats(&json!({"layers": {"2": map_row}}), 2).unwrap()[..]
        );
        assert!(load_allocation(&p, "nope", SlabGeometry::FN).is_err());
    }

    #[test]
    fn selection_is_balanced_and_tops_up_the_short_side() {
        assert_eq!(select_counts(100, 100, 24), (12, 12));
        assert_eq!(select_counts(100, 4, 24), (20, 4));
        assert_eq!(select_counts(3, 100, 24), (3, 21));
        assert_eq!(select_counts(100, 100, 2), (5, 5)); // a top-10 lane needs 10 experts
        let v: Vec<u32> = (0..100).collect();
        let s = spread(&v, 12);
        assert!(s.windows(2).all(|w| w[0] < w[1]) && s[0] == 0 && *s.last().unwrap() < 100);
    }

    #[test]
    fn wide_c_lists_parse_ranges_and_reject_bad_lanes() {
        assert_eq!(
            parse_c_list("1-3,9,16-17,32").unwrap(),
            vec![1, 2, 3, 9, 16, 17, 32]
        );
        assert_eq!(parse_c_list("32,1,1").unwrap(), vec![1, 32]);
        for bad in ["", "0", "65", "5-3", "a", "1-65", ",1", "1,", "1,,2", "1-"] {
            assert!(parse_c_list(bad).is_err(), "{bad}");
        }
        let c = parse_args(&args(&[
            "--codeobject",
            "/co",
            "--wide-only",
            "--wide-c",
            "9-10",
        ]))
        .unwrap();
        assert!((c.wide, c.wide_only, c.wide_c.clone()) == (true, true, vec![9, 10]));
        assert!(!c.wide_slab_only);
        let s = parse_args(&args(&["--codeobject", "/co", "--wide-slab-only"])).unwrap();
        assert!(s.wide && s.wide_only && s.wide_slab_only);
        assert_eq!(s.wide_c, (1..=64).collect::<Vec<_>>());
        assert!(parse_args(&args(&[
            "--codeobject",
            "/co",
            "--wide",
            "--timing",
            "--card",
            "c",
            "--fabric-owner",
            "o",
            "--fabric-wrapper",
            "w"
        ]))
        .is_err());
    }

    #[test]
    fn wide_cli_subsets_keep_cartesian_counts_and_require_a_wide_mode() {
        let c = parse_args(&args(&[
            "--codeobject",
            "/co",
            "--wide-slab-only",
            "--wide-c",
            "1",
        ]))
        .unwrap();
        assert_eq!(c.wide_c, vec![1]);
        assert_eq!(slab_only_case_names(&c.wide_c).len(), 48);
        assert_eq!(
            slab_only_case_names(&(1..=MAX_ROWS).collect::<Vec<_>>()).len(),
            3072
        );
        for tail in [
            vec!["--wide-c", "1"],
            vec!["--wide-only", "--wide-c"],
            vec!["--wide-only", "--wide-c", ""],
            vec!["--wide-only", "--wide-c", "0-1"],
            vec!["--wide-only", "--wide-c", "64-65"],
            vec!["--wide-only", "--unknown"],
        ] {
            let mut argv = args(&["--codeobject", "/co"]);
            argv.extend(args(&tail));
            assert!(parse_args(&argv).is_err(), "{tail:?}");
        }
    }

    #[test]
    fn halo_uuid_is_optional_only_without_an_explicit_uuid_pin() {
        let defaults = parse_args(&args(&["--codeobject", "/co"])).unwrap();
        assert_eq!(defaults.expect_uuid.as_deref(), Some(DEFAULT_UUID));
        assert_eq!(defaults.expect_bdf, DEFAULT_BDF);
        assert_eq!(defaults.expect_arch, DEFAULT_ARCH);
        let halo = parse_args(&args(&[
            "--codeobject",
            "/co",
            "--expect-arch",
            "gfx1151",
            "--expect-bdf",
            "0000:C0:00.0",
        ]))
        .unwrap();
        assert_eq!(halo.expect_uuid, None);
        assert_eq!(halo.expect_bdf, "0000:c0:00.0");
        assert_eq!(halo.expect_arch, "gfx1151");
        for arch in ["gfx1151", "gfx1100", "gfx1201"] {
            let pinned = parse_args(&args(&[
                "--codeobject",
                "/co",
                "--expect-arch",
                arch,
                "--expect-bdf",
                "0000:c0:00.0",
                "--expect-uuid",
                "GPU-authoritative",
            ]))
            .unwrap();
            assert_eq!(pinned.expect_uuid.as_deref(), Some("GPU-authoritative"));
            assert_eq!(pinned.expect_bdf, "0000:c0:00.0");
        }
        for flag in ["--expect-uuid", "--expect-bdf", "--expect-arch"] {
            assert!(parse_args(&args(&["--codeobject", "/co", flag, ""])).is_err());
        }
    }

    #[test]
    fn wide_case_names_cover_every_lane_count_style_seed_and_representation() {
        let all: Vec<usize> = (1..=64).collect();
        let names = wide_case_names(&all);
        // Eight valid prefixes (plain, second seed, e8p, e8p seed, purew3, purew3 seed,
        // puremq4, puremq4 seed), two legacy indexed, four invalid (w3-invalid and e8p-invalid
        // plus second seeds), four route styles each.
        assert_eq!(names.len(), 64 * (8 * 4 + 2 * 4 + 4 * 4));
        for c in all {
            for prefix in [
                "",
                "seed1_",
                "e8p_",
                "e8p_seed1_",
                "purew3_",
                "purew3_seed1_",
                "puremq4_",
                "puremq4_seed1_",
                "legacy_",
                "legacy_seed1_",
            ] {
                for style in ["pool", "ident", "disjoint", "ties"] {
                    assert!(names.contains(&format!("w{c}_{prefix}{style}")));
                }
            }
            for prefix in [
                "invalid_",
                "invalid_seed1_",
                "e8p_invalid_",
                "e8p_invalid_seed1_",
            ] {
                for style in ["pool", "ident", "disjoint", "ties"] {
                    assert!(names.contains(&format!("w{c}_{prefix}{style}")));
                }
            }
        }
    }

    /// Disjoint means each eight-row union, not 640 distinct experts in a 512-expert model.
    #[test]
    fn wide_routes_are_valid_lane_unions() {
        for c in 1..=64 {
            for seed in [0, 1] {
                for style in ["pool", "ident", "disjoint", "ties"] {
                    let mut rng = Rng::new(name_seed(&format!("w{c}_{seed}_{style}")) ^ 0x71de);
                    let (ids, route) = wide_plan(&mut rng, c, style);
                    assert_eq!(route.len(), 10 * c);
                    assert!(ids.len() <= 512 && ids.iter().all(|&id| id < 512));
                    for lane in route.chunks_exact(10) {
                        assert_eq!(lane.iter().copied().collect::<BTreeSet<_>>().len(), 10);
                        assert!(lane.iter().all(|e| ids.contains(e)));
                    }
                    // Bucket-1-heavy membership bytes (even ids byte 0, odd ids byte 1).
                    let membership: Vec<u8> = (0..512).map(|id| u8::from(id % 2 == 0)).collect();
                    let signed: Vec<i32> = route.iter().map(|&id| id as i32).collect();
                    let descriptors =
                        cpu_rowtile_descriptors(&signed, &membership, c, None).unwrap();
                    assert_descriptor_semantics(&signed, &membership, c, None, &descriptors);
                    for tile in route.chunks(80) {
                        let bucket = build_bucket(tile, &|_| Codec::W3, Codec::W3);
                        let counts: Vec<u32> =
                            bucket.offs.windows(2).map(|w| w[1] - w[0]).collect();
                        assert!(counts.iter().all(|&n| n <= 8));
                        match style {
                            "ident" | "ties" => {
                                assert!(counts.iter().all(|&n| n as usize == tile.len() / 10))
                            }
                            "disjoint" => assert_eq!(
                                (bucket.ids.len(), counts.iter().max().copied()),
                                (tile.len(), Some(1))
                            ),
                            _ => {}
                        }
                    }
                    if c == 64 && style == "disjoint" {
                        assert_eq!((route.len(), ids.len()), (640, 80));
                        assert_eq!(&route[..80], &route[560..640]);
                    }
                }
            }
        }
    }

    /// Independently group by map then sort by first occurrence, checking every field/tail.
    fn assert_descriptor_semantics(
        route: &[i32],
        codec: &[u8],
        rows: usize,
        uniform: Option<usize>,
        words: &[u32],
    ) {
        assert_eq!(words.len(), rows.div_ceil(8) * 482);
        let mut seen = vec![false; rows * 10];
        for (t, descriptor) in words.chunks_exact(482).enumerate() {
            let lo = t * 80;
            let hi = (lo + 80).min(route.len());
            for bucket in 0..2 {
                let mut groups: BTreeMap<i32, Vec<u32>> = BTreeMap::new();
                for s in lo..hi {
                    let id = route[s];
                    if !(0..512).contains(&id) {
                        continue;
                    }
                    // Exact production law: byte 1 is bucket 0, every other byte bucket 1.
                    let b = uniform.unwrap_or_else(|| if codec[id as usize] == 1 { 0 } else { 1 });
                    if b == bucket {
                        groups.entry(id).or_default().push(s as u32);
                    }
                }
                let mut groups: Vec<_> = groups.into_iter().collect();
                groups.sort_by_key(|(_, slots)| slots[0]);
                let ids = &descriptor[bucket * 80..(bucket + 1) * 80];
                let offsets = &descriptor[160 + bucket * 81..160 + (bucket + 1) * 81];
                let slots = &descriptor[322 + bucket * 80..322 + (bucket + 1) * 80];
                let mut cursor = 0;
                for (e, (id, expected_slots)) in groups.iter().enumerate() {
                    assert_eq!(ids[e], *id as u32);
                    assert_eq!(offsets[e] as usize, cursor);
                    assert!(expected_slots.len() <= 8);
                    assert_eq!(
                        &slots[cursor..cursor + expected_slots.len()],
                        expected_slots
                    );
                    for &slot in expected_slots {
                        assert!(!seen[slot as usize]);
                        seen[slot as usize] = true;
                    }
                    cursor += expected_slots.len();
                    assert_eq!(offsets[e + 1] as usize, cursor);
                }
                assert!(ids[groups.len()..].iter().all(|&id| id == 0));
                assert!(offsets[groups.len()..]
                    .iter()
                    .all(|&o| o as usize == cursor));
                assert!(slots[cursor..].iter().all(|&s| s == 0));
            }
        }
        for (s, &id) in route.iter().enumerate() {
            assert_eq!(seen[s], (0..512).contains(&id), "absolute slot {s}");
        }
    }

    #[test]
    fn legacy_aggregate_names_keep_native_mtp_out_of_target_layer_namespace() {
        assert_eq!(
            legacy_aggregate_name(0, ExpertProj::GateUp),
            "model.language_model.layers.0.mlp.experts.gate_up_proj"
        );
        assert_eq!(
            legacy_aggregate_name(47, ExpertProj::Down),
            "model.language_model.layers.47.mlp.experts.down_proj"
        );
        assert_eq!(
            legacy_aggregate_name(48, ExpertProj::GateUp),
            "mtp.layers.0.mlp.experts.gate_up_proj"
        );
        assert_eq!(
            legacy_aggregate_name(48, ExpertProj::Down),
            "mtp.layers.0.mlp.experts.down_proj"
        );
    }

    #[test]
    fn cpu_pm_planner_semantics_all_rows_invalid_ids_uniform_and_membership() {
        // Exact production law: bucket 0 is byte 1; 0/2/255 are bucket 1 (the `!= 1` law,
        // including arbitrary non-admitted membership bytes, which behave as MQ4).
        let codec: Vec<u8> = (0..512).map(|id| [1, 0, 2, 255][id % 4]).collect();
        for rows in 1..=64 {
            for style in ["pool", "ident", "disjoint", "ties"] {
                let (_, route) = wide_plan(&mut Rng::new((rows * 19) as u64), rows, style);
                for invalid in [false, true] {
                    let route: Vec<i32> = route
                        .iter()
                        .enumerate()
                        .map(|(s, &id)| {
                            if invalid && s % 13 == 0 {
                                [-1, 512, i32::MIN, i32::MAX][(s / 13) % 4]
                            } else {
                                id as i32
                            }
                        })
                        .collect();
                    for uniform in [None, Some(0), Some(1)] {
                        let words = cpu_rowtile_descriptors(&route, &codec, rows, uniform).unwrap();
                        assert_descriptor_semantics(&route, &codec, rows, uniform, &words);
                    }
                }
            }
        }
        let route = [511, 1, 17, 2, 31, 3, 4, 5, 6, 7];
        let words = cpu_rowtile_descriptors(&route, &vec![1; 512], 1, None).unwrap();
        assert_eq!(
            &words[..10],
            &route.map(|id| id as u32),
            "first occurrence, NOT sorted IDs"
        );
        assert_eq!(&words[322..332], &(0..10).collect::<Vec<_>>());
        assert_eq!(words[240], 10, "bucket0 terminal offset");
        assert_eq!(words[321], 0, "bucket1 terminal offset");
        for rows in [0, 65, usize::MAX] {
            assert!(cpu_rowtile_descriptors(&[], &codec, rows, None).is_err());
        }
        assert!(cpu_rowtile_descriptors(&[0; 9], &codec, 1, None).is_err());
        assert!(cpu_rowtile_descriptors(&[0; 10], &codec[..511], 1, None).is_err());
        assert!(cpu_rowtile_descriptors(&[0; 10], &codec, 1, Some(2)).is_err());
    }

    #[test]
    #[should_panic(expected = "cannot pick")]
    fn impossible_distinct_expert_request_refuses_instead_of_looping() {
        pick_ids(&mut Rng::new(1), 640, true);
    }

    /// The padded production descriptor: live experts first, `n == 0` tail experts up to `S = 10*C`, monotone offsets
    /// ending at the bucket's slot count, slots zero filled to `S`.
    #[test]
    fn padded_descriptors_keep_live_prefix_and_empty_tail() {
        let route: Vec<u32> = (0..20).map(|i| 100 + (i % 12) as u32).collect();
        let b = build_bucket(&route, &|_| Codec::Mq4, Codec::Mq4);
        let (ids, offs, slots) = pad_descriptor(&b, 20);
        assert_eq!((ids.len(), offs.len(), slots.len()), (20, 21, 20));
        assert_eq!(&ids[..b.ids.len()], &b.ids[..]);
        assert_eq!(&offs[..b.offs.len()], &b.offs[..]);
        let total = *b.offs.last().unwrap();
        assert!(
            offs[b.ids.len()..].iter().all(|&o| o == total),
            "empty tail experts"
        );
        assert!(offs.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(&slots[..b.slots.len()], &b.slots[..]);
        assert!(slots[b.slots.len()..].iter().all(|&s| s == 0));
    }

    #[test]
    fn guard_bands_are_checked_around_the_payload() {
        let mut buf = vec![0xa5u8; GUARD_WORDS * 8 + 12];
        assert_eq!(guard_violations(&buf, 3), 0);
        buf[3] = 0;
        buf[GUARD_WORDS * 4 + 12 + 5] = 1;
        assert_eq!(guard_violations(&buf, 3), 2);
        buf[GUARD_WORDS * 4 + 4] = 9; // payload bytes are not guards
        assert_eq!(guard_violations(&buf, 3), 2);
    }
    /// E8P planar wire layout: `[words M*K/8 u16 LE][scales M*K/128 F16 LE]`, writer-guarded
    /// scales, exact synthetic sizes on the frozen slab shapes (gate/up 870400 B, down 435200 B).
    #[test]
    fn e8p_wire_layout_sizes_and_writer_guard() {
        for kind in Kind::ALL {
            let t = gen_tensor(Codec::E8p, kind, 7, true);
            assert_eq!(t.len(), kind.tensor_bytes(Codec::E8p));
            let (m, k) = (kind.m(), kind.k());
            assert_eq!(t.len(), m * k / 4 + 2 * m * k / 128);
            let (nw, ns) = (m * k / 8, m * k / 128);
            assert_eq!(nw * 2 + ns * 2, t.len());
            for c in t[nw * 2..].chunks_exact(2) {
                let d = u16::from_le_bytes([c[0], c[1]]);
                assert!(e8p::writer_guard(d).is_ok(), "scale {d:#06x} rejected");
            }
            for c in t[..nw * 2].chunks_exact(2).take(64) {
                let w = u16::from_le_bytes([c[0], c[1]]);
                let (ix, s7, sh) = e8p::unpack_word_fields(w);
                assert_eq!(e8p::pack_word(ix, s7, sh), w);
            }
        }
        assert_eq!(Kind::GateUp.tensor_bytes(Codec::E8p), 870_400);
        assert_eq!(Kind::Down.tensor_bytes(Codec::E8p), 435_200);
    }

    /// The E8P CPU twin is deterministic on synthetic wire and checks every slot.
    #[test]
    fn e8p_cpu_twin_is_deterministic() {
        let mut rng = Rng::new(0xe8b1);
        let ids = pick_ids(&mut rng, 10, true);
        let experts = experts_for(&ids, |_, _| Codec::E8p);
        let mut route = ids.clone();
        rng.shuffle(&mut route);
        let case = build_case(
            "e8p_twin_probe",
            "synthetic",
            1,
            route,
            experts,
            false,
            false,
            0,
        );
        let buckets = buckets_of(&case);
        for kind in Kind::ALL {
            let got = cpu_expected(&case, kind, &buckets);
            let again = cpu_expected(&case, kind, &buckets);
            assert_eq!(got.y0, again.y0);
            assert_eq!(got.y1, again.y1);
            assert!(got.y0.iter().all(|v| v.is_finite()));
            assert!(got.twin_slots_checked > 0);
        }
    }

    /// Cold-codec mapping: MQ4-only layers carry cold codec 0 (no cold bucket), one cold codec
    /// maps to its route byte (1 W3, 3 E8P), two cold codecs refuse; W2 has no `Codec`
    /// variant and cannot appear. Membership is cold = 1, MQ4 = 0.
    #[test]
    fn case_cold_codec_mapping_is_single_cold() {
        let mk = |codec: fn(usize, u32) -> Codec| {
            let mut rng = Rng::new(0xc01d);
            let ids = pick_ids(&mut rng, 10, true);
            let experts = experts_for(&ids, codec);
            let mut route = ids.clone();
            rng.shuffle(&mut route);
            build_case(
                "cold_probe",
                "synthetic",
                1,
                route,
                experts,
                false,
                false,
                0,
            )
        };
        assert_eq!(case_cold_codec(&mk(|_, _| Codec::Mq4)).unwrap(), 0);
        assert_eq!(
            case_cold_codec(&mk(|i, _| if i < 5 { Codec::W3 } else { Codec::Mq4 })).unwrap(),
            1
        );
        assert_eq!(
            case_cold_codec(&mk(|i, _| if i < 5 { Codec::E8p } else { Codec::Mq4 })).unwrap(),
            3
        );
        assert_eq!(case_cold_codec(&mk(|_, _| Codec::E8p)).unwrap(), 3);
        assert_eq!(case_cold_codec(&mk(|_, _| Codec::W3)).unwrap(), 1);
        assert!(case_cold_codec(&mk(|i, _| if i < 5 { Codec::W3 } else { Codec::E8p })).is_err());
        assert_eq!(membership_of(Codec::W3, 1), 1);
        assert_eq!(membership_of(Codec::Mq4, 1), 0);
        assert_eq!(membership_of(Codec::E8p, 3), 1);
        assert_eq!(membership_of(Codec::Mq4, 3), 0);
        assert_eq!(membership_of(Codec::W3, 0), 0);
        assert_eq!(membership_of(Codec::E8p, 0), 0);
    }

    /// Source admission is authoritative: per-arch MQ4/W3/E8P matrix, W2 refused everywhere,
    /// W3 refused on gfx1100, one cold codec per layer. C1 serves cold 0/1 only (cold 3 is K1).
    #[test]
    fn source_admission_matrix_and_single_cold() {
        for (arch, mq4, w3, e8p) in [
            ("gfx1201", true, true, true),
            ("gfx1100", true, false, true),
            ("gfx1151", true, true, true),
        ] {
            assert_eq!(qwen4_slab_codec_admitted(arch, 0), mq4, "{arch} MQ4");
            assert_eq!(qwen4_slab_codec_admitted(arch, 1), w3, "{arch} W3");
            assert_eq!(qwen4_slab_codec_admitted(arch, 3), e8p, "{arch} E8P");
            assert!(!qwen4_slab_codec_admitted(arch, 2), "{arch} W2 refused");
        }
        assert!(qwen4_slab_layer_cold_codec("gfx1201", &[0, 1, 3]).is_err());
        assert_eq!(qwen4_slab_layer_cold_codec("gfx1201", &[0, 1, 1]), Ok(1));
        assert_eq!(qwen4_slab_layer_cold_codec("gfx1201", &[3, 0]), Ok(3));
        assert_eq!(qwen4_slab_layer_cold_codec("gfx1100", &[0, 0]), Ok(0));
        assert_eq!(qwen4_slab_layer_cold_codec("gfx1100", &[0, 3]), Ok(3));
        assert!(qwen4_slab_layer_cold_codec("gfx1100", &[0, 1]).is_err());
        assert!(qwen4_slab_layer_cold_codec("gfx1201", &[0, 2]).is_err());
        assert!(slab_moe::k1_rows8_cold_codec(0) && slab_moe::k1_rows8_cold_codec(1));
        assert!(!slab_moe::k1_rows8_cold_codec(3));
    }

    /// Invalid-ID frozen inputs: tile-boundary slots 79/80/159/160/319/320/639 when inside
    /// `10*R` plus the ragged tail, exact `-1/512/MIN/MAX` encoding, valid slots untouched.
    /// Both admitted cold codecs are covered (`invalid_` W3-cold, `e8p_invalid_` E8P-cold,
    /// including pure-E8P-invalid disjoint); pure prefixes run pure buckets on every style.
    #[test]
    fn invalid_positions_cover_tile_boundaries_and_tail() {
        const BAD: [i32; 4] = [-1, 512, i32::MIN, i32::MAX];
        for rows in 1..=64 {
            let pos = invalid_positions(rows);
            assert!(!pos.is_empty() && pos.iter().all(|&s| s < 10 * rows));
            assert!(pos.contains(&(10 * rows - 1)), "ragged tail R={rows}");
            for s in [79usize, 80, 159, 160, 319, 320, 639] {
                assert_eq!(pos.contains(&s), s < 10 * rows, "R={rows} slot {s}");
            }
            let route: Vec<u32> = (0..10 * rows as u32).map(|i| i % 512).collect();
            let frozen = invalid_topk_for(&route, rows);
            assert_eq!(frozen.len(), 10 * rows);
            for (s, &v) in frozen.iter().enumerate() {
                if let Some(k) = pos.iter().position(|&q| q == s) {
                    assert_eq!(v, BAD[k % BAD.len()], "R={rows} slot {s}");
                } else {
                    assert_eq!(v, route[s] as i32, "R={rows} valid slot {s} untouched");
                }
            }
        }
        assert!(invalid_positions(64).contains(&639));
        assert!(invalid_positions(8).contains(&79));
        assert!(!invalid_positions(8).contains(&80));
    }

    /// E8P buckets partition mixed routes exactly like the narrow buckets (ascending ids,
    /// stable original slot order, every slot in exactly one bucket).
    #[test]
    fn e8p_buckets_partition_mixed_routes() {
        let mut rng = Rng::new(0xe8b);
        let pool = pick_ids(&mut rng, 14, true);
        let codecs: BTreeMap<u32, Codec> = pool
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, if i % 2 == 0 { Codec::E8p } else { Codec::Mq4 }))
            .collect();
        let f = |e: u32| codecs[&e];
        let route = lane_routes(&mut rng, &pool, 8);
        let mut seen = vec![false; route.len()];
        for codec in [Codec::E8p, Codec::Mq4] {
            let b = build_bucket(&route, &f, codec);
            assert_eq!(b.offs.len(), b.ids.len() + 1);
            assert!(b.ids.windows(2).all(|w| w[0] < w[1]));
            for (u, &id) in b.ids.iter().enumerate() {
                let seg = &b.slots[b.offs[u] as usize..b.offs[u + 1] as usize];
                assert!(!seg.is_empty() && seg.windows(2).all(|w| w[0] < w[1]));
                for &s in seg {
                    assert_eq!(route[s as usize], id);
                    assert!(!seen[s as usize]);
                    seen[s as usize] = true;
                }
            }
        }
        assert!(seen.iter().all(|&s| s));
    }

    /// Whole-slab selector counts: eight valid prefixes plus four invalid-ID prefixes (w3-invalid
    /// and e8p-invalid, incl. pure-E8P-invalid disjoint), four route styles each, per R; legacy
    /// indexed is excluded from the slab-only cohort.
    #[test]
    fn slab_only_cohort_counts_exclude_legacy() {
        let all = wide_case_names(&(1..=64).collect::<Vec<_>>());
        assert_eq!(all.len(), 64 * (8 * 4 + 2 * 4 + 4 * 4));
        let (mut plain, mut e8p, mut purew3, mut puremq4, mut invalid, mut legacy) =
            (0, 0, 0, 0, 0, 0);
        for n in &all {
            if n.contains("_legacy_") {
                legacy += 1;
            } else if n.contains("e8p_invalid_") {
                invalid += 1;
            } else if n.contains("_invalid_") {
                invalid += 1;
            } else if n.contains("purew3_") {
                purew3 += 1;
            } else if n.contains("puremq4_") {
                puremq4 += 1;
            } else if n.contains("_e8p_") {
                e8p += 1;
            } else {
                plain += 1;
            }
        }
        assert_eq!(
            (plain, e8p, purew3, puremq4, invalid, legacy),
            (512, 512, 512, 512, 1024, 512)
        );
        let slab: Vec<_> = all.iter().filter(|n| !n.contains("_legacy_")).collect();
        assert_eq!(slab.len(), 64 * 48);
    }
    /// Name->codec applicability matches the built case metadata on representative names, with
    /// no tensor synthesis in the helper path.
    #[test]
    fn wide_case_name_applicability_matches_built_cases() {
        for name in [
            "w1_pool",
            "w1_seed1_ties",
            "w8_e8p_pool",
            "w8_e8p_disjoint",
            "w8_purew3_ident",
            "w8_puremq4_ties",
            "w1_legacy_pool",
            "w1_invalid_pool",
            "w1_e8p_invalid_pool",
            "w8_e8p_invalid_disjoint",
        ] {
            let case = build_wide_synthetic(name);
            let mut bytes: Vec<u8> = case
                .experts
                .values()
                .map(|e| e.codec.route_byte())
                .collect();
            bytes.sort_unstable();
            bytes.dedup();
            let mut want = wide_case_route_bytes(name).to_vec();
            want.sort_unstable();
            want.dedup();
            assert_eq!(bytes, want, "{name}");
            for arch in ["gfx1201", "gfx1100", "gfx1151"] {
                assert_eq!(
                    wide_case_admitted_on(arch, name),
                    want.iter().all(|&b| qwen4_slab_codec_admitted(arch, b)),
                    "{arch} {name}"
                );
            }
        }
        assert!(!wide_case_admitted_on("gfx1100", "w1_pool"));
        assert!(wide_case_admitted_on("gfx1201", "w1_pool"));
        assert!(wide_case_admitted_on("gfx1100", "w1_puremq4_pool"));
        assert!(wide_case_admitted_on("gfx1100", "w1_e8p_pool"));
        assert!(!wide_case_admitted_on("gfx1100", "w1_invalid_pool"));
        assert!(wide_case_admitted_on("gfx1100", "w1_e8p_invalid_disjoint"));
    }
    /// Receipt resource formulas: chain capacity words/bytes per arm, descriptor extents,
    /// per-expert weight bytes, input/router extents.
    #[test]
    fn receipt_resource_formulas_match_cited_extents() {
        assert_eq!(64 * (3 * 10 * 640 + 10 * 2560 + 2560), 3_031_040);
        assert_eq!(3_031_040 * 4 + 5 * 2 * 64 * 4, 12_126_720);
        assert_eq!(2 * 12_126_720, 24_253_440);
        assert_eq!(8 * 482 * 4, 15_424);
        assert_eq!(4 * (640 + 2 * 64) * 4, 12_288);
        assert_eq!(
            Kind::GateUp.tensor_bytes(Codec::Mq4) + Kind::Down.tensor_bytes(Codec::Mq4),
            1_740_800 + 870_400
        );
        assert_eq!(
            Kind::GateUp.tensor_bytes(Codec::E8p) + Kind::Down.tensor_bytes(Codec::E8p),
            870_400 + 435_200
        );
        assert_eq!(2 * 2 * 64 * 4, 1_024);
    }
}
