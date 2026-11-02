// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Strict gfx1201 lane-row gate, carried from bi-land's H0 harness.
//!
//! Required: --out DIR --singleton-pins reviewed.json. Optional: --family
//! mq6|bf16|all, --mode bytes|timing|both, repeated --seed N, --iters N,
//! --cold-mib N, --budget-s N, --sweep-c, repeated --captured MANIFEST.json,
//! --captured-only (requires at least one --captured; omits seeded cases).
//! Defaults: bytes, two seeds.
//! Pins are a reviewed JSON object {"runtime_singleton_symbol":"sha256"}.
//!
//! Every candidate total width 1..=64 uses a direct frozen t8 launch: gfx1201
//! MQ6 S2, ordinary BF16 S1, or HC k4. Each output byte must match the actual
//! runtime singleton on that input row AND independent PM t8 total-R=1 calls.
//! PM C1 is an additional hardware comparison, never the runtime reference.
//! Inactive input rows, inactive output rows and head/tail guards are poisoned;
//! full input/weight allocations are checked for mutation after every byte arm.
//! All x4 live/empty masks, differing heights, output tails, arbitrary BF16 K,
//! quant-group tails, raw/activated HC and HC exceptional inputs are included.
//! This executable makes no CPU-emulator or universal exceptional-domain proof.
//!
//! Real-capture arm (`--captured MANIFEST.json`, repeatable, bytes mode only):
//! each manifest is a SHA-pinned `qwen4-rowtile-operands-v1` (arch gfx1201,
//! rows 64) with `provenance.source` pinning its `qwen4-rowtile-source-v1`
//! source-index (`capture_mode="eager-liveprojection-operands-only"`,
//! `retained_identity_claim=false`, 64 sequential row recipes with stable
//! singleton occurrence/grid/block/shared_mem, four immutable weight bindings,
//! model/prompt pins). Weights and X (64-row LE f32) are SHA-pinned with exact
//! extents before upload; manifests bind name/family/form/ms/k/rows/weights/x/
//! hc_act/scale to the source row. Admitted here: (`mq6`,`single`/`x4`),
//! (`bf16`,`single` ordinary scalar reference / one-slot native PM,
//! `x4` S1 ordinary incl lone, `k4` raw/activated HC with finite
//! actual scale binding hc_act==scale.is_some()). Seeded gates stay separate.
//! Full 1..64 real-capture reader arms reuse the strict seeded byte path with
//! actual singleton pins; missing/changed captures are refusals, never skips.
//!
//! --incumbent-only is explicitly NONCANDIDATE diagnostics (also permits q8).
//! There is no gfx1201 Q8 PM alias to Halo. Q8 captures are refused here (the
//! Q8 reader owns them). Incumbent rows are tested only at
//! 2..=8 and their differences are labeled diagnostic, not candidate passes.
//! Timing is a separate diagnostic arm, never a full-window performance claim.
//! Writes h0.json and H0.md; exit 3 on candidate differences/launch failures,
//! 2 on refusal/reference/setup errors. Timing-only and incumbent-only reports
//! explicitly cannot qualify row exactness. No absent device/image/pin is PASS.

use hip_bridge::{DeviceBuffer, Event, KernargBlob};
use rdna_compute::{kernels, DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::time::Instant;

type Result<T, E = String> = std::result::Result<T, E>;
const MAXC: usize = 64;
const PAD_ROWS: usize = 8;
const GUARD: usize = 64;
const SENTINEL: u8 = 0xff;
const HC_SCALE: f32 = 0.25;
const WARMUP: usize = 8;
const AMPS: [f32; 8] = [1.0, 1.0 / 1024.0, 1.0 / 16.0, 4.0, 64.0, 0.5, 0.0, 1024.0];

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}
fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        Self((z ^ (z >> 31)) | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn fill(&mut self, out: &mut [u8]) {
        let (chunks, rest) = out.as_chunks_mut::<8>();
        for c in chunks {
            *c = self.next().to_le_bytes();
        }
        if !rest.is_empty() {
            let tail = self.next().to_le_bytes();
            rest.copy_from_slice(&tail[..rest.len()]);
        }
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
    /// A finite f16: mostly `2^-lo..2^hi`, sometimes `+-0` or subnormal.
    fn f16(&mut self, lo_exp: u16, hi_exp: u16) -> u16 {
        let sign = (self.below(2) as u16) << 15;
        match self.below(32) {
            0 => sign,
            1 => sign | (1 + self.below(0x3ff) as u16),
            _ => {
                sign | ((15 - lo_exp + self.below(u64::from(lo_exp + hi_exp + 1)) as u16) << 10)
                    | self.below(0x400) as u16
            }
        }
    }
}

/// MQ6G256V2: `[m][k/256]` groups of 8 header bytes (`s0 z0 s1 z1`, f16) and 192 payload bytes.
fn gen_mq6(m: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let mut bytes = vec![0u8; m * (k / 256) * 200];
    for g in bytes.as_chunks_mut::<200>().0.iter_mut() {
        rng.fill(&mut g[8..]);
        for half in 0..2 {
            let (s, z) = (rng.f16(9, 2), rng.f16(7, 5));
            g[4 * half..4 * half + 2].copy_from_slice(&s.to_le_bytes());
            g[4 * half + 2..4 * half + 4].copy_from_slice(&z.to_le_bytes());
        }
    }
    bytes
}

/// Q8_0: `[m][k/32]` blocks of an f16 scale and 32 int8.
fn gen_q8(m: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let mut bytes = vec![0u8; m * (k / 32) * 34];
    for blk in bytes.as_chunks_mut::<34>().0.iter_mut() {
        rng.fill(&mut blk[2..]);
        blk[..2].copy_from_slice(&rng.f16(8, 0).to_le_bytes());
    }
    bytes
}

/// BF16 `[m][k]`: finite, exponent band `2^-15..2^13`, with `+-0` and subnormals.
fn gen_bf16(m: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let mut bytes = vec![0u8; m * k * 2];
    for e in bytes.as_chunks_mut::<2>().0.iter_mut() {
        let sign = (rng.below(2) as u16) << 15;
        let v = match rng.below(64) {
            0 => sign,
            1 => sign | rng.below(0x80) as u16,
            _ => sign | ((112 + rng.below(29) as u16) << 7) | rng.below(0x80) as u16,
        };
        e.copy_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// `[MAXC][k]` activations with the per-row magnitudes of [`AMPS`], `+-0`, F32
/// subnormals and full-mantissa values.
fn gen_x(k: usize, rng: &mut Rng) -> Vec<f32> {
    let mut x = Vec::with_capacity(MAXC * k);
    for c in 0..MAXC {
        let amp = AMPS[c % AMPS.len()];
        for _ in 0..k {
            x.push(match rng.below(64) {
                _ if amp == 0.0 => {
                    if rng.below(2) == 0 {
                        0.0
                    } else {
                        -0.0
                    }
                }
                0 => 0.0,
                1 => -0.0,
                2..=5 => f32::from_bits(((rng.below(2) as u32) << 31) | rng.below(1 << 23) as u32),
                _ => rng.unit() * amp,
            });
        }
    }
    x
}

// ---------------------------------------------------------------------------
// Shapes.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Fam {
    Mq6,
    Q8,
    Bf16,
}

impl Fam {
    fn tag(self) -> &'static str {
        match self {
            Self::Mq6 => "mq6",
            Self::Q8 => "q8",
            Self::Bf16 => "bf16",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    /// One matrix: MQ6 `gemv_mq6g256v2`, Q8 `gemv_q8_0`.
    Single,
    /// Up to four matrices of one `x` (`ms` has one to four heights): MQ6 x4, BF16 x4.
    X4,
    /// BF16 HC k4 with the given `hc_act_scale`.
    K4(Option<f32>),
}

#[derive(Clone, Debug)]
struct Spec {
    fam: Fam,
    name: String,
    kind: Kind,
    ms: Vec<usize>,
    k: usize,
    /// Weights read per C = 1 launch are timed (the real Flash-Next shapes).
    timed: bool,
    /// `Some(variant)`: the controlled HC epilogue-extremes data (`ctl_weights` / `ctl_x`).
    ctl: Option<usize>,
}

impl Spec {
    fn single(fam: Fam, name: &str, m: usize, k: usize, timed: bool) -> Self {
        Self {
            fam,
            name: format!("{name} {m}x{k}"),
            kind: Kind::Single,
            ms: vec![m],
            k,
            timed,
            ctl: None,
        }
    }
    fn x4(fam: Fam, name: &str, ms: &[usize], k: usize, timed: bool) -> Self {
        Self {
            fam,
            name: format!("{name} x4 {ms:?} k={k}"),
            kind: Kind::X4,
            ms: ms.to_vec(),
            k,
            timed,
            ctl: None,
        }
    }
    fn k4(name: &str, m: usize, k: usize, hc: bool, timed: bool) -> Self {
        let scale = hc.then_some(HC_SCALE);
        let tag = if hc {
            "hc_act=Some(0.25)"
        } else {
            "hc_act=None"
        };
        Self {
            fam: Fam::Bf16,
            name: format!("{name} k4 {m}x{k} {tag}"),
            kind: Kind::K4(scale),
            ms: vec![m],
            k,
            timed,
            ctl: None,
        }
    }
    /// Bytes-only BF16 HC k4 epilogue-extremes case: M = 8, `W[n,0]` the only non-zero weight,
    /// `x[0]` of row `c` the extreme `T` value of `variant`, the rest of `x` alternating signed zeros.
    fn hc_extremes(k: usize, scale: Option<f32>, variant: usize) -> Self {
        Self {
            fam: Fam::Bf16,
            name: format!("hc_epilogue_extremes k4 8x{k} scale={scale:?} v{variant}"),
            kind: Kind::K4(scale),
            ms: vec![8],
            k,
            timed: false,
            ctl: Some(variant),
        }
    }
    /// A BF16 group of one matrix (production's lone-projection shape).
    fn lone(&self) -> bool {
        self.fam == Fam::Bf16 && self.kind == Kind::X4 && self.ms.len() == 1
    }
    fn weight_bytes(&self, m: usize) -> usize {
        match self.fam {
            Fam::Mq6 => m * (self.k / 256) * 200,
            Fam::Q8 => m * (self.k / 32) * 34,
            Fam::Bf16 => m * self.k * 2,
        }
    }
    fn group_weight_bytes(&self) -> usize {
        self.ms.iter().map(|&m| self.weight_bytes(m)).sum()
    }
}

/// Flash-Next decode shapes (`hipfire-arch-qwen4` `config.rs` `validate_flash_next`
/// and `weights.rs`): hidden 2560, vocab 248320, 24 heads x 256 (q_dim 6144, q
/// projection `2 * q_dim`), 2 kv heads (kv_dim 512), GDN qkv 10240 / z, out K
/// 6144 / a, b 48, QSA indexer `(4 + 1) * 128 = 640`, 512 experts, shared
/// intermediate 640, HC wide `4 * 2560 = 10240` and rank 320. The x4 groups are
/// the ones `layer_ops::project_weights` / `pipeline/mod.rs` form: GDN
/// `[qkv, z, b, a]`, QSA `[indexer_qk, q, k, v]`, MoE `[router, shared_gate_scalar,
/// shared_gate, shared_up]`.
fn carried_specs(fam: Fam) -> Vec<Spec> {
    const GDN: [usize; 4] = [10240, 6144, 48, 48];
    const QSA: [usize; 4] = [640, 12288, 512, 512];
    match fam {
        Fam::Mq6 => vec![
            Spec::single(Fam::Mq6, "head", 248_320, 2560, true),
            Spec::single(Fam::Mq6, "gdn_out/qsa_o", 2560, 6144, true),
            Spec::x4(Fam::Mq6, "gdn_qkv_z_b_a", &GDN, 2560, true),
            Spec::x4(Fam::Mq6, "qsa_idx_q_k_v", &QSA, 2560, true),
            Spec::single(Fam::Mq6, "stress", 13, 1280, false),
            Spec::single(Fam::Mq6, "stress", 9, 768, false),
            Spec::single(Fam::Mq6, "stress", 1, 256, false),
            Spec::x4(Fam::Mq6, "stress", &[17, 5, 48, 9], 768, false),
        ],
        Fam::Q8 => vec![
            Spec::single(Fam::Q8, "head", 248_320, 2560, true),
            Spec::single(Fam::Q8, "gdn_qkv", 10240, 2560, true),
            Spec::single(Fam::Q8, "gdn_z", 6144, 2560, true),
            Spec::single(Fam::Q8, "qsa_k/v", 512, 2560, true),
            Spec::single(Fam::Q8, "gdn_a/b", 48, 2560, true),
            Spec::single(Fam::Q8, "gdn_out/qsa_o", 2560, 6144, true),
            Spec::single(Fam::Q8, "shared_down", 2560, 640, true),
            Spec::single(Fam::Q8, "hc_up", 10240, 320, true),
            Spec::single(Fam::Q8, "hc_down", 320, 10240, true),
            Spec::single(Fam::Q8, "stress", 13, 1568, false),
            Spec::single(Fam::Q8, "stress", 9, 1536, false),
            Spec::single(Fam::Q8, "stress", 13, 96, false),
            Spec::single(Fam::Q8, "stress", 5, 1344, false),
            Spec::single(Fam::Q8, "stress", 1, 32, false),
        ],
        Fam::Bf16 => {
            let mut v = vec![
                Spec::x4(
                    Fam::Bf16,
                    "moe_router_gate_gate_up",
                    &[512, 1, 640, 640],
                    2560,
                    true,
                ),
                Spec::x4(Fam::Bf16, "gdn_qkv_z_b_a", &GDN, 2560, true),
                Spec::x4(Fam::Bf16, "router", &[512], 2560, true),
                Spec::x4(Fam::Bf16, "shared_down", &[2560], 640, true),
                Spec::k4("hc_down", 320, 10240, false, true),
                Spec::k4("hc_down", 320, 10240, true, true),
                Spec::k4("hc_block_inject", 4, 10240, false, true),
                Spec::k4("hc_block_inject", 4, 10240, true, true),
                Spec::k4("hc_up", 10240, 320, false, true),
                Spec::k4("hc_up", 10240, 320, true, true),
                Spec::x4(Fam::Bf16, "stress", &[17, 5, 48, 9], 2568, false),
                Spec::x4(Fam::Bf16, "stress_k%8!=0", &[9, 5, 7, 3], 2563, false),
                Spec::x4(Fam::Bf16, "stress", &[7], 96, false),
                Spec::k4("stress", 7, 640, false, false),
                Spec::k4("stress", 7, 640, true, false),
            ];
            // HC epilogue extremes: K%32 == 0, both K, scales Some(3.7) / Some(0.5) and None.
            for k in [160, 640] {
                for scale in [Some(3.7f32), Some(0.5), None] {
                    for variant in 0..CTL_VARIANTS {
                        v.push(Spec::hc_extremes(k, scale, variant));
                    }
                }
            }
            v
        }
    }
}

/// Number of `hc_epilogue_extremes` variants: three over `CTL_T`, the fourth the NaN rows.
const CTL_VARIANTS: usize = 4;

/// The epilogue extremes of variants 0..=2 (row `c` of variant `v` takes `CTL_T[(c + 8 v) % len]`).
const CTL_T: [f32; 22] = [
    3e-30,
    -3e-30,
    1e-3,
    -1e-3,
    0.5,
    -0.5,
    1.0,
    -1.0,
    1.4e-45,
    -1.4e-45,
    1e-38,
    -1e-38,
    3.0e38,
    -3.0e38,
    f32::INFINITY,
    f32::NEG_INFINITY,
    257.0 / 256.0,
    -257.0 / 256.0,
    33.0,
    -88.7,
    88.7,
    104.0,
];

/// Variant 3: quiet and signaling NaN patterns of both signs, then four finite rows.
fn ctl_t(variant: usize, c: usize) -> f32 {
    if variant == 3 {
        [
            f32::from_bits(0x7fc0_0001),
            f32::from_bits(0x7f80_0001),
            f32::from_bits(0xffc0_0001),
            f32::from_bits(0xff80_0001),
            1.0,
            -1.0,
            0.5,
            -0.5,
        ][c % 8]
    } else {
        CTL_T[(c + 8 * variant) % CTL_T.len()]
    }
}

/// BF16 `[8][k]`: `W[n,0] = [1,-1,0.5,2,0.75,3,-0.125,1.5][n]`, every other weight `0x0000`.
fn ctl_weights(k: usize) -> Vec<u8> {
    let w0 = [1.0f32, -1.0, 0.5, 2.0, 0.75, 3.0, -0.125, 1.5];
    let mut bytes = vec![0u8; 8 * k * 2];
    for (n, w) in w0.iter().enumerate() {
        bytes[n * k * 2..n * k * 2 + 2]
            .copy_from_slice(&((w.to_bits() >> 16) as u16).to_le_bytes());
    }
    bytes
}

/// `[MAXC][k]`: `x[c][0] = ctl_t(variant, c)`, the rest alternating `+0.0` / `-0.0`.
fn ctl_x(k: usize, variant: usize) -> Vec<f32> {
    let mut x = Vec::with_capacity(MAXC * k);
    for c in 0..MAXC {
        x.push(ctl_t(variant, c));
        x.extend((1..k).map(|j| if j % 2 == 1 { 0.0 } else { -0.0 }));
    }
    x
}

// ---------------------------------------------------------------------------
// Device data.
// ---------------------------------------------------------------------------

/// One output allocation: `GUARD | (MAXC + PAD_ROWS) * m f32 | GUARD`.
struct Out {
    full: GpuTensor,
    m: usize,
}

impl Out {
    fn new(gpu: &mut Gpu, m: usize) -> Result<Self> {
        let full = gpu
            .zeros(&[2 * GUARD + (MAXC + PAD_ROWS) * m], DType::F32)
            .map_err(err)?;
        Ok(Self { full, m })
    }
    fn words(&self) -> usize {
        2 * GUARD + (MAXC + PAD_ROWS) * self.m
    }
    fn reset(&self, gpu: &Gpu) -> Result<()> {
        gpu.hip
            .memset(&self.full.buf, i32::from(SENTINEL), self.words() * 4)
            .map_err(err)
    }
    fn download(&self, gpu: &Gpu) -> Result<Vec<u8>> {
        gpu.hip.device_synchronize().map_err(err)?;
        let mut v = vec![0u8; self.full.buf.size()];
        gpu.hip.memcpy_dtoh(&mut v, &self.full.buf).map_err(err)?;
        v.truncate(self.words() * 4);
        Ok(v)
    }
    /// Sentinel everywhere, rows `0..c` of `reference` in place.
    fn expected(&self, reference: &[u8], c: usize) -> Vec<u8> {
        let mut e = vec![SENTINEL; self.words() * 4];
        let (lo, hi) = (GUARD * 4, GUARD * 4 + c * self.m * 4);
        e[lo..hi].copy_from_slice(&reference[lo..hi]);
        e
    }
}

/// One weight matrix: `copies` rotating copies in one allocation, its output and
/// per-row views.
struct Mat {
    m: usize,
    wbytes: usize,
    big: GpuTensor,
    views: Vec<GpuTensor>,
    out: Out,
    yfull: GpuTensor,
    yrow: Vec<GpuTensor>,
}

struct Case {
    spec: Spec,
    mats: Vec<Mat>,
    x: GpuTensor,
    xbig: GpuTensor,
    host_x: Vec<u8>,
    weight_hashes: Vec<String>,
    xrow: Vec<GpuTensor>,
    copies: usize,
}

fn upload_weights(
    gpu: &Gpu,
    host: &[u8],
    copies: usize,
    bf16: bool,
) -> Result<(GpuTensor, Vec<GpuTensor>)> {
    gpu.bind_thread().map_err(err)?;
    let stride = (host.len() + 2 * GUARD * 4).div_ceil(256) * 256;
    let total = stride * copies;
    let (dtype, esz) = if bf16 {
        (DType::BF16, 2)
    } else {
        (DType::Raw, 1)
    };
    let buf: DeviceBuffer = gpu.hip.malloc(total).map_err(err)?;
    let big = GpuTensor {
        buf,
        shape: vec![total / esz],
        dtype,
    };
    gpu.hip
        .memset(&big.buf, i32::from(SENTINEL), total)
        .map_err(err)?;
    for c in 0..copies {
        if !host.is_empty() {
            let view = big.sub_offset((c * stride + GUARD * 4) / esz, host.len() / esz);
            gpu.hip.memcpy_htod(&view.buf, host).map_err(err)?;
        }
    }
    // Empty x4 slots still have distinct, guarded pointer allocations.
    let views = (0..copies)
        .map(|c| big.sub_offset((c * stride + GUARD * 4) / esz, host.len() / esz))
        .collect();
    Ok((big, views))
}

fn build_case(gpu: &mut Gpu, spec: &Spec, seed: u64, cold_bytes: usize) -> Result<Case> {
    let mut h = Sha256::new();
    h.update(spec.name.as_bytes());
    h.update(spec.fam.tag().as_bytes());
    let d = h.finalize();
    let mut rng = Rng::new(seed ^ u64::from_le_bytes(d[..8].try_into().unwrap()));
    let copies = if cold_bytes == 0 {
        1
    } else {
        cold_bytes
            .div_ceil(spec.group_weight_bytes().max(1))
            .clamp(1, 4096)
    };
    let mut mats = Vec::new();
    for &m in &spec.ms {
        let host = match (spec.fam, spec.ctl) {
            (_, Some(_)) => ctl_weights(spec.k),
            (Fam::Mq6, None) => gen_mq6(m, spec.k, &mut rng),
            (Fam::Q8, None) => gen_q8(m, spec.k, &mut rng),
            (Fam::Bf16, None) => gen_bf16(m, spec.k, &mut rng),
        };
        let (big, views) = upload_weights(gpu, &host, copies, spec.fam == Fam::Bf16)?;
        let out = Out::new(gpu, m)?;
        let yfull = out.full.sub_offset(GUARD, MAXC * m);
        let yrow = (0..MAXC).map(|c| yfull.sub_offset(c * m, m)).collect();
        mats.push(Mat {
            m,
            wbytes: host.len(),
            big,
            views,
            out,
            yfull,
            yrow,
        });
    }
    let xs = match spec.ctl {
        Some(v) => ctl_x(spec.k, v),
        None => gen_x(spec.k, &mut rng),
    };
    let host_x: Vec<u8> = xs.iter().flat_map(|v| v.to_le_bytes()).collect();
    let xbig = gpu
        .zeros(&[2 * GUARD + (MAXC + PAD_ROWS) * spec.k], DType::F32)
        .map_err(err)?;
    let x = xbig.sub_offset(GUARD, MAXC * spec.k);
    let weight_hashes = mats
        .iter()
        .map(|m| tensor_sha(gpu, &m.big))
        .collect::<Result<Vec<_>>>()?;
    let xrow = (0..MAXC)
        .map(|c| x.sub_offset(c * spec.k, spec.k))
        .collect();
    Ok(Case {
        spec: spec.clone(),
        mats,
        x,
        xbig,
        host_x,
        weight_hashes,
        xrow,
        copies,
    })
}
/// Build a byte-path case from SHA-pinned real-capture operands (no seeded
/// synthesis): captured weights per matrix slot (empty slots keep guarded
/// allocations) and the captured 64-row X, with the source singleton for
/// strict identity checks in the caller.
fn build_captured_case(gpu: &mut Gpu, loaded: &LoadedCapture) -> Result<Case> {
    let spec = loaded.spec.clone();
    if loaded
        .manifest_ms
        .iter()
        .copied()
        .filter(|&m| m > 0)
        .collect::<Vec<_>>()
        != spec.ms
    {
        return Err("captured live-slot packing disagrees".into());
    }
    let mut mats = Vec::new();
    let mut slot = 0usize;
    for &m in spec.ms.iter() {
        while loaded.manifest_ms.get(slot).copied().unwrap_or(0) == 0 {
            slot += 1;
        }
        let host = loaded.weights.get(slot).cloned().unwrap_or_default();
        slot += 1;
        let expected = spec.weight_bytes(m);
        if host.len() != expected {
            return Err(format!(
                "captured slot {slot}: weight extent {} != {expected}",
                host.len()
            ));
        }
        let (big, views) = upload_weights(gpu, &host, 1, spec.fam == Fam::Bf16)?;
        let out = Out::new(gpu, m)?;
        let yfull = out.full.sub_offset(GUARD, MAXC * m);
        let yrow = (0..MAXC).map(|c| yfull.sub_offset(c * m, m)).collect();
        mats.push(Mat {
            m,
            wbytes: host.len(),
            big,
            views,
            out,
            yfull,
            yrow,
        });
    }
    if loaded.host_x.len() != MAXC * spec.k {
        return Err("captured X row count/extent mismatch".into());
    }
    let host_x: Vec<u8> = loaded.host_x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let xbig = gpu
        .zeros(&[2 * GUARD + (MAXC + PAD_ROWS) * spec.k], DType::F32)
        .map_err(err)?;
    let x = xbig.sub_offset(GUARD, MAXC * spec.k);
    let weight_hashes = mats
        .iter()
        .map(|m| tensor_sha(gpu, &m.big))
        .collect::<Result<Vec<_>>>()?;
    let xrow = (0..MAXC)
        .map(|c| x.sub_offset(c * spec.k, spec.k))
        .collect();
    Ok(Case {
        spec,
        mats,
        x,
        xbig,
        host_x,
        weight_hashes,
        xrow,
        copies: 1,
    })
}
fn free_case(gpu: &mut Gpu, case: Case) -> Result<()> {
    for mat in case.mats {
        gpu.hip.free(mat.big.buf).map_err(err)?;
        gpu.free_tensor(mat.out.full).map_err(err)?;
    }
    gpu.free_tensor(case.xbig).map_err(err)
}

impl Case {
    fn reset(&self, gpu: &Gpu) -> Result<()> {
        self.mats.iter().try_for_each(|m| m.out.reset(gpu))
    }
    fn download(&self, gpu: &Gpu) -> Result<Vec<Vec<u8>>> {
        self.mats.iter().map(|m| m.out.download(gpu)).collect()
    }
    fn expected(&self, reference: &[Vec<u8>], c: usize) -> Vec<Vec<u8>> {
        self.mats
            .iter()
            .zip(reference)
            .map(|(m, r)| m.out.expected(r, c))
            .collect()
    }
    fn group_bytes(&self) -> usize {
        self.mats.iter().map(|m| m.wbytes).sum()
    }
    fn y_elems(&self) -> usize {
        self.mats.iter().map(|m| m.m).sum()
    }
}

/// Small ragged shapes complement the carried production-shape census.
fn specs(fam: Fam) -> Vec<Spec> {
    let mut v = carried_specs(fam);
    if fam != Fam::Q8 {
        for m in [1, 2, 3, 7, 8, 9, 17] {
            if fam == Fam::Mq6 {
                for k in [256, 512, 768, 1280] {
                    v.push(Spec::single(fam, "output_tail", m, k, false));
                }
            } else {
                for k in [1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 160] {
                    v.push(Spec::x4(fam, "ordinary_scalar_tail", &[m], k, false));
                }
                for k in [32, 64, 96, 160, 640] {
                    for act in [false, true] {
                        v.push(Spec::k4("quarter_tail", m, k, act, false));
                    }
                }
            }
        }
        // Slot positions matter: do not compact empty slots or alias their outputs.
        for mask in 0u8..16 {
            let ms: Vec<usize> = [1, 3, 8, 9]
                .iter()
                .enumerate()
                .map(|(i, &m)| if mask & (1 << i) != 0 { m } else { 0 })
                .collect();
            let ks: &[usize] = if fam == Fam::Mq6 {
                &[256, 768]
            } else {
                &[7, 32, 33]
            };
            for &k in ks {
                v.push(Spec::x4(
                    fam,
                    &format!("slot_mask_{mask:04b}"),
                    &ms,
                    k,
                    false,
                ));
            }
        }
    }
    v
}

fn tensor_sha(gpu: &Gpu, tensor: &GpuTensor) -> Result<String> {
    gpu.hip.device_synchronize().map_err(err)?;
    let esz = match tensor.dtype {
        DType::BF16 => 2,
        DType::F32 => 4,
        _ => 1,
    };
    let mut hash = Sha256::new();
    let mut stage = vec![0u8; tensor.buf.size().min(8 << 20)];
    let mut offset = 0;
    while offset < tensor.buf.size() {
        let n = stage.len().min(tensor.buf.size() - offset);
        let view = tensor.sub_offset(offset / esz, n / esz);
        gpu.hip
            .memcpy_dtoh(&mut stage[..n], &view.buf)
            .map_err(err)?;
        hash.update(&stage[..n]);
        offset += n;
    }
    Ok(hash.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

impl Case {
    fn empty(&self) -> bool {
        self.mats.iter().all(|m| m.m == 0)
    }
    /// Poison every inactive activation row, including an entire extra tile.
    fn prepare_rows(&self, gpu: &Gpu, rows: usize) -> Result<String> {
        gpu.hip
            .memset(&self.xbig.buf, i32::from(SENTINEL), self.xbig.buf.size())
            .map_err(err)?;
        if rows > 0 {
            gpu.hip
                .memcpy_htod(&self.x.buf, &self.host_x[..rows * self.spec.k * 4])
                .map_err(err)?;
        }
        tensor_sha(gpu, &self.xbig)
    }
    fn check_inputs(&self, gpu: &Gpu, x_sha: &str) -> Result<()> {
        if tensor_sha(gpu, &self.xbig)? != x_sha {
            return Err("activation/guard mutation".into());
        }
        for (i, mat) in self.mats.iter().enumerate() {
            if tensor_sha(gpu, &mat.big)? != self.weight_hashes[i] {
                return Err(format!("matrix {i}: weight/guard mutation"));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Route {
    Single,
    SingleAlt,
    Hip,
    Tile,
    PmC1,
}
impl Route {
    fn label(self) -> &'static str {
        match self {
            Self::Single => "runtime_singleton",
            Self::SingleAlt => "noncandidate_singleton_alt",
            Self::Hip => "noncandidate_incumbent_rows",
            Self::Tile => "candidate_t8",
            Self::PmC1 => "additional_pm_t8_c1_per_row",
        }
    }
    fn candidate(self) -> bool {
        matches!(self, Self::Tile | Self::PmC1)
    }
}

const MQ6: &str = "gemv_mq6g256v2_rows_t8_pm_gfx1201";
const MQ6_X4: &str = "gemv_mq6g256v2_rows_x4_t8_pm_gfx1201";
const BF16: &str = "gemv_bf16_xf32_rows_x4_t8_pm_gfx1201";
const HC: &str = "gemv_bf16_xf32_k4_rows_t8_pm_gfx1201";
fn tile_symbol(spec: &Spec) -> Result<&'static str> {
    match (spec.fam, spec.kind) {
        (Fam::Mq6, Kind::Single) => Ok(MQ6),
        (Fam::Mq6, Kind::X4) => Ok(MQ6_X4),
        (Fam::Bf16, Kind::Single | Kind::X4) => Ok(BF16),
        (Fam::Bf16, Kind::K4(_)) => Ok(HC),
        _ => Err("no gfx1201 PM twin for this representation (no Halo Q8 alias)".into()),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Identity {
    module: String,
    symbol: String,
    sha: String,
    container_sha: String,
}
impl Identity {
    fn json(&self) -> Value {
        json!({"module":self.module,"symbol":self.symbol,"object_sha256":self.sha,"container_sha256":self.container_sha})
    }
}
fn identity(gpu: &Gpu, symbol: &str) -> Result<Identity> {
    let (module, container_sha, sha) = gpu.lab_kernel_device_identity(symbol).map_err(err)?;
    for pin in [&container_sha, &sha] {
        if pin.len() != 64 || !pin.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(format!("{symbol}: invalid loaded identity SHA {pin:?}"));
        }
    }
    Ok(Identity {
        module,
        symbol: symbol.into(),
        sha,
        container_sha,
    })
}
fn singleton_identity(gpu: &Gpu, pins: &BTreeMap<String, String>, case: &Case) -> Result<Identity> {
    let symbol = if case.empty() {
        if case.spec.fam == Fam::Mq6 {
            "gemv_mq6g256v2_x4"
        } else {
            "gemv_bf16_xf32_x4"
        }
    } else {
        gpu.last_launched_kernel()
            .ok_or("singleton route not recorded")?
    };
    let actual = identity(gpu, symbol)?;
    let want = pins
        .get(symbol)
        .ok_or_else(|| format!("missing reviewed singleton pin: {symbol}"))?;
    if &actual.sha != want {
        return Err(format!(
            "{}/{symbol}: actual SHA {} differs from reviewed {want}",
            actual.module, actual.sha
        ));
    }
    Ok(actual)
}
fn load_tile(gpu: &mut Gpu, spec: &Spec) -> Result<Identity> {
    let symbol = tile_symbol(spec)?;
    gpu.ensure_row_tile_pm_entry(symbol).map_err(err)?;
    let actual = identity(gpu, symbol)?;
    let (bytes, pin, module) = match spec.fam {
        Fam::Mq6 => (
            kernels::LANE_ROWS_MQ6_PM_GFX1201,
            kernels::LANE_ROWS_MQ6_PM_GFX1201_SHA256,
            "lane_rows_mq6_pm_gfx1201",
        ),
        Fam::Bf16 => (
            kernels::LANE_ROWS_BF16_PM_GFX1201,
            kernels::LANE_ROWS_BF16_PM_GFX1201_SHA256,
            "lane_rows_bf16_pm_gfx1201",
        ),
        _ => return Err("no gfx1201 Q8 PM object".into()),
    };
    let pin = pin.trim();
    if pin.len() != 64
        || sha256_hex(bytes) != pin
        || actual.container_sha != pin
        || actual.module != module
    {
        return Err(format!(
            "{symbol}: embedded/loaded object does not match registered module and SHA pin"
        ));
    }
    Ok(actual)
}

/// Raw frozen ABI: total logical rows, full-window matrix strides, grid-y tiles.
fn tile_launch(gpu: &mut Gpu, case: &Case, rows: usize, first: usize, copy: usize) -> Result<()> {
    if !(1..=MAXC).contains(&rows) || first + rows > MAXC {
        return Err(format!("invalid tile rows={rows}, first={first}"));
    }
    let symbol = tile_symbol(&case.spec)?;
    let k = case.spec.k;
    let x = case.x.sub_offset(first * k, rows * k);
    let mut kb = KernargBlob::new();
    let (grid, block, lds, abi) = match (case.spec.fam, case.spec.kind) {
        (Fam::Mq6, Kind::Single) => {
            let mat = &case.mats[0];
            kb.push_ptr(mat.views[copy].buf.as_ptr());
            kb.push_ptr(x.buf.as_ptr());
            kb.push_ptr(
                mat.yfull
                    .sub_offset(first * mat.m, rows * mat.m)
                    .buf
                    .as_ptr(),
            );
            kb.push_i32(mat.m as i32);
            kb.push_i32(k as i32);
            kb.push_i32(rows as i32);
            (mat.m.div_ceil(8), 256, 8192, 36)
        }
        (Fam::Mq6 | Fam::Bf16, Kind::X4) | (Fam::Bf16, Kind::Single) => {
            let ps = parts(case, copy, |m| &m.yfull);
            for (w, _, _) in ps {
                kb.push_ptr(w.buf.as_ptr());
            }
            kb.push_ptr(x.buf.as_ptr());
            for (_, y, m) in ps {
                kb.push_ptr(y.sub_offset(first * m, rows * m).buf.as_ptr());
            }
            for (_, _, m) in ps {
                kb.push_i32(m as i32);
            }
            kb.push_i32(k as i32);
            kb.push_i32(rows as i32);
            if case.spec.fam == Fam::Mq6 {
                (
                    ps.iter().map(|p| p.2.div_ceil(8)).sum::<usize>(),
                    256,
                    8192,
                    96,
                )
            } else {
                (ps.iter().map(|p| p.2).sum::<usize>(), 32, 0, 96)
            }
        }
        (Fam::Bf16, Kind::K4(scale)) => {
            let mat = &case.mats[0];
            kb.push_ptr(mat.views[copy].buf.as_ptr());
            kb.push_ptr(x.buf.as_ptr());
            kb.push_ptr(
                mat.yfull
                    .sub_offset(first * mat.m, rows * mat.m)
                    .buf
                    .as_ptr(),
            );
            kb.push_i32(mat.m as i32);
            kb.push_i32(k as i32);
            kb.push_f32(scale.unwrap_or(1.0));
            kb.push_i32(i32::from(scale.is_some()));
            kb.push_i32(rows as i32);
            (mat.m, 128, 128, 44)
        }
        _ => return Err("no candidate tile for this family".into()),
    };
    if kb.len() != abi {
        return Err(format!("{symbol}: ABI {} != {abi}", kb.len()));
    }
    // HC's logical44B ABI has a 48-byte kernarg segment.
    if symbol == HC {
        kb.push_i32(0);
    }
    // A wholly empty x4 grid still exercises the kernel's uniform dead-WG exit.
    gpu.launch_kernel_blob(
        symbol,
        [grid.max(1) as u32, rows.div_ceil(8) as u32, 1],
        [block, 1, 1],
        lds,
        kb.as_mut_slice(),
    )
    .map_err(err)
}

fn incumbent_launch(gpu: &mut Gpu, case: &Case, rows: usize, copy: usize) -> Result<()> {
    let (k, x) = (case.spec.k, &case.x);
    let mat = &case.mats[0];
    match (case.spec.fam, case.spec.kind) {
        (Fam::Mq6, Kind::Single) => gpu
            .gemm_mq6g256v2_f32_rows(&mat.views[copy], x, &mat.yfull, mat.m, k, rows)
            .map_err(err),
        (Fam::Mq6, Kind::X4) => gpu
            .gemm_mq6g256v2_f32_rows_x4(parts(case, copy, |m| &m.yfull), x, k, rows)
            .map_err(err),
        (Fam::Bf16, Kind::Single | Kind::X4) => gpu
            .gemv_bf16_xf32_x4_rows(parts(case, copy, |m| &m.yfull), x, k, rows)
            .map_err(err),
        (Fam::Bf16, Kind::K4(scale)) => gpu
            .gemv_bf16_xf32_k4_rows(&mat.views[copy], x, &mat.yfull, mat.m, k, scale, rows)
            .map_err(err),
        (Fam::Q8, Kind::Single) => gpu
            .gemm_q8_0_batched_f32_chunked(&mat.views[copy], x, &mat.yfull, mat.m, k, rows)
            .map_err(err),
        _ => Err("unsupported diagnostic incumbent".into()),
    }
}

fn launch(
    gpu: &mut Gpu,
    case: &Case,
    route: Route,
    rows: usize,
    sample: Option<usize>,
) -> Result<()> {
    let copy = sample.map_or(0, |s| s % case.copies);
    match route {
        Route::Single | Route::SingleAlt => {
            for row in 0..rows {
                let copy = sample.map_or(0, |s| (s * rows + row) % case.copies);
                single_row(gpu, case, route, row, copy)?;
            }
            Ok(())
        }
        Route::PmC1 => {
            for row in 0..rows {
                tile_launch(gpu, case, 1, row, copy)?;
            }
            Ok(())
        }
        Route::Tile => tile_launch(gpu, case, rows, 0, copy),
        Route::Hip => incumbent_launch(gpu, case, rows, copy),
    }
}
type Parts<'a> = [(&'a GpuTensor, &'a GpuTensor, usize); 4];

fn parts<'a>(case: &'a Case, copy: usize, y: impl Fn(&'a Mat) -> &'a GpuTensor) -> Parts<'a> {
    let slot = |i: usize| {
        let live = i < case.mats.len();
        let mat = &case.mats[if live { i } else { 0 }];
        (&mat.views[copy], y(mat), if live { mat.m } else { 0 })
    };
    [slot(0), slot(1), slot(2), slot(3)]
}

/// One singleton launch of row `row` of `route` (`Single` or `SingleAlt`).
fn single_row(gpu: &mut Gpu, case: &Case, route: Route, row: usize, copy: usize) -> Result<()> {
    let (k, x) = (case.spec.k, &case.xrow[row]);
    if case.empty() {
        // Production APIs use grid.x=sum(Mi), which is zero for the all-empty
        // domain. Exercise that already-loaded actual singleton object's dead
        // workgroup with grid.x=1 instead; no substitute arithmetic is involved.
        let symbol = if case.spec.fam == Fam::Mq6 {
            "gemv_mq6g256v2_x4"
        } else {
            "gemv_bf16_xf32_x4"
        };
        gpu.lab_kernel_device_identity(symbol).map_err(err)?;
        let ps = parts(case, copy, |m| &m.yrow[row]);
        let mut kb = KernargBlob::new();
        for (w, _, _) in ps {
            kb.push_ptr(w.buf.as_ptr());
        }
        kb.push_ptr(x.buf.as_ptr());
        for (_, y, _) in ps {
            kb.push_ptr(y.buf.as_ptr());
        }
        for _ in 0..4 {
            kb.push_i32(0);
        }
        kb.push_i32(k as i32);
        if kb.len() != 92 {
            return Err("empty singleton x4 ABI != 92".into());
        }
        return gpu
            .launch_kernel_blob(symbol, [1, 1, 1], [32, 1, 1], 0, kb.as_mut_slice())
            .map_err(err);
    }
    let m0 = &case.mats[0];
    match (case.spec.fam, case.spec.kind) {
        (Fam::Mq6, Kind::Single) => gpu
            .gemv_mq6g256v2(&m0.views[copy], x, &m0.yrow[row], m0.m, k)
            .map_err(err),
        (Fam::Q8, Kind::Single) => gpu
            .gemv_q8_0(&m0.views[copy], x, &m0.yrow[row], m0.m, k)
            .map_err(err),
        (Fam::Mq6, Kind::X4) => gpu
            .gemv_mq6g256v2_x4(parts(case, copy, |m| &m.yrow[row]), x, k)
            .map_err(err),
        (Fam::Bf16, Kind::Single) => gpu
            .gemv_bf16_xf32(&m0.views[copy], x, &m0.yrow[row], m0.m, k)
            .map_err(err),
        (Fam::Bf16, Kind::X4) => {
            // Lone: primary `gemv_bf16_xf32`, alt the one-slot x4. Group: primary x4, alt per matrix.
            let per_matrix = (route == Route::Single) == case.spec.lone();
            if per_matrix {
                for mat in case.mats.iter().filter(|m| m.m > 0) {
                    gpu.gemv_bf16_xf32(&mat.views[copy], x, &mat.yrow[row], mat.m, k)
                        .map_err(err)?;
                }
                Ok(())
            } else {
                gpu.gemv_bf16_xf32_x4(parts(case, copy, |m| &m.yrow[row]), x, k)
                    .map_err(err)
            }
        }
        (Fam::Bf16, Kind::K4(hc)) => gpu
            .gemv_bf16_xf32_k4(&m0.views[copy], x, &m0.yrow[row], m0.m, k, hc)
            .map_err(err),
        (fam, kind) => Err(format!("no singleton for {fam:?} {kind:?}")),
    }
}
#[derive(Clone, Debug)]
struct Mismatch {
    region: &'static str,
    matrix: usize,
    row: usize,
    output: usize,
    got: u32,
    want: u32,
}

/// `(mismatching words, first mismatch)` of `got` against `want`, `c` live rows.
fn compare(case: &Case, got: &[Vec<u8>], want: &[Vec<u8>], c: usize) -> (usize, Option<Mismatch>) {
    let mut bad = 0;
    let mut first = None;
    for (mi, ((g, w), mat)) in got.iter().zip(want).zip(&case.mats).enumerate() {
        for (i, (a, b)) in g
            .as_chunks::<4>()
            .0
            .iter()
            .zip(w.as_chunks::<4>().0)
            .enumerate()
        {
            let (a, b) = (u32::from_le_bytes(*a), u32::from_le_bytes(*b));
            if a == b {
                continue;
            }
            bad += 1;
            if first.is_none() {
                let (region, row, output) = if i < GUARD {
                    ("guard_head", 0, i)
                } else if i >= GUARD + (MAXC + PAD_ROWS) * mat.m {
                    ("guard_tail", 0, i - GUARD - (MAXC + PAD_ROWS) * mat.m)
                } else {
                    let off = i - GUARD;
                    let (row, output) = (off / mat.m, off % mat.m);
                    (if row < c { "live" } else { "inactive_row" }, row, output)
                };
                first = Some(Mismatch {
                    region,
                    matrix: mi,
                    row,
                    output,
                    got: a,
                    want: b,
                });
            }
        }
    }
    (bad, first)
}

fn mismatch_json(m: &Option<Mismatch>) -> Value {
    match m {
        None => Value::Null,
        Some(m) => json!({
            "region": m.region, "matrix": m.matrix, "row": m.row, "output": m.output,
            "got_bits": format!("{:#010x}", m.got), "want_bits": format!("{:#010x}", m.want),
        }),
    }
}

struct Reference {
    bytes: Vec<Vec<u8>>,
    identity: Identity,
    x_sha: String,
}
fn reference(gpu: &mut Gpu, case: &Case, pins: &BTreeMap<String, String>) -> Result<Reference> {
    let x_sha = case.prepare_rows(gpu, MAXC)?;
    case.reset(gpu)?;
    let mut launched = None;
    for row in 0..MAXC {
        single_row(gpu, case, Route::Single, row, 0)?;
        let actual = singleton_identity(gpu, pins, case)?;
        if let Some(previous) = &launched {
            if previous != &actual {
                return Err(format!("singleton row {row}: object/route changed"));
            }
        }
        launched = Some(actual);
    }
    let bytes = case.download(gpu)?;
    let (bad, first) = compare(case, &bytes, &case.expected(&bytes, MAXC), MAXC);
    if bad != 0 {
        return Err(format!(
            "singleton overwrote guards/inactive rows: {}",
            mismatch_json(&first)
        ));
    }
    case.check_inputs(gpu, &x_sha)?;
    case.reset(gpu)?;
    launch(gpu, case, Route::Single, MAXC, None)?;
    let repeat = case.download(gpu)?;
    let (bad, first) = compare(case, &repeat, &bytes, MAXC);
    if bad != 0 {
        return Err(format!(
            "singleton repeat is not deterministic: {}",
            mismatch_json(&first)
        ));
    }
    case.check_inputs(gpu, &x_sha)?;
    Ok(Reference {
        bytes,
        identity: launched.ok_or("no runtime singleton identity")?,
        x_sha,
    })
}

struct ByteCell {
    route: Route,
    rows: usize,
    ok: bool,
    value: Value,
}
fn byte_cell(
    gpu: &mut Gpu,
    case: &Case,
    route: Route,
    rows: usize,
    actual: &Reference,
    pm_c1: Option<&[Vec<u8>]>,
    tile: Option<&Identity>,
) -> Result<(ByteCell, Vec<Vec<u8>>)> {
    let x_sha = case.prepare_rows(gpu, rows)?;
    case.reset(gpu)?;
    let launch_error = launch(gpu, case, route, rows, None).err();
    if let Some(error) = launch_error {
        return Ok((
            ByteCell {
                route,
                rows,
                ok: false,
                value: json!({
                    "route":route.label(),"candidate":route.candidate(),"R":rows,
                    "ok":false,"launched":false,"error":error,"identity":tile.map(Identity::json),
                    "input_immutability_error":case.check_inputs(gpu, &x_sha).err()
                }),
            },
            Vec::new(),
        ));
    }
    let got = case.download(gpu)?;
    let (bad, first) = compare(case, &got, &case.expected(&actual.bytes, rows), rows);
    let pm_bad = pm_c1.map(|pm| compare(case, &got, &case.expected(pm, rows), rows));
    let input_error = case.check_inputs(gpu, &x_sha).err();
    let ok = bad == 0 && pm_bad.as_ref().is_none_or(|p| p.0 == 0) && input_error.is_none();
    let launched = if route.candidate() {
        tile.map(Identity::json)
    } else {
        gpu.last_launched_kernel()
            .and_then(|s| identity(gpu, s).ok())
            .map(|id| id.json())
    };
    let value = json!({"route":route.label(),"candidate":route.candidate(),"R":rows,
        "first_row":0,"ok":ok,"launched":true,"identity":launched,
        "mismatch_words_vs_runtime_singleton":bad,"first_mismatch":mismatch_json(&first),
        "mismatch_words_vs_pm_c1":pm_bad.as_ref().map(|p| p.0),
        "first_mismatch_vs_pm_c1":pm_bad.as_ref().map(|p| mismatch_json(&p.1)),
        "input_immutability_error":input_error,"grid_y":if route == Route::Tile {rows.div_ceil(8)} else {1},
        "independent_launches":if matches!(route, Route::Single | Route::SingleAlt | Route::PmC1) {rows} else {1},
        "logical_x_bytes":rows * case.spec.k * 4,
        "logical_y_bytes":rows * case.y_elems() * 4,
        "output_allocation_bytes":case.mats.iter().map(|m| m.out.words()*4).collect::<Vec<_>>(),
        "input_object_sha256":x_sha,"weight_allocation_sha256s":case.weight_hashes,
    });
    Ok((
        ByteCell {
            route,
            rows,
            ok,
            value,
        },
        got,
    ))
}

fn bytes_phase(
    gpu: &mut Gpu,
    case: &Case,
    actual: &Reference,
    tile: Option<&Identity>,
) -> Result<Vec<ByteCell>> {
    let mut cells = Vec::new();
    if tile.is_some() {
        // Independent PM C1 calls on all 64 distinct rows, checked against AR.
        let (cell, pm) = byte_cell(gpu, case, Route::PmC1, MAXC, actual, None, tile)?;
        let usable = !pm.is_empty();
        cells.push(cell);
        for rows in 1..=MAXC {
            let (mut cell, _) = byte_cell(
                gpu,
                case,
                Route::Tile,
                rows,
                actual,
                if usable { Some(&pm) } else { None },
                tile,
            )?;
            if !usable {
                cell.ok = false;
                cell.value["ok"] = json!(false);
                cell.value["error"] = json!("mandatory independent PM C1 comparison unavailable");
            }
            cells.push(cell);
        }
    }
    if case.spec.fam == Fam::Bf16 && matches!(case.spec.kind, Kind::X4) && !case.empty() {
        cells.push(byte_cell(gpu, case, Route::SingleAlt, MAXC, actual, None, None)?.0);
    }
    // A diagnostic difference is not a candidate success or candidate evidence.
    for rows in 2..=8 {
        cells.push(byte_cell(gpu, case, Route::Hip, rows, actual, None, None)?.0);
    }
    Ok(cells)
}

fn timing_cells(sweep: bool, candidate: bool) -> Vec<(Route, usize)> {
    let mut cells = Vec::new();
    let widths: Vec<usize> = if sweep {
        (1..=MAXC).collect()
    } else {
        vec![1, 2, 4, 8, 9, 16, 17, 32, 33, 48, 63, 64]
    };
    for rows in widths {
        cells.push((Route::Single, rows));
        if rows <= 8 && rows > 1 {
            cells.push((Route::Hip, rows));
        }
        if candidate {
            cells.push((Route::Tile, rows));
        }
    }
    if candidate {
        cells.push((Route::PmC1, 1));
    }
    cells
}

fn time_cell(
    gpu: &mut Gpu,
    events: &[Event],
    case: &Case,
    route: Route,
    rows: usize,
    cold: bool,
    iters: usize,
) -> Result<Value> {
    case.prepare_rows(gpu, rows)?;
    case.reset(gpu)?;
    let sample = |s| cold.then_some(s);
    for s in 0..WARMUP {
        launch(gpu, case, route, rows, sample(s))?;
    }
    gpu.hip.device_synchronize().map_err(err)?;
    gpu.hip.event_record(&events[0], None).map_err(err)?;
    for s in 0..iters {
        launch(gpu, case, route, rows, sample(WARMUP + s))?;
        gpu.hip.event_record(&events[s + 1], None).map_err(err)?;
    }
    gpu.hip.event_synchronize(&events[iters]).map_err(err)?;
    let mut samples = Vec::with_capacity(iters);
    for s in 0..iters {
        samples.push(
            f64::from(
                gpu.hip
                    .event_elapsed_ms(&events[s], &events[s + 1])
                    .map_err(err)?,
            ) * 1000.0,
        );
    }
    samples.sort_by(f64::total_cmp);
    let pct = |p: f64| samples[((samples.len() - 1) as f64 * p).round() as usize];
    let median = pct(0.5);
    Ok(
        json!({"route":route.label(),"R":rows,"temperature":if cold {"cold"} else {"warm"},
        "diagnostic_only":true,"median_us":median,"us_per_row":median/rows as f64,
        "p10_us":pct(0.1),"p90_us":pct(0.9),"sorted_samples_us":samples,
        "weight_copies":case.copies,"logical_w_bytes":case.group_bytes() *
            if matches!(route, Route::Single | Route::SingleAlt | Route::PmC1) {rows} else {1},
        "logical_x_bytes":rows * case.spec.k * 4,"logical_y_bytes":rows * case.y_elems() * 4,
        "cold_footprint_bytes":case.copies * case.group_bytes()}),
    )
}

fn sha_pin(value: &Value) -> Result<String> {
    let pin = value.as_str().ok_or("missing SHA256 string")?;
    if pin.len() != 64 || !pin.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid SHA256 pin".into());
    }
    Ok(pin.to_ascii_lowercase())
}

fn capture_rel(base: &std::path::Path, value: &Value) -> Result<PathBuf> {
    let raw = value
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or("capture path must be nonempty relative")?;
    let rel = std::path::Path::new(raw);
    if rel.is_absolute()
        || rel
            .components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return Err("capture/operand path must stay relative to its fixture directory".into());
    }
    Ok(base.join(rel))
}

fn capture_blob(
    base: &std::path::Path,
    descriptor: &Value,
    extent: Option<usize>,
) -> Result<Vec<u8>> {
    let path = capture_rel(base, &descriptor["path"])?;
    let bytes = std::fs::read(&path).map_err(err)?;
    let pin = sha_pin(&descriptor["sha256"])?;
    if let Some(n) = extent {
        if bytes.len() != n {
            return Err(format!(
                "{}: byte extent {} != {n}",
                path.display(),
                bytes.len()
            ));
        }
    }
    if sha256_hex(&bytes) != pin {
        return Err(format!("{}: SHA256 does not match {pin}", path.display()));
    }
    Ok(bytes)
}

/// Real-capture manifest + pinned source-index provenance for the gfx1201 byte
/// path. Seeded gates stay separate; this loader is strict and never skips.
/// `manifest_ms` preserves the original four source slots (zeros for empty);
/// `spec.ms` packs only live slots for the strict seeded byte path.
struct LoadedCapture {
    manifest_path: PathBuf,
    manifest_sha: String,
    source_path: PathBuf,
    source_sha: String,
    manifest_ms: [usize; 4],
    spec: Spec,
    weights: Vec<Vec<u8>>,
    host_x: Vec<f32>,
    singleton: Identity,
}

fn load_capture(path: &std::path::Path) -> Result<LoadedCapture> {
    let manifest_bytes = std::fs::read(path).map_err(err)?;
    let manifest_sha = sha256_hex(&manifest_bytes);
    let manifest: Value = serde_json::from_slice(&manifest_bytes).map_err(err)?;
    if manifest["schema"] != "qwen4-rowtile-operands-v1"
        || manifest["arch"] != "gfx1201"
        || manifest["rows"] != MAXC as u64
    {
        return Err("capture requires qwen4-rowtile-operands-v1, exact gfx1201 and rows64".into());
    }
    let base = path.parent().unwrap_or(std::path::Path::new("."));
    let source_path = capture_rel(base, &manifest["provenance"]["source"])?;
    let source_bytes = std::fs::read(&source_path).map_err(err)?;
    let source_pin = sha_pin(&manifest["provenance"]["source_sha256"])?;
    if sha256_hex(&source_bytes) != source_pin {
        return Err("capture provenance SHA mismatch".into());
    }
    let source: Value = serde_json::from_slice(&source_bytes).map_err(err)?;
    if source["schema"] != "qwen4-rowtile-source-v1"
        || source["capture_mode"] != "eager-liveprojection-operands-only"
        || source["retained_identity_claim"] != false
    {
        return Err("capture lacks actual eager live-projection provenance".into());
    }
    for key in [
        "name", "family", "form", "ms", "k", "rows", "weights", "x", "hc_act", "scale",
    ] {
        if manifest.get(key).is_none() || source.get(key) != manifest.get(key) {
            return Err(format!("capture/source binding differs at {key}"));
        }
    }
    let name = manifest["name"]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or("missing capture name")?
        .to_string();
    let family = manifest["family"]
        .as_str()
        .ok_or("missing capture family")?
        .to_string();
    let form = manifest["form"]
        .as_str()
        .ok_or("missing capture form")?
        .to_string();
    let k = manifest["k"]
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .filter(|&v| v > 0 && v <= i32::MAX as usize)
        .ok_or("invalid capture K")?;
    let heights = manifest["ms"]
        .as_array()
        .filter(|a| a.len() == 4)
        .ok_or("capture ms needs four slots")?;
    let mut ms = [0usize; 4];
    for (m, value) in ms.iter_mut().zip(heights) {
        *m = value
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .filter(|&v| v <= i32::MAX as usize)
            .ok_or("invalid capture M")?;
    }
    if family == "q8" {
        return Err(
            "gfx1201 has no qualified Q8 PM twin; Q8 captures belong to the Q8 reader".into(),
        );
    }
    let hc_act = manifest["hc_act"]
        .as_bool()
        .ok_or("capture needs hc_act bool")?;
    let scale = if manifest["scale"].is_null() {
        None
    } else {
        let value = manifest["scale"].as_f64().ok_or("invalid HC scale")? as f32;
        if !value.is_finite() {
            return Err("nonfinite HC scale is not admitted".into());
        }
        Some(value)
    };
    if hc_act != scale.is_some() {
        return Err("HC activation/optional scale binding disagrees".into());
    }
    let fam = match family.as_str() {
        "mq6" => Fam::Mq6,
        "bf16" => Fam::Bf16,
        _ => return Err("capture family not admitted here".into()),
    };
    let spec = match (fam, form.as_str()) {
        (Fam::Mq6, "single") => {
            if ms[0] == 0 || ms[1..].iter().any(|&m| m != 0) || !k.is_multiple_of(256) {
                return Err("invalid captured MQ6 single shape".into());
            }
            Spec {
                fam,
                name: format!("captured {name} single {}x{k}", ms[0]),
                kind: Kind::Single,
                ms: vec![ms[0]],
                k,
                timed: false,
                ctl: None,
            }
        }
        (Fam::Mq6, "x4") => {
            if ms.iter().all(|&m| m == 0) || !k.is_multiple_of(256) {
                return Err("invalid captured MQ6 x4 shape".into());
            }
            let live: Vec<usize> = ms.iter().copied().filter(|&m| m > 0).collect();
            Spec {
                fam,
                name: format!("captured {name} x4 {ms:?} k={k}"),
                kind: Kind::X4,
                ms: live,
                k,
                timed: false,
                ctl: None,
            }
        }
        (Fam::Bf16, "single") => {
            if ms[0] == 0 || ms[1..].iter().any(|&m| m != 0) {
                return Err("invalid captured BF16 single shape".into());
            }
            Spec {
                fam,
                name: format!("captured {name} single {}x{k}", ms[0]),
                kind: Kind::Single,
                ms: vec![ms[0]],
                k,
                timed: false,
                ctl: None,
            }
        }
        (Fam::Bf16, "x4") => {
            if ms.iter().all(|&m| m == 0) {
                return Err("invalid captured BF16 x4 shape".into());
            }
            let live: Vec<usize> = ms.iter().copied().filter(|&m| m > 0).collect();
            Spec {
                fam,
                name: format!("captured {name} x4 {ms:?} k={k}"),
                kind: Kind::X4,
                ms: live,
                k,
                timed: false,
                ctl: None,
            }
        }
        (Fam::Bf16, "hc_k4") => {
            if ms[0] == 0 || ms[1..].iter().any(|&m| m != 0) || k % 32 != 0 {
                return Err("invalid captured BF16 k4 shape".into());
            }
            let expected = if hc_act {
                Some(scale.ok_or("HC scale missing")?)
            } else {
                None
            };
            if scale != expected {
                return Err("HC scale binding disagrees".into());
            }
            Spec {
                fam,
                name: format!(
                    "captured {name} k4 {}x{k} scale={scale:?} hc_act={hc_act}",
                    ms[0]
                ),
                kind: Kind::K4(scale),
                ms: vec![ms[0]],
                k,
                timed: false,
                ctl: None,
            }
        }
        _ => {
            return Err(format!(
                "capture route not admitted here: {family}/{form} ms={ms:?} K={k}"
            ))
        }
    };
    let captures = source["captures"]
        .as_array()
        .filter(|a| a.len() == MAXC)
        .ok_or("capture needs all64 row recipes")?;
    let start = captures[0]["position"]
        .as_u64()
        .ok_or("missing capture position")?;
    let occurrence = captures[0]["occurrence"]
        .as_u64()
        .ok_or("missing capture occurrence")?;
    for (row, recipe) in captures.iter().enumerate() {
        if recipe["row"].as_u64() != Some(row as u64)
            || recipe["position"].as_u64() != start.checked_add(row as u64)
            || recipe["occurrence"].as_u64() != Some(occurrence)
            || recipe["token"].as_u64().is_none_or(|v| v > u32::MAX as u64)
        {
            return Err(format!(
                "capture row {row} lacks stable sequential singleton binding"
            ));
        }
        for field in ["grid", "block"] {
            let dims = recipe[field]
                .as_array()
                .filter(|a| a.len() == 3)
                .ok_or("missing singleton launch geometry")?;
            if dims
                .iter()
                .any(|v| v.as_u64().is_none_or(|n| n == 0 || n > u32::MAX as u64))
                || recipe[field] != captures[0][field]
            {
                return Err("capture singleton launch geometry changed across rows".into());
            }
        }
        if recipe["shared_mem"].as_u64().is_none()
            || recipe["shared_mem"] != captures[0]["shared_mem"]
        {
            return Err("capture lacks stable singleton LDS recipe".into());
        }
    }
    let descriptors = manifest["weights"]
        .as_array()
        .filter(|a| a.len() == 4)
        .ok_or("capture needs four weight descriptors")?;
    let bindings = source["weight_bindings"]
        .as_array()
        .filter(|a| a.len() == 4)
        .ok_or("capture needs four immutable weight bindings")?;
    let mut weights: Vec<Vec<u8>> = Vec::with_capacity(4);
    for i in 0..4 {
        let extent = match fam {
            Fam::Mq6 => ms[i].checked_mul(k / 256).and_then(|v| v.checked_mul(200)),
            Fam::Bf16 => ms[i].checked_mul(k).and_then(|v| v.checked_mul(2)),
            Fam::Q8 => None,
        }
        .filter(|&n| n <= u32::MAX as usize)
        .ok_or("capture weight byte extent overflow")?;
        if bindings[i]["bytes"].as_u64() != Some(extent as u64)
            || bindings[i]["byte_offset"].as_u64().is_none()
            || !(bindings[i]["name"].is_null() || bindings[i]["name"].is_string())
            || !(bindings[i]["layer"].is_null() || bindings[i]["layer"].is_i64())
        {
            return Err("capture immutable weight binding/extent missing".into());
        }
        if ms[i] == 0 && extent != 0 {
            return Err("capture zero slot must carry zero weight bytes".into());
        }
        weights.push(capture_blob(base, &descriptors[i], Some(extent))?);
    }
    let x_bytes = k
        .checked_mul(MAXC)
        .and_then(|v| v.checked_mul(4))
        .filter(|&n| n <= u32::MAX as usize)
        .ok_or("capture activation extent overflow")?;
    let raw = capture_blob(base, &manifest["x"], Some(x_bytes))?;
    let mut host_x = Vec::with_capacity(MAXC * k);
    for word in raw.chunks_exact(4) {
        let value = f32::from_le_bytes([word[0], word[1], word[2], word[3]]);
        if !value.is_finite() {
            return Err("nonfinite live captured X".into());
        }
        host_x.push(value);
    }
    let singleton_value = &source["singleton"];
    let singleton = Identity {
        module: singleton_value["module"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or("missing actual singleton module")?
            .to_string(),
        symbol: singleton_value["symbol"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or("missing actual singleton symbol")?
            .to_string(),
        sha: sha_pin(&singleton_value["object_sha256"])?,
        container_sha: sha_pin(&singleton_value["container_sha256"])?,
    };
    let model_pin = sha_pin(&source["model"]["sha256"])?;
    let prompt_pin = sha_pin(&source["prompt"]["sha256"])?;
    println!("real-capture manifest={} manifest_sha256={manifest_sha} source_sha256={source_pin} model_sha256={model_pin} prompt_sha256={prompt_pin} source_singleton={}/{} object_sha256={} container_sha256={} retained_identity_claim=false",
        path.display(), singleton.module, singleton.symbol, singleton.sha, singleton.container_sha);
    Ok(LoadedCapture {
        manifest_path: path.to_path_buf(),
        manifest_sha,
        source_path: source_path.clone(),
        source_sha: source_pin,
        manifest_ms: ms,
        spec,
        weights,
        host_x,
        singleton,
    })
}
struct Args {
    out: PathBuf,
    pins_path: PathBuf,
    fams: Vec<Fam>,
    bytes: bool,
    timing: bool,
    seeds: Vec<u64>,
    iters: usize,
    cold_mib: usize,
    budget_s: u64,
    incumbent_only: bool,
    sweep: bool,
    captured: Vec<PathBuf>,
    captured_only: bool,
}
fn parse_args() -> Result<Args> {
    let mut a = Args {
        out: PathBuf::new(),
        pins_path: PathBuf::new(),
        fams: vec![Fam::Mq6, Fam::Bf16],
        bytes: true,
        timing: false,
        seeds: Vec::new(),
        iters: 50,
        cold_mib: 512,
        budget_s: 420,
        incumbent_only: false,
        sweep: false,
        captured: Vec::new(),
        captured_only: false,
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut val = || args.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--out" => a.out = PathBuf::from(val()?),
            "--singleton-pins" => a.pins_path = PathBuf::from(val()?),
            "--family" => {
                a.fams = match val()?.as_str() {
                    "mq6" => vec![Fam::Mq6],
                    "bf16" => vec![Fam::Bf16],
                    "q8" => vec![Fam::Q8],
                    "all" => vec![Fam::Mq6, Fam::Q8, Fam::Bf16],
                    s => return Err(format!("unknown family {s}")),
                }
            }
            "--mode" => {
                (a.bytes, a.timing) = match val()?.as_str() {
                    "bytes" => (true, false),
                    "timing" => (false, true),
                    "both" => (true, true),
                    s => return Err(format!("unknown mode {s}")),
                }
            }
            "--seed" => a.seeds.push(val()?.parse().map_err(err)?),
            "--iters" => a.iters = val()?.parse().map_err(err)?,
            "--cold-mib" => a.cold_mib = val()?.parse().map_err(err)?,
            "--budget-s" => a.budget_s = val()?.parse().map_err(err)?,
            "--incumbent-only" => a.incumbent_only = true,
            "--sweep-c" => a.sweep = true,
            "--captured" => a.captured.push(PathBuf::from(val()?)),
            "--captured-only" => a.captured_only = true,
            s => return Err(format!("unknown argument {s}")),
        }
    }
    if a.out.as_os_str().is_empty() || a.pins_path.as_os_str().is_empty() {
        return Err("required: --out DIR --singleton-pins reviewed.json".into());
    }
    if a.captured_only && (a.captured.is_empty() || !a.bytes || a.incumbent_only) {
        return Err(
            "--captured-only requires at least one --captured and candidate bytes mode".into(),
        );
    }
    if a.iters == 0 || a.cold_mib.checked_mul(1 << 20).is_none() {
        return Err("positive --iters and representable --cold-mib required".into());
    }
    if !a.incumbent_only && a.fams.contains(&Fam::Q8) {
        if a.fams.len() == 1 {
            return Err("gfx1201 Q8 has no qualified PM twin; use --incumbent-only for explicitly noncandidate diagnostics".into());
        }
        a.fams.retain(|f| *f != Fam::Q8);
        eprintln!("Q8 excluded from candidate families: no gfx1201 object proof, no Halo alias");
    }
    if a.seeds.is_empty() {
        a.seeds = vec![0x4831_3230_315f_4830, 0x9e37_79b9_7f4a_7c15];
    }
    Ok(a)
}

fn read_pins(a: &Args) -> Result<(BTreeMap<String, String>, String)> {
    let bytes = std::fs::read(&a.pins_path).map_err(err)?;
    let pins: BTreeMap<String, String> = serde_json::from_slice(&bytes).map_err(err)?;
    if pins.is_empty()
        || pins.iter().any(|(s, h)| {
            s.is_empty()
                || h.len() != 64
                || !h
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        })
    {
        return Err("singleton pins must be a nonempty symbol -> lowercase SHA256 JSON map".into());
    }
    Ok((pins, sha256_hex(&bytes)))
}

fn run() -> Result<i32> {
    let a = parse_args()?;
    let (pins, pins_sha) = read_pins(&a)?;
    let mut gpu = Gpu::init().map_err(err)?;
    if gpu.arch.as_str() != "gfx1201" {
        return Err(format!("requires exact gfx1201; got {}", gpu.arch.as_str()));
    }
    gpu.lane_rows_scope = false;
    std::fs::create_dir_all(&a.out).map_err(err)?;
    let events: Vec<Event> = if a.timing {
        (0..=a.iters)
            .map(|_| gpu.hip.event_create().map_err(err))
            .collect::<Result<_>>()?
    } else {
        Vec::new()
    };
    let start = Instant::now();
    let mut shapes = Vec::new();
    let mut candidate_failures = 0usize;
    let mut diagnostic_failures = 0usize;
    let mut setup_errors = 0usize;
    let mut timing_skipped = Vec::new();
    let mut timing_elapsed_s = 0.0;
    let mut planned = 0usize;
    for &seed in a.seeds.iter().filter(|_| !a.captured_only) {
        for &fam in &a.fams {
            for spec in specs(fam) {
                planned += 1;
                let timed = a.timing && spec.timed;
                let copies_bytes = if timed { a.cold_mib * (1 << 20) } else { 0 };
                let case = build_case(&mut gpu, &spec, seed, copies_bytes)?;
                let mut shape = json!({"family":fam.tag(),"name":spec.name,"seed":seed,
                    "ms":spec.ms,"K":spec.k,"kind":format!("{:?}",spec.kind),
                    "weight_bytes":case.group_bytes(),"weight_copies":case.copies,
                    "all_empty_singleton_launch":"actual runtime object; grid.x=1 dead workgroup (runtime API grid would be zero)",
                    "all_empty":case.empty(),"cpu_object_emulation_proven":false,
                    "reference":Value::Null,"candidate_identity":Value::Null,
                    "bytes":[],"timing":[],"error":Value::Null});
                let reference = reference(&mut gpu, &case, &pins);
                let mut shape_bad = false;
                match reference {
                    Err(e) => {
                        setup_errors += 1;
                        shape["error"] = json!(format!("runtime reference refusal: {e}"));
                    }
                    Ok(actual) => {
                        shape["reference"] = json!({"identity":actual.identity.json(),
                            "reviewed_singleton_pin_match":true,"deterministic_repeat":true,
                            "guarded_input_sha256":actual.x_sha,
                            "oracle":"actual architecture runtime singleton object; independent row calls"});
                        let tile = if a.incumbent_only {
                            Ok(None)
                        } else {
                            load_tile(&mut gpu, &spec).map(Some)
                        };
                        match tile {
                            Err(e) => {
                                setup_errors += 1;
                                shape["error"] = json!(format!("candidate object refusal: {e}"));
                            }
                            Ok(tile) => {
                                shape["candidate_identity"] =
                                    tile.as_ref().map_or(Value::Null, Identity::json);
                                if a.bytes {
                                    match bytes_phase(&mut gpu, &case, &actual, tile.as_ref()) {
                                        Err(e) => {
                                            if a.incumbent_only {
                                                setup_errors += 1;
                                            } else {
                                                candidate_failures += 1;
                                            }
                                            shape_bad = true;
                                            shape["error"] = json!(format!("byte arm failed: {e}"));
                                        }
                                        Ok(cells) => {
                                            let reached: Vec<usize> = cells
                                                .iter()
                                                .filter(|c| {
                                                    c.route == Route::Tile
                                                        && c.value["launched"] == true
                                                })
                                                .map(|c| c.rows)
                                                .collect();
                                            let bad = cells
                                                .iter()
                                                .filter(|c| c.route.candidate() && !c.ok)
                                                .count();
                                            candidate_failures += bad;
                                            diagnostic_failures += cells
                                                .iter()
                                                .filter(|c| !c.route.candidate() && !c.ok)
                                                .count();
                                            let mutations = cells
                                                .iter()
                                                .filter(|c| {
                                                    !c.value["input_immutability_error"].is_null()
                                                })
                                                .count();
                                            setup_errors += mutations;
                                            shape_bad = bad > 0 || mutations > 0;
                                            shape["candidate_widths_reached"] = json!(reached);
                                            shape["bytes"] = json!(cells
                                                .into_iter()
                                                .map(|c| c.value)
                                                .collect::<Vec<_>>());
                                        }
                                    }
                                }
                                if timed && !shape_bad {
                                    let mut times = Vec::new();
                                    for cold in [false, true] {
                                        for (route, rows) in
                                            timing_cells(a.sweep, !a.incumbent_only)
                                        {
                                            if timing_elapsed_s >= a.budget_s as f64 {
                                                timing_skipped.push(json!({"seed":seed,"name":spec.name,"route":route.label(),"R":rows,"cold":cold}));
                                                continue;
                                            }
                                            let timing_start = Instant::now();
                                            match time_cell(
                                                &mut gpu, &events, &case, route, rows, cold,
                                                a.iters,
                                            ) {
                                                Ok(cell) => times.push(cell),
                                                Err(e) => {
                                                    if route.candidate() {
                                                        candidate_failures += 1;
                                                    } else {
                                                        diagnostic_failures += 1;
                                                    }
                                                    times.push(json!({"route":route.label(),"R":rows,"cold":cold,"error":e,"diagnostic_only":true}));
                                                }
                                            }
                                            timing_elapsed_s +=
                                                timing_start.elapsed().as_secs_f64();
                                        }
                                    }
                                    shape["timing"] = json!(times);
                                }
                            }
                        }
                    }
                }
                eprintln!(
                    "{} seed={seed} {}",
                    spec.name,
                    if !shape["error"].is_null() {
                        "REFUSAL/ERROR"
                    } else if shape_bad {
                        "CANDIDATE FAIL"
                    } else if a.incumbent_only || !a.bytes {
                        "NONQUALIFYING DIAGNOSTIC"
                    } else {
                        "candidate bytes match"
                    }
                );
                shapes.push(shape);
                free_case(&mut gpu, case)?;
            }
        }
    }
    // Real-capture byte arms (full 1..64, strict seeded path, actual singleton
    // pins): SHA-pinned manifests only; no fake/random captures, no skip=PASS.
    let mut captured_shapes = 0usize;
    if !a.captured.is_empty() && !a.incumbent_only && a.bytes {
        for path in &a.captured {
            planned += 1;
            let mut shape = json!({"family":Value::Null,"name":path.display().to_string(),
                "seed":Value::Null,"manifest":path.display().to_string(),
                "reference":Value::Null,"candidate_identity":Value::Null,
                "bytes":[],"error":Value::Null,"origin":"real-capture"});
            let loaded = load_capture(path);
            let loaded = match loaded {
                Err(e) => {
                    setup_errors += 1;
                    shape["error"] = json!(format!("capture refusal: {e}"));
                    shapes.push(shape);
                    continue;
                }
                Ok(loaded) => loaded,
            };
            shape["family"] = json!(loaded.spec.fam.tag());
            shape["name"] = json!(format!(
                "real-capture {} {}",
                loaded.spec.fam.tag(),
                loaded.spec.name
            ));
            shape["ms"] = json!(loaded.spec.ms);
            shape["K"] = json!(loaded.spec.k);
            shape["kind"] = json!(format!("{:?}", loaded.spec.kind));
            shape["manifest_sha256"] = json!(loaded.manifest_sha);
            shape["source"] = json!(loaded.source_path.display().to_string());
            shape["source_sha256"] = json!(loaded.source_sha);
            shape["captured_singleton"] = loaded.singleton.json();
            let case = build_captured_case(&mut gpu, &loaded);
            let mut case = match case {
                Err(e) => {
                    setup_errors += 1;
                    shape["error"] = json!(format!("capture case refusal: {e}"));
                    shapes.push(shape);
                    continue;
                }
                Ok(case) => case,
            };
            // Strict source-singleton binding before the byte path: the actual
            // loaded object must equal the captured source identity, and the
            // singleton pin map must cover it.
            let bound = (|| -> Result<Identity> {
                if loaded.spec.fam == Fam::Bf16 && matches!(loaded.spec.kind, Kind::Single) {
                    // Same managed source/module load as the actual scalar C1
                    // route; captured-only has no incidental seeded preload.
                    gpu.ensure_kernel_public(
                        "gemv_bf16_xf32",
                        kernels::gemv_bf16_xf32_src(gpu.arch_caps.is_gfx1151()),
                        "gemv_bf16_xf32",
                    )
                    .map_err(err)?;
                }
                let actual = identity(&gpu, &loaded.singleton.symbol)?;
                if actual != loaded.singleton {
                    return Err(format!(
                        "captured singleton identity differs: actual {actual:?} captured {:?}",
                        loaded.singleton
                    ));
                }
                let want = pins.get(&loaded.singleton.symbol).ok_or_else(|| {
                    format!(
                        "missing reviewed singleton pin: {}",
                        loaded.singleton.symbol
                    )
                })?;
                if &actual.sha != want {
                    return Err(format!(
                        "captured singleton SHA {} != reviewed {want}",
                        actual.sha
                    ));
                }
                Ok(actual)
            })();
            let bound = match bound {
                Err(e) => {
                    setup_errors += 1;
                    shape["error"] = json!(format!("capture singleton refusal: {e}"));
                    free_case(&mut gpu, case)?;
                    shapes.push(shape);
                    continue;
                }
                Ok(bound) => bound,
            };
            let reference = reference(&mut gpu, &case, &pins);
            let mut shape_bad = false;
            match reference {
                Err(e) => {
                    setup_errors += 1;
                    shape["error"] = json!(format!("runtime reference refusal: {e}"));
                }
                Ok(actual) => {
                    if actual.identity != bound {
                        setup_errors += 1;
                        shape["error"] =
                            json!("captured source identity differs from live reference route");
                    } else {
                        shape["reference"] = json!({"identity": actual.identity.json(),
                            "reviewed_singleton_pin_match": true, "deterministic_repeat": true,
                            "guarded_input_sha256": actual.x_sha,
                            "oracle": "actual architecture runtime singleton object; independent row calls; captured operands"});
                        let tile = load_tile(&mut gpu, &case.spec).map(Some);
                        match tile {
                            Err(e) => {
                                setup_errors += 1;
                                shape["error"] = json!(format!("candidate object refusal: {e}"));
                            }
                            Ok(tile) => {
                                shape["candidate_identity"] =
                                    tile.as_ref().map_or(Value::Null, Identity::json);
                                match bytes_phase(&mut gpu, &case, &actual, tile.as_ref()) {
                                    Err(e) => {
                                        candidate_failures += 1;
                                        shape_bad = true;
                                        shape["error"] = json!(format!("byte arm failed: {e}"));
                                    }
                                    Ok(cells) => {
                                        let reached: Vec<usize> = cells
                                            .iter()
                                            .filter(|c| {
                                                c.route == Route::Tile
                                                    && c.value["launched"] == true
                                            })
                                            .map(|c| c.rows)
                                            .collect();
                                        let bad = cells
                                            .iter()
                                            .filter(|c| c.route.candidate() && !c.ok)
                                            .count();
                                        candidate_failures += bad;
                                        diagnostic_failures += cells
                                            .iter()
                                            .filter(|c| !c.route.candidate() && !c.ok)
                                            .count();
                                        let mutations = cells
                                            .iter()
                                            .filter(|c| {
                                                !c.value["input_immutability_error"].is_null()
                                            })
                                            .count();
                                        setup_errors += mutations;
                                        shape_bad = bad > 0 || mutations > 0;
                                        if reached.len() != MAXC
                                            || reached.iter().enumerate().any(|(i, &r)| r != i + 1)
                                        {
                                            candidate_failures += 1;
                                            shape_bad = true;
                                            shape["error"] = json!(
                                                "real-capture reader did not reach full 1..=64"
                                            );
                                        }
                                        shape["candidate_widths_reached"] = json!(reached);
                                        shape["bytes"] = json!(cells
                                            .into_iter()
                                            .map(|c| c.value)
                                            .collect::<Vec<_>>());
                                    }
                                }
                            }
                        }
                    }
                }
            }
            eprintln!(
                "real-capture {} {}",
                loaded.spec.name,
                if !shape["error"].is_null() {
                    "REFUSAL/ERROR"
                } else if shape_bad {
                    "CANDIDATE FAIL"
                } else {
                    "candidate bytes match"
                }
            );
            shapes.push(shape);
            captured_shapes += 1;
            free_case(&mut gpu, case)?;
        }
    } else if !a.captured.is_empty() && (a.incumbent_only || !a.bytes) {
        return Err("real-capture arms require candidate bytes mode (no --incumbent-only, --mode bytes|both)".into());
    }
    let qualified = a.bytes
        && !a.incumbent_only
        && candidate_failures == 0
        && setup_errors == 0
        && shapes.len() == planned
        && planned > 0;
    let status = if setup_errors > 0 {
        2
    } else if candidate_failures > 0 {
        3
    } else {
        0
    };
    let report = json!({"schema":"gfx1201_lane_rowtiles_h0.v2","tool":"lane_rows_pm_gfx1201_h0",
        "arch":gpu.arch.as_str(),"row_exact_component_qualified":qualified,
        "qualification_scope":if a.captured_only {
            "supplied real captures only; not the full original seeded-plus-captured qualification"
        } else {
            "listed seeded-stress and controlled HC cases plus supplied real captures; no full-model claim"
        },
        "captured_only":a.captured_only,
        "cpu_emulator_proof":false,"performance_qualified":false,"timing_diagnostic_only":true,
        "incumbent_only_noncandidate":a.incumbent_only,"byte_arm_enabled":a.bytes,
        "seeds":a.seeds,"singleton_pins_file":a.pins_path,"singleton_pins_file_sha256":pins_sha,
        "reviewed_singleton_pins":pins,"planned_shapes":planned,"completed_shapes":shapes.len(),
        "captured_manifests":a.captured.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
        "captured_shapes":captured_shapes,
        "candidate_failures":candidate_failures,"diagnostic_failures":diagnostic_failures,
        "setup_errors":setup_errors,"timing_skipped_by_budget":timing_skipped,
        "elapsed_s":start.elapsed().as_secs_f64(),"exit_status":status,"shapes":shapes});
    std::fs::write(
        a.out.join("h0.json"),
        serde_json::to_string_pretty(&report).map_err(err)?,
    )
    .map_err(err)?;
    let mut md = format!("# gfx1201 lane-row component gate\n\nComponent byte-qualified: **{qualified}**. Exit: {status}.\n\nActual reviewed runtime singleton object is the reference; independent PM t8 C1 is additional.\nCPU/emulator proof: not claimed. Performance qualification: not claimed.\n\nCandidate failures: {candidate_failures}; refusals: {setup_errors}; noncandidate diagnostic differences: {diagnostic_failures}.\n\n| Family | Shape | Seed | Candidate widths reached | Error |\n|---|---|---:|---|---|\n");
    for shape in report["shapes"].as_array().ok_or("missing shape report")? {
        writeln!(
            md,
            "| {} | {} | {} | {} | {} |",
            shape["family"],
            shape["name"],
            shape["seed"],
            shape["candidate_widths_reached"],
            shape["error"]
        )
        .map_err(err)?;
    }
    std::fs::write(a.out.join("H0.md"), md).map_err(err)?;
    Ok(status)
}
fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("lane_rows_pm_gfx1201_h0: refusal: {e}");
            std::process::exit(2);
        }
    }
}
