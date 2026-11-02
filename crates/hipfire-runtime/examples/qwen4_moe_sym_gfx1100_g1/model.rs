// SPDX-License-Identifier: Apache-2.0
//! CPU model for the gfx1100 K2 G1 oracle: geometry, tensor layouts (QT44/QT53 anchor and MQ-I4 planar),
//! independent signed-weight fold, A4 activation sets, expected-output assembly through
//! the gfx1100 adapter `epilogue::grouped_prefill_epilogue` (the public `grouped_prefill_epilogue_with` plus the reviewed
//! device `Finish::MulUp` NaN rule), per-class comparison, and the mutated CPU "kernels" used by `--self-check`.
//! Nothing here reads a candidate kernel's output to form an expectation.

use half::f16;
use hipfire_runtime::e8p_i4_reference as e8p;
use hipfire_runtime::mq_i4::{self, MqI4Codec, MqI4Profile, MqI4Rotation};
use hipfire_runtime::mq_i4_reference::{self, A4Block, DecodeKind, EpilogueIntrinsics};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub type Res<T> = Result<T, String>;
pub fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

pub const BLK: usize = 72;
pub const POISON32: u32 = 0xa5a5_a5a5;
pub const POISON16: u16 = 0xa5a5;
pub const SLAB_ALIGN: usize = 256;

pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

pub struct Rng(pub u32);
impl Rng {
    pub fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
    pub fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Codec {
    Mq4,
    W2,
    W3,
    /// E8P-I4 (`qwen4_moe_*_iu4_sym_e8p_pm_gfx1100`): planar `[M*K/8 u16 words][M*K/128 f16 scales]`, K-E0 arithmetic.
    E8p,
}

impl Codec {
    pub fn parse(s: &str) -> Res<Self> {
        match s {
            "mq4" => Ok(Self::Mq4),
            "w2" => Ok(Self::W2),
            "w3" => Ok(Self::W3),
            "e8p" => Ok(Self::E8p),
            _ => Err(format!("--codec must be mq4|w2|w3|e8p, got {s:?}")),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Mq4 => "mq4",
            Self::W2 => "w2",
            Self::W3 => "w3",
            Self::E8p => "e8p",
        }
    }
    /// Text inserted after `sym` in the entry names.
    pub fn infix(self) -> &'static str {
        match self {
            Self::Mq4 => "",
            Self::W2 => "_w2",
            Self::W3 => "_w3",
            Self::E8p => "_e8p",
        }
    }
    /// MQ-I4 planar codec (W2/W3 only; E8P has its own words/scales planes and MQ4 is the QT layout).
    pub fn planar(self) -> Option<MqI4Codec> {
        match self {
            Self::Mq4 | Self::E8p => None,
            Self::W2 => Some(MqI4Codec::W2),
            Self::W3 => Some(MqI4Codec::W3),
        }
    }
    /// `L`: planar weight is `2c - L`.
    pub fn levels(self) -> i32 {
        match self {
            Self::Mq4 | Self::E8p => 0,
            Self::W2 => 3,
            Self::W3 => 7,
        }
    }
    /// Allowed signed weights (MQ4 uses the complete -8..=7 domain). E8P has no per-weight set (codeword lattice).
    pub fn weight_set(self) -> Vec<i32> {
        match self {
            Self::Mq4 => (-8..=7).collect(),
            Self::W2 => vec![-3, -1, 1, 3],
            Self::W3 => vec![-7, -5, -3, -1, 1, 3, 5, 7],
            Self::E8p => Vec::new(),
        }
    }
    /// Qualified quant-type bytes of the (gate_up, down) tensors.
    pub fn qt(self, kind: Kind) -> u8 {
        match (self, kind) {
            (Self::Mq4, Kind::GateUp) => 44,
            (Self::Mq4, Kind::Down) => 53,
            (Self::W3, Kind::GateUp) => 131,
            (Self::W3, Kind::Down) => 132,
            (Self::W2, Kind::GateUp) => 133,
            (Self::W2, Kind::Down) => 134,
            (Self::E8p, Kind::GateUp) => 135,
            (Self::E8p, Kind::Down) => 136,
        }
    }
    /// Negative controls this codec's G1 requires (the F32 gate detects all of them but the W2/W3 pack selector).
    pub fn negative_names(self) -> &'static [&'static str] {
        match self {
            Self::E8p => &[
                "wrong-lut",
                "wrong-sign-mask",
                "reversed-epoch",
                "wrong-fold-epoch",
                "wrong-scale-row",
                "wrong-expert-pointer",
            ],
            _ => &[
                "wrong-scale-row",
                "reversed-epoch",
                "wrong-pack-selector",
                "wrong-expert-pointer",
            ],
        }
    }
    pub fn mutations(self) -> &'static [Mutation] {
        match self {
            Self::E8p => &Mutation::E8P,
            _ => &Mutation::ALL,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    GateUp,
    Down,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Self::GateUp => "gate_up",
            Self::Down => "down",
        }
    }
    pub fn m(self) -> usize {
        match self {
            Self::GateUp => 1280,
            Self::Down => 2560,
        }
    }
    pub fn k(self) -> usize {
        match self {
            Self::GateUp => 2560,
            Self::Down => 640,
        }
    }
    pub fn out_cols(self) -> usize {
        match self {
            Self::GateUp => 640,
            Self::Down => 2560,
        }
    }
    pub fn groups(self) -> usize {
        self.k() / 128
    }
    pub fn threads(self) -> u32 {
        match self {
            Self::GateUp => 64,
            Self::Down => 128,
        }
    }
    pub fn grid_x(self) -> u32 {
        (self.m() / 64) as u32
    }
    pub fn decode_kind(self) -> DecodeKind {
        match self {
            Self::GateUp => DecodeKind::GateUp,
            Self::Down => DecodeKind::Down,
        }
    }
    pub fn rotation(self) -> MqI4Rotation {
        match self {
            Self::GateUp => MqI4Rotation::R256,
            Self::Down => MqI4Rotation::R128,
        }
    }
    /// QT44 (136 B per K256) / QT53 (68 B per K128) group bytes.
    pub fn qt_group_bytes(self) -> usize {
        match self {
            Self::GateUp => 136,
            Self::Down => 68,
        }
    }
    pub fn qt_len(self) -> usize {
        match self {
            Self::GateUp => self.m() * (self.groups() / 2) * 136,
            Self::Down => self.m() * self.groups() * 68,
        }
    }
    pub fn entry(self, codec: Codec, acc: bool) -> String {
        let base = match self {
            Self::GateUp => "qwen4_moe_gate_up_silu_iu4_sym",
            Self::Down => "qwen4_moe_down_iu4_sym",
        };
        format!(
            "{base}{}_pm_{}{}",
            codec.infix(),
            arch_name(),
            if acc { "_acc" } else { "" }
        )
    }
}

/// Target architecture of every `_pm_<arch>` kernel entry (`--expect-arch`; gfx1100 unless [`set_arch`] ran first).
static ARCH_NAME: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
pub fn set_arch(name: &'static str) {
    let _ = ARCH_NAME.set(name);
}
pub fn arch_name() -> &'static str {
    ARCH_NAME.get().copied().unwrap_or("gfx1100")
}

pub fn profile(planar: MqI4Codec, kind: Kind) -> MqI4Profile {
    MqI4Profile {
        codec: planar,
        rotation: kind.rotation(),
    }
}

pub fn f16f(bits: u16) -> f32 {
    f16::from_bits(bits).to_f32()
}

// ---------------------------------------------------------------------------------------------------------------
// QT44 / QT53 tensors (the independent MQ4 layout; also the same-device anchor encoding of W2/W3 values)
// ---------------------------------------------------------------------------------------------------------------

/// `(header offset, nibble-plane offset)` of K128 epoch `h` of `row`.
fn qt_loc(kind: Kind, row: usize, h: usize) -> (usize, usize) {
    let g = kind.groups();
    match kind {
        Kind::GateUp => {
            let base = row * (g / 2) * 136 + (h / 2) * 136;
            (base + 4 * (h % 2), base + 8 + 64 * (h % 2))
        }
        Kind::Down => {
            let base = row * g * 68 + h * 68;
            (base, base + 4)
        }
    }
}

/// Symmetric zero-point header: `-8 * sc` when exactly an f16, else 0. The `_sym` kernels never read it.
pub fn sym_zp(sc: u16) -> u16 {
    let v = -8.0f32 * f16f(sc);
    let h = f16::from_f32(v);
    if v.is_finite() && h.to_f32() == v {
        h.to_bits()
    } else {
        0
    }
}

pub fn qt_build(kind: Kind, w: &[i8], scales: &[u16]) -> Vec<u8> {
    let (m, k, g) = (kind.m(), kind.k(), kind.groups());
    let mut out = vec![0u8; kind.qt_len()];
    for r in 0..m {
        for h in 0..g {
            let (ho, no) = qt_loc(kind, r, h);
            let sc = scales[r * g + h];
            out[ho..ho + 2].copy_from_slice(&sc.to_le_bytes());
            out[ho + 2..ho + 4].copy_from_slice(&sym_zp(sc).to_le_bytes());
            for i in 0..128 {
                let nib = (i32::from(w[r * k + h * 128 + i]) + 8) as u8;
                out[no + i / 2] |= nib << (4 * (i % 2));
            }
        }
    }
    out
}

/// Parse raw QT44/QT53 bytes into signed weights (nibble - 8), scale bits and the count of headers whose zero
/// point is not the symmetric `-8*sc` (diagnostic only).
pub fn qt_parse(kind: Kind, bytes: &[u8]) -> Res<(Vec<i8>, Vec<u16>, usize)> {
    if bytes.len() != kind.qt_len() {
        return Err(format!(
            "QT tensor is {} bytes, expected {} for {}",
            bytes.len(),
            kind.qt_len(),
            kind.name()
        ));
    }
    let (m, k, g) = (kind.m(), kind.k(), kind.groups());
    let mut w = vec![0i8; m * k];
    let mut scales = vec![0u16; m * g];
    let mut asym = 0usize;
    for r in 0..m {
        for h in 0..g {
            let (ho, no) = qt_loc(kind, r, h);
            let sc = u16::from_le_bytes([bytes[ho], bytes[ho + 1]]);
            let zp = u16::from_le_bytes([bytes[ho + 2], bytes[ho + 3]]);
            scales[r * g + h] = sc;
            if zp != sym_zp(sc) {
                asym += 1;
            }
            for i in 0..128 {
                let byte = bytes[no + i / 2];
                let nib = if i % 2 == 0 { byte & 15 } else { byte >> 4 };
                w[r * k + h * 128 + i] = (nib as i32 - 8) as i8;
            }
        }
    }
    Ok((w, scales, asym))
}

// ---------------------------------------------------------------------------------------------------------------
// Experts
// ---------------------------------------------------------------------------------------------------------------

pub struct Expert {
    pub codec: Codec,
    pub kind: Kind,
    /// Candidate-codec bytes (QT44/QT53 for MQ4, planar MQ-I4 for W2/W3, `[words][scales]` for E8P).
    pub bytes: Vec<u8>,
    /// Same-device MQ4 anchor encoding of the same signed values (== `bytes` for MQ4; EMPTY for E8P: its `|v|` up to 11
    /// lies outside the MQ4 -8..7 domain, so no MQ4 anchor exists).
    pub anchor: Vec<u8>,
    /// Signed weights `[m*k]` decoded from `bytes` (empty for E8P).
    pub w: Vec<i8>,
    pub scales: Vec<u16>,
    /// Planar codes (empty for MQ4/E8P), the `iu4_gemm_f32_parts` input.
    pub codes: Vec<u8>,
    /// E8P wire words `[m*k/8]` (empty for every other codec).
    pub words: Vec<u16>,
    /// E8P scales failing `e8p::writer_guard` (non-finite, zero, subnormal, negative); 0 for every other codec.
    pub guard_trips: usize,
    pub asym_headers: usize,
    pub origin: String,
    pub sha256: String,
}

impl Expert {
    pub fn from_bytes(codec: Codec, kind: Kind, bytes: Vec<u8>, origin: String) -> Res<Self> {
        let sha256 = sha256_hex(&bytes);
        if codec == Codec::E8p {
            let (words, scales) = e8p_split(kind, &bytes)?;
            let guard_trips = scales
                .iter()
                .filter(|&&d| e8p::writer_guard(d).is_err())
                .count();
            return Ok(Self {
                codec,
                kind,
                bytes,
                anchor: Vec::new(),
                w: Vec::new(),
                scales,
                codes: Vec::new(),
                words,
                guard_trips,
                asym_headers: 0,
                origin,
                sha256,
            });
        }
        match codec.planar() {
            None => {
                let (w, scales, asym) = qt_parse(kind, &bytes)?;
                Ok(Self {
                    codec,
                    kind,
                    anchor: bytes.clone(),
                    bytes,
                    w,
                    scales,
                    codes: Vec::new(),
                    words: Vec::new(),
                    guard_trips: 0,
                    asym_headers: asym,
                    origin,
                    sha256,
                })
            }
            Some(pc) => {
                let t = mq_i4::decode_tensor(profile(pc, kind), kind.m(), kind.k(), &bytes)
                    .map_err(err)?;
                let l = codec.levels();
                let w: Vec<i8> = t
                    .codes
                    .iter()
                    .map(|&c| (2 * i32::from(c) - l) as i8)
                    .collect();
                let anchor = qt_build(kind, &w, &t.scale_bits);
                let (w2, s2, _) = qt_parse(kind, &anchor)?;
                if w2 != w || s2 != t.scale_bits {
                    return Err("anchor re-encoding does not round-trip the planar values".into());
                }
                Ok(Self {
                    codec,
                    kind,
                    bytes,
                    anchor,
                    w,
                    scales: t.scale_bits,
                    codes: t.codes,
                    words: Vec::new(),
                    guard_trips: 0,
                    asym_headers: 0,
                    origin,
                    sha256,
                })
            }
        }
    }

    /// E8P expert from wire words and FP16 scales (`words.len() == m*k/8`, `scales.len() == m*k/128`).
    pub fn from_e8p(kind: Kind, words: &[u16], scales: &[u16], origin: String) -> Res<Self> {
        let mut bytes = Vec::with_capacity(words.len() * 2 + scales.len() * 2);
        for w in words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        for s in scales {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        Self::from_bytes(Codec::E8p, kind, bytes, origin)
    }

    /// Encode signed weights + scales in the candidate codec.
    pub fn from_values(
        codec: Codec,
        kind: Kind,
        w: &[i8],
        scales: &[u16],
        origin: String,
    ) -> Res<Self> {
        let bytes = match codec.planar() {
            None => qt_build(kind, w, scales),
            Some(pc) => {
                let l = codec.levels();
                let codes: Vec<u8> = w
                    .iter()
                    .map(|&v| {
                        let c = (i32::from(v) + l) / 2;
                        debug_assert!((0..=l).contains(&c));
                        c as u8
                    })
                    .collect();
                mq_i4::encode_tensor(profile(pc, kind), kind.m(), kind.k(), &codes, scales)
                    .map_err(err)?
            }
        };
        Self::from_bytes(codec, kind, bytes, origin)
    }
}

/// Split E8P planar bytes `[m*k/8 u16 LE words][m*k/128 u16 LE FP16 scales]` (exact length required).
pub fn e8p_split(kind: Kind, bytes: &[u8]) -> Res<(Vec<u16>, Vec<u16>)> {
    let (nw, ns) = (kind.m() * kind.k() / 8, kind.m() * kind.k() / 128);
    if bytes.len() != 2 * (nw + ns) {
        return Err(format!(
            "E8P {} tensor is {} bytes, want {} ({nw} words + {ns} scales)",
            kind.name(),
            bytes.len(),
            2 * (nw + ns)
        ));
    }
    let u = |b: &[u8]| -> Vec<u16> {
        b.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    };
    Ok((u(&bytes[..2 * nw]), u(&bytes[2 * nw..])))
}

// ---------------------------------------------------------------------------------------------------------------
// Synthetic generators
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Var {
    Ties,
    SignedZero,
    RowId,
    Overflow,
}

#[derive(Clone, Copy, Debug)]
pub enum Gen {
    Random,
    Extreme,
    EveryCode,
    MaxDot,
    /// `scales[0]` forced (0xFF / 0x00 low-byte sentinels for the unaligned last-row chunk).
    Sentinel(u16),
    Targeted(Var),
}

pub const SCALE_POOL: [u16; 6] = [0x3000, 0x3800, 0x3c00, 0x4000, 0x2c00, 0x3555];
pub const FLT_MAX_BITS: u32 = 0x7f7f_ffff;
/// Activation scales of the exceptional x rows (finite max, +/-inf, quiet and signalling NaN payloads).
pub const EXC_D: [u32; 7] = [
    FLT_MAX_BITS,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0000,
    0x7fc1_2345,
    0x7f81_2345,
    0xffc5_4321,
];

/// Fixed epoch-0 activation values of the targeted-dot scenarios (-7..=7, both parities).
pub fn pattern_p() -> [i8; 128] {
    let mut p = [0i8; 128];
    for (j, v) in p.iter_mut().enumerate() {
        *v = (((j * 5 + 3) % 15) as i32 - 7) as i8;
    }
    p
}

const OFF: i32 = 7168;

/// Reachability DP: weights drawn from `set` against a fixed activation block; picks weights hitting a target
/// integer dot (or the nearest reachable one).
pub struct Solver {
    set: Vec<i32>,
    x: [i32; 128],
    reach: Vec<Vec<bool>>,
}

impl Solver {
    pub fn new(set: &[i32], x: &[i8; 128]) -> Self {
        let width = (2 * OFF + 1) as usize;
        let mut xi = [0i32; 128];
        for j in 0..128 {
            xi[j] = i32::from(x[j]);
        }
        let mut reach = vec![vec![false; width]; 129];
        reach[0][OFF as usize] = true;
        for j in 0..128 {
            let (a, b) = reach.split_at_mut(j + 1);
            let (cur, nxt) = (&a[j], &mut b[0]);
            for s in 0..width {
                if cur[s] {
                    for &wt in set {
                        let t = s as i32 + wt * xi[j];
                        if t >= 0 && (t as usize) < width {
                            nxt[t as usize] = true;
                        }
                    }
                }
            }
        }
        Self {
            set: set.to_vec(),
            x: xi,
            reach,
        }
    }

    fn nearest(&self, target: i32) -> i32 {
        let width = 2 * OFF + 1;
        for d in 0..=2 * OFF {
            for cand in [target + d, target - d] {
                let idx = cand + OFF;
                if idx >= 0 && idx < width && self.reach[128][idx as usize] {
                    return cand;
                }
            }
        }
        0
    }

    pub fn solve(&self, target: i32, rng: &mut Rng) -> (Vec<i32>, i32) {
        let width = (2 * OFF + 1) as usize;
        let t = self.nearest(target);
        let mut s = t + OFF;
        let mut out = vec![0i32; 128];
        for j in (0..128).rev() {
            let mut opts = Vec::new();
            for &wt in &self.set {
                let p = s - wt * self.x[j];
                if p >= 0 && (p as usize) < width && self.reach[j][p as usize] {
                    opts.push(wt);
                }
            }
            let pick = opts[rng.below(opts.len() as u32) as usize];
            out[j] = pick;
            s -= pick * self.x[j];
        }
        (out, t)
    }
}

pub fn target_for(var: Var, kind: Kind, r: usize) -> i32 {
    let oc = kind.out_cols();
    let (idx, is_up) = match kind {
        Kind::GateUp => (r % oc, r >= oc),
        Kind::Down => (r, false),
    };
    match var {
        Var::Ties => {
            // exact BF16 ties (low half 0x8000) on both LSB parities and both signs. Planar codecs against
            // pattern_p() (odd activation sum) only reach odd dots, i.e. the 257..511 tie family; the even
            // 514/518/1028/1036 family is reachable for MQ4. Unreachable targets move to the nearest reachable dot.
            const T: [i32; 18] = [
                514, 518, -514, -518, 257, 259, -257, -259, 1028, 1036, -1028, -1036, 0, 261, -261,
                2, -2, 263,
            ];
            T[(idx + if is_up { 5 } else { 0 }) % 18]
        }
        Var::SignedZero => {
            const NEG: [i32; 6] = [-1, -2, -514, -518, -1028, -1036];
            let z = (idx / 16) % 2 == 0;
            match kind {
                Kind::Down => {
                    if idx % 2 == 0 {
                        0
                    } else {
                        NEG[idx % 6]
                    }
                }
                Kind::GateUp => {
                    if z != is_up {
                        0
                    } else {
                        NEG[idx % 6]
                    }
                }
            }
        }
        Var::RowId => ((r as i32 % 1000) - 500) * 2,
        Var::Overflow => [1, 2, -1, -2, 0, 3][idx % 6],
    }
}

pub fn gen_expert(codec: Codec, kind: Kind, gen: Gen, seed: u32) -> Res<Expert> {
    if codec == Codec::E8p {
        return crate::e8p::gen_expert_e8p(kind, gen, seed);
    }
    let (m, k, g) = (kind.m(), kind.k(), kind.groups());
    let mut rng = Rng(seed | 1);
    let set = codec.weight_set();
    let (lo, hi) = (set[0], *set.last().unwrap());
    let mid_w: i32 = match codec.planar() {
        Some(_) => 1, // midpoint code 2^(b-1) <=> weight +1 for both planar codecs
        None => 0,
    };
    let mut w: Vec<i8> = (0..m * k)
        .map(|_| set[rng.below(set.len() as u32) as usize] as i8)
        .collect();
    let mut scales: Vec<u16> = (0..m * g)
        .map(|_| SCALE_POOL[rng.below(SCALE_POOL.len() as u32) as usize])
        .collect();
    match gen {
        Gen::Random => {}
        Gen::Extreme => {
            for (i, v) in w.iter_mut().enumerate() {
                *v = if (i / 3) % 2 == 0 { lo as i8 } else { hi as i8 };
            }
            for (i, s) in scales.iter_mut().enumerate() {
                *s = [0x7bff, 0x0001, 0x0400, 0x3c00, 0x03ff][i % 5];
            }
            // zero-scale groups: planar codecs must hold only the midpoint code
            for r in (0..m).step_by(97) {
                for gg in 0..g {
                    if gg % 5 == 0 {
                        scales[r * g + gg] = 0;
                        if codec.planar().is_some() {
                            for j in 0..128 {
                                w[r * k + gg * 128 + j] = mid_w as i8;
                            }
                        }
                    }
                }
            }
        }
        Gen::EveryCode => {
            for (i, v) in w.iter_mut().enumerate() {
                *v = set[(i + i / k) % set.len()] as i8;
            }
        }
        Gen::MaxDot => {
            for r in 0..m {
                let v = if r % 2 == 0 { lo } else { hi };
                for j in 0..k {
                    w[r * k + j] = v as i8;
                }
            }
        }
        Gen::Sentinel(first) => {
            scales[0] = first;
        }
        Gen::Targeted(var) => {
            let solver = Solver::new(&set, &pattern_p());
            for r in 0..m {
                let target = target_for(var, kind, r);
                if target == 0 {
                    // exact zero for every codec: zero scale (planar: midpoint codes, weight +1) kills epoch 0
                    for j in 0..128 {
                        w[r * k + j] = mid_w as i8;
                    }
                    scales[r * g] = 0;
                    continue;
                }
                let (w0, _achieved) = solver.solve(target, &mut rng);
                for j in 0..128 {
                    w[r * k + j] = w0[j] as i8;
                }
                scales[r * g] = 0x3c00;
            }
        }
    }
    Expert::from_values(
        codec,
        kind,
        &w,
        &scales,
        format!("synthetic:{gen:?}:seed={seed}"),
    )
}

// ---------------------------------------------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------------------------------------------

pub fn block_from_values(d: f32, values: &[i8; 128]) -> A4Block {
    let mut qs = [0u8; 64];
    let mut s = 0i32;
    for (j, &v) in values.iter().enumerate() {
        qs[j / 2] |= ((v as u8) & 0xf) << (4 * (j % 2));
        s += i32::from(v);
    }
    A4Block { d, s, qs }
}

#[derive(Clone, Debug)]
pub enum X0 {
    None,
    /// Row 0: epoch 0 = `pattern_p()` with scale `d0`; epochs >= 1 zero blocks.
    Pattern {
        d0: f32,
    },
    /// Rows 0..EXC_D.len(): pattern_p epoch 0 with the exceptional scales.
    Exceptional,
}

impl X0 {
    fn block(&self, r: usize, g: usize) -> Option<A4Block> {
        let d = match self {
            X0::None => return None,
            X0::Pattern { d0 } => {
                if r == 0 {
                    *d0
                } else {
                    return None;
                }
            }
            X0::Exceptional => {
                if r < EXC_D.len() {
                    f32::from_bits(EXC_D[r])
                } else {
                    return None;
                }
            }
        };
        Some(if g == 0 {
            block_from_values(d, &pattern_p())
        } else {
            block_from_values(1.0, &[0i8; 128])
        })
    }
}

#[derive(Clone)]
pub struct XSet {
    /// Token-major `[rows][groups]`.
    pub tok: Vec<A4Block>,
    pub rows: usize,
    pub groups: usize,
}

pub fn make_x(groups: usize, rows: usize, seed: u32, x0: &X0) -> XSet {
    let mut rng = Rng(seed | 1);
    let ds = [1.0f32, 0.5, 1.0 / 64.0, 1.0e-3, 3.0, 0.0, 65536.0, 1.0e-20];
    let mut tok = Vec::with_capacity(rows * groups);
    for r in 0..rows {
        for g in 0..groups {
            if let Some(b) = x0.block(r, g) {
                tok.push(b);
                continue;
            }
            let mut vals = [0i8; 128];
            for (j, v) in vals.iter_mut().enumerate() {
                *v = match (r + g) % 9 {
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
            }
            tok.push(block_from_values(ds[(r * 3 + g) % ds.len()], &vals));
        }
    }
    XSet { tok, rows, groups }
}

/// Epoch-major `[g][row]` 72-byte blocks; rows not in `referenced` are 0xFF poison (NaN scale, junk nibbles).
pub fn epoch_major(x: &XSet, referenced: &[bool]) -> Vec<u8> {
    let (rows, groups) = (x.rows, x.groups);
    let mut out = vec![0u8; rows * groups * BLK];
    for g in 0..groups {
        for r in 0..rows {
            let o = (g * rows + r) * BLK;
            if !referenced[r] {
                out[o..o + BLK].fill(0xff);
                continue;
            }
            let b = &x.tok[r * groups + g];
            out[o..o + 4].copy_from_slice(&b.d.to_le_bytes());
            out[o + 4..o + 8].copy_from_slice(&b.s.to_le_bytes());
            out[o + 8..o + BLK].copy_from_slice(&b.qs);
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------------------------
// Folds
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
pub struct FoldOpts {
    /// Mutation: scale of row `r ^ 1` instead of `r`.
    pub scale_row_xor1: bool,
    /// Mutation: ascending-epoch order reversed.
    pub reverse_epochs: bool,
    /// E8P mutation: LUT row `idx ^ 1`.
    pub lut_xor1: bool,
    /// E8P mutation: sign-mask entry `sign7 ^ 1`.
    pub mask_xor1: bool,
    /// E8P mutation: activation sum `s` dropped in every epoch (signed dot and quarter multiplier kept: `2C/4` instead of `(2C+s)/4`).
    pub drop_fold_sum: bool,
}

/// Independent signed-weight fold: `sum = fma(RN(sc*d), float(C_h), sum)`, +0 start, ascending epochs, with the
/// exact integer dot `C_h = sum(w_i * x_i)` over each K128 epoch. Same algorithm as `rdna-compute`
/// `fold_reference` (full signed MQ4 domain), generalized to any signed weight array.
pub fn fold_generic(
    kind: Kind,
    e: &Expert,
    xrows: &[usize],
    x: &XSet,
    o: FoldOpts,
) -> Vec<Vec<f32>> {
    let (m, k, g) = (kind.m(), kind.k(), kind.groups());
    let n = xrows.len();
    let mut xs = vec![0i8; n * g * 128];
    let mut xd = vec![0f32; n * g];
    for (jj, &j) in xrows.iter().enumerate() {
        for h in 0..g {
            let b = &x.tok[j * g + h];
            xd[jj * g + h] = b.d;
            for i in 0..128 {
                xs[(jj * g + h) * 128 + i] = b.element(i);
            }
        }
    }
    let per_row: Vec<Vec<f32>> = (0..m)
        .into_par_iter()
        .map(|r| {
            let mut sums = vec![0f32; n];
            let order: Vec<usize> = if o.reverse_epochs {
                (0..g).rev().collect()
            } else {
                (0..g).collect()
            };
            let sr = if o.scale_row_xor1 { r ^ 1 } else { r };
            for &h in &order {
                let sc = f16f(e.scales[sr * g + h]);
                let wrow = &e.w[r * k + h * 128..r * k + h * 128 + 128];
                for jj in 0..n {
                    let xv = &xs[(jj * g + h) * 128..(jj * g + h) * 128 + 128];
                    let mut c = 0i32;
                    for i in 0..128 {
                        c += i32::from(wrow[i]) * i32::from(xv[i]);
                    }
                    let scale = sc * xd[jj * g + h];
                    sums[jj] = scale.mul_add(c as f32, sums[jj]);
                }
            }
            sums
        })
        .collect();
    (0..n)
        .map(|jj| (0..m).map(|r| per_row[r][jj]).collect())
        .collect()
}

/// Port of `rdna-compute/examples/qwen4_moe_sym.rs::fold_reference` straight from the raw QT44/QT53 bytes and the
/// raw A4 nibbles (header/nibble offsets via `qt_loc`, sparse zero-skipping integer dot, full signed -8..7 domain),
/// deliberately not reusing `qt_parse` / `fold_generic` so the MQ4 expectation has its own decode path.
pub fn fold_reference_port(kind: Kind, w: &[u8], xrows: &[usize], x: &XSet) -> Vec<Vec<f32>> {
    let (m, g) = (kind.m(), kind.groups());
    let n = xrows.len();
    let mut xs: Vec<Vec<(u8, i32)>> = vec![Vec::new(); g * n];
    let mut xd = vec![0f32; g * n];
    for h in 0..g {
        for (jj, &j) in xrows.iter().enumerate() {
            let b = &x.tok[j * g + h];
            xd[h * n + jj] = b.d;
            for i in 0..128usize {
                let byte = b.qs[i / 2];
                let nib = if i % 2 == 0 { byte & 15 } else { byte >> 4 };
                let v = ((nib << 4) as i8 >> 4) as i32;
                if v != 0 {
                    xs[h * n + jj].push((i as u8, v));
                }
            }
        }
    }
    let per_row: Vec<Vec<f32>> = (0..m)
        .into_par_iter()
        .map(|row| {
            let mut sums = vec![0f32; n];
            let mut ws = [0i32; 128];
            let mut wnz: Vec<(u8, i32)> = Vec::with_capacity(128);
            for h in 0..g {
                let (hdr, nb) = qt_loc(kind, row, h);
                let sc = f16f(u16::from_le_bytes([w[hdr], w[hdr + 1]]));
                wnz.clear();
                for i in 0..128usize {
                    let byte = w[nb + i / 2];
                    ws[i] = if i % 2 == 0 {
                        (byte & 15) as i32 - 8
                    } else {
                        (byte >> 4) as i32 - 8
                    };
                    if ws[i] != 0 {
                        wnz.push((i as u8, ws[i]));
                    }
                }
                for jj in 0..n {
                    let xl = &xs[h * n + jj];
                    let c: i32 = if xl.len() <= wnz.len() {
                        xl.iter().map(|&(i, v)| ws[i as usize] * v).sum()
                    } else {
                        let mut xv = [0i32; 128];
                        for &(i, v) in xl {
                            xv[i as usize] = v;
                        }
                        wnz.iter().map(|&(i, v)| xv[i as usize] * v).sum()
                    };
                    let scale = sc * xd[h * n + jj];
                    sums[jj] = scale.mul_add(c as f32, sums[jj]);
                }
            }
            sums
        })
        .collect();
    (0..n)
        .map(|jj| (0..m).map(|r| per_row[r][jj]).collect())
        .collect()
}

/// The CPU twin the spec names: `iu4_gemm_f32_parts` for W2/W3, the signed MQ4 fold for MQ4.
pub fn fold_twin(kind: Kind, e: &Expert, xrows: &[usize], x: &XSet) -> Vec<Vec<f32>> {
    if e.codec == Codec::E8p {
        return crate::e8p::fold_e8p(kind, e, xrows, x, FoldOpts::default(), true);
    }
    match e.codec.planar() {
        None => fold_reference_port(kind, &e.bytes, xrows, x),
        Some(pc) => {
            let (m, k, g) = (kind.m(), kind.k(), kind.groups());
            let blocks: Vec<A4Block> = xrows
                .iter()
                .flat_map(|&j| x.tok[j * g..(j + 1) * g].iter().copied())
                .collect();
            let p = profile(pc, kind);
            blocks
                .par_chunks(g)
                .map(|b| mq_i4_reference::iu4_gemm_f32_parts(p, m, k, &e.codes, &e.scales, b))
                .collect()
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Expected outputs
// ---------------------------------------------------------------------------------------------------------------

pub type Tile = (i32, [i32; 16]);

pub fn tile(expert: i32, slots: &[i32]) -> Tile {
    let mut t = [-1i32; 16];
    t[..slots.len()].copy_from_slice(slots);
    (expert, t)
}

pub fn tiles_for(expert: i32, slots: Vec<i32>) -> Vec<Tile> {
    slots.chunks(16).map(|c| tile(expert, c)).collect()
}

/// x rows referenced by live slots.
pub fn referenced_rows(tiles: &[Tile], div: usize, rows: usize) -> Vec<bool> {
    let mut r = vec![false; rows];
    for t in tiles {
        if t.0 >= 0 {
            for &s in &t.1 {
                if s >= 0 {
                    r[s as usize / div] = true;
                }
            }
        }
    }
    r
}

/// expert index (as used by `expert_of`) -> sorted x rows needed.
pub fn needed_pairs(
    tiles: &[Tile],
    div: usize,
    expert_of: &dyn Fn(i32) -> usize,
) -> BTreeMap<usize, Vec<usize>> {
    let mut set: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for t in tiles {
        if t.0 >= 0 {
            for &s in &t.1 {
                if s >= 0 {
                    set.entry(expert_of(t.0))
                        .or_default()
                        .insert(s as usize / div);
                }
            }
        }
    }
    set.into_iter()
        .map(|(e, rows)| (e, rows.into_iter().collect()))
        .collect()
}

pub struct Expectation {
    pub m_total: usize,
    /// 0 live slot, 1 padding slot in a live tile, 2 sentinel tile.
    pub row_class: Vec<u8>,
    pub acc: Vec<u32>,
    pub h: Vec<u16>,
    pub coverage: Value,
}

pub type Dense = HashMap<(usize, usize), Vec<f32>>;

pub fn build_dense(
    kind: Kind,
    experts: &[Expert],
    x: &XSet,
    pairs: &BTreeMap<usize, Vec<usize>>,
    f: &dyn Fn(Kind, &Expert, &[usize], &XSet) -> Vec<Vec<f32>>,
) -> Dense {
    let mut dense = HashMap::new();
    for (&ei, rows) in pairs {
        let res = f(kind, &experts[ei], rows, x);
        for (jj, &j) in rows.iter().enumerate() {
            dense.insert((ei, j), res[jj].clone());
        }
    }
    dense
}

fn bf16_class(
    bits: u32,
    tie_even: &mut u64,
    tie_odd: &mut u64,
    nonfinite: &mut u64,
    subnormal: &mut u64,
) {
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        *nonfinite += 1;
        return;
    }
    if bits & 0x7f80_0000 == 0 && bits & 0x007f_ffff != 0 {
        *subnormal += 1;
    }
    if bits & 0xffff == 0x8000 {
        if (bits >> 16) & 1 == 0 {
            *tie_even += 1;
        } else {
            *tie_odd += 1;
        }
    }
}

/// Assemble expected `_acc` words and BF16 bytes. `hooks` is the ONLY source of exp2/ldexp/divide results; GateUp
/// NaN-result outputs follow the reviewed MulUp rule of `epilogue` (everything else is the public twin).
/// `pack_swap` is a self-check mutation (adjacent output pair swapped).
pub fn assemble<I: EpilogueIntrinsics>(
    kind: Kind,
    tiles: &[Tile],
    div: usize,
    dense: &Dense,
    expert_of: &dyn Fn(i32) -> usize,
    hooks: &I,
    pack_swap: bool,
) -> Expectation {
    let (m, oc) = (kind.m(), kind.out_cols());
    let m_total = tiles.len() * 16;
    let mut row_class = vec![0u8; m_total];
    let mut acc = vec![0u32; m_total * m];
    let mut h = vec![0u16; m_total * oc];
    let (mut tie_even, mut tie_odd, mut nonfinite, mut subnormal) = (0u64, 0u64, 0u64, 0u64);
    let (
        mut neg_zero_out,
        mut pos_zero_out,
        mut inf_out,
        mut nan_out,
        mut live_rows,
        mut pad_rows,
        mut sentinel_tiles,
    ) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    let (mut neg_sums, mut pos_sums) = (0u64, 0u64);
    for (ti, t) in tiles.iter().enumerate() {
        if t.0 < 0 {
            sentinel_tiles += 1;
            for lane in 0..16 {
                let p = ti * 16 + lane;
                row_class[p] = 2;
                acc[p * m..(p + 1) * m].fill(POISON32);
                h[p * oc..(p + 1) * oc].fill(POISON16);
            }
            continue;
        }
        let ei = expert_of(t.0);
        let mut sums = vec![0f32; 16 * m];
        for lane in 0..16 {
            let p = ti * 16 + lane;
            if t.1[lane] >= 0 {
                let v = &dense[&(ei, t.1[lane] as usize / div)];
                sums[lane * m..(lane + 1) * m].copy_from_slice(v);
                for (r, f) in v.iter().enumerate() {
                    acc[p * m + r] = f.to_bits();
                    if f.is_finite() && *f != 0.0 {
                        if f.is_sign_negative() {
                            neg_sums += 1;
                        } else {
                            pos_sums += 1;
                        }
                    }
                }
                live_rows += 1;
            } else {
                row_class[p] = 1;
                pad_rows += 1;
            }
        }
        let mut out =
            crate::epilogue::grouped_prefill_epilogue(kind.decode_kind(), m, &sums, &t.1, hooks);
        if pack_swap {
            for lane in 0..16 {
                if t.1[lane] >= 0 {
                    for c in (0..oc).step_by(2) {
                        out.swap(lane * oc + c, lane * oc + c + 1);
                    }
                }
            }
        }
        for lane in 0..16 {
            let p = ti * 16 + lane;
            if t.1[lane] < 0 {
                continue;
            }
            for r in 0..m {
                // input roundings of the epilogue (GateUp rounds g and u, Down rounds its single value)
                bf16_class(
                    acc[p * m + r],
                    &mut tie_even,
                    &mut tie_odd,
                    &mut nonfinite,
                    &mut subnormal,
                );
            }
            for c in 0..oc {
                let v = out[lane * oc + c];
                h[p * oc + c] = v;
                if v == 0x8000 {
                    neg_zero_out += 1;
                } else if v == 0 {
                    pos_zero_out += 1;
                } else if v & 0x7f80 == 0x7f80 {
                    if v & 0x007f == 0 {
                        inf_out += 1;
                    } else {
                        nan_out += 1;
                    }
                }
            }
        }
    }
    let coverage = json!({"live_rows": live_rows, "padding_rows": pad_rows, "sentinel_tiles": sentinel_tiles,
        "f32_sum_ties_round_down_even": tie_even, "f32_sum_ties_round_up_odd": tie_odd, "f32_sum_nonfinite": nonfinite,
        "f32_sum_subnormal": subnormal, "f32_sum_negative_nonzero": neg_sums, "f32_sum_positive_nonzero": pos_sums,
        "bf16_out_negative_zero": neg_zero_out, "bf16_out_positive_zero": pos_zero_out, "bf16_out_infinity": inf_out, "bf16_out_nan": nan_out});
    Expectation {
        m_total,
        row_class,
        acc,
        h,
        coverage,
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------------------------------------------

#[derive(Default, Clone)]
pub struct ClassCount {
    pub checked: u64,
    pub bad: u64,
    pub first: Option<Value>,
}

#[derive(Default, Clone)]
pub struct Cmp {
    pub live: ClassCount,
    pub pad: ClassCount,
    pub sentinel: ClassCount,
}

impl Cmp {
    pub fn bad_total(&self) -> u64 {
        self.live.bad + self.pad.bad + self.sentinel.bad
    }
    pub fn checked_total(&self) -> u64 {
        self.live.checked + self.pad.checked + self.sentinel.checked
    }
    pub fn to_json(&self) -> Value {
        let c =
            |c: &ClassCount| json!({"checked": c.checked, "mismatches": c.bad, "first": c.first});
        json!({"live": c(&self.live), "padding_in_live_tile": c(&self.pad), "sentinel_tile_poison": c(&self.sentinel),
               "mismatches_total": self.bad_total(), "checked_total": self.checked_total()})
    }
}

fn note(cc: &mut ClassCount, ok: bool, first: impl FnOnce() -> Value) {
    cc.checked += 1;
    if !ok {
        cc.bad += 1;
        if cc.first.is_none() {
            cc.first = Some(first());
        }
    }
}

pub fn cmp_acc(got: &[u32], exp: &Expectation, m: usize) -> Cmp {
    let mut c = Cmp::default();
    for p in 0..exp.m_total {
        let cc = match exp.row_class[p] {
            0 => &mut c.live,
            1 => &mut c.pad,
            _ => &mut c.sentinel,
        };
        for r in 0..m {
            let (g, w) = (got[p * m + r], exp.acc[p * m + r]);
            note(
                cc,
                g == w,
                || json!({"tile": p / 16, "lane": p % 16, "grouped_row": p, "row": r, "got": format!("{g:#010x}"), "want": format!("{w:#010x}")}),
            );
        }
    }
    c
}

pub fn cmp_h(got: &[u16], exp: &Expectation, oc: usize) -> Cmp {
    let mut c = Cmp::default();
    for p in 0..exp.m_total {
        let cc = match exp.row_class[p] {
            0 => &mut c.live,
            1 => &mut c.pad,
            _ => &mut c.sentinel,
        };
        for col in 0..oc {
            let (g, w) = (got[p * oc + col], exp.h[p * oc + col]);
            note(
                cc,
                g == w,
                || json!({"tile": p / 16, "lane": p % 16, "grouped_row": p, "col": col, "got": format!("{g:#06x}"), "want": format!("{w:#06x}")}),
            );
        }
    }
    c
}

/// Element-wise equality of two device outputs (anchor vs candidate): count and first differing index.
pub fn cmp_pair_u16(a: &[u16], b: &[u16]) -> (u64, Option<Value>) {
    let (mut bad, mut first) = (0u64, None);
    for i in 0..a.len().min(b.len()) {
        if a[i] != b[i] {
            bad += 1;
            if first.is_none() {
                first = Some(
                    json!({"index": i, "candidate": format!("{:#06x}", a[i]), "anchor": format!("{:#06x}", b[i])}),
                );
            }
        }
    }
    if a.len() != b.len() {
        bad += 1;
    }
    (bad, first)
}

pub fn cmp_pair_u32(a: &[u32], b: &[u32]) -> (u64, Option<Value>) {
    let (mut bad, mut first) = (0u64, None);
    for i in 0..a.len().min(b.len()) {
        if a[i] != b[i] {
            bad += 1;
            if first.is_none() {
                first = Some(
                    json!({"index": i, "candidate": format!("{:#010x}", a[i]), "anchor": format!("{:#010x}", b[i])}),
                );
            }
        }
    }
    if a.len() != b.len() {
        bad += 1;
    }
    (bad, first)
}

pub fn le_u32s(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub fn le_u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

// ---------------------------------------------------------------------------------------------------------------
// Self-check: mutated CPU "kernels"
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mutation {
    None,
    WrongScaleRow,
    ReversedEpoch,
    WrongPackSelector,
    WrongExpertPointer,
    /// E8P: LUT row `idx ^ 1`.
    WrongLut,
    /// E8P: MASK entry `sign7 ^ 1`.
    WrongSignMask,
    /// E8P: the fold's activation sum `s` is dropped in every epoch (term `2C/4` instead of `(2C+s)/4`).
    WrongFoldEpoch,
}

impl Mutation {
    pub const ALL: [Mutation; 4] = [
        Self::WrongScaleRow,
        Self::ReversedEpoch,
        Self::WrongPackSelector,
        Self::WrongExpertPointer,
    ];
    /// E8P has no BF16 pack-selector negative; its unpack stage contributes the LUT and the sign mask instead.
    pub const E8P: [Mutation; 6] = [
        Self::WrongLut,
        Self::WrongSignMask,
        Self::ReversedEpoch,
        Self::WrongFoldEpoch,
        Self::WrongScaleRow,
        Self::WrongExpertPointer,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::WrongScaleRow => "wrong-scale-row",
            Self::ReversedEpoch => "reversed-epoch",
            Self::WrongPackSelector => "wrong-pack-selector",
            Self::WrongExpertPointer => "wrong-expert-pointer",
            Self::WrongLut => "wrong-lut",
            Self::WrongSignMask => "wrong-sign-mask",
            Self::WrongFoldEpoch => "wrong-fold-epoch",
        }
    }
    /// The F32 `_acc` gate must detect everything except the BF16 pack selector.
    pub fn detected_in_f32(self) -> bool {
        self != Self::WrongPackSelector
    }
}

/// Independent CPU fold of one expert: raw weights through `fold_generic` (MQ4/W2/W3) or the `decode8` model (E8P).
pub fn fold_model(kind: Kind, e: &Expert, rows: &[usize], x: &XSet, o: FoldOpts) -> Vec<Vec<f32>> {
    if e.codec == Codec::E8p {
        crate::e8p::fold_e8p(kind, e, rows, x, o, false)
    } else {
        fold_generic(kind, e, rows, x, o)
    }
}

/// An independent CPU model of the candidate kernel (raw weights through `fold_model`), optionally mutated.
pub fn simulate<I: EpilogueIntrinsics>(
    mutation: Mutation,
    kind: Kind,
    experts: &[Expert],
    tiles: &[Tile],
    div: usize,
    x: &XSet,
    hooks: &I,
) -> Expectation {
    let n = experts.len();
    let expert_of = move |e: i32| -> usize {
        if mutation == Mutation::WrongExpertPointer {
            (e as usize + 1) % n
        } else {
            e as usize
        }
    };
    let pairs = needed_pairs(tiles, div, &expert_of);
    let opts = FoldOpts {
        scale_row_xor1: mutation == Mutation::WrongScaleRow,
        reverse_epochs: mutation == Mutation::ReversedEpoch,
        lut_xor1: mutation == Mutation::WrongLut,
        mask_xor1: mutation == Mutation::WrongSignMask,
        drop_fold_sum: mutation == Mutation::WrongFoldEpoch,
    };
    let dense = build_dense(kind, experts, x, &pairs, &|k, e, rows, x| {
        fold_model(k, e, rows, x, opts)
    });
    assemble(
        kind,
        tiles,
        div,
        &dense,
        &expert_of,
        hooks,
        mutation == Mutation::WrongPackSelector,
    )
}

// ---------------------------------------------------------------------------------------------------------------
// Per-codec tile masks (K4 codec masks)
// ---------------------------------------------------------------------------------------------------------------

/// `masked_ids[t] = original_eid[t]` iff the tile is valid and its expert has `codec`, else `-1`.
pub fn mask_tiles(tiles: &[Tile], experts: &[Expert], codec: Codec) -> Vec<i32> {
    tiles
        .iter()
        .map(|t| {
            if t.0 >= 0 && experts[t.0 as usize].codec == codec {
                t.0
            } else {
                -1
            }
        })
        .collect()
}
