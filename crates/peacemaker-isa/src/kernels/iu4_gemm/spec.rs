// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Typed variant axes of the builder-emitted MQ4V2 x int4 GEMM family and the
//! geometry every generator derives from them (plan §3.1-§3.5).
use crate::{kernels::iu4_k1::Variant, Arch, KernargLayout};

/// Fold granularity. `K128` is bit-exact with hipcc K1. The per-256 folds are
/// closed at Q0 (plan §5) and are rejected by the product emitter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Fold {
    K128,
    K256Shared,
    K256Pow2,
}
/// Workgroup tile: rows x tokens x waves.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tile {
    T128x128x8,
    T256x128x16,
}
/// Number of int32 accumulator sets. `Two` is dropped by G0g (plan §8).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Cacc {
    One,
    Two,
}
/// Epilogue family. `QkvzaGdn` is the fused GDN input projection: one SET
/// launch over the QKV weight and the load-time Z fold (Z, beta, alpha),
/// whose QKV tiles run `gdn_chunk_prep` in the epilogue (`gdn_epilogue`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Epi {
    Set,
    Add,
    /// ADD with the shared-expert sigmoid residual store, hipcc `full_add_v3`
    /// `add == 2`: `Y = fma(sigmoid(gate[token]), 0.0f + acc, Y)`. Emitted for the
    /// token-layout `_b1` module only.
    AddSigmoid,
    GateUpSilu,
    GateUpSiluBf16,
    QkvzaGdn,
}
/// Activation layout inside each K128 block of `Xq` (N tokens, N*72 bytes
/// either way). `Token`: one 72-B `block_i4_128` record per token.
/// `Slab`: the same bytes as planes `[d: 4N][s: 4N][qs[0..32): 32N]
/// [qs[32..64): 32N]`, so a wave's 8-token staging load covers 256
/// contiguous bytes and its 16-token `d` load 64 bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ALayout {
    Token,
    Slab,
}

/// The matrix instruction of the K-loop. `Dense` is the shipped
/// `v_wmma_i32_16x16x32_iu4`. `S24X64` / `S24X32` run the weights through
/// SWMMAC (`v_swmmac_i32_16x16x64_iu4` / `v_swmmac_i32_16x16x32_iu4`): the
/// weights are the 2:4-sparse operand A, packed by [`super::sparse24`] into
/// the same MQ4V2 blocks, so staging, LDS layout, fold and epilogue do not
/// change (the format saves no weight bytes: it measures the MMA-issue
/// saving inside the production pipeline).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mma {
    Dense,
    S24X64,
    S24X32,
}
impl Mma {
    pub fn name(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::S24X64 => "s24x64",
            Self::S24X32 => "s24x32",
        }
    }
    pub fn is_sparse(self) -> bool {
        self != Self::Dense
    }
}
impl std::str::FromStr for Mma {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "dense" => Ok(Self::Dense),
            "s24x64" => Ok(Self::S24X64),
            "s24x32" => Ok(Self::S24X32),
            _ => Err(format!("unknown mma {s} (dense|s24x64|s24x32)")),
        }
    }
}

/// Depth of the A/W payload rings (the DS/SZ planes stay per-block).
/// `Three` publishes two slabs ahead: a third A/W LDS slab, a second prefetch
/// register set and a block's next-but-one slabs in flight. The K-loop trip is
/// three groups (12 slabs) so every LDS address stays an immediate; see
/// [`Spec::validate`] for what it is generated for.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RingDepth {
    #[default]
    Two,
    Three,
}
impl RingDepth {
    pub fn name(self) -> &'static str {
        match self {
            Self::Two => "2",
            Self::Three => "3",
        }
    }
}
impl std::str::FromStr for RingDepth {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "2" => Ok(Self::Two),
            "3" => Ok(Self::Three),
            _ => Err(format!("unknown ring depth {s} (2|3)")),
        }
    }
}

/// Where a K-loop block publishes its next slab: the `flip_weights` + LDS
/// stores, which carry the vector-memory wait (`s_wait_loadcnt`) for the
/// slab's global fetch (the builder places that wait at the first use of
/// the fetched registers), relative to the slab's WMMA bundles. The stores
/// target the ring's free next buffer, so every placement is legal for the
/// typed rings; the block's drain (`s_wait_dscnt`) and rendezvous still
/// follow the last WMMA. Scheduling only: every result bit is unchanged.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum WaitPlace {
    /// After the slab's last WMMA bundle (the shipped order).
    #[default]
    AfterMma,
    /// Between the slab's two 32-K sub-block bundles.
    MidMma,
    /// After the slab's fragment loads, before its first WMMA bundle.
    BeforeMma,
    /// The B1 rendezvous window (dense, ring depth 2, 128x128x8 tile only): the token-scale
    /// (`d`) loads move from phase B to right after phase A's fragment loads; the slab-1
    /// publish follows slab 0's WMMA group; the rendezvous is split (`s_barrier_signal`
    /// after the stores drain) and sub-block 1's WMMA group (registers only) runs inside
    /// its window. The shipped placement of the 27B gate/up `_b1s` export
    /// ([`Tune::shipped_gate_up_slab`]); phase B keeps the shipped order.
    Window,
}
impl WaitPlace {
    pub fn name(self) -> &'static str {
        match self {
            Self::AfterMma => "after",
            Self::MidMma => "mid",
            Self::BeforeMma => "before",
            Self::Window => "window",
        }
    }
}
impl std::str::FromStr for WaitPlace {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "after" => Ok(Self::AfterMma),
            "mid" => Ok(Self::MidMma),
            "before" => Ok(Self::BeforeMma),
            "window" => Ok(Self::Window),
            _ => Err(format!(
                "unknown wait placement {s} (after|mid|before|window)"
            )),
        }
    }
}

/// Stage-2 accumulate of the fold. `Fma` is hipcc K1's
/// `sum = fma(t, float(C) - magic, sum)`; `MulAdd` is the deliberately
/// numeric twin `sum = sum + RN(t * (float(C) - magic))` (non-fused
/// `v_mul_f32` + `v_add_f32`): not bit-identical to the shipped kernel.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FoldOp {
    #[default]
    Fma,
    MulAdd,
}
impl FoldOp {
    pub fn name(self) -> &'static str {
        match self {
            Self::Fma => "fma",
            Self::MulAdd => "muladd",
        }
    }
}
impl std::str::FromStr for FoldOp {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "fma" => Ok(Self::Fma),
            "muladd" => Ok(Self::MulAdd),
            _ => Err(format!("unknown fold op {s} (fma|muladd)")),
        }
    }
}

/// The schedule of the K-loop fold (the dense 128x128x8 `cacc1` K128 block at ring depth 2).
/// `Paired` is the shipped order: stage 1 (`float(C) - magic`) of the first four accumulators
/// between the slab-1 WMMAs, stage 2 of each carrying stage 1 of its row-group-1 partner.
/// `Chain` (F3-01-lead1rot-prio, GPU-confirmed on the gate/up slab) keeps every per-element
/// operation (`x = C - magic`, `t = RN(sc * d)`, `acc = fma(t, x, acc)`) and changes only
/// where they issue and which registers hold `x`:
///
/// - only accumulator `k_0` is unbiased between the WMMAs (after WMMA 2, pair-swapped so
///   `float(C[j])` lands in slot `j ^ 1`); `k_1` is unbiased by group 0's `subrev` halves;
/// - stage 2 runs in bundle order, each accumulator's `fmac` packets carrying the `subrev`
///   of the NEXT accumulator, whose unbiased values rotate one slot down the element order
///   (see `fold::chain`) so each packet reads four distinct banks;
/// - the last accumulator has no partner (four `fmac::fmac` packets);
/// - wave priority is 2 from each block's head (after the descriptor step) and 1 from the fold
///   on, so the WMMA groups win arbitration over co-resident folds. The epilogue stays 0.
///
/// Scheduling and register naming only: every stored bit is the `Paired` result.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FoldSched {
    #[default]
    Paired,
    Chain,
}
impl FoldSched {
    pub fn name(self) -> &'static str {
        match self {
            Self::Paired => "paired",
            Self::Chain => "chain",
        }
    }
}
impl std::str::FromStr for FoldSched {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "paired" => Ok(Self::Paired),
            "chain" => Ok(Self::Chain),
            _ => Err(format!("unknown fold schedule {s} (paired|chain)")),
        }
    }
}

/// What the fused GDN projection's QKV row tiles do with their accumulators
/// (`Epi::QkvzaGdn` only). `Gdn` (the shipped kernel) runs `gdn_chunk_prep` on them
/// (`gdn_epilogue`) and stores only the raw rows its completion pass reads. `Set`
/// stores every QKV row like the plain SET kernel (stride `M0`, into the pointer
/// the kernel's `X` argument names) and skips the GDN preparation: one launch over
/// the QKV weight and the Z fold with the verify path's plain consumers
/// (symbol `..._qkvzaset`). The per-element math is the SET kernel's, so the
/// outputs equal the separate launches' bit for bit.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum QkvStore {
    #[default]
    Gdn,
    Set,
}
impl QkvStore {
    pub fn name(self) -> &'static str {
        match self {
            Self::Gdn => "gdn",
            Self::Set => "set",
        }
    }
}
impl std::str::FromStr for QkvStore {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "gdn" => Ok(Self::Gdn),
            "set" => Ok(Self::Set),
            _ => Err(format!("unknown qkv store {s} (gdn|set)")),
        }
    }
}

/// Where the weight (W) slab fetches of a K128 block issue relative to each other.
/// `Split` (the shipped schedule) fetches the next block's slab 0 after the B1
/// rendezvous and its slab 1 after the B2 signal, a fold's length later. `PairW`
/// issues both slabs' weight loads back to back (a block's 64 B of payload per row
/// leave the memory system as one burst; the activation loads keep their places) into
/// a third 4-VGPR weight set that holds slab 1 until its publish. Scheduling only:
/// every result bit is unchanged.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FetchPlace {
    #[default]
    Split,
    PairW,
}
impl FetchPlace {
    pub fn name(self) -> &'static str {
        match self {
            Self::Split => "split",
            Self::PairW => "pairw",
        }
    }
}
impl std::str::FromStr for FetchPlace {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "split" => Ok(Self::Split),
            "pairw" => Ok(Self::PairW),
            _ => Err(format!("unknown fetch placement {s} (split|pairw)")),
        }
    }
}

/// What the waves of a 128x128x8 workgroup do when the tile's second token half
/// (tokens 64..127, the waves with `tok_half = 1`) holds no token (`N - bs <= 64`):
/// those waves compute only zeros whose stores are masked. `Off` (the shipped
/// schedule) computes them anyway; `Skip` branches each barrier-free compute run
/// (fragment loads, WMMA groups, scale loads, fold) of those waves over (the LDS
/// stores, global fetches, waits and barriers stay on every wave); `SkipHi`
/// additionally remaps the waves so the token half is `wave >> 2` and the row
/// pair `wave & 3` (dead waves 4..7 instead of the odd ones), whatever
/// wave-to-SIMD placement the hardware uses. Every stored byte is unchanged.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum IdleHalf {
    #[default]
    Off,
    Skip,
    SkipHi,
}
impl IdleHalf {
    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Skip => "skip",
            Self::SkipHi => "skiphi",
        }
    }
}
impl std::str::FromStr for IdleHalf {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "off" => Ok(Self::Off),
            "skip" => Ok(Self::Skip),
            "skiphi" => Ok(Self::SkipHi),
            _ => Err(format!("unknown idle-half policy {s} (off|skip|skiphi)")),
        }
    }
}

/// Generator knobs that leave the product's math unchanged (`ring`, `wait`)
/// or change it on purpose (`fold_op`). The default is today's exact codegen:
/// every emitted byte of a default-`Tune` spec is unchanged.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Tune {
    pub ring: RingDepth,
    pub wait: WaitPlace,
    pub fold_op: FoldOp,
    pub fold_sched: FoldSched,
    pub fetch: FetchPlace,
    pub idle: IdleHalf,
    pub qkv: QkvStore,
    /// Naming only: marks the small-N (`n <= 128` rows, verify/short batch) export of a
    /// symbol (`_v64`) so it can sit beside the shipped one in the same module. Emits no
    /// different code by itself; the schedule is the other knobs.
    pub n64: bool,
}
impl Tune {
    pub fn is_default(self) -> bool {
        self == Self::default()
    }
    /// The tune the shipped `_b1s` module emits for its 27B gate/up export
    /// `gemm_mq4g256v2_gate_up_silu_mmq_iu4_b1s` (`Epi::GateUpSilu`, slab layout,
    /// 128x128x8 tile): the B1 rendezvous window ([`WaitPlace::Window`], GPU-timed
    /// R2GateStack-05, 0.9966 of the previous schedule) with the chained fold and wave
    /// priorities ([`FoldSched::Chain`], F3-01-lead1rot-prio, GPU-confirmed; every output
    /// bit is the `Paired` schedule's); ring depth 2 and every other knob are the default. Every other symbol of every module keeps `Tune::default()`.
    pub fn shipped_gate_up_slab() -> Self {
        Self {
            wait: WaitPlace::Window,
            fold_sched: FoldSched::Chain,
            ..Self::default()
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Spec {
    pub fold: Fold,
    pub tile: Tile,
    pub cacc: Cacc,
    pub epi: Epi,
    pub act: ALayout,
    pub arch: Arch,
    pub mma: Mma,
    pub tune: Tune,
}

/// Every variant keeps occupancy only at or below this many VGPRs (G0g:
/// 4 WG/WGP for 8-wave and 2 WG/WGP for 16-wave workgroups).
pub const VGPR_CEILING: u16 = 192;
/// Row tiles per raster band (IU4_RASTER_BAND).
pub const RASTER_BAND: u32 = 8;
/// gfx1201 LDS: the most one workgroup may allocate, and the whole WGP's.
pub const LDS_PER_WORKGROUP: u32 = 65536;
pub const LDS_PER_WGP: u32 = 131072;
/// Rows of the fused projection's Z fold: Z, then beta in rows 0..47 and
/// alpha in rows 64..111 of one extra 128-row tile (zero rows elsewhere), so
/// each beta/alpha half is one pair of wave row groups.
pub const GDN_FOLD_EXTRA_ROWS: u32 = 128;
pub const GDN_FOLD_ALPHA_ROW: u32 = 64;
/// Explicit kernarg bytes of the fused projection (hidden arguments follow).
pub const GDN_KERNARG_BYTES: u32 = 120;

/// LDS byte layout. Every plane is a separate builder slot so each publish and
/// retirement is a checked transition. `DS`/`SZ` hold one f32 per token/row
/// (`d` and the converted `sc`): the symmetric fold reads nothing else.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    /// Payload slab slots (A, W). A depth-2 layout has two, a depth-3 layout three;
    /// the unused entry is `u32::MAX` (the typed ring never names it).
    pub a: [u32; 3],
    pub w: [u32; 3],
    pub ds: [u32; 2],
    pub sz: [u32; 2],
    pub a_bytes: u32,
    pub w_bytes: u32,
    pub ds_bytes: u32,
    pub sz_bytes: u32,
    pub end: u32,
    /// Dynamic LDS bytes a launch passes: the depth-2 size at every ring depth, so
    /// the product launcher's constant stays correct for a depth-3 object.
    pub launch: u32,
    /// Static LDS the kernel declares (`group_segment_fixed_size`): 0 at depth
    /// 2; at depth 3 the third A/W slab, so `fixed + launch` covers `end`.
    pub fixed: u32,
}

impl Tile {
    pub fn rows(self) -> u32 {
        match self {
            Self::T128x128x8 => 128,
            Self::T256x128x16 => 256,
        }
    }
    pub fn tokens(self) -> u32 {
        128
    }
    pub fn waves(self) -> u32 {
        match self {
            Self::T128x128x8 => 8,
            Self::T256x128x16 => 16,
        }
    }
    pub fn threads(self) -> u32 {
        self.waves() * 32
    }
    /// Staging rounds per 64-K slab (each lane moves 8 B per round).
    pub fn a_rounds(self) -> u32 {
        self.tokens() * 32 / (self.threads() * 8)
    }
    pub fn w_rounds(self) -> u32 {
        self.rows() * 32 / (self.threads() * 8)
    }
    /// Slab rows covered by one staging round (`R = r * round_rows + tid / 4`).
    pub fn round_rows(self) -> u32 {
        self.threads() / 4
    }
    pub fn layout(self) -> Layout {
        self.layout_for(RingDepth::Two)
    }
    /// The LDS layout of the payload rings at `ring` depth. Depth 3 appends a
    /// third A/W slab after the depth-2 pair and moves the metadata planes
    /// behind it; the depth-2 layout is the shipped one, unchanged.
    pub fn layout_for(self, ring: RingDepth) -> Layout {
        let a_bytes = self.tokens() * 32;
        let w_bytes = self.rows() * 32;
        let slab = a_bytes + w_bytes;
        let (a, w, ds, sz, ds_bytes, sz_bytes, end, launch) = match self {
            // A0 W0 | A1 W1 | DS0 DS1 | SZ0 SZ1. The two DS planes are adjacent so
            // one d address reaches both with ds_load_2addr_b32 offsets.
            Self::T128x128x8 => (
                [0, 8192, u32::MAX],
                [4096, 12288, u32::MAX],
                [16384, 16896],
                [17408, 17920],
                512,
                512,
                18432,
                20480,
            ),
            Self::T256x128x16 => (
                [0, 12288, u32::MAX],
                [4096, 16384, u32::MAX],
                [24576, 25088],
                [25600, 26624],
                512,
                1024,
                27648,
                30720,
            ),
        };
        match ring {
            RingDepth::Two => Layout {
                a,
                w,
                ds,
                sz,
                a_bytes,
                w_bytes,
                ds_bytes,
                sz_bytes,
                end,
                launch,
                fixed: 0,
            },
            // A0 W0 | A1 W1 | A2 W2 | DS0 DS1 | SZ0 SZ1: the metadata planes
            // keep their offsets relative to each other, one slab later.
            RingDepth::Three => Layout {
                a: [a[0], a[1], a[1] + slab],
                w: [w[0], w[1], w[1] + slab],
                ds: ds.map(|o| o + slab),
                sz: sz.map(|o| o + slab),
                a_bytes,
                w_bytes,
                ds_bytes,
                sz_bytes,
                end: end + slab,
                launch,
                fixed: slab,
            },
        }
    }
}

impl Fold {
    pub fn packets_per_k128(self) -> u32 {
        match self {
            Self::K128 => 96,
            Self::K256Shared | Self::K256Pow2 => 48,
        }
    }
}

impl Epi {
    pub fn variant(self) -> Variant {
        match self {
            Self::Set | Self::QkvzaGdn => Variant::FullSet,
            Self::Add => Variant::FullAdd,
            Self::AddSigmoid => Variant::FullAddSigmoid,
            Self::GateUpSilu | Self::GateUpSiluBf16 => Variant::GateUpSilu,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Set => "set",
            Self::Add => "add",
            Self::AddSigmoid => "addsigmoid",
            Self::GateUpSilu => "silu",
            Self::GateUpSiluBf16 => "silu-bf16",
            Self::QkvzaGdn => "qkvzagdn",
        }
    }
    pub fn is_silu(self) -> bool {
        matches!(self, Self::GateUpSilu | Self::GateUpSiluBf16)
    }
}

impl Spec {
    /// The CLI knob strings (`--fold --tile --cacc --epi --alayout`) as a
    /// `Spec`, parsed in that order through the axis `FromStr` impls. The
    /// result is not validated (`emit` validates). `fold` additionally accepts
    /// the long names `k256shared` / `k256pow2`. `mma` is `Dense`: set
    /// `spec.mma` afterwards for a sparse kernel (`--mma`).
    pub fn from_knobs(
        fold: &str,
        tile: &str,
        cacc: &str,
        epi: &str,
        act: &str,
        arch: Arch,
    ) -> Result<Self, String> {
        let (fold, tile, cacc, act) = (fold.parse()?, tile.parse()?, cacc.parse()?, act.parse()?);
        Ok(Self {
            fold,
            tile,
            cacc,
            epi: epi.parse()?,
            act,
            arch,
            mma: Mma::Dense,
            tune: Tune::default(),
        })
    }
    pub fn control(epi: Epi) -> Self {
        Self {
            fold: Fold::K128,
            tile: Tile::T128x128x8,
            cacc: Cacc::One,
            epi,
            act: ALayout::Token,
            arch: Arch::Gfx1201,
            mma: Mma::Dense,
            tune: Tune::default(),
        }
    }
    /// Product symbol suffix. The control keeps the plan's `_b1`; the 16-wave
    /// tile needs different launch geometry, so it is a distinct module; the
    /// slab activation layout reads different bytes, so it is one too.
    pub fn suffix(self) -> &'static str {
        match (self.tile, self.act) {
            (Tile::T128x128x8, ALayout::Token) => "_b1",
            (Tile::T128x128x8, ALayout::Slab) => "_b1s",
            (Tile::T256x128x16, _) => "_b1t256",
        }
    }
    /// The fused projection's QKV tiles run the GDN preparation (`gdn_epilogue`).
    pub fn gdn_prep(self) -> bool {
        self.epi == Epi::QkvzaGdn && self.tune.qkv == QkvStore::Gdn
    }
    pub fn module(self) -> String {
        format!("gemm_mq4g256v2_residual_mmq_iu4_gfx12{}", self.suffix())
    }
    pub fn symbol(self) -> String {
        let stem = match self.epi {
            Epi::Set => "gemm_mq4g256v2_residual_mmq_iu4_full_set",
            Epi::Add => "gemm_mq4g256v2_residual_mmq_iu4_full_add",
            Epi::AddSigmoid => "gemm_mq4g256v2_residual_mmq_iu4_full_add_sigmoid",
            Epi::GateUpSilu | Epi::GateUpSiluBf16 => "gemm_mq4g256v2_gate_up_silu_mmq_iu4",
            Epi::QkvzaGdn if self.tune.qkv == QkvStore::Set => {
                "gemm_mq4g256v2_residual_mmq_iu4_qkvzaset"
            }
            Epi::QkvzaGdn => "gemm_mq4g256v2_residual_mmq_iu4_qkvzagdn",
        };
        format!(
            "{stem}{}{}{}{}",
            self.suffix(),
            if self.epi == Epi::GateUpSiluBf16 {
                "_bf16"
            } else {
                ""
            },
            if self.mma.is_sparse() {
                format!("_{}", self.mma.name())
            } else {
                String::new()
            },
            if self.tune.n64 { "_v64" } else { "" }
        )
    }
    /// The fused projection's ABI: pointers Wqkv, Wzfold, Xq, Yz, Ybeta,
    /// Yalpha; M0, Mz, K, N; pointers ConvW, ConvState, Q, K16, V, X (raw
    /// rows for `gdn_chunk_prep_fixup`); QScale, Eps; then K1's hidden block.
    pub fn kernargs(self) -> KernargLayout {
        if self.epi != Epi::QkvzaGdn {
            return self.epi.variant().kernargs();
        }
        let mut args = KernargLayout::new(GDN_KERNARG_BYTES + 256);
        for (i, name) in ["Wqkv", "Wz", "Xq", "Yz", "Ybeta", "Yalpha"]
            .iter()
            .enumerate()
        {
            args = args.pointer(name, 8 * i as u32);
        }
        for (i, name) in ["M0", "Mz", "K", "N"].iter().enumerate() {
            args = args.hidden(name, 48 + 4 * i as u32, 4, "by_value");
        }
        for (i, name) in ["ConvW", "ConvState", "Q", "K16", "V", "X"]
            .iter()
            .enumerate()
        {
            args = args.pointer(name, 64 + 8 * i as u32);
        }
        args = args
            .hidden("QScale", 112, 4, "by_value")
            .hidden("Eps", 116, 4, "by_value");
        let h = GDN_KERNARG_BYTES;
        for (i, name) in ["x", "y", "z"].iter().enumerate() {
            args = args.hidden(
                &format!("hidden_block_count_{name}"),
                h + 4 * i as u32,
                4,
                &format!("hidden_block_count_{name}"),
            );
        }
        for (i, name) in ["x", "y", "z"].iter().enumerate() {
            args = args.hidden(
                &format!("hidden_group_size_{name}"),
                h + 12 + 2 * i as u32,
                2,
                &format!("hidden_group_size_{name}"),
            );
        }
        for (i, name) in ["x", "y", "z"].iter().enumerate() {
            args = args.hidden(
                &format!("hidden_remainder_{name}"),
                h + 18 + 2 * i as u32,
                2,
                &format!("hidden_remainder_{name}"),
            );
        }
        for (i, name) in ["x", "y", "z"].iter().enumerate() {
            args = args.hidden(
                &format!("hidden_global_offset_{name}"),
                h + 40 + 8 * i as u32,
                8,
                &format!("hidden_global_offset_{name}"),
            );
        }
        args.hidden("hidden_grid_dims", h + 64, 2, "hidden_grid_dims")
            .hidden(
                "hidden_dynamic_lds_size",
                h + 120,
                4,
                "hidden_dynamic_lds_size",
            )
    }
    pub fn variant_name(self) -> String {
        format!(
            "{}-{}-{}{}-{}{}{}",
            match self.fold {
                Fold::K128 => "k128",
                Fold::K256Shared => "k256s",
                Fold::K256Pow2 => "k256p",
            },
            match self.tile {
                Tile::T128x128x8 => "128x128x8",
                Tile::T256x128x16 => "256x128x16",
            },
            match self.cacc {
                Cacc::One => "cacc1",
                Cacc::Two => "cacc2",
            },
            match self.act {
                ALayout::Token => "",
                ALayout::Slab => "-slab",
            },
            self.epi.name(),
            if self.mma.is_sparse() {
                format!("-{}", self.mma.name())
            } else {
                String::new()
            },
            self.tune_suffix()
        )
    }
    /// Variant-name suffix of the non-default generator knobs (empty for the
    /// default, so every existing variant name is unchanged).
    fn tune_suffix(self) -> String {
        let t = self.tune;
        let mut s = String::new();
        if t.ring != RingDepth::Two {
            s += &format!("-r{}", t.ring.name());
        }
        if t.wait != WaitPlace::AfterMma {
            s += &format!("-w{}", t.wait.name());
        }
        if t.fold_op != FoldOp::Fma {
            s += &format!("-f{}", t.fold_op.name());
        }
        if t.fold_sched != FoldSched::Paired {
            s += &format!("-fs{}", t.fold_sched.name());
        }
        if t.fetch != FetchPlace::Split {
            s += &format!("-x{}", t.fetch.name());
        }
        if t.idle != IdleHalf::Off {
            s += &format!("-idle{}", t.idle.name());
        }
        if t.qkv != QkvStore::Gdn {
            s += &format!("-q{}", t.qkv.name());
        }
        if t.n64 {
            s += "-n64";
        }
        s
    }
    /// The LDS layout of this spec's payload rings (`Tune::ring`).
    pub fn layout(self) -> Layout {
        self.tile.layout_for(self.tune.ring)
    }
    /// Dynamic LDS bytes a launch of this spec must pass: the depth-2 size at
    /// every ring depth (20480 for the 128x128x8 tile), so the product
    /// launcher's constant is right for a depth-3 object too.
    pub fn lds_launch(self) -> u32 {
        self.layout().launch
    }
    /// Static LDS the kernel declares (`group_segment_fixed_size`): 0 at ring
    /// depth 2, the third A/W slab at depth 3.
    pub fn lds_fixed(self) -> u32 {
        self.layout().fixed
    }
    /// LDS one workgroup holds: static plus dynamic.
    pub fn lds_total(self) -> u32 {
        let l = self.layout();
        l.fixed + l.launch
    }
    /// What ring depth 3 asks of the machine for this spec, `Err` (with the
    /// numbers) where it does not fit: the second prefetch set in the VGPRs
    /// depth 3 frees, the VGPR occupancy ceiling, and the LDS of the third A/W
    /// slab per workgroup and per WGP at the occupancy [`VGPR_CEILING`] keeps.
    fn validate_ring3(self) -> Result<(), String> {
        if self.mma.is_sparse() {
            return Err(format!(
                "ring depth 3 is generated for the dense K-loop block only (mma axis is {}): \
                 the depth-3 schedule (`ring3.rs`, `const RING = 3`) composes the dense fine steps; \
                 the sparse fine steps have no depth-3 composition yet",
                self.mma.name()
            ));
        }
        if self.epi == Epi::QkvzaGdn {
            return Err(
                "ring depth 3 is not generated for the fused GDN projection: its epilogue \
                 re-lays out the K-loop's LDS (`gdn_epilogue`, ring bytes vs the launch size) and \
                 the depth-3 map is not handled there"
                    .into(),
            );
        }
        let pf_set = 2 * (self.tile.a_rounds() + self.tile.w_rounds());
        if pf_set > 8 {
            return Err(format!(
                "ring depth 3 needs a second prefetch set of {pf_set} VGPRs, but only the 8 freed \
                 by seeding the accumulators in cacc[3][1] instead of magic8 are available"
            ));
        }
        let vgprs = super::Gen::build(self).vgprs();
        if vgprs > VGPR_CEILING {
            return Err(format!(
                "ring depth 3 needs {vgprs} VGPRs, the occupancy ceiling is {VGPR_CEILING}"
            ));
        }
        let (end, total) = (self.layout().end, self.lds_total());
        let per_wgp = if self.tile.waves() == 8 { 4 } else { 2 };
        if end > total || total > LDS_PER_WORKGROUP || total * per_wgp > LDS_PER_WGP {
            return Err(format!(
                "ring depth 3 needs {end} B of LDS per workgroup ({total} B launched: {} B static \
                 + {} B dynamic; {} B for the {per_wgp} workgroups per WGP the {VGPR_CEILING}-VGPR \
                 ceiling allows); the budget is {LDS_PER_WORKGROUP} B per workgroup and \
                 {LDS_PER_WGP} B per WGP",
                self.lds_fixed(),
                self.lds_launch(),
                total * per_wgp
            ));
        }
        Ok(())
    }
    /// Reject combinations that have no generator: closed fold tracks, the
    /// dropped second accumulator set, and non-gfx12 targets.
    pub fn validate(self) -> Result<(), String> {
        if self.arch != Arch::Gfx1201 {
            return Err("iu4_gemm targets gfx1201 only".into());
        }
        if self.mma.is_sparse() {
            let m = self.mma.name();
            if self.fold != Fold::K128 {
                return Err(format!(
                    "sparse mma {m} needs Fold::K128 (fold axis is {:?})",
                    self.fold
                ));
            }
            if self.tile != Tile::T128x128x8 {
                return Err(format!(
                    "sparse mma {m} needs Tile::T128x128x8 (tile axis is {:?})",
                    self.tile
                ));
            }
            if self.cacc != Cacc::One {
                return Err(format!(
                    "sparse mma {m} needs Cacc::One (cacc axis is {:?})",
                    self.cacc
                ));
            }
            if self.act != ALayout::Token {
                return Err(format!(
                    "sparse mma {m} needs ALayout::Token (activation layout axis is {:?})",
                    self.act
                ));
            }
            if !matches!(self.epi, Epi::Set | Epi::Add) {
                return Err(format!(
                    "sparse mma {m} supports Epi::Set and Epi::Add only (epilogue axis is {:?})",
                    self.epi
                ));
            }
        }
        if self.fold != Fold::K128 {
            return Err("K256 folds are closed at Q0 (plan §5); no product generator".into());
        }
        if self.cacc != Cacc::One {
            return Err(
                "Cacc::Two is dropped by G0g: it cannot fit the 192-VGPR occupancy ceiling".into(),
            );
        }
        if self.epi == Epi::GateUpSiluBf16 && self.tile != Tile::T128x128x8 {
            return Err("packed bf16 h is emitted only for the production _b1 tile".into());
        }
        if self.act == ALayout::Slab && self.tile != Tile::T128x128x8 {
            return Err(
                "the slab activation layout is emitted only for the production _b1 tile".into(),
            );
        }
        if self.epi == Epi::AddSigmoid
            && (self.tile != Tile::T128x128x8 || self.act != ALayout::Token)
        {
            return Err(
                "the sigmoid residual ADD is emitted only for the token-layout production _b1 tile"
                    .into(),
            );
        }
        if self.epi == Epi::QkvzaGdn && self.tile != Tile::T128x128x8 {
            return Err(
                "the fused GDN projection needs the 128-row tile (one head per tile)".into(),
            );
        }
        if self.tune.ring == RingDepth::Three {
            self.validate_ring3()?;
        }
        if self.tune.wait != WaitPlace::AfterMma && self.mma.is_sparse() {
            return Err(format!(
                "wait placement {} is generated for the dense K-loop block only (mma axis is {})",
                self.tune.wait.name(),
                self.mma.name()
            ));
        }
        if self.tune.wait == WaitPlace::Window {
            if self.tune.ring != RingDepth::Two {
                return Err("wait placement window is generated at ring depth 2 only".into());
            }
            if self.tile != Tile::T128x128x8 {
                return Err(format!(
                    "wait placement window is generated for Tile::T128x128x8 only (tile axis is {:?})",
                    self.tile
                ));
            }
        }
        if self.tune.fold_sched == FoldSched::Chain {
            if self.mma.is_sparse() {
                return Err(format!(
                    "fold schedule chain is generated for the dense K-loop block only (mma axis is {})",
                    self.mma.name()
                ));
            }
            if self.tune.ring != RingDepth::Two {
                return Err("fold schedule chain is generated at ring depth 2 only".into());
            }
            if self.tile != Tile::T128x128x8 || self.cacc != Cacc::One || self.fold != Fold::K128 {
                return Err(format!(
                    "fold schedule chain is generated for the k128 Tile::T128x128x8 cacc1 fold only (fold axis is {:?}, tile axis is {:?}, cacc axis is {:?})",
                    self.fold, self.tile, self.cacc
                ));
            }
            if self.tune.fold_op != FoldOp::Fma {
                return Err(
                    "fold schedule chain is the fused-accumulate fold (fold op fma) only".into(),
                );
            }
        }
        if self.tune.n64
            && (self.tile != Tile::T128x128x8 || self.act != ALayout::Slab || self.mma.is_sparse())
        {
            return Err(format!(
                "the n64 (`_v64`) export is generated for the dense slab Tile::T128x128x8 module only (tile axis is {:?}, activation layout axis is {:?}, mma axis is {})",
                self.tile,
                self.act,
                self.mma.name()
            ));
        }
        if self.tune.qkv == QkvStore::Set && self.epi != Epi::QkvzaGdn {
            return Err(format!(
                "qkv store set is a knob of the fused GDN projection only (epilogue axis is {:?})",
                self.epi
            ));
        }
        if self.tune.fetch == FetchPlace::PairW {
            if self.mma.is_sparse() {
                return Err(format!(
                    "fetch placement pairw is generated for the dense K-loop block only (mma axis is {})",
                    self.mma.name()
                ));
            }
            if self.tune.ring != RingDepth::Two {
                return Err("fetch placement pairw is generated at ring depth 2 only".into());
            }
            if self.tile != Tile::T128x128x8 || self.epi == Epi::QkvzaGdn {
                return Err(format!(
                    "fetch placement pairw is generated for the Tile::T128x128x8 SET/ADD/SiLU kernels only (tile axis is {:?}, epilogue axis is {:?})",
                    self.tile, self.epi
                ));
            }
            let vgprs = super::Gen::build(self).vgprs();
            if vgprs > VGPR_CEILING {
                return Err(format!(
                    "fetch placement pairw needs {vgprs} VGPRs, the occupancy ceiling is {VGPR_CEILING}"
                ));
            }
        }
        if self.tune.idle != IdleHalf::Off {
            let idle = self.tune.idle.name();
            if self.mma.is_sparse() {
                return Err(format!(
                    "idle-half {idle} is generated for the dense K-loop block only (mma axis is {})",
                    self.mma.name()
                ));
            }
            if self.tile != Tile::T128x128x8 {
                return Err(format!(
                    "idle-half {idle} is generated for Tile::T128x128x8 only (tile axis is {:?}): it skips the token-half-1 waves of the 8-wave tile",
                    self.tile
                ));
            }
            if self.tune.ring != RingDepth::Two {
                return Err(format!(
                    "idle-half {idle} is generated at ring depth 2 only (the depth-3 schedule is not wrapped)"
                ));
            }
            if !matches!(self.tune.wait, WaitPlace::AfterMma | WaitPlace::Window) {
                return Err(format!(
                    "idle-half {idle} needs wait placement after or window (mid/before interleave the LDS publish between WMMA groups; wait axis is {})",
                    self.tune.wait.name()
                ));
            }
            if self.epi == Epi::QkvzaGdn && self.tune.qkv != QkvStore::Set {
                return Err(format!(
                    "idle-half {idle} is not generated for the GDN-preparation projection: its epilogue exchanges across waves through LDS (qkv store axis is {}; use qkv set)",
                    self.tune.qkv.name()
                ));
            }
        }
        Ok(())
    }
}

impl std::str::FromStr for Fold {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "k128" => Ok(Self::K128),
            "k256s" | "k256shared" => Ok(Self::K256Shared),
            "k256p" | "k256pow2" => Ok(Self::K256Pow2),
            _ => Err(format!("unknown fold {s}")),
        }
    }
}
impl std::str::FromStr for Tile {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "128x128x8" => Ok(Self::T128x128x8),
            "256x128x16" => Ok(Self::T256x128x16),
            _ => Err(format!("unknown tile {s}")),
        }
    }
}
impl std::str::FromStr for Cacc {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "1" => Ok(Self::One),
            "2" => Ok(Self::Two),
            _ => Err(format!("unknown cacc {s}")),
        }
    }
}
impl std::str::FromStr for Epi {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "set" => Ok(Self::Set),
            "add" => Ok(Self::Add),
            "addsigmoid" => Ok(Self::AddSigmoid),
            "silu" => Ok(Self::GateUpSilu),
            "silu-bf16" => Ok(Self::GateUpSiluBf16),
            "qkvzagdn" => Ok(Self::QkvzaGdn),
            _ => Err(format!("unknown epilogue {s}")),
        }
    }
}
impl std::str::FromStr for ALayout {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "token" => Ok(Self::Token),
            "slab" => Ok(Self::Slab),
            _ => Err(format!("unknown activation layout {s}")),
        }
    }
}
