//! Typed variant axes of the builder-emitted MQ4V2 x int4 GEMM family and the
//! geometry every generator derives from them (plan §3.1-§3.5).
use crate::{Arch, KernargLayout, kernels::iu4_k1::Variant};

/// Fold granularity. `K128` is bit-exact with hipcc K1. The per-256 folds are
/// closed at Q0 (plan §5) and are rejected by the product emitter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Fold { K128, K256Shared, K256Pow2 }
/// Workgroup tile: rows x tokens x waves.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tile { T128x128x8, T256x128x16 }
/// Number of int32 accumulator sets. `Two` is dropped by G0g (plan §8).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Cacc { One, Two }
/// Epilogue family. `QkvzaGdn` is the fused GDN input projection: one SET
/// launch over the QKV weight and the load-time Z fold (Z, beta, alpha),
/// whose QKV tiles run `gdn_chunk_prep` in the epilogue (`gdn_epilogue`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Epi { Set, Add, GateUpSilu, GateUpSiluBf16, QkvzaGdn }
/// Activation layout inside each K128 block of `Xq` (N tokens, N*72 bytes
/// either way). `Token`: one 72-B `block_i4_128` record per token.
/// `Slab`: the same bytes as planes `[d: 4N][s: 4N][qs[0..32): 32N]
/// [qs[32..64): 32N]`, so a wave's 8-token staging load covers 256
/// contiguous bytes and its 16-token `d` load 64 bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ALayout { Token, Slab }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Spec { pub fold: Fold, pub tile: Tile, pub cacc: Cacc, pub epi: Epi, pub act: ALayout, pub arch: Arch }

/// Every variant keeps occupancy only at or below this many VGPRs (G0g:
/// 4 WG/WGP for 8-wave and 2 WG/WGP for 16-wave workgroups).
pub const VGPR_CEILING: u16 = 192;
/// Row tiles per raster band (IU4_RASTER_BAND).
pub const RASTER_BAND: u32 = 8;
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
pub struct Layout { pub a: [u32; 2], pub w: [u32; 2], pub ds: [u32; 2], pub sz: [u32; 2], pub a_bytes: u32, pub w_bytes: u32, pub ds_bytes: u32, pub sz_bytes: u32, pub end: u32, pub launch: u32 }

impl Tile {
    pub fn rows(self) -> u32 { match self { Self::T128x128x8 => 128, Self::T256x128x16 => 256 } }
    pub fn tokens(self) -> u32 { 128 }
    pub fn waves(self) -> u32 { match self { Self::T128x128x8 => 8, Self::T256x128x16 => 16 } }
    pub fn threads(self) -> u32 { self.waves() * 32 }
    /// Staging rounds per 64-K slab (each lane moves 8 B per round).
    pub fn a_rounds(self) -> u32 { self.tokens() * 32 / (self.threads() * 8) }
    pub fn w_rounds(self) -> u32 { self.rows() * 32 / (self.threads() * 8) }
    /// Slab rows covered by one staging round (`R = r * round_rows + tid / 4`).
    pub fn round_rows(self) -> u32 { self.threads() / 4 }
    pub fn layout(self) -> Layout {
        let a_bytes = self.tokens() * 32;
        let w_bytes = self.rows() * 32;
        match self {
            // A0 W0 | A1 W1 | DS0 DS1 | SZ0 SZ1. The two DS planes are adjacent so
            // one d address reaches both with ds_load_2addr_b32 offsets.
            Self::T128x128x8 => Layout { a: [0, 8192], w: [4096, 12288], ds: [16384, 16896], sz: [17408, 17920],
                a_bytes, w_bytes, ds_bytes: 512, sz_bytes: 512, end: 18432, launch: 20480 },
            Self::T256x128x16 => Layout { a: [0, 12288], w: [4096, 16384], ds: [24576, 25088], sz: [25600, 26624],
                a_bytes, w_bytes, ds_bytes: 512, sz_bytes: 1024, end: 27648, launch: 30720 },
        }
    }
}

impl Fold {
    pub fn packets_per_k128(self) -> u32 { match self { Self::K128 => 96, Self::K256Shared | Self::K256Pow2 => 48 } }
}

impl Epi {
    pub fn variant(self) -> Variant { match self { Self::Set | Self::QkvzaGdn => Variant::FullSet, Self::Add => Variant::FullAdd, Self::GateUpSilu | Self::GateUpSiluBf16 => Variant::GateUpSilu } }
    pub fn name(self) -> &'static str { match self { Self::Set => "set", Self::Add => "add", Self::GateUpSilu => "silu", Self::GateUpSiluBf16 => "silu-bf16", Self::QkvzaGdn => "qkvzagdn" } }
    pub fn is_silu(self) -> bool { matches!(self, Self::GateUpSilu | Self::GateUpSiluBf16) }
}

impl Spec {
    pub fn control(epi: Epi) -> Self { Self { fold: Fold::K128, tile: Tile::T128x128x8, cacc: Cacc::One, epi, act: ALayout::Token, arch: Arch::Gfx1201 } }
    /// Product symbol suffix. The control keeps the plan's `_b1`; the 16-wave
    /// tile needs different launch geometry, so it is a distinct module; the
    /// slab activation layout reads different bytes, so it is one too.
    pub fn suffix(self) -> &'static str {
        match (self.tile, self.act) { (Tile::T128x128x8, ALayout::Token) => "_b1", (Tile::T128x128x8, ALayout::Slab) => "_b1s", (Tile::T256x128x16, _) => "_b1t256" }
    }
    pub fn module(self) -> String { format!("gemm_mq4g256v2_residual_mmq_iu4_gfx12{}", self.suffix()) }
    pub fn symbol(self) -> String {
        let stem = match self.epi {
            Epi::Set => "gemm_mq4g256v2_residual_mmq_iu4_full_set",
            Epi::Add => "gemm_mq4g256v2_residual_mmq_iu4_full_add",
            Epi::GateUpSilu | Epi::GateUpSiluBf16 => "gemm_mq4g256v2_gate_up_silu_mmq_iu4",
            Epi::QkvzaGdn => "gemm_mq4g256v2_residual_mmq_iu4_qkvzagdn",
        };
        format!("{stem}{}{}", self.suffix(), if self.epi == Epi::GateUpSiluBf16 { "_bf16" } else { "" })
    }
    /// The fused projection's ABI: pointers Wqkv, Wzfold, Xq, Yz, Ybeta,
    /// Yalpha; M0, Mz, K, N; pointers ConvW, ConvState, Q, K16, V, X (raw
    /// rows for `gdn_chunk_prep_fixup`); QScale, Eps; then K1's hidden block.
    pub fn kernargs(self) -> KernargLayout {
        if self.epi != Epi::QkvzaGdn { return self.epi.variant().kernargs() }
        let mut args = KernargLayout::new(GDN_KERNARG_BYTES + 256);
        for (i, name) in ["Wqkv", "Wz", "Xq", "Yz", "Ybeta", "Yalpha"].iter().enumerate() { args = args.pointer(name, 8 * i as u32); }
        for (i, name) in ["M0", "Mz", "K", "N"].iter().enumerate() { args = args.hidden(name, 48 + 4 * i as u32, 4, "by_value"); }
        for (i, name) in ["ConvW", "ConvState", "Q", "K16", "V", "X"].iter().enumerate() { args = args.pointer(name, 64 + 8 * i as u32); }
        args = args.hidden("QScale", 112, 4, "by_value").hidden("Eps", 116, 4, "by_value");
        let h = GDN_KERNARG_BYTES;
        for (i, name) in ["x", "y", "z"].iter().enumerate() {
            args = args.hidden(&format!("hidden_block_count_{name}"), h + 4 * i as u32, 4, &format!("hidden_block_count_{name}"));
        }
        for (i, name) in ["x", "y", "z"].iter().enumerate() {
            args = args.hidden(&format!("hidden_group_size_{name}"), h + 12 + 2 * i as u32, 2, &format!("hidden_group_size_{name}"));
        }
        for (i, name) in ["x", "y", "z"].iter().enumerate() {
            args = args.hidden(&format!("hidden_remainder_{name}"), h + 18 + 2 * i as u32, 2, &format!("hidden_remainder_{name}"));
        }
        for (i, name) in ["x", "y", "z"].iter().enumerate() {
            args = args.hidden(&format!("hidden_global_offset_{name}"), h + 40 + 8 * i as u32, 8, &format!("hidden_global_offset_{name}"));
        }
        args.hidden("hidden_grid_dims", h + 64, 2, "hidden_grid_dims")
            .hidden("hidden_dynamic_lds_size", h + 120, 4, "hidden_dynamic_lds_size")
    }
    pub fn variant_name(self) -> String {
        format!("{}-{}-{}{}-{}",
            match self.fold { Fold::K128 => "k128", Fold::K256Shared => "k256s", Fold::K256Pow2 => "k256p" },
            match self.tile { Tile::T128x128x8 => "128x128x8", Tile::T256x128x16 => "256x128x16" },
            match self.cacc { Cacc::One => "cacc1", Cacc::Two => "cacc2" },
            match self.act { ALayout::Token => "", ALayout::Slab => "-slab" }, self.epi.name())
    }
    /// Reject combinations that have no generator: closed fold tracks, the
    /// dropped second accumulator set, and non-gfx12 targets.
    pub fn validate(self) -> Result<(), String> {
        if self.arch != Arch::Gfx1201 { return Err("iu4_gemm targets gfx1201 only".into()) }
        if self.fold != Fold::K128 { return Err("K256 folds are closed at Q0 (plan §5); no product generator".into()) }
        if self.cacc != Cacc::One { return Err("Cacc::Two is dropped by G0g: it cannot fit the 192-VGPR occupancy ceiling".into()) }
        if self.epi == Epi::GateUpSiluBf16 && self.tile != Tile::T128x128x8 { return Err("packed bf16 h is emitted only for the production _b1 tile".into()) }
        if self.act == ALayout::Slab && self.tile != Tile::T128x128x8 { return Err("the slab activation layout is emitted only for the production _b1 tile".into()) }
        if self.epi == Epi::QkvzaGdn && self.tile != Tile::T128x128x8 { return Err("the fused GDN projection needs the 128-row tile (one head per tile)".into()) }
        Ok(())
    }
}

impl std::str::FromStr for Fold { type Err = String; fn from_str(s: &str) -> Result<Self, String> { match s { "k128" => Ok(Self::K128), "k256s" => Ok(Self::K256Shared), "k256p" => Ok(Self::K256Pow2), _ => Err(format!("unknown fold {s}")) } } }
impl std::str::FromStr for Tile { type Err = String; fn from_str(s: &str) -> Result<Self, String> { match s { "128x128x8" => Ok(Self::T128x128x8), "256x128x16" => Ok(Self::T256x128x16), _ => Err(format!("unknown tile {s}")) } } }
impl std::str::FromStr for Cacc { type Err = String; fn from_str(s: &str) -> Result<Self, String> { match s { "1" => Ok(Self::One), "2" => Ok(Self::Two), _ => Err(format!("unknown cacc {s}")) } } }
impl std::str::FromStr for Epi { type Err = String; fn from_str(s: &str) -> Result<Self, String> { match s { "set" => Ok(Self::Set), "add" => Ok(Self::Add), "silu" => Ok(Self::GateUpSilu), "silu-bf16" => Ok(Self::GateUpSiluBf16), "qkvzagdn" => Ok(Self::QkvzaGdn), _ => Err(format!("unknown epilogue {s}")) } } }
impl std::str::FromStr for ALayout { type Err = String; fn from_str(s: &str) -> Result<Self, String> { match s { "token" => Ok(Self::Token), "slab" => Ok(Self::Slab), _ => Err(format!("unknown activation layout {s}")) } } }
