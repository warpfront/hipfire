// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Variant axes and fixed geometry of the MQ6G256V2 x `block_i8_128` GEMM.
pub use crate::kernels::iu4_gemm::RingDepth;
use crate::{kernels::iu4_k1::Variant, Arch, KernargLayout};

/// Epilogue: F32 overwrite, BF16 bits (`isfinite(v) ? RNE : u >> 16`, `v = 0 + y`), or F32 overwrite of
/// a Flash-Next QSA projection with the head RMSNorm + weight + HalfSplit RoPE of
/// `qsa_norm_rope_head` (`kernels/src/tensor_ops.hip`) applied to the head rows the tile covers
/// (`SetNr`: `T256x128` only; see `nr_epilogue`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Epi {
    Set,
    SetBf16,
    SetNr,
}
/// CTA raster. `Banded` is the product (hipcc `IU4_G12_RASTER`, band 8). `Linear`
/// (`rs = x * 128`, `bs = y * 128`) exists only so the CPU interpreter, which has no
/// SALU float-reciprocal semantics, can run the data path of the same kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Raster {
    Banded,
    Linear,
}
/// CTA tile: weight rows x tokens per workgroup (the wave arrangement follows).
///
/// Every wave keeps the shipped 32-row x 64-token fragment grid (eight 16x16 accumulator
/// fragments: `cacc` 64 + `acc` 64 VGPRs), so the register map, the K-loop bundles and the fold
/// are the same for every tile and the VGPR count stays at the 192 ceiling. A bigger CTA tile is
/// therefore more waves, not bigger waves:
///
/// | tile | waves | threads | per-wave | LDS launch | WG/WGP (VGPR) | WG/WGP (LDS) | waves/WGP |
/// |---|---|---|---|---|---|---|---|
/// | `T128x128` (shipped) | 8 | 256 | 32 x 64 | 36864 B | 4 | 3 | 24 |
/// | `T256x128` | 16 | 512 | 32 x 64 | 55296 B | 2 | 2 | 32 |
/// | `T128x256` | 16 | 512 | 32 x 64 | 55296 B | 2 | 2 | 32 |
///
/// (VGPR column: 192 VGPRs per lane allow 8 waves per SIMD32, 32 per WGP; LDS column: 128 KiB per
/// WGP over the launch bytes.) A 256-wide tile moves one operand's staging to a single round of
/// the 512 threads and doubles the other's; its arithmetic intensity per staged byte is 1.33x the
/// shipped tile's (256x128 outputs per K step over 384 staged rows+tokens vs 128x128 over 256).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Tile {
    T128x128,
    T256x128,
    T128x256,
}
/// Generator knobs that leave the product's math unchanged: scheduling only, every result bit
/// equals the default's. The default is today's exact codegen (every emitted byte, register map
/// and LDS map of a default-`Tune` spec is unchanged).
///
/// `ring` is the depth of the A/W payload rings (the DS/SZ metadata planes stay two-deep):
/// `Two` is the shipped schedule; `Three` publishes two slabs ahead (two raw prefetch register
/// sets, three A/W slabs in LDS, the K-loop unrolled over a three-group period; see
/// [`ring3_budget`] for what it costs and [`Spec::validate`] for where it fits).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Tune {
    pub ring: RingDepth,
}
impl Tune {
    pub fn is_default(self) -> bool {
        self == Self::default()
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Spec {
    pub epi: Epi,
    pub raster: Raster,
    pub tile: Tile,
    pub arch: Arch,
    pub tune: Tune,
}

/// The shipped tile: 128 weight rows x 128 tokens x 8 waves (32 rows x 64 tokens per wave). The
/// constants below describe it; the other tiles' geometry is [`Tile`]'s methods.
pub const TILE_ROWS: u32 = 128;
pub const TILE_TOKENS: u32 = 128;
pub const THREADS: u32 = 256;
pub const VGPR_CEILING: u16 = 192;
pub const RASTER_BAND: u32 = 8;
/// `block_i8_128`: `{f32 d, i32 s, int8 qs[128]}`.
pub const XBLK_BYTES: u32 = 136;
/// MQ6G256V2 group: 8 header bytes + 192 payload bytes.
pub const GROUP_BYTES: u32 = 200;

/// LDS byte layout (every plane a separate builder slot).
///   slot s: A_s = tokens * 64 (token-group pairs x 4 K16 steps x 512 B),
///           W_s = rows * 64 (row-group pairs x 4 K16 steps x 512 B, decoded codes)
///   DS_p = tokens * 8 per K128 parity: per chunk of 128 tokens, planar token `d` [128 f32]
///          then `float(s)` [128 f32] (chunk `i` at `1024 * i`)
///   SZ_p = rows * 8 per K128 parity: per chunk of 128 rows, planar row `sc` [128 f32] then
///          `z'` [128 f32]
/// The chunking keeps the `ds_load_2addr_b32` offsets of the fold (`d` -> `s` = 128 dwords)
/// inside their 8-bit field for every tile; with one chunk it is the shipped planar layout.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub a: [u32; 2],
    pub w: [u32; 2],
    pub ds: [u32; 2],
    pub sz: [u32; 2],
    pub a_bytes: u32,
    pub w_bytes: u32,
    pub ds_bytes: u32,
    pub sz_bytes: u32,
    pub launch: u32,
}
/// The shipped tile's layout.
pub const LAYOUT: Layout = Layout {
    a: [0, 16384],
    w: [8192, 24576],
    ds: [32768, 33792],
    sz: [34816, 35840],
    a_bytes: 8192,
    w_bytes: 8192,
    ds_bytes: 1024,
    sz_bytes: 1024,
    launch: 36864,
};
/// The dynamic LDS bytes the shipped tile's launch must pass (the same figure as the hipcc
/// twin's `MQ6_I8_GFX12_LDS_BYTES`); see [`Tile::layout`] for the others.
pub const LAUNCH_LDS_BYTES: u32 = 36864;

impl Layout {
    /// Base of A slab buffer `slot`: `0`, `1` of every layout, and `2`, the third buffer of a depth-3
    /// ring, placed after the depth-2 launch region (so the depth-2 buffers keep their offsets).
    pub fn a_slot(&self, slot: usize) -> Result<u32, String> {
        match slot {
            0 | 1 => Ok(self.a[slot]),
            2 => Ok(self.launch),
            _ => Err(format!("A slab buffer {slot} out of range (0..3)")),
        }
    }
    /// Base of W slab buffer `slot` (see [`Layout::a_slot`]; the third buffer follows the third A).
    pub fn w_slot(&self, slot: usize) -> Result<u32, String> {
        match slot {
            0 | 1 => Ok(self.w[slot]),
            2 => Ok(self.launch + self.a_bytes),
            _ => Err(format!("W slab buffer {slot} out of range (0..3)")),
        }
    }
    /// The dynamic LDS bytes of a launch whose A/W rings are `depth` deep (`2` or `3`).
    pub fn launch_for(&self, depth: RingDepth) -> u32 {
        match depth {
            RingDepth::Two => self.launch,
            RingDepth::Three => self.launch + self.a_bytes + self.w_bytes,
        }
    }
}

/// gfx1201 limits the depth-3 legality check reads: the per-workgroup LDS limit, the LDS of a
/// WGP, the VGPRs of a SIMD32 (wave32), and the architectural VGPRs per lane.
pub const WG_LDS_LIMIT: u32 = 65536;
pub const WGP_LDS: u32 = 131072;
pub const SIMD_VGPRS: u32 = 1536;
pub const VGPR_ARCH_MAX: u32 = 256;

/// What a depth-3 A/W ring costs on `tile` against what the target allows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Ring3Budget {
    /// VGPRs of the second raw prefetch set (`2 * a_rounds + 3 * w_rounds`).
    pub pf_vgprs: u32,
    /// VGPRs the depth-3 kernel needs: the depth-2 [`VGPR_CEILING`] map plus the second set.
    pub vgprs: u32,
    /// The VGPR budget at the occupancy the depth-3 LDS footprint leaves: workgroups per WGP by
    /// LDS (`WGP_LDS / lds`, capped at what fits), times the tile's waves, over four SIMDs, then
    /// `SIMD_VGPRS` over that, in granules of eight, capped at [`VGPR_ARCH_MAX`].
    pub vgpr_budget: u32,
    /// Dynamic LDS bytes of the depth-3 launch (three A+W slabs plus the DS/SZ planes).
    pub lds: u32,
    /// The per-workgroup LDS limit ([`WG_LDS_LIMIT`]).
    pub lds_limit: u32,
    /// Workgroups per WGP the depth-3 LDS leaves (evaluated at the launch limit when `lds`
    /// exceeds it).
    pub wg_per_wgp: u32,
}
impl Ring3Budget {
    pub fn lds_fits(&self) -> bool {
        self.lds <= self.lds_limit
    }
    pub fn vgprs_fit(&self) -> bool {
        self.vgprs <= self.vgpr_budget
    }
}
/// The depth-3 need-vs-budget numbers of `tile` (the feasibility record: see [`Spec::validate`]).
pub fn ring3_budget(tile: Tile) -> Ring3Budget {
    let pf_vgprs = tile.a_rounds() * 2 + tile.w_rounds() * 3;
    let lds = tile.layout().launch_for(RingDepth::Three);
    // Workgroups per WGP by LDS; a footprint over the launch limit is evaluated at the limit
    // (the best occupancy any trimmed layout could reach), so the budget is still reported.
    let wg_per_wgp = WGP_LDS / lds.min(WG_LDS_LIMIT);
    let waves_wgp = wg_per_wgp * tile.waves();
    let per_simd = waves_wgp.div_ceil(4);
    let vgpr_budget = (SIMD_VGPRS / per_simd / 8 * 8).min(VGPR_ARCH_MAX);
    Ring3Budget {
        pf_vgprs,
        vgprs: u32::from(VGPR_CEILING) + pf_vgprs,
        vgpr_budget,
        lds,
        lds_limit: WG_LDS_LIMIT,
        wg_per_wgp,
    }
}

impl Tile {
    pub const ALL: [Tile; 3] = [Tile::T128x128, Tile::T256x128, Tile::T128x256];
    /// Weight rows per workgroup.
    pub fn rows(self) -> u32 {
        match self {
            Self::T128x128 | Self::T128x256 => 128,
            Self::T256x128 => 256,
        }
    }
    /// Tokens per workgroup.
    pub fn tokens(self) -> u32 {
        match self {
            Self::T128x128 | Self::T256x128 => 128,
            Self::T128x256 => 256,
        }
    }
    pub fn waves(self) -> u32 {
        match self {
            Self::T128x128 => 8,
            Self::T256x128 | Self::T128x256 => 16,
        }
    }
    pub fn threads(self) -> u32 {
        self.waves() * 32
    }
    /// Staging rounds per 64-K slab (each lane moves one 8-byte chunk of its token / one 12-byte
    /// weight quad per round): activations / weights.
    pub fn a_rounds(self) -> u32 {
        self.tokens() * 4 / self.threads()
    }
    pub fn w_rounds(self) -> u32 {
        self.rows() * 4 / self.threads()
    }
    /// Slab rows covered by one staging round (`R = r * round_rows + tid / 4`).
    pub fn round_rows(self) -> u32 {
        self.threads() / 4
    }
    /// Token halves (64 tokens each) and row pairs (32 rows each): wave `w` owns token half
    /// `w % tok_halves` and row pair `w / tok_halves`.
    pub fn tok_halves(self) -> u32 {
        self.tokens() / 64
    }
    pub fn row_pairs(self) -> u32 {
        self.rows() / 32
    }
    /// `ds_store` / `buffer` immediates: LDS bytes between a staging round and the next.
    pub fn round_lds_stride(self) -> u32 {
        self.threads() / 128 * 2048
    }
    /// The CLI / variant grammar name: rows x tokens x waves.
    pub fn name(self) -> &'static str {
        match self {
            Self::T128x128 => "128x128x8",
            Self::T256x128 => "256x128x16",
            Self::T128x256 => "128x256x16",
        }
    }
    /// The shipped tile keeps the product symbol; the others add this suffix.
    pub fn symbol_suffix(self) -> &'static str {
        match self {
            Self::T128x128 => "",
            Self::T256x128 => "_t256x128",
            Self::T128x256 => "_t128x256",
        }
    }
    pub fn layout(self) -> Layout {
        match self {
            Self::T128x128 => LAYOUT,
            // Slot stride 24576: A0 W0 | A1 W1 | DS0 DS1 | SZ0 SZ1.
            Self::T256x128 => Layout {
                a: [0, 24576],
                w: [8192, 32768],
                ds: [49152, 50176],
                sz: [51200, 53248],
                a_bytes: 8192,
                w_bytes: 16384,
                ds_bytes: 1024,
                sz_bytes: 2048,
                launch: 55296,
            },
            Self::T128x256 => Layout {
                a: [0, 24576],
                w: [16384, 40960],
                ds: [49152, 51200],
                sz: [53248, 54272],
                a_bytes: 16384,
                w_bytes: 8192,
                ds_bytes: 2048,
                sz_bytes: 1024,
                launch: 55296,
            },
        }
    }
}
impl std::str::FromStr for Tile {
    type Err = String;
    /// `128x128`, `256x128`, `128x256` (rows x tokens), or the variant-name spellings
    /// `128x128x8`, `256x128x16`, `128x256x16`.
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "128x128" | "128x128x8" => Ok(Self::T128x128),
            "256x128" | "256x128x16" => Ok(Self::T256x128),
            "128x256" | "128x256x16" => Ok(Self::T128x256),
            _ => Err(format!(
                "unknown mq6_i8_gemm tile {s} (128x128|256x128|128x256)"
            )),
        }
    }
}

impl Epi {
    pub fn name(self) -> &'static str {
        match self {
            Self::Set => "set",
            Self::SetBf16 => "set_bf16",
            Self::SetNr => "set_nr",
        }
    }
}
impl Spec {
    pub fn product(epi: Epi) -> Self {
        Self {
            epi,
            raster: Raster::Banded,
            tile: Tile::T128x128,
            arch: Arch::Gfx1201,
            tune: Tune::default(),
        }
    }
    /// The interpreter twin: identical data path, linear raster.
    pub fn emulation(epi: Epi) -> Self {
        Self {
            epi,
            raster: Raster::Linear,
            tile: Tile::T128x128,
            arch: Arch::Gfx1201,
            tune: Tune::default(),
        }
    }
    /// The same spec on another tile.
    pub fn with_tile(self, tile: Tile) -> Self {
        Self { tile, ..self }
    }
    /// The same spec with other generator knobs.
    pub fn with_tune(self, tune: Tune) -> Self {
        Self { tune, ..self }
    }
    /// Whether the kernel takes `s_delay_alu` hints and (at ring depth 2) the stacked work-queue
    /// schedule: every epilogue but `SetNr`, whose shipped bytes are the plain schedule's.
    pub fn stacked(self) -> bool {
        self.epi != Epi::SetNr
    }
    pub fn module() -> &'static str {
        "gemm_mq6g256v2_mmq_i8_pm_gfx1201"
    }
    pub fn symbol(self) -> String {
        let base = format!(
            "gemm_mq6g256v2_mmq_i8_pm_{}_gfx1201{}",
            self.epi.name(),
            self.tile.symbol_suffix()
        );
        match self.raster {
            Raster::Banded => base,
            Raster::Linear => format!("{base}_linear"),
        }
    }
    /// The variant name; a non-default generator knob adds a suffix (`-r3`), so every existing
    /// variant name is unchanged.
    pub fn variant_name(self) -> String {
        format!(
            "{}-{}{}{}",
            self.tile.name(),
            self.epi.name(),
            match self.raster {
                Raster::Banded => "",
                Raster::Linear => "-linear",
            },
            match self.tune.ring {
                RingDepth::Two => String::new(),
                RingDepth::Three => format!("-r{}", self.tune.ring.name()),
            }
        )
    }
    /// The hipcc twin's ABI `(A, Xq, Y, M, K, N)` followed by K1's hidden block. `SetNr` keeps that
    /// block where it is and adds, past it, the norm weight and the RoPE table pointers; the
    /// seventh word (`unused`, byte 36) carries the head row stride minus one.
    pub fn kernargs(self) -> KernargLayout {
        let layout = Variant::FullSet.kernargs();
        if self.epi != Epi::SetNr {
            return layout;
        }
        let mut layout = layout;
        for a in &mut layout.args {
            if a.name == "unused" {
                a.name = "head_mask".into();
            }
        }
        layout.size = NR_KERNARG_BYTES;
        layout
            .pointer("norm", NR_NORM_OFFSET)
            .pointer("rope", NR_ROPE_OFFSET)
    }
    /// Reject combinations that have no generator, with the numbers when a budget is the reason.
    ///
    /// Ring depth 3 (`tune.ring`) needs, besides the unrolled three-group K-loop, a third A/W slab in
    /// LDS and a second raw prefetch set in VGPRs ([`ring3_budget`]): it is refused where either
    /// does not fit, and for the epilogue that reuses LDS after the K-loop (`SetNr`).
    pub fn validate(self) -> Result<(), String> {
        if self.arch != Arch::Gfx1201 {
            return Err("mq6_i8_gemm targets gfx1201 only".into());
        }
        if self.epi == Epi::SetNr && self.tile != Tile::T256x128 {
            return Err("mq6_i8_gemm set_nr runs on the 256x128 tile only".into());
        }
        if self.tune.ring == RingDepth::Three {
            let n = ring3_budget(self.tile);
            if self.epi == Epi::SetNr {
                return Err(
                    "ring depth 3 is not generated for set_nr: its epilogue reuses the K-loop's \
                     LDS rings (the depth-3 rings end in another state)"
                        .into(),
                );
            }
            if !n.lds_fits() {
                return Err(format!(
                    "ring depth 3 does not fit the {} tile: the launch needs {} B of LDS (three \
                     {} B A + {} B W slabs and {} B of DS/SZ planes), the workgroup limit is {} B \
                     (the depth-2 launch is {} B)",
                    self.tile.name(),
                    n.lds,
                    self.tile.layout().a_bytes,
                    self.tile.layout().w_bytes,
                    2 * (self.tile.layout().ds_bytes + self.tile.layout().sz_bytes),
                    n.lds_limit,
                    self.tile.layout().launch,
                ));
            }
            if !n.vgprs_fit() {
                return Err(format!(
                    "ring depth 3 does not fit the {} tile: the second prefetch set needs {} VGPRs \
                     on top of the {} of the depth-2 map ({} total), the budget at {} workgroup(s) \
                     per WGP is {}",
                    self.tile.name(),
                    n.pf_vgprs,
                    VGPR_CEILING,
                    n.vgprs,
                    n.wg_per_wgp,
                    n.vgpr_budget,
                ));
            }
        }
        Ok(())
    }
    /// The VGPR budget of this spec's register plan: the depth-2 [`VGPR_CEILING`], or at depth 3
    /// the occupancy budget of [`ring3_budget`].
    pub fn vgpr_budget(self) -> u16 {
        match self.tune.ring {
            RingDepth::Two => VGPR_CEILING,
            RingDepth::Three => ring3_budget(self.tile).vgpr_budget as u16,
        }
    }
    /// The dynamic LDS bytes this spec's launch passes: always the depth-2 launch (the product
    /// launcher passes one fixed size per symbol and the tune table carries none). A depth-3
    /// object declares the third A/W slab as static LDS ([`Spec::static_lds`]), so fixed +
    /// dynamic covers the depth-3 layout ([`Spec::total_lds`]).
    pub fn launch_lds(self) -> u32 {
        self.tile.layout().launch
    }
    /// The kernel's `group_segment_fixed_size`: `0` at depth 2, the third A+W slab at depth 3.
    pub fn static_lds(self) -> u32 {
        self.total_lds() - self.tile.layout().launch
    }
    /// LDS bytes the kernel addresses: fixed + dynamic (`Layout::launch_for`).
    pub fn total_lds(self) -> u32 {
        self.tile.layout().launch_for(self.tune.ring)
    }
}
impl std::str::FromStr for Epi {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "set" => Ok(Self::Set),
            "set_bf16" => Ok(Self::SetBf16),
            "set_nr" => Ok(Self::SetNr),
            _ => Err(format!("unknown mq6_i8_gemm epilogue {s}")),
        }
    }
}

/// `SetNr` kernel arguments past the hidden block: the bf16 norm weight pointer, the RoPE table
/// pointer (`[rows][64]` f32: cosines of bases 0..32 then sines), and the segment size.
pub const NR_NORM_OFFSET: u32 = 296;
pub const NR_ROPE_OFFSET: u32 = 304;
pub const NR_KERNARG_BYTES: u32 = 312;
