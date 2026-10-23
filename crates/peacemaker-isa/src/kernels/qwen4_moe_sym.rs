// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Builder-emitted Qwen4 grouped symmetric IU4 MoE GEMMs (fn-moe-sym route,
//! `HIPFIRE_QWEN4_MOE_SYM_IU4=1`): the SwiGLU gate/up and the down projection
//! over packed `block_i4_128` activations, for gfx1151
//! (`v_wmma_i32_16x16x16_iu4`, eight K16 steps per K128 epoch) and gfx1201
//! (`v_wmma_i32_16x16x32_iu4`, four K32 steps).
//!
//! Same 64-byte ABI (the 60-byte frozen ABI plus a trailing `n_experts`), grid, block and output bytes as the hand-written
//! `kernels/src/qwen4_moe_iu4_sym.gfx1151.hip`:
//! `(expert_weight_ptrs, expert_tile_ids, sorted_slot_index, Xq, Y_grouped,
//! M, K, x_row_div, m_total, x_src_rows, n_experts)`.
//! - gate/up: block 64 (two waves), grid `(M/64, m_total/16)`; wave `w` of
//!   CTA `x` owns gate rows `32x + 16w ..+16` and the up rows `M/2` further,
//!   and stores `rt(silu(rt(g)) * rt(u))` into `Y[P][M/2]` (BF16 bits).
//! - down: block 128 (four waves), grid `(M/64, m_total/16)`; wave `w` owns
//!   rows `64x + 16w ..+16` and stores `rt(sum)` into `Y[P][M]`.
//!
//! Every wave is independent: operands are loaded straight into registers
//! (the next K128 epoch prefetched while the current one is consumed), no
//! LDS and no barrier. The tile of 16 grouped slots is `blockIdx.y`; a tile
//! past `m_total` or with an expert id outside `[0, n_experts)` (the -1 sentinel, or a
//! corrupt id) returns before any load or store.
//! Each lane loads its slot `s = sorted[16*tile + lane%16]`; a padding slot
//! (`s < 0`) gets an out-of-range buffer offset, so its activation loads
//! return 0 without touching memory, and its outputs store +0.
//!
//! Numerics, per output and ascending epoch `h`: `C_h` is the exact int32
//! WMMA chain (A nibbles rebiased with XOR 0x88888888, signed A and X)
//! seeded with the magic 0x4b400000; `sum = fma(RN(d_h * sc_h), C_h -
//! 12582912.0, sum)` from `+0`, the HIP kernel's `fma(RN(sc*d), float(C),
//! sum)` exactly (`|C_h| <= 8192`). The row scale of each accumulator comes
//! from a `ds_swizzle_b32` broadcast: lane `(hi, k)` loads the header of the
//! row its broadcast partner needs (gfx11 C rows `2j + hi`, gfx12 `8hi + j`).
//! BF16 rounding is RNE with non-finite values passed through; the SiLU is
//! the hipcc `g / (1 + expf(-g)) * u` DAG ([`crate::kernels::gemm_uk::Epilogue::silu_dense`]).
//!
//! The gfx1151 down NT4 entry also builds row-repeat variants `..._nt4r2` and
//! `..._nt4r4` (`Spec::rr` = 2 or 4; see `nt`): one CTA folds `rr` contiguous
//! 64-row blocks of the same expert run, so the grid is `(M/(64*rr), m_total/16)`
//! and CTA `x` owns rows `64*rr*x + 64*blk + 16*wave`. `rr = 1` is the frozen
//! entry, byte for byte.
use super::bf16::Bf16;
use super::common::{
    self, add64, add64_imm, bload, bstore_b128, lit, op, s, s_add_i32, smem, sop, srd_tail, v,
    GatherTemps, SRD_WORD3,
};
use super::iu4_fold::{self, MAGIC, REBIAS};
use crate::{
    insn::{Instruction, MemoryClass},
    reg::Live,
    Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan, V,
};
use peacemaker_author::{End, Gfx1151, Gfx1201, Scc, Target, WgUniform, Workgroup};

type Wg<'b, T> = Workgroup<'b, T, Builder>;

/// `WeightCodec::Mq3I4` NT1 gate/up and down (own generators: `qwen4_moe_sym_w3.rs` for gfx1201,
/// whose `emit` dispatches gfx1151 to its sibling `qwen4_moe_sym_w3_gfx1151.rs`; the MQ4 entries
/// above and below are untouched).
#[path = "qwen4_moe_sym_w3.rs"]
pub mod w3;

/// E8P-I4 NT1 entries (own generator; existing MQ4/W3 modules are unchanged).
#[path = "qwen4_moe_sym_e8p.rs"]
pub mod e8p;

/// A `K`-derived trip counter: workgroup-uniform.
fn trips_insn(cmp: &str, trips: u8) -> Instruction {
    Instruction::new(format!("{cmp} s{trips}, 0"), vec![], vec![s(trips)])
}

fn trips_cmp<T: Target>(wg: &mut Wg<T>, cmp: &str, trips: u8) -> Result<WgUniform<Scc>, String> {
    wg.scmp_wg_uniform(trips_insn(cmp, trips))
}

/// Run the generic kernel body `$body::<T>(&mut Workgroup<T, Builder>, ..)`
/// with `T` the target type of `$arch`.
macro_rules! on_target {
    ($arch:expr, $b:expr, $body:ident($($arg:expr),*)) => {
        match $arch {
            Arch::Gfx1151 => $body(&mut Workgroup::<Gfx1151, Builder>::new($b)?, $($arg),*),
            Arch::Gfx1201 => $body(&mut Workgroup::<Gfx1201, Builder>::new($b)?, $($arg),*),
            a => Err(format!("qwen4_moe_sym: no typed target for {}", a.name())),
        }
    };
}

pub const KERNARG_BYTES: u32 = 64;
pub const VGPR_CEILING: u16 = 256;
/// Slot indices (`x_row_div * x_src_rows`) the activation gather resolves exactly.
pub const MAX_SLOTS: u32 = common::GATHER_MAX_INDEX;
/// SiLU elements per interleaved group (two groups per lane).
const SILU_N: u8 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    GateUp,
    Down,
}
impl Kind {
    pub const ALL: [Kind; 2] = [Kind::GateUp, Kind::Down];
    pub fn tag(self) -> &'static str {
        match self {
            Self::GateUp => "gate_up",
            Self::Down => "down",
        }
    }
    /// Weight tensors folded per wave (gate and up, or down).
    fn tensors(self) -> u8 {
        match self {
            Self::GateUp => 2,
            Self::Down => 1,
        }
    }
    /// Output columns of one CTA (`blockIdx.x` stride of the row base).
    pub fn cta_rows(self) -> u32 {
        match self {
            Self::GateUp => 32,
            Self::Down => 64,
        }
    }
    pub fn group_bytes(self) -> u32 {
        match self {
            Self::GateUp => 136,
            Self::Down => 68,
        }
    }
}
impl std::str::FromStr for Kind {
    type Err = String;
    fn from_str(t: &str) -> Result<Self, String> {
        match t {
            "gate_up" => Ok(Self::GateUp),
            "down" => Ok(Self::Down),
            _ => Err(format!("qwen4_moe_sym kind {t} (gate_up|down)")),
        }
    }
}

/// Expert weight codec of a grouped IU4 GEMM. `Mq4Sym` is the shipped
/// pointer-addressed QT44/QT53 symmetric MQ4 (every existing symbol and byte);
/// `Mq3I4` is the planar qt131/132 W3 grid (`d * (2c - 7)`, codes plane then
/// f16 scale plane, see `hipfire_runtime::mq_i4`) with the same exact IU4 fold.
/// Its numerics are frozen in `mq_i4_reference` (`iu4_gemm_f32`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WeightCodec {
    Mq4Sym,
    Mq3I4,
    E8pI4,
}

/// One entry: arch, GEMM kind, expert-run tile width `nt` (16-slot tiles
/// per weight stream) and row repeat `rr` (contiguous 64-row blocks folded
/// per CTA, gfx1151 down NT4 only). `nt == 1` is the frozen 16-slot entry,
/// the byte anchor of every wider tile; `rr == 1` is the anchor of every
/// row repeat.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Spec {
    pub arch: Arch,
    pub kind: Kind,
    pub nt: u8,
    pub rr: u8,
    pub codec: WeightCodec,
}

/// Tile widths built per arch: the anchor and the shipped expert-run tiles
/// (gfx1151 NT4; gfx1201 NT4 for VRAM-resident, NT8 for host-mapped experts).
pub fn tile_widths(arch: Arch) -> &'static [u8] {
    match arch {
        Arch::Gfx1201 => &[1, 4, 8],
        _ => &[1, 4],
    }
}

/// Row repeats of the down NT4 entry built per arch, after every
/// [`tile_widths`] entry (gfx1151 only).
pub fn row_repeats(arch: Arch) -> &'static [u8] {
    match arch {
        Arch::Gfx1151 => &[2, 4],
        _ => &[],
    }
}

/// Every entry of one architecture's module, in emission order: both kinds at
/// every tile width, then the down NT4 row repeats.
pub fn module_specs(arch: Arch) -> Vec<Spec> {
    module_specs_codec(arch, WeightCodec::Mq4Sym)
}

/// [`module_specs`] for one weight codec. The MQ3-I4 module (gfx1151, gfx1201) holds the
/// NT1 gate/up and down entries; the MQ4 module is the shipped one.
pub fn module_specs_codec(arch: Arch, codec: WeightCodec) -> Vec<Spec> {
    if matches!(codec, WeightCodec::Mq3I4 | WeightCodec::E8pI4) {
        let admitted = match codec {
            WeightCodec::Mq3I4 => matches!(arch, Arch::Gfx1151 | Arch::Gfx1201),
            _ => arch == Arch::Gfx1201,
        };
        return if admitted {
            Kind::ALL
                .into_iter()
                .map(|kind| Spec {
                    arch,
                    kind,
                    nt: 1,
                    rr: 1,
                    codec,
                })
                .collect()
        } else {
            Vec::new()
        };
    }
    tile_widths(arch)
        .iter()
        .flat_map(|&nt| {
            Kind::ALL.into_iter().map(move |kind| Spec {
                arch,
                kind,
                nt,
                rr: 1,
                codec,
            })
        })
        .chain(row_repeats(arch).iter().map(|&rr| Spec {
            arch,
            kind: Kind::Down,
            nt: 4,
            rr,
            codec,
        }))
        .collect()
}

impl Spec {
    /// One MQ4 entry from its CLI `--epi` spelling: `gate_up`/`down` (the
    /// 16-slot NT1 entry), `gate_up_ntN`/`down_ntN` (expert-run tile width
    /// `N`) or `down_nt4rR` (`R` contiguous 64-row blocks per CTA). The result
    /// is not validated (`emit` validates).
    pub fn from_epi(epi: &str, arch: Arch) -> Result<Self, String> {
        let (kind, tile) = match epi.rsplit_once("_nt") {
            Some((k, n)) => (k, Some(n)),
            None => (epi, None),
        };
        let (nt, rr) = match tile {
            Some(t) => {
                let (n, r) = t.split_once('r').map_or((t, None), |(n, r)| (n, Some(r)));
                (
                    n.parse::<u8>().map_err(|e| format!("--epi {epi}: {e}"))?,
                    match r {
                        Some(r) => r.parse::<u8>().map_err(|e| format!("--epi {epi}: {e}"))?,
                        None => 1,
                    },
                )
            }
            None => (1, 1),
        };
        Ok(Self {
            arch,
            kind: kind.parse()?,
            nt,
            rr,
            codec: WeightCodec::Mq4Sym,
        })
    }
    pub fn symbol(self) -> String {
        let infix = match self.codec {
            WeightCodec::Mq4Sym => "",
            WeightCodec::Mq3I4 => "_w3",
            WeightCodec::E8pI4 => "_e8p",
        };
        let base = match self.kind {
            Kind::GateUp => format!(
                "qwen4_moe_gate_up_silu_iu4_sym{infix}_pm_{}",
                self.arch.name()
            ),
            Kind::Down => format!("qwen4_moe_down_iu4_sym{infix}_pm_{}", self.arch.name()),
        };
        if self.nt == 1 {
            base
        } else if self.rr == 1 {
            format!("{base}_nt{}", self.nt)
        } else {
            format!("{base}_nt{}r{}", self.nt, self.rr)
        }
    }
    /// Contract/variant tag: `gate_up`, `down_nt4`, `down_nt4r2`; a non-MQ4
    /// codec appends `_w3`.
    pub fn variant(self) -> String {
        let tag = if self.nt == 1 {
            self.kind.tag().into()
        } else if self.rr == 1 {
            format!("{}_nt{}", self.kind.tag(), self.nt)
        } else {
            format!("{}_nt{}r{}", self.kind.tag(), self.nt, self.rr)
        };
        match self.codec {
            WeightCodec::Mq4Sym => tag,
            WeightCodec::Mq3I4 => format!("{tag}_w3"),
            WeightCodec::E8pI4 => format!("{tag}_e8p"),
        }
    }
    /// The committed MQ4 module (`WeightCodec::Mq4Sym`).
    pub fn module(arch: Arch) -> String {
        Self::module_codec(arch, WeightCodec::Mq4Sym)
    }
    pub fn module_codec(arch: Arch, codec: WeightCodec) -> String {
        match codec {
            WeightCodec::Mq4Sym => format!("qwen4_moe_iu4_sym_pm_{}", arch.name()),
            WeightCodec::Mq3I4 => format!("qwen4_moe_iu4_sym_w3_pm_{}", arch.name()),
            WeightCodec::E8pI4 => format!("qwen4_moe_iu4_sym_e8p_pm_{}", arch.name()),
        }
    }
    pub fn validate(self) -> Result<(), String> {
        if !matches!(self.arch, Arch::Gfx1151 | Arch::Gfx1201) {
            return Err("qwen4_moe_sym: built for gfx1151 and gfx1201 only".into());
        }
        // gfx11 operands are twice gfx12's per lane: NT8 exceeds the VGPR file.
        let max_nt = if self.arch.gfx12() { 8 } else { 4 };
        if !matches!(self.nt, 1 | 2 | 4 | 8) || self.nt > max_nt {
            return Err(format!(
                "qwen4_moe_sym: tile width {} (1, 2, 4{})",
                self.nt,
                if max_nt == 8 { " or 8" } else { "" }
            ));
        }
        if !matches!(self.rr, 1 | 2 | 4) {
            return Err(format!("qwen4_moe_sym: row repeat {} (1, 2 or 4)", self.rr));
        }
        if self.rr > 1 && (self.kind != Kind::Down || self.nt != 4 || self.arch.gfx12()) {
            return Err(format!(
                "qwen4_moe_sym: row repeat {} is built for the gfx1151 down NT4 entry only",
                self.rr
            ));
        }
        if self.codec == WeightCodec::Mq3I4
            && !(matches!(self.arch, Arch::Gfx1151 | Arch::Gfx1201) && self.nt == 1 && self.rr == 1)
        {
            return Err("qwen4_moe_sym: the MQ3-I4 codec is built for the gfx1151 and gfx1201 NT1 entries only".into());
        }
        if self.codec == WeightCodec::E8pI4
            && !(self.arch == Arch::Gfx1201 && self.nt == 1 && self.rr == 1)
        {
            return Err(
                "qwen4_moe_sym: the E8P-I4 codec is built for the gfx1201 NT1 entries only".into(),
            );
        }
        Ok(())
    }
    /// Waves per CTA: the 16-slot gate/up pairs gate and up rows in one wave
    /// (2 waves); every expert-run gate/up and every down uses 4.
    pub fn waves(self) -> u32 {
        if self.kind == Kind::GateUp && self.nt == 1 {
            2
        } else {
            4
        }
    }
    pub fn threads(self) -> u32 {
        self.waves() * 32
    }
    /// LDS bytes: the expert-run gate/up hands RNE(gate) rows to the up waves.
    pub fn lds_bytes(self) -> u32 {
        if self.codec == WeightCodec::E8pI4 {
            1536
        } else if self.kind == Kind::GateUp && self.nt > 1 {
            u32::from(self.nt) * NT_GATE_BYTES
        } else {
            0
        }
    }
    pub fn kernargs(self) -> KernargLayout {
        KernargLayout::new(KERNARG_BYTES)
            .pointer("expert_weight_ptrs", 0)
            .pointer("expert_tile_ids", 8)
            .pointer("sorted_slot_index", 16)
            .pointer("Xq", 24)
            .pointer("Y_grouped", 32)
            .hidden("M", 40, 4, "by_value")
            .hidden("K", 44, 4, "by_value")
            .hidden("x_row_div", 48, 4, "by_value")
            .hidden("m_total", 52, 4, "by_value")
            .hidden("x_src_rows", 56, 4, "by_value")
            .hidden("n_experts", 60, 4, "by_value")
    }
    /// Operand dwords per lane per K128 epoch of one 16-row/16-token fragment:
    /// gfx11 lanes `r` and `r+16` both carry all 16; gfx12 lanes split K32.
    fn aw(self) -> u8 {
        if self.arch.gfx12() {
            8
        } else {
            16
        }
    }
}

/// Routed-expert shape of one MoE layer. `hidden` is the activation width `H`
/// (the gate/up K and the down M), `intermediate` the per-expert width `I`
/// (the down K; the gate/up M is `2I`: gate rows then up rows).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertGeometry {
    pub experts: u32,
    pub top_k: u32,
    pub hidden: u32,
    pub intermediate: u32,
}

/// Largest `top_k` the plan admits (the flat slot index is `token * top_k + k`).
pub const MAX_TOP_K: u32 = 16;

/// One projection of a [`GeometryPlan`]: the kernarg `M`/`K`/`x_row_div` of its
/// entry and the quantities derived from them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProjectionGeometry {
    pub kind: Kind,
    /// Kernarg `M` (gate/up: `2I`; down: `H`).
    pub m: u32,
    /// Kernarg `K` (gate/up: `H`; down: `I`).
    pub k: u32,
    /// K128 epochs per output, `k / 128` (an odd count only for the down tail).
    pub epochs: u32,
    /// Kernarg `x_row_div`: gate/up reads the token row `slot / top_k`, down
    /// reads the flat slot row (`1`).
    pub x_row_div: u32,
    /// Grid x: `m / 64` CTAs (see [`Spec`]; the gfx1151 row repeats are not planned).
    pub grid_x: u32,
    /// Bytes of one weight row, `epochs * 68` (`(K/256)*136` for QT44,
    /// `(K/128)*68` for QT53).
    pub row_bytes: u32,
}

/// Validated grouped-GEMM geometry of one MoE layer for one arch and codec.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeometryPlan {
    pub arch: Arch,
    pub codec: WeightCodec,
    pub geometry: ExpertGeometry,
    pub gate_up: ProjectionGeometry,
    pub down: ProjectionGeometry,
    /// [`tile_widths`] of the arch.
    pub tile_widths: &'static [u8],
    /// Bytes of the `expert_weight_ptrs` descriptor table (`experts * 8`).
    pub descriptor_table_bytes: u32,
}

impl ExpertGeometry {
    /// Plan the gate/up (`M = 2I, K = H`) and down (`M = H, K = I`) entries.
    ///
    /// Refused unless: `experts >= 1`; `top_k` in `1..=16`; `Mq4Sym` codec on an
    /// arch whose every [`tile_widths`] entry [`Spec::validate`]s; gate/up
    /// `M % 64 == 0` and `K % 256 == 0`; down `M % 64 == 0` and `K % 128 == 0`;
    /// the `experts * 8` descriptor table fits `u32`; and the kernel's 32-bit
    /// row arithmetic holds (`row_bytes < 2^24` for the `v_mul_u32_u24` lane
    /// offsets, `M * row_bytes < 2^32` for the `s_mul_i32` weight row base).
    pub fn plan(self, arch: Arch, codec: WeightCodec) -> Result<GeometryPlan, String> {
        const WHAT: &str = "qwen4_moe_sym plan";
        if codec != WeightCodec::Mq4Sym {
            return Err(format!(
                "{WHAT}: codec {codec:?} (the QT44/QT53 symmetric MQ4 objects are Mq4Sym)"
            ));
        }
        let widths = tile_widths(arch);
        for &nt in widths {
            for kind in Kind::ALL {
                Spec {
                    arch,
                    kind,
                    nt,
                    rr: 1,
                    codec,
                }
                .validate()?;
            }
        }
        if self.experts == 0 {
            return Err(format!("{WHAT}: experts must be >= 1"));
        }
        if !(1..=MAX_TOP_K).contains(&self.top_k) {
            return Err(format!("{WHAT}: top_k {} (1..={MAX_TOP_K})", self.top_k));
        }
        let descriptor_table_bytes = self.experts.checked_mul(8).ok_or_else(|| {
            format!(
                "{WHAT}: {} experts: the expert_weight_ptrs table exceeds u32 bytes",
                self.experts
            )
        })?;
        let gate_up_m = self.intermediate.checked_mul(2).ok_or_else(|| {
            format!(
                "{WHAT}: intermediate {} overflows gate/up M = 2I",
                self.intermediate
            )
        })?;
        let gate_up = projection(Kind::GateUp, gate_up_m, self.hidden, 256, self.top_k)?;
        let down = projection(Kind::Down, self.hidden, self.intermediate, 128, 1)?;
        Ok(GeometryPlan {
            arch,
            codec,
            geometry: self,
            gate_up,
            down,
            tile_widths: widths,
            descriptor_table_bytes,
        })
    }
}

fn projection(
    kind: Kind,
    m: u32,
    k: u32,
    k_multiple: u32,
    x_row_div: u32,
) -> Result<ProjectionGeometry, String> {
    let tag = kind.tag();
    if m == 0 || m % 64 != 0 {
        return Err(format!(
            "qwen4_moe_sym plan: {tag} M {m} must be a nonzero multiple of 64"
        ));
    }
    if k == 0 || k % k_multiple != 0 {
        return Err(format!(
            "qwen4_moe_sym plan: {tag} K {k} must be a nonzero multiple of {k_multiple}"
        ));
    }
    let epochs = k / 128;
    let row_bytes = epochs
        .checked_mul(68)
        .filter(|&b| b < 1 << 24)
        .ok_or_else(|| {
            format!("qwen4_moe_sym plan: {tag} K {k}: row bytes exceed the 24-bit lane offset")
        })?;
    if u64::from(m) * u64::from(row_bytes) > u64::from(u32::MAX) {
        return Err(format!(
            "qwen4_moe_sym plan: {tag} {m} rows x {row_bytes} B exceed the 32-bit weight row base"
        ));
    }
    Ok(ProjectionGeometry {
        kind,
        m,
        k,
        epochs,
        x_row_div,
        grid_x: m / 64,
        row_bytes,
    })
}

impl GeometryPlan {
    /// The validated NT entry for `kind` at tile width `nt` (one of [`Self::tile_widths`]).
    pub fn spec(&self, kind: Kind, nt: u8) -> Result<Spec, String> {
        if !self.tile_widths.contains(&nt) {
            return Err(format!(
                "qwen4_moe_sym plan: tile width {nt} is not built for {}",
                self.arch.name()
            ));
        }
        let spec = Spec {
            arch: self.arch,
            kind,
            nt,
            rr: 1,
            codec: self.codec,
        };
        spec.validate()?;
        Ok(spec)
    }
}

/// Default-off schedule experiment. Chains belong to distinct output tiles;
/// no output's K-step or K128-fold order changes. Not part of `emit_module`.
#[derive(Clone, Copy, Debug)]
pub struct R4Spec {
    pub base: Spec,
    pub chains: u8,
    /// Register-direct epoch schedule: distance one when true, zero otherwise.
    pub prefetch: bool,
    /// Compact heaviest-first list of pad16 tile leaders. The experimental
    /// ABI is 72 bytes: the 64-byte ABI (ending in `n_experts` at 60), then a pointer
    /// to little-endian u32 leader indices at offset 64. Grid.y is its length.
    /// Use `gemm_uk::sched::plan(counts, 16 * nt)`, converting each item's
    /// expert-relative first_row to `(expert_pad16_prefix + first_row) / 16`.
    pub grouped: bool,
}

pub fn emit_r4(spec: R4Spec) -> Result<Emitted, String> {
    spec.base.validate()?;
    if spec.base.nt == 1
        || spec.base.rr != 1
        || !matches!(spec.chains, 1 | 2 | 4 | 8)
        || spec.chains > spec.base.nt
    {
        return Err("qwen4_moe_sym R4: NT >= 2, row repeat 1 and chains <= NT required".into());
    }
    let nt = u16::from(spec.base.nt);
    let aw = u16::from(spec.base.aw());
    let end =
        (72 + 8 * nt + 2 * aw + 2 + nt).div_ceil(8) * 8 + aw * nt + 8 * u16::from(spec.chains - 1);
    if end > VGPR_CEILING {
        return Err(format!(
            "qwen4_moe_sym R4: {end} VGPR exceeds {VGPR_CEILING}"
        ));
    }
    nt::emit_with(spec)
}

// VGPRs.
const TID: u8 = 0;
const XOFF: u8 = 16; // d offset of this lane's slot row (or X_OOB)
const XQOFF: u8 = 17; // nibble base offset (gfx12: + 8*hi)
const WOFF: u8 = 18; // lr*row_bytes (gfx12: + 8*hi)
const HOFF: u8 = 19; // header row * row_bytes
const YOFF: u8 = 20;
const MAGIC8: u8 = 24;
const CACC: u8 = 32; // int32 chains, 8 per tensor
const SUM: u8 = 48; // f32 sums, 8 per tensor
const SCF: u8 = 64; // broadcast row scales, 8 per tensor
const TPROD: u8 = 80; // fold products d*sc
const HF: u8 = 88; // f32 scale of this lane's header row, per tensor
const SET0: u8 = 92;
// Epilogue (aliases dead K-loop registers).
const RT_TMP: u8 = 32;
const OWN: u8 = 64;
const SEND: u8 = 68;
const RECV: u8 = 70;
const SEL: u8 = 72;
const OUT: u8 = 76;
const PACK: u8 = 80;

// SGPRs.
const KARG: u8 = 0;
const T0: u8 = 4;
const T1: u8 = 5;
const T64: u8 = 6;
const ARGS0: u8 = 8; // ptrs 8:9, tiles 10:11, sorted 12:13, xq 14:15
const ARGS1: u8 = 16; // Y 16:17, M 18, K 19
const ARGS2: u8 = 20; // x_row_div 20, m_total 21
const ARGS3: u8 = 22; // x_src_rows
const EID: u8 = 23;
const SRD_W: u8 = 24;
const SRD_U: u8 = 28;
const SRD_X: [u8; 2] = [32, 36];
const SRD_S: u8 = 40;
const TRIPS: u8 = 44;
const XS2: u8 = 45;
const ROWB: u8 = 46;
const TILE16: u8 = 47;
const WAVE: u8 = 48;
const RBASE: u8 = 49;
const WGX: u8 = 50;
const WGY: u8 = 51;
const LIVE: u8 = 52;
const HIMASK: u8 = 54;
const PSEL: u8 = 56;
const WPTR: u8 = 58;
const MASKT: [u8; 2] = [60, 62];
const SILU_MASK: u8 = 24;
const SRD_Y: u8 = 64;

struct Gen {
    spec: Spec,
    acc: bool,
}

impl Gen {
    fn arch(&self) -> Arch {
        self.spec.arch
    }
    /// Local labels are unique per module entry: the MQ3-I4 NT1 entry and its F32 qualification entry share a module.
    fn label(&self, name: &str) -> String {
        let entry = match (self.spec.codec, self.acc) {
            (WeightCodec::Mq4Sym, _) => "",
            (WeightCodec::Mq3I4, false) => "_w3",
            (WeightCodec::Mq3I4, true) => "_w3acc",
            (WeightCodec::E8pI4, false) => "_e8p",
            (WeightCodec::E8pI4, true) => "_e8pacc",
        };
        format!(".Lq4s_{}{entry}_{name}", self.spec.kind.tag())
    }
    fn t(&self) -> u8 {
        self.spec.kind.tensors()
    }
    fn aw(&self) -> u8 {
        self.spec.aw()
    }
    /// Set `p`'s registers: A of tensor t, X, header word of tensor t, d.
    fn set_base(&self, p: usize) -> u8 {
        SET0 + p as u8 * self.set_len()
    }
    fn set_len(&self) -> u8 {
        (self.aw() * (self.t() + 1) + self.t() + 1).div_ceil(4) * 4
    }
    fn a(&self, p: usize, t: u8) -> u8 {
        self.set_base(p) + self.aw() * t
    }
    fn x(&self, p: usize) -> u8 {
        self.set_base(p) + self.aw() * self.t()
    }
    fn h(&self, p: usize, t: u8) -> u8 {
        self.x(p) + self.aw() + t
    }
    fn d(&self, p: usize) -> u8 {
        self.h(p, self.t())
    }
    fn srd(&self, t: u8) -> u8 {
        if t == 0 {
            SRD_W
        } else {
            SRD_U
        }
    }
    /// Header and nibble byte offsets of set `p` inside one two-epoch trip.
    fn header_off(&self, p: usize, t: u8) -> u32 {
        let _ = t;
        match self.spec.kind {
            Kind::GateUp => 4 * p as u32,
            Kind::Down => 68 * p as u32,
        }
    }
    fn nibble_off(&self, p: usize) -> u32 {
        match self.spec.kind {
            Kind::GateUp => 8 + 64 * p as u32,
            Kind::Down => 4 + 68 * p as u32,
        }
    }

    fn plan(&self) -> Result<RegPlan, String> {
        let mut p = RegPlan::new(VGPR_CEILING, 104)?;
        let l = |n: &str| self.label(n);
        let kernel = || Live::Between("entry".into(), l("epilogue"));
        let pro = || Live::Between("entry".into(), l("k_begin"));
        let epi = || Live::Between(l("epilogue"), l("end"));
        p.v::<1>("tid", TID, pro())?;
        for i in 1..16u8 {
            p.v::<1>("prologue_tmp", i, pro())?;
        }
        for (name, r) in [("x_off", XOFF), ("w_off", WOFF), ("h_off", HOFF)] {
            p.v::<1>(name, r, kernel())?;
        }
        if self.arch().gfx12() {
            p.v::<1>("xq_off", XQOFF, kernel())?;
        }
        p.v::<1>("y_off", YOFF, Live::Whole)?;
        p.v::<8>("magic8", MAGIC8, kernel())?;
        for t in 0..self.t() {
            p.v::<8>("cacc", CACC + 8 * t, kernel())?;
            p.v::<8>("sum", SUM + 8 * t, Live::Whole)?;
            p.v::<8>("scale_rows", SCF + 8 * t, kernel())?;
            p.v::<1>("scale_f32", HF + t, kernel())?;
        }
        p.v::<8>("fold_t", TPROD, kernel())?;
        // Operand sets at their load granularity (gfx11 b128, gfx12 b64):
        // A per tensor, X, then the header words and d.
        for set in 0..2 {
            for chunk in 0..(self.t() + 1) * 4 {
                let base = self.a(set, 0) + chunk * self.aw() / 4;
                if self.arch().gfx12() {
                    p.v::<2>("operand", base, kernel())?;
                } else {
                    p.v::<4>("operand", base, kernel())?;
                }
            }
            for t in 0..self.t() {
                p.v::<1>("header", self.h(set, t), kernel())?;
            }
            p.v::<1>("d", self.d(set), kernel())?;
        }
        // Epilogue registers over the dead K-loop ranges.
        for i in 0..8u8 {
            p.v::<1>("rt_tmp", RT_TMP + i, epi())?;
        }
        if self.spec.kind == Kind::GateUp {
            for i in 0..(6 * SILU_N) {
                p.v::<1>("silu_tmp", SET0 + i, epi())?;
            }
            for i in 0..(3 * SILU_N) {
                p.s::<2>("silu_mask", SILU_MASK + 2 * i, epi())?;
            }
        }
        p.v::<4>("out", OUT, epi())?;
        if self.arch().gfx12() {
            // Packed rows come straight from the sums.
        } else {
            for i in 0..4u8 {
                p.v::<1>("own", OWN + i, epi())?;
            }
            for i in 0..2u8 {
                p.v::<1>("send", SEND + i, epi())?;
                p.v::<1>("recv", RECV + i, epi())?;
                p.v::<1>("sel", SEL + i, epi())?;
            }
            for i in 0..4u8 {
                p.v::<1>("pack", PACK + i, epi())?;
            }
            p.s::<2>("hi_mask", HIMASK, epi())?;
            p.s::<1>("permlane_sel", PSEL, epi())?;
        }
        p.s::<2>("kernarg_ptr", KARG, Live::Whole)?;
        p.s::<4>("srd_y", SRD_Y, epi())?;
        if !self.arch().gfx12() {
            p.s::<1>("wg_x_in", 2, Live::Whole)?;
            p.s::<1>("wg_y_in", 3, Live::Whole)?;
        }
        p.s::<1>("s_tmp0", T0, kernel())?;
        p.s::<1>("s_tmp1", T1, kernel())?;
        p.s::<2>("s_tmp64", T64, kernel())?;
        p.s::<8>("kernargs_0x00", ARGS0, kernel())?;
        p.s::<2>("y_base", ARGS1, Live::Whole)?;
        p.s::<2>("kernargs_m_k", ARGS1 + 2, kernel())?;
        p.s::<2>("kernargs_0x30", ARGS2, kernel())?;
        p.s::<1>("x_src_rows", ARGS3, kernel())?;
        p.s::<1>("expert", EID, kernel())?;
        for t in 0..self.t() {
            p.s::<4>("srd_w", self.srd(t), kernel())?;
        }
        for r in SRD_X {
            p.s::<4>("srd_x", r, kernel())?;
        }
        p.s::<4>("srd_sorted", SRD_S, kernel())?;
        for (name, r) in [
            ("trips", TRIPS),
            ("x_stride2", XS2),
            ("row_bytes", ROWB),
            ("tile16", TILE16),
            ("wave", WAVE),
            ("row_base", RBASE),
            ("wg_x", WGX),
            ("wg_y", WGY),
        ] {
            p.s::<1>(name, r, kernel())?;
        }
        p.s::<2>("w_ptr", WPTR, kernel())?;
        p.s::<2>("live", LIVE, Live::Whole)?;
        for m in MASKT {
            p.s::<2>("cmp_mask", m, Live::Whole)?;
        }
        // Keep the reservation a whole 8-VGPR granule, so the descriptor's
        // allocation equals the metadata count.
        let top = p.next_free_vgpr();
        if top % 8 != 0 {
            p.v::<1>("granule_pad", (top.div_ceil(8) * 8 - 1) as u8, Live::Whole)?;
        }
        Ok(p)
    }
}

/// Kernel arguments, workgroup and lane ids, the wave index, `TILE16`.
fn pro_front(b: &mut Builder, g: &Gen) -> Result<(), String> {
    let a = g.arch();
    smem(b, ARGS0, 8, KARG, 0)?;
    smem(b, ARGS1, 2, KARG, 0x20)?;
    smem(b, ARGS1 + 2, 2, KARG, 0x28)?;
    smem(b, ARGS2, 2, KARG, 0x30)?;
    smem(b, ARGS3, 1, KARG, 0x38)?;
    if a.gfx12() {
        // Workgroup ids: ttmp9 (x), ttmp7[15:0] (y).
        sop(b, format!("s_mov_b32 s{WGX}, ttmp9"), &[WGX], &[])?;
        sop(b, format!("s_and_b32 s{WGY}, ttmp7, 0xffff"), &[WGY], &[])?;
    } else {
        sop(b, format!("s_mov_b32 s{WGX}, s2"), &[WGX], &[2])?;
        sop(b, format!("s_mov_b32 s{WGY}, s3"), &[WGY], &[3])?;
    }
    // v1 = wave, v2 = lane, v3 = lr, v4 = hi.
    op(b, "v_lshrrev_b32_e32 v1, 5, v0", &[v(1)], &[v(TID)])?;
    op(b, "v_and_b32_e32 v2, 31, v0", &[v(2)], &[v(TID)])?;
    op(
        b,
        format!("v_readfirstlane_b32 s{WAVE}, v1"),
        &[s(WAVE)],
        &[v(1)],
    )?;
    op(b, "v_and_b32_e32 v3, 15, v2", &[v(3)], &[v(2)])?;
    op(b, "v_lshrrev_b32_e32 v4, 4, v2", &[v(4)], &[v(2)])?;
    // A tile at or past m_total, or a sentinel tile, returns before any store.
    sop(
        b,
        format!("s_lshl_b32 s{TILE16}, s{WGY}, 4"),
        &[TILE16],
        &[WGY],
    )?;
    Ok(())
}

/// `s_cmp_ge_i32 TILE16, m_total`: the tile is past the end (the workgroup returns).
fn outside_insn() -> Instruction {
    Instruction::new(
        format!("s_cmp_ge_i32 s{TILE16}, s{}", ARGS2 + 1),
        vec![],
        vec![s(TILE16), s(ARGS2 + 1)],
    )
}

/// The tile's expert id and `n_experts`.
fn pro_expert(b: &mut Builder) -> Result<(), String> {
    sop(b, format!("s_lshl_b32 s{T0}, s{WGY}, 2"), &[T0], &[WGY])?;
    add64(b, T64, ARGS0 + 2, T0)?;
    smem(b, EID, 1, T64, 0)?;
    // Fail closed: an expert id outside [0, n_experts) (the -1 sentinel, or a corrupt / non-finite-router id) exits
    // before any `expert_weight_ptrs[id]` load. One unsigned compare covers both bounds (negative ids are huge).
    smem(b, T1, 1, KARG, 0x3c)?;
    Ok(())
}

/// `s_cmp_ge_u32 expert, n_experts`: a sentinel or corrupt expert id (the workgroup returns).
fn sentinel_insn() -> Instruction {
    Instruction::new(
        format!("s_cmp_ge_u32 s{EID}, s{T1}"),
        vec![],
        vec![s(EID), s(T1)],
    )
}

/// The expert's weight pointer, trips, row base, descriptors, lane offsets and the first set's loads.
fn pro_setup(b: &mut Builder, g: &Gen) -> Result<(), String> {
    let a = g.arch();
    let kind = g.spec.kind;
    sop(b, format!("s_lshl_b32 s{T0}, s{EID}, 3"), &[T0], &[EID])?;
    add64(b, T64, ARGS0, T0)?;
    smem(b, WPTR, 2, T64, 0)?;
    // row_bytes = (K/256)*136 (QT44) or (K/128)*68 (QT53).
    let k = ARGS1 + 3;
    match kind {
        Kind::GateUp => {
            sop(b, format!("s_lshr_b32 s{ROWB}, s{k}, 8"), &[ROWB], &[k])?;
            sop(b, format!("s_mulk_i32 s{ROWB}, 0x88"), &[ROWB], &[ROWB])?;
        }
        Kind::Down => {
            sop(b, format!("s_lshr_b32 s{ROWB}, s{k}, 7"), &[ROWB], &[k])?;
            sop(b, format!("s_mulk_i32 s{ROWB}, 0x44"), &[ROWB], &[ROWB])?;
        }
    }
    // Two-epoch trips before the tail: (K/128 - 1) / 2.
    sop(b, format!("s_lshr_b32 s{TRIPS}, s{k}, 7"), &[TRIPS], &[k])?;
    sop(
        b,
        format!("{} s{TRIPS}, s{TRIPS}, -1", s_add_i32(a)),
        &[TRIPS],
        &[TRIPS],
    )?;
    sop(
        b,
        format!("s_lshr_b32 s{TRIPS}, s{TRIPS}, 1"),
        &[TRIPS],
        &[TRIPS],
    )?;
    // Row base of this wave: x*cta_rows + 16*wave.
    let shift = if kind == Kind::GateUp { 5 } else { 6 };
    sop(
        b,
        format!("s_lshl_b32 s{RBASE}, s{WGX}, {shift}"),
        &[RBASE],
        &[WGX],
    )?;
    sop(b, format!("s_lshl_b32 s{T0}, s{WAVE}, 4"), &[T0], &[WAVE])?;
    sop(
        b,
        format!("{} s{RBASE}, s{RBASE}, s{T0}", s_add_i32(a)),
        &[RBASE],
        &[RBASE, T0],
    )?;
    // Sorted slots of this tile.
    sop(b, format!("s_lshl_b32 s{T0}, s{WGY}, 6"), &[T0], &[WGY])?;
    add64(b, SRD_S, ARGS0 + 4, T0)?;
    sop(
        b,
        format!("s_mov_b32 s{}, 64", SRD_S + 2),
        &[SRD_S + 2],
        &[],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, {}", SRD_S + 3, lit(SRD_WORD3)),
        &[SRD_S + 3],
        &[],
    )?;
    op(b, "v_lshlrev_b32_e32 v5, 2, v3", &[v(5)], &[v(3)])?;
    bload(b, 1, 6, 5, SRD_S, 0)?;
    // Activation descriptors: epoch 0 (X0) and 1 (X1), x_src_rows*72 records.
    sop(
        b,
        format!("s_mul_i32 s{T0}, s{ARGS3}, 0x48"),
        &[T0],
        &[ARGS3],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{}", SRD_X[0], ARGS0 + 6),
        &[SRD_X[0]],
        &[ARGS0 + 6],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{}", SRD_X[0] + 1, ARGS0 + 7),
        &[SRD_X[0] + 1],
        &[ARGS0 + 7],
    )?;
    srd_tail(b, SRD_X[0], Some(T0))?;
    add64(b, SRD_X[1], ARGS0 + 6, T0)?;
    srd_tail(b, SRD_X[1], Some(T0))?;
    sop(b, format!("s_lshl_b32 s{XS2}, s{T0}, 1"), &[XS2], &[T0])?;
    // Weight descriptors: rows rbase.. of the expert (up rows M/2 further).
    sop(
        b,
        format!("s_mul_i32 s{T0}, s{RBASE}, s{ROWB}"),
        &[T0],
        &[RBASE, ROWB],
    )?;
    add64(b, SRD_W, WPTR, T0)?;
    srd_tail(b, SRD_W, None)?;
    if kind == Kind::GateUp {
        sop(
            b,
            format!("s_lshr_b32 s{T1}, s{}, 1", ARGS1 + 2),
            &[T1],
            &[ARGS1 + 2],
        )?;
        sop(
            b,
            format!("s_mul_i32 s{T1}, s{T1}, s{ROWB}"),
            &[T1],
            &[T1, ROWB],
        )?;
        add64(b, SRD_U, SRD_W, T1)?;
        srd_tail(b, SRD_U, None)?;
    }
    // Lane offsets. Weights: lr*row_bytes (gfx12 + 8*hi, its K32 half).
    op(
        b,
        format!("v_mul_u32_u24_e32 v{WOFF}, s{ROWB}, v3"),
        &[v(WOFF)],
        &[s(ROWB), v(3)],
    )?;
    if a.gfx12() {
        op(
            b,
            format!("v_lshl_add_u32 v{WOFF}, v4, 3, v{WOFF}"),
            &[v(WOFF)],
            &[v(4), v(WOFF)],
        )?;
    }
    // Header row: the row whose scale this lane's broadcast slot serves.
    if a.gfx12() {
        op(b, "v_and_b32_e32 v7, 7, v3", &[v(7)], &[v(3)])?;
        op(b, "v_lshl_or_b32 v7, v4, 3, v7", &[v(7)], &[v(4), v(7)])?;
    } else {
        op(b, "v_and_b32_e32 v7, 14, v3", &[v(7)], &[v(3)])?;
        op(b, "v_or_b32_e32 v7, v7, v4", &[v(7)], &[v(7), v(4)])?;
    }
    op(
        b,
        format!("v_mul_u32_u24_e32 v{HOFF}, s{ROWB}, v7"),
        &[v(HOFF)],
        &[s(ROWB), v(7)],
    )?;
    // Y offset: ((16*tile + lr)*row_len + rbase + 8*hi)*2.
    match kind {
        Kind::GateUp => sop(
            b,
            format!("s_lshr_b32 s{T1}, s{}, 1", ARGS1 + 2),
            &[T1],
            &[ARGS1 + 2],
        )?,
        Kind::Down => sop(
            b,
            format!("s_mov_b32 s{T1}, s{}", ARGS1 + 2),
            &[T1],
            &[ARGS1 + 2],
        )?,
    }
    op(
        b,
        format!("v_add_nc_u32_e32 v8, s{TILE16}, v3"),
        &[v(8)],
        &[s(TILE16), v(3)],
    )?;
    op(
        b,
        format!("v_mul_lo_u32 v8, v8, s{T1}"),
        &[v(8)],
        &[v(8), s(T1)],
    )?;
    op(
        b,
        format!("v_lshl_add_u32 v9, v4, 3, s{RBASE}"),
        &[v(9)],
        &[v(4), s(RBASE)],
    )?;
    op(b, "v_add_nc_u32_e32 v8, v8, v9", &[v(8)], &[v(8), v(9)])?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{YOFF}, 1, v8"),
        &[v(YOFF)],
        &[v(8)],
    )?;
    // Activation row offset: (slot / x_row_div) * 72, or the gather's
    // out-of-range offset for a padding slot so its loads read 0.
    common::gather_offset(
        b,
        XOFF,
        6,
        Some(ARGS2),
        iu4_fold::XBLK_BYTES,
        LIVE,
        GatherTemps {
            v: [11, 12, 13, 10],
            mask: MASKT[0],
        },
    )?;
    if a.gfx12() {
        op(
            b,
            format!("v_lshl_add_u32 v{XQOFF}, v4, 3, v{XOFF}"),
            &[v(XQOFF)],
            &[v(4), v(XOFF)],
        )?;
    }
    load_set(b, g, 0)?;
    Ok(())
}

/// The fold magic and the zeroed sums.
fn pro_init(b: &mut Builder, g: &Gen) -> Result<(), String> {
    for j in 0..8u8 {
        op(
            b,
            format!("v_mov_b32_e32 v{}, {}", MAGIC8 + j, lit(MAGIC)),
            &[v(MAGIC8 + j)],
            &[],
        )?;
    }
    for t in 0..g.t() {
        for j in 0..8u8 {
            op(
                b,
                format!("v_mov_b32_e32 v{}, 0", SUM + 8 * t + j),
                &[v(SUM + 8 * t + j)],
                &[],
            )?;
        }
    }
    Ok(())
}

fn prologue<T: Target>(wg: &mut Wg<T>, g: &Gen, end: &End) -> Result<(), String> {
    pro_front(wg.isa(), g)?;
    let outside = wg.scmp_wg_uniform(outside_insn())?;
    wg.exit_if(outside, end)?;
    pro_expert(wg.isa())?;
    let sentinel = wg.scmp_wg_uniform(sentinel_insn())?;
    wg.exit_if(sentinel, end)?;
    pro_setup(wg.isa(), g)?;
    pro_init(wg.isa(), g)
}

/// Issue set `p`'s loads (epoch parity p of the current trip): header words,
/// d, the activation nibbles, then the weight nibbles, as one clause.
fn load_set(b: &mut Builder, g: &Gen, p: usize) -> Result<(), String> {
    let gfx12 = g.arch().gfx12();
    let xq = if gfx12 { XQOFF } else { XOFF };
    let (width, step) = if gfx12 { (2u8, 2u8) } else { (4u8, 4u8) };
    b.clause(|b| {
        for t in 0..g.t() {
            bload(b, 1, g.h(p, t), HOFF, g.srd(t), g.header_off(p, t))?;
        }
        bload(b, 1, g.d(p), XOFF, SRD_X[p], 0)?;
        for i in 0..4u8 {
            bload(
                b,
                width,
                g.x(p) + step * i,
                xq,
                SRD_X[p],
                8 + 16 * u32::from(i),
            )?;
        }
        for t in 0..g.t() {
            for i in 0..4u8 {
                bload(
                    b,
                    width,
                    g.a(p, t) + step * i,
                    WOFF,
                    g.srd(t),
                    g.nibble_off(p) + 16 * u32::from(i),
                )?;
            }
        }
        Ok(())
    })
}

/// Fold set `p` (one K128 epoch) into the sums.
fn compute(b: &mut Builder, g: &Gen, p: usize) -> Result<(), String> {
    let a = g.arch();
    // Row scales: this lane's header word -> f32 -> the eight rows of its
    // accumulators by ds_swizzle broadcast within each 16-lane half.
    for t in 0..g.t() {
        op(
            b,
            format!("v_cvt_f32_f16_e64 v{}, v{}.l", HF + t, g.h(p, t)),
            &[v(HF + t)],
            &[v(g.h(p, t))],
        )?;
        for j in 0..8u8 {
            let k = if a.gfx12() { j } else { 2 * j };
            let text = format!(
                "ds_swizzle_b32 v{}, v{} offset:swizzle(BROADCAST,16,{k})",
                SCF + 8 * t + j,
                HF + t
            );
            b.ds_crosslane(
                crate::insn::Instruction::new(text, vec![v(SCF + 8 * t + j)], vec![v(HF + t)])
                    .memory(MemoryClass::DsLoad),
            )?;
        }
    }
    for t in 0..g.t() {
        for i in 0..g.aw() {
            let r = g.a(p, t) + i;
            op(
                b,
                format!("v_xor_b32_e32 v{r}, {}, v{r}", lit(REBIAS)),
                &[v(r)],
                &[v(r)],
            )?;
        }
    }
    for i in 0..g.aw() / 2 {
        for t in 0..g.t() {
            iu4_fold::wmma_step(
                b,
                a,
                V::<8>(CACC + 8 * t),
                V::<2>(g.a(p, t) + 2 * i),
                V::<2>(g.x(p) + 2 * i),
                i == 0,
                V::<8>(MAGIC8),
            )?;
        }
    }
    for t in 0..g.t() {
        iu4_fold::fold_pass(b, CACC + 8 * t, SUM + 8 * t, SCF + 8 * t, g.d(p), TPROD)?;
    }
    Ok(())
}

fn advance(b: &mut Builder, g: &Gen) -> Result<(), String> {
    for t in 0..g.t() {
        add64_imm(b, g.srd(t), 136)?;
    }
    for r in SRD_X {
        add64(b, r, r, XS2)?;
    }
    Ok(())
}

/// One trip done: `s_add_i32 s{TRIPS}, s{TRIPS}, -1`.
fn trips_dec(b: &mut Builder) -> Result<(), String> {
    op(
        b,
        format!("{} s{TRIPS}, s{TRIPS}, -1", s_add_i32(b.spec.arch)),
        &[s(TRIPS)],
        &[s(TRIPS)],
    )
}

/// An odd K/128 leaves one epoch after the trips, not two: `s_bitcmp1_b32 K, 7`.
fn odd_insn() -> Instruction {
    Instruction::new(
        format!("s_bitcmp1_b32 s{}, 7", ARGS1 + 3),
        vec![],
        vec![s(ARGS1 + 3)],
    )
}

/// The two epochs left after the trips: set 1's loads, then folds of sets 0 and 1.
fn tail_pair(b: &mut Builder, g: &Gen) -> Result<(), String> {
    load_set(b, g, 1)?;
    compute(b, g, 0)?;
    compute(b, g, 1)
}

/// Two-epoch trips over `TRIPS` (skipped at zero trips), then the tail
/// epochs.
fn kloop<T: Target>(wg: &mut Wg<T>, g: &Gen) -> Result<(), String> {
    wg.label(&g.label("k_begin"))?;
    let (head, done) = (g.label("k_loop"), g.label("k_loop_end"));
    let entry = wg.isa().ledger.shape();
    let no_trip = trips_cmp(wg, "s_cmp_eq_u32", TRIPS)?;
    wg.wg_skip_if(no_trip, &done, (), |wg, ()| {
        wg.loop_carried(&head, (), |wg, ()| {
            let b = wg.isa();
            load_set(b, g, 1)?;
            compute(b, g, 0)?;
            advance(b, g)?;
            load_set(b, g, 0)?;
            compute(b, g, 1)?;
            trips_dec(b)?;
            Ok(((), trips_cmp(wg, "s_cmp_lg_u32", TRIPS)?))
        })?;
        if wg.isa().ledger.shape() != entry {
            return Err("k loop exit ledger differs from its entry".into());
        }
        Ok(())
    })?;
    match g.spec.kind {
        // K % 256 == 0: an even epoch count, two epochs left.
        Kind::GateUp => tail_pair(wg.isa(), g),
        // K % 128 == 0: one epoch left when K/128 is odd, else two.
        Kind::Down => {
            let odd = wg.scmp(odd_insn())?;
            wg.if_else(
                odd,
                &g.label("tail_odd"),
                &g.label("epilogue"),
                |w| tail_pair(w.isa(), g),
                |w| compute(w.isa(), g, 0),
            )
        }
    }
}

/// The epilogue's label (the join of the tail's `if_else`).
fn epilogue_label(b: &mut Builder, g: &Gen) -> Result<(), String> {
    b.label(&g.label("epilogue"))
}

/// `gate_up`: g and u rounded to BF16, then the shipped SwiGLU expression, in the g registers.
fn swiglu(b: &mut Builder, g: &Gen) -> Result<(), String> {
    if g.spec.kind != Kind::GateUp {
        return Err("swiglu: the gate_up entry only".into());
    }
    for t in 0..2u8 {
        for j in 0..8u8 {
            Bf16::rne_finite_passthrough(
                b,
                SUM + 8 * t + j,
                RT_TMP + j,
                MASKT[usize::from(j % 2)],
                true,
            )?;
        }
    }
    for grp in 0..2u8 {
        crate::kernels::gemm_uk::Epilogue::silu_dense(
            b,
            SUM + SILU_N * grp,
            SUM + 8 + SILU_N * grp,
            SET0,
            SILU_MASK,
            SILU_N,
        )?;
    }
    Ok(())
}

/// The rounded BF16 rows of this lane, packed and stored at `Y`.
fn store(b: &mut Builder, g: &Gen) -> Result<(), String> {
    let vals = SUM;
    for j in 0..8u8 {
        Bf16::rne_finite_passthrough(b, vals + j, RT_TMP + j, MASKT[usize::from(j % 2)], false)?;
    }
    // Padding slots store +0.
    for j in 0..8u8 {
        op(
            b,
            format!("v_cndmask_b32_e64 v{0}, 0, v{0}, s{LIVE}", vals + j),
            &[v(vals + j)],
            &[v(vals + j), s(LIVE)],
        )?;
    }
    let perm =
        |b: &mut Builder, dst: u8, hi_src: u8, lo_src: u8, sel: String| -> Result<(), String> {
            let mut uses = vec![v(hi_src), v(lo_src)];
            if sel.starts_with('v') {
                uses.push(v(sel[1..].parse::<u8>().map_err(|e| e.to_string())?));
            }
            op(
                b,
                format!("v_perm_b32 v{dst}, v{hi_src}, v{lo_src}, {sel}"),
                &[v(dst)],
                &uses,
            )
        };
    if g.arch().gfx12() {
        // Lane (hi, lr) holds rows 8hi..8hi+7 of token lr: pack and store.
        for k in 0..4u8 {
            perm(b, OUT + k, vals + 2 * k + 1, vals + 2 * k, lit(0x0706_0302))?;
        }
    } else {
        // Lane (hi, lr) holds rows 2j+hi of token lr. Exchange with the
        // other half so lane hi stores rows 8hi..8hi+7: lane 0 sends rows
        // 8..14 (even), lane 1 sends rows 1..7 (odd), via v_permlanex16.
        // A VOP3 lane-mask operand is a pair for M7; the high word is unused in wave32.
        op(
            b,
            format!("s_mov_b32 s{HIMASK}, 0xffff0000"),
            &[s(HIMASK)],
            &[],
        )?;
        op(
            b,
            format!("s_mov_b32 s{}, 0", HIMASK + 1),
            &[s(HIMASK + 1)],
            &[],
        )?;
        op(b, format!("s_mov_b32 s{PSEL}, 0x76543210"), &[s(PSEL)], &[])?;
        for k in 0..4u8 {
            op(
                b,
                format!(
                    "v_cndmask_b32_e64 v{}, v{}, v{}, s{HIMASK}",
                    OWN + k,
                    vals + k,
                    vals + 4 + k
                ),
                &[v(OWN + k)],
                &[v(vals + k), v(vals + 4 + k), s(HIMASK)],
            )?;
        }
        for m in 0..2u8 {
            // hi = 0 sends pack(H[4+2m], H[5+2m]); hi = 1 sends pack(H[2m], H[2m+1]).
            perm(
                b,
                PACK + 2 * m,
                vals + 5 + 2 * m,
                vals + 4 + 2 * m,
                lit(0x0706_0302),
            )?;
            perm(
                b,
                PACK + 2 * m + 1,
                vals + 1 + 2 * m,
                vals + 2 * m,
                lit(0x0706_0302),
            )?;
            op(
                b,
                format!(
                    "v_cndmask_b32_e64 v{}, v{}, v{}, s{HIMASK}",
                    SEND + m,
                    PACK + 2 * m,
                    PACK + 2 * m + 1
                ),
                &[v(SEND + m)],
                &[v(PACK + 2 * m), v(PACK + 2 * m + 1), s(HIMASK)],
            )?;
            op(
                b,
                format!("v_mov_b32_e32 v{}, v{}", RECV + m, SEND + m),
                &[v(RECV + m)],
                &[v(SEND + m)],
            )?;
            op(
                b,
                format!(
                    "v_permlanex16_b32 v{}, v{}, s{PSEL}, 0xfedcba98",
                    RECV + m,
                    SEND + m
                ),
                &[v(RECV + m)],
                &[v(RECV + m), v(SEND + m), s(PSEL)],
            )?;
        }
        // Byte selectors (src0 = received word, src1 = own row): hi = 0
        // puts its own row low, hi = 1 puts the received row low.
        for (i, (lo, hi)) in [(0x0504_0302u32, 0x0302_0504u32), (0x0706_0302, 0x0302_0706)]
            .into_iter()
            .enumerate()
        {
            let r = SEL + i as u8;
            op(b, format!("v_mov_b32_e32 v{r}, {}", lit(hi)), &[v(r)], &[])?;
            op(
                b,
                format!("v_cndmask_b32_e64 v{r}, {}, v{r}, s{HIMASK}", lit(lo)),
                &[v(r)],
                &[v(r), s(HIMASK)],
            )?;
        }
        for k in 0..4u8 {
            perm(
                b,
                OUT + k,
                RECV + k / 2,
                OWN + k,
                format!("v{}", SEL + k % 2),
            )?;
        }
    }
    // Raw buffer store over Y (the same VMEM family as every load).
    sop(
        b,
        format!("s_mov_b32 s{SRD_Y}, s{ARGS1}"),
        &[SRD_Y],
        &[ARGS1],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{}", SRD_Y + 1, ARGS1 + 1),
        &[SRD_Y + 1],
        &[ARGS1 + 1],
    )?;
    srd_tail(b, SRD_Y, None)?;
    bstore_b128(b, OUT, YOFF, SRD_Y, 0)
}

fn epilogue(b: &mut Builder, g: &Gen) -> Result<(), String> {
    epilogue_label(b, g)?;
    if g.spec.kind == Kind::GateUp {
        swiglu(b, g)?;
    }
    store(b, g)
}

/// The position-indexed-activation twin of a gfx1201 MQ4 down NT4 / NT8 entry (`qwen4_moe_down_iu4_sym_gp_pm_gfx1201_nt{4,8}`):
/// the same grid, tile walk, fold order, rounding and stores, reading the down A4 plane written in grouped-position order
/// (record `g * m_total + 16 * tile + lane`, see `moe_rotate_i4_pm::emit_gp`) instead of gathering `sorted[p]`-indexed records.
/// Outputs are byte-identical to the flat-slot entry's over the same activations; a padding lane or dead tile still reads
/// zeros through an out-of-range offset. `x_src_rows` is ignored (the plane stride is `m_total * 72`).
pub fn emit_down_gp(spec: Spec) -> Result<Emitted, String> {
    spec.validate()?;
    if spec.arch != Arch::Gfx1201
        || spec.kind != Kind::Down
        || spec.codec != WeightCodec::Mq4Sym
        || spec.rr != 1
        || !matches!(spec.nt, 4 | 8)
    {
        return Err("qwen4_moe_sym gp: the gfx1201 MQ4 down NT4/NT8 entries only".into());
    }
    nt::emit_gp(spec)
}

pub fn emit(spec: Spec) -> Result<Emitted, String> {
    spec.validate()?;
    if spec.codec == WeightCodec::Mq3I4 {
        return w3::emit(spec);
    }
    if spec.codec == WeightCodec::E8pI4 {
        return e8p::emit(spec);
    }
    if spec.nt > 1 {
        return nt::emit(spec);
    }
    let g = Gen { spec, acc: false };
    let mut b = Builder::new(kernel_spec_nt1(spec), g.plan()?);
    b.enable_delay_alu();
    on_target!(spec.arch, &mut b, body(&g))?;
    b.finish()
}

fn body<T: Target>(wg: &mut Wg<T>, g: &Gen) -> Result<(), String> {
    let end = wg.exit(&g.label("end"))?;
    prologue(wg, g, &end)?;
    kloop(wg, g)?;
    epilogue(wg.isa(), g)?;
    wg.end(end)
}

/// The MQ4 NT1 entries (`gate_up` over two waves, `down` over four) as builder-level blocks, so a
/// `.rip` script (`qwen4_moe_sym.rip`) owns the kernel's structure: the exits, the K-loop's trips
/// and the order of set loads and folds inside a trip, the tail and the exit. Every block is a
/// straight run of instructions (or one `Instruction` for a compare), emitted by the same function
/// the Rust [`emit`] calls. The NT1 entries have no LDS and no hand-off.
pub mod nt1_blocks {
    use super::*;

    /// Register assignment, labels and launch contract of one NT1 entry.
    pub struct Gen {
        g: super::Gen,
    }

    impl Gen {
        /// The MQ4 NT1 entry `spec`. The same refusals as [`emit`].
        pub fn new(spec: Spec) -> Result<Gen, String> {
            spec.validate()?;
            if spec.codec != WeightCodec::Mq4Sym || spec.nt != 1 || spec.rr != 1 {
                return Err("qwen4_moe nt1 blocks: the MQ4 NT1 entries only".into());
            }
            Ok(Gen {
                g: super::Gen { spec, acc: false },
            })
        }
        pub fn spec(&self) -> Spec {
            self.g.spec
        }
        pub fn kernel_spec(&self) -> KernelSpec {
            kernel_spec_nt1(self.g.spec)
        }
        pub fn plan(&self) -> Result<RegPlan, String> {
            self.g.plan()
        }
        /// A block label of this entry (`k_begin`, `k_loop`, `k_loop_end`, `tail_odd`,
        /// `epilogue`, `end`).
        pub fn label(&self, name: &str) -> String {
            self.g.label(name)
        }
        /// The kernel exit's label.
        pub fn exit_label(&self) -> String {
            self.g.label("end")
        }
    }

    fn parity(p: usize) -> Result<usize, String> {
        if p < 2 {
            Ok(p)
        } else {
            Err(format!("set parity {p}: expected 0 or 1"))
        }
    }

    /// Kernel arguments, workgroup and lane ids, the wave index, `TILE16`.
    pub fn pro_front(b: &mut Builder, g: &Gen) -> Result<(), String> {
        super::pro_front(b, &g.g)
    }
    /// `s_cmp_ge_i32 TILE16, m_total`: the tile is past the end (the workgroup returns).
    pub fn outside_insn() -> Instruction {
        super::outside_insn()
    }
    /// The tile's expert id and `n_experts`.
    pub fn pro_expert(b: &mut Builder) -> Result<(), String> {
        super::pro_expert(b)
    }
    /// `s_cmp_ge_u32 expert, n_experts`: a sentinel or corrupt expert id (the workgroup returns).
    pub fn sentinel_insn() -> Instruction {
        super::sentinel_insn()
    }
    /// Weight pointer, trips, row base, descriptors, lane offsets and set 0's loads.
    pub fn pro_setup(b: &mut Builder, g: &Gen) -> Result<(), String> {
        super::pro_setup(b, &g.g)
    }
    /// The fold magic and the zeroed sums.
    pub fn pro_init(b: &mut Builder, g: &Gen) -> Result<(), String> {
        super::pro_init(b, &g.g)
    }
    /// Issue set `p`'s loads (epoch parity `p`).
    pub fn load_set(b: &mut Builder, g: &Gen, p: usize) -> Result<(), String> {
        super::load_set(b, &g.g, parity(p)?)
    }
    /// Fold set `p` (one K128 epoch) into the sums.
    pub fn compute(b: &mut Builder, g: &Gen, p: usize) -> Result<(), String> {
        super::compute(b, &g.g, parity(p)?)
    }
    /// Step every weight descriptor and both activation descriptors one trip's epoch pair on.
    pub fn advance(b: &mut Builder, g: &Gen) -> Result<(), String> {
        super::advance(b, &g.g)
    }
    /// `s_cmp_eq_u32 TRIPS, 0`: no full trip (workgroup-uniform).
    pub fn trips_zero_insn() -> Instruction {
        super::trips_insn("s_cmp_eq_u32", TRIPS)
    }
    /// `s_cmp_lg_u32 TRIPS, 0`: trips remain (workgroup-uniform back-edge condition).
    pub fn trips_more_insn() -> Instruction {
        super::trips_insn("s_cmp_lg_u32", TRIPS)
    }
    /// One trip done: `s_add_i32 TRIPS, TRIPS, -1`.
    pub fn trips_dec(b: &mut Builder) -> Result<(), String> {
        super::trips_dec(b)
    }
    /// `s_bitcmp1_b32 K, 7`: K/128 is odd, so one epoch (not two) is left after the trips.
    pub fn odd_insn() -> Instruction {
        super::odd_insn()
    }
    /// The two epochs left after the trips: set 1's loads, then folds of sets 0 and 1.
    pub fn tail_pair(b: &mut Builder, g: &Gen) -> Result<(), String> {
        super::tail_pair(b, &g.g)
    }
    /// The epilogue's label (the join label of the tail's `if_else`).
    pub fn epilogue_label(b: &mut Builder, g: &Gen) -> Result<(), String> {
        super::epilogue_label(b, &g.g)
    }
    /// `gate_up`: g and u rounded to BF16, then the shipped SwiGLU expression.
    pub fn swiglu(b: &mut Builder, g: &Gen) -> Result<(), String> {
        super::swiglu(b, &g.g)
    }
    /// The rounded BF16 rows of this lane, packed and stored at `Y`.
    pub fn store(b: &mut Builder, g: &Gen) -> Result<(), String> {
        super::store(b, &g.g)
    }
}

/// The NT1 entry's kernel descriptor.
fn kernel_spec_nt1(spec: Spec) -> KernelSpec {
    KernelSpec {
        kernel_id: "qwen4_moe_sym".into(),
        variant: spec.kind.tag().into(),
        arch: spec.arch,
        symbol: spec.symbol(),
        kernargs: spec.kernargs(),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: true,
        workgroup_size: spec.threads() as u16,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    }
}

/// Every entry of one architecture ([`module_specs`]: both kinds at every
/// [`tile_widths`] width, then the [`row_repeats`]) as one code object.
pub fn emit_module(
    arch: Arch,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = module_specs(arch)
        .into_iter()
        .map(emit)
        .collect::<Result<Vec<_>, _>>()?;
    let (text, proof) = super::iu4_gemm::module(&emitted, &Spec::module(arch))?;
    Ok((emitted, text, proof))
}

/// One architecture's module for one weight codec. [`WeightCodec::Mq4Sym`] is
/// [`emit_module`] (the committed bundle); [`WeightCodec::Mq3I4`] is its own
/// module ([`Spec::module_codec`]), so no existing MQ4 byte moves.
pub fn emit_module_codec(
    arch: Arch,
    codec: WeightCodec,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let mut emitted = module_specs_codec(arch, codec)
        .into_iter()
        .map(emit)
        .collect::<Result<Vec<_>, _>>()?;
    if codec == WeightCodec::Mq3I4 {
        // The qualification entry follows the shipped one in the module.
        for spec in module_specs_codec(arch, codec) {
            emitted.push(w3::emit_acc(spec)?);
        }
    }
    if codec == WeightCodec::E8pI4 {
        for spec in module_specs_codec(arch, codec) {
            emitted.push(e8p::emit_acc(spec)?);
        }
    }
    let (text, proof) = super::iu4_gemm::module(&emitted, &Spec::module_codec(arch, codec))?;
    Ok((emitted, text, proof))
}

/// LDS bytes per 16-slot tile of the expert-run gate/up handoff: RNE(gate)
/// of 2 row pairs x 32 lanes x 8 f32.
pub const NT_GATE_BYTES: u32 = 2048;

/// gfx1151 gate/up over eight pad16 tiles per weight panel, with two
/// rotating operand buffers (208 VGPR, 16 KiB gate handoff LDS). Each output
/// retains the incumbent's ascending K128 fold and BF16/SwiGLU rounding.
/// The 64-byte ABI and grid are unchanged. Ordinary Spec/R4 emissions keep
/// their original register layout and prefetch policy; this entry has its
/// own object and an explicit measured-shape runtime selector.
pub fn emit_gate_up_stream8(arch: Arch) -> Result<Emitted, String> {
    if arch != Arch::Gfx1151 {
        return Err("qwen4_moe_sym stream8: measured gfx1151 gate/up only".into());
    }
    nt::emit_buffered(
        R4Spec {
            base: Spec {
                arch,
                kind: Kind::GateUp,
                nt: 8,
                rr: 1,
                codec: WeightCodec::Mq4Sym,
            },
            chains: 1,
            prefetch: true,
            grouped: false,
        },
        2,
    )
}

/// Expert-run tiles (`nt` 16-slot tiles of one expert per weight stream).
///
/// Grid and ABI are the 16-slot entry's; the block is 128 (four waves). CTA
/// `(x, tile)` leads when `tile`'s position inside its expert's run of tiles
/// (counted by a 32-lane walk back over `expert_tile_ids`) is a multiple of
/// `nt`; others return. The leader takes `cnt` = up to `nt` consecutive tiles
/// of the same expert (bounded by `m_total/16`); tiles past `cnt` are dead:
/// their slots read -1 (gathered from nothing), their folds are skipped by
/// uniform branches and they store nothing. Per K128 epoch every wave loads
/// its 16 weight rows once (the next epoch prefetched) and folds them with
/// each live tile's activations, which are reloaded for the next epoch right
/// after the tile's fold: each output keeps the 16-slot entry's ascending
/// per-epoch fold and rounding, so all bytes equal its bytes.
///
/// Gate/up waves are (row pair `w & 1`) x (gate if `w < 2`, else up rows
/// `M/2` further). After the K loop every wave rounds its sums to BF16
/// (round trip); the gate waves store theirs to LDS and the up waves, after
/// one barrier, read their pair's gate values in the same lane layout and
/// store `rt(silu(rt(g)) * rt(u))`. Down waves own rows `64x + 16w` as in
/// the 16-slot entry.
///
/// Row repeat (`rr` = 2 or 4, gfx1151 down NT4): the grid is
/// `(M/(64*rr), m_total/16)` and CTA `x` folds `rr` contiguous 64-row blocks
/// of its run, block `b` owning rows `64*rr*x + 64*b + 16w ..+16` of wave
/// `w`; adjacent blocks therefore stream adjacent weight rows. The prologue
/// (run walk, leader test, `cnt`, slots, activation offsets, magic) runs
/// once. The blocks are unrolled statically, each with its own labels, and
/// each block resets the activation descriptors, the K trip count and the
/// weight descriptor (`w_ptr + row_base * row_bytes`) and zeroes its sums.
/// Each block's output is the 16-slot-entry fold of those rows, so every
/// output byte equals the `rr = 1` entry's. Block `b` stores at `Y` offset
/// `+128*b` bytes (store immediate), and a dead tile skips its store
/// instead of leaving the kernel, so later blocks still run.
///
/// Between blocks the next block's first-epoch weights are issued before
/// the current block's stores, into the weight set the current block loaded
/// second (`rr = 1` parity swaps every block: block `b` reads logical set
/// `p` from physical set `p ^ (b & 1)`), so no copy is needed and the
/// stores and the next epoch's activation reload (issued after them) overlap
/// the weight latency. The store temporaries (`rt_tmp`, `own`, `send`,
/// `recv`, `sel`, `out`) move onto the dead fold registers (`scale_rows`,
/// `cacc`, `fold_t`) so no live weight set is overwritten, and the
/// gfx11 `pack` words alias the dead tile 0 activations `x(0)[0..4]`
/// until the reload. Pending stores are drained (`vscnt`) before the K loop
/// so its entry and exit ledgers agree. The register plan gives every block
/// its own adjacent body/epilogue intervals on the aliased registers.
pub use nt::blocks;

mod nt {
    use super::*;
    use peacemaker_author::{End, Free, LdsRegion, StoreTarget, Wave};

    /// The gate/up handoff region: RNE(gate) of every tile.
    enum Gate {}

    // VGPRs (the gfx12 NT8 maximum is v231).
    const XOFF: u8 = 16; // + j: activation row offset of tile j (or X_OOB)
    const XQOFF: u8 = 24; // + j: gfx12 nibble offset (+ 8*hi)
    const WOFF: u8 = 32;
    const HOFF: u8 = 33;
    const YOFF: u8 = 34;
    const LDSA: u8 = 35; // gate/up: (32*(w&1) + lane) * 32
    const HF: u8 = 36;
    const MAGIC8: u8 = 40;
    const CACC: u8 = 48;
    const TPROD: u8 = 56;
    const SCF: u8 = 64;
    const SUM0: u8 = 72; // + 8j
                         // Epilogue aliases (dead K-loop ranges).
    const SILU_TMP: u8 = 40; // 24: magic, cacc, fold products
    const RT_TMP: u8 = 64; // 8: row scales

    // SGPRs.
    const SRD_W: u8 = 24;
    const WPTR: u8 = 28;
    const BACK: u8 = 30;
    const POS: u8 = 31;
    const SRD_X: [u8; 2] = [32, 36];
    const SRD_S: u8 = 40;
    const TRIPS: u8 = 44;
    const XS2: u8 = 45;
    const ROWB: u8 = 46;
    const TILE16: u8 = 47;
    const WAVE: u8 = 48;
    const RBASE: u8 = 49;
    const WGX: u8 = 50;
    const WGY: u8 = 51;
    const CNT: u8 = 52;
    const YSTEP: u8 = 53;
    const MASKA: u8 = 54; // prologue lane mask; epilogue `hi_mask` (gfx11)
    const MASKT: [u8; 2] = [56, 58];
    const LIVE0: u8 = 60; // + 2j
    const SRD_Y: u8 = 76;
    const PSEL: u8 = 80;
    const SILU_MASK: u8 = 24;

    #[derive(Clone, Copy)]
    struct G {
        spec: Spec,
        blk: u8,
        chains: u8,
        prefetch: bool,
        grouped: bool,
        /// Position-indexed activation plane (`..._gp` entries): record
        /// `(16 * tile + lane) * 72` of the epoch's `m_total`-record plane instead
        /// of the flat-slot gather. The fold, rounding and stores are unchanged.
        gpos: bool,
        buffered: u8,
    }
    impl G {
        fn arch(&self) -> Arch {
            self.spec.arch
        }
        fn nt(&self) -> u8 {
            self.spec.nt
        }
        fn aw(&self) -> u8 {
            self.spec.aw()
        }
        fn streamed(&self) -> bool {
            self.buffered < self.nt()
        }
        fn buffered_tiles(&self) -> u8 {
            self.buffered
        }
        fn x_end(&self) -> u8 {
            self.x(0) + self.aw() * self.buffered_tiles()
        }
        fn gate_up(&self) -> bool {
            self.spec.kind == Kind::GateUp
        }
        fn rr(&self) -> u8 {
            self.spec.rr
        }
        /// Row block `blk` of the same entry.
        fn at(&self, blk: u8) -> G {
            G { blk, ..*self }
        }
        /// Block-local label; `rr = 1` keeps the frozen entry's names.
        fn label(&self, name: &str) -> String {
            if self.gpos {
                format!(".Lq4s_{}_nt{}_gp_{name}", self.spec.kind.tag(), self.nt())
            } else if self.rr() == 1 {
                format!(".Lq4s_{}_nt{}_{name}", self.spec.kind.tag(), self.nt())
            } else {
                format!(
                    ".Lq4s_{}_nt{}r{}b{}_{name}",
                    self.spec.kind.tag(),
                    self.nt(),
                    self.rr(),
                    self.blk
                )
            }
        }
        fn exit_label(&self) -> String {
            if self.rr() == 1 {
                self.label("end")
            } else {
                format!(
                    ".Lq4s_{}_nt{}r{}_end",
                    self.spec.kind.tag(),
                    self.nt(),
                    self.rr()
                )
            }
        }
        /// A default-off R4 schedule: the original emission has none of the
        /// per-dead-tile guards below.
        fn experimental(&self) -> bool {
            self.chains != 1 || !self.prefetch || self.grouped
        }
        /// Schedules whose dead-tile guards join differently-shaped load
        /// sets (`maybe` pendings): the prologue and every trip end settle all
        /// loads, so the K loop's entry and back-edge ledgers are both empty.
        fn settles_loads(&self) -> bool {
            self.experimental() && (self.chains != 1 || !self.prefetch)
        }
        fn sum(&self, j: u8) -> u8 {
            SUM0 + 8 * j
        }
        fn sums_end(&self) -> u8 {
            SUM0 + 8 * self.nt()
        }
        /// Parity of the physical weight sets: odd row blocks swap them.
        fn flip(&self) -> usize {
            usize::from(self.blk & 1)
        }
        /// Physical weight nibble set / header word `set`.
        fn wa_set(&self, set: usize) -> u8 {
            self.sums_end() + self.aw() * set as u8
        }
        fn wh_set(&self, set: usize) -> u8 {
            self.sums_end() + 2 * self.aw() + set as u8
        }
        /// Weight nibbles / header of epoch parity `p` in this block: odd
        /// row blocks swap the physical sets (see the module note).
        fn wa(&self, p: usize) -> u8 {
            self.wa_set(p ^ self.flip())
        }
        fn wh(&self, p: usize) -> u8 {
            self.wh_set(p ^ self.flip())
        }
        fn d(&self, j: u8) -> u8 {
            self.sums_end() + 2 * self.aw() + 2 + j % self.buffered_tiles()
        }
        fn x(&self, j: u8) -> u8 {
            let base =
                (self.sums_end() + 2 * self.aw() + 2 + self.buffered_tiles()).div_ceil(8) * 8;
            base + self.aw() * (j % self.buffered_tiles())
        }
        fn live(&self, j: u8) -> u8 {
            LIVE0 + 2 * j
        }
        fn cacc(&self, j: u8) -> u8 {
            if j == 0 {
                CACC
            } else {
                self.x_end() + 8 * (j - 1)
            }
        }
        // Epilogue temporaries: gate values over tile 0's activations, the
        // gfx11 lane exchange over the weight sets.
        fn gval(&self) -> u8 {
            self.x(0)
        }
        // Row repeats keep every weight set live across the transition, so
        // their store temporaries sit on the dead fold registers instead.
        fn own(&self) -> u8 {
            if self.rr() > 1 {
                CACC
            } else {
                self.wa(0)
            }
        }
        fn send(&self) -> u8 {
            if self.rr() > 1 {
                CACC + 4
            } else {
                self.wa(0) + 4
            }
        }
        fn recv(&self) -> u8 {
            if self.rr() > 1 {
                CACC + 6
            } else {
                self.wa(0) + 6
            }
        }
        fn sel(&self) -> u8 {
            if self.rr() > 1 {
                TPROD
            } else {
                self.wa(0) + 8
            }
        }
        fn out(&self) -> u8 {
            if self.rr() > 1 {
                TPROD + 4
            } else {
                self.wa(0) + 12
            }
        }
        fn pack(&self) -> u8 {
            if self.rr() > 1 {
                self.x(0)
            } else {
                self.wa(0) + 16
            }
        }
        fn header_off(&self, p: usize) -> u32 {
            match self.spec.kind {
                Kind::GateUp => 4 * p as u32,
                Kind::Down => 68 * p as u32,
            }
        }
        fn nibble_off(&self, p: usize) -> u32 {
            match self.spec.kind {
                Kind::GateUp => 8 + 64 * p as u32,
                Kind::Down => 4 + 68 * p as u32,
            }
        }

        fn plan(&self) -> Result<RegPlan, String> {
            let mut p = RegPlan::new(VGPR_CEILING, 104)?;
            let rr = self.rr();
            let blk = |b: u8| self.at(b);
            let l = |n: &str| blk(0).label(n);
            let exit = self.exit_label();
            // Whole-run registers: live to the last block's epilogue.
            let kernel = || Live::Between("entry".into(), blk(rr - 1).label("epilogue"));
            let pro = || Live::Between("entry".into(), l("k_begin"));
            // Registers the store temporaries alias: block `b`'s body runs
            // from the previous block's reload (entry for block 0) to its
            // epilogue, and its temporaries from there to its own reload
            // (the exit for the last block). `rr = 1` is `kernel`/`epi`.
            let body = |b: u8| {
                Live::Between(
                    if b == 0 {
                        "entry".to_string()
                    } else {
                        blk(b - 1).label("xreload")
                    },
                    blk(b).label("epilogue"),
                )
            };
            let epi = |b: u8| {
                Live::Between(
                    blk(b).label("epilogue"),
                    if b + 1 == rr {
                        exit.clone()
                    } else {
                        blk(b).label("xreload")
                    },
                )
            };
            let gfx12 = self.arch().gfx12();
            p.v::<1>("tid", 0, pro())?;
            for i in 1..16u8 {
                p.v::<1>("prologue_tmp", i, pro())?;
            }
            for j in 0..self.nt() {
                p.v::<1>("x_off", XOFF + j, kernel())?;
                if gfx12 {
                    p.v::<1>("xq_off", XQOFF + j, kernel())?;
                }
                p.v::<8>("sum", self.sum(j), Live::Whole)?;
                if j < self.buffered_tiles() {
                    p.v::<1>("d", self.d(j), kernel())?;
                    for c in 0..4u8 {
                        if gfx12 {
                            p.v::<2>("x", self.x(j) + 2 * c, kernel())?;
                        } else if j == 0 && c == 0 {
                            for b in 0..rr {
                                p.v::<4>("x", self.x(0), body(b))?;
                            }
                        } else {
                            p.v::<4>("x", self.x(j) + 4 * c, kernel())?;
                        }
                    }
                }
                p.s::<2>("live", self.live(j), Live::Whole)?;
            }
            for (name, r) in [("w_off", WOFF), ("h_off", HOFF), ("scale_f32", HF)] {
                p.v::<1>(name, r, kernel())?;
            }
            p.v::<1>("y_off", YOFF, Live::Whole)?;
            if self.gate_up() {
                p.v::<1>("lds_addr", LDSA, Live::Whole)?;
            }
            p.v::<8>("magic8", MAGIC8, kernel())?;
            for b in 0..rr {
                p.v::<8>("cacc", CACC, body(b))?;
            }
            for j in 1..self.chains {
                p.v::<8>("tile_cacc", self.cacc(j), kernel())?;
            }
            for b in 0..rr {
                p.v::<8>("fold_t", TPROD, body(b))?;
            }
            for b in 0..rr {
                p.v::<8>("scale_rows", SCF, body(b))?;
            }
            for set in 0..2 {
                for c in 0..4u8 {
                    if gfx12 {
                        p.v::<2>("w", self.wa_set(set) + 2 * c, kernel())?;
                    } else {
                        p.v::<4>("w", self.wa_set(set) + 4 * c, kernel())?;
                    }
                }
                p.v::<1>("header", self.wh_set(set), kernel())?;
            }
            for b in 0..rr {
                for i in 0..8u8 {
                    p.v::<1>("rt_tmp", RT_TMP + i, epi(b))?;
                }
                p.v::<4>("out", self.out(), epi(b))?;
                if self.gate_up() {
                    p.v::<8>("gate", self.gval(), epi(b))?;
                    for i in 0..(6 * SILU_N) {
                        p.v::<1>("silu_tmp", SILU_TMP + i, epi(b))?;
                    }
                    for i in 0..(3 * SILU_N) {
                        p.s::<2>("silu_mask", SILU_MASK + 2 * i, epi(b))?;
                    }
                }
                if !gfx12 {
                    for i in 0..4u8 {
                        p.v::<1>("own", self.own() + i, epi(b))?;
                        p.v::<1>("pack", self.pack() + i, epi(b))?;
                    }
                    for i in 0..2u8 {
                        p.v::<1>("send", self.send() + i, epi(b))?;
                        p.v::<1>("recv", self.recv() + i, epi(b))?;
                        p.v::<1>("sel", self.sel() + i, epi(b))?;
                    }
                    p.s::<2>("hi_mask", MASKA, epi(b))?;
                    p.s::<1>("permlane_sel", PSEL, epi(b))?;
                }
            }
            if !gfx12 {
                p.s::<1>("wg_x_in", 2, Live::Whole)?;
                p.s::<1>("wg_y_in", 3, Live::Whole)?;
            }
            p.s::<2>("kernarg_ptr", 0, Live::Whole)?;
            for b in 0..rr {
                p.s::<4>("srd_y", SRD_Y, epi(b))?;
            }
            p.s::<1>("s_tmp0", 4, kernel())?;
            p.s::<1>("s_tmp1", 5, kernel())?;
            p.s::<2>("s_tmp64", 6, kernel())?;
            p.s::<8>("kernargs_0x00", 8, kernel())?;
            p.s::<2>("y_base", 16, Live::Whole)?;
            p.s::<2>("kernargs_m_k", 18, kernel())?;
            p.s::<2>("kernargs_0x30", 20, kernel())?;
            p.s::<1>("x_src_rows", 22, kernel())?;
            p.s::<1>("expert", 23, kernel())?;
            p.s::<4>("srd_w", SRD_W, kernel())?;
            p.s::<2>("w_ptr", WPTR, kernel())?;
            p.s::<1>("run_back", BACK, pro())?;
            p.s::<1>("run_pos", POS, pro())?;
            for r in SRD_X {
                p.s::<4>("srd_x", r, kernel())?;
            }
            p.s::<4>("srd_tiles_sorted", SRD_S, kernel())?;
            for (name, r) in [
                ("trips", TRIPS),
                ("x_stride2", XS2),
                ("row_bytes", ROWB),
                ("tile16", TILE16),
                ("row_base", RBASE),
                ("wg_x", WGX),
                ("wg_y", WGY),
            ] {
                p.s::<1>(name, r, kernel())?;
            }
            p.s::<1>("wave", WAVE, Live::Whole)?;
            p.s::<1>("tile_count", CNT, Live::Whole)?;
            p.s::<1>("y_step", YSTEP, Live::Whole)?;
            p.s::<2>("lane_mask", MASKA, pro())?;
            for m in MASKT {
                p.s::<2>("cmp_mask", m, Live::Whole)?;
            }
            let top = p.next_free_vgpr();
            if top % 8 != 0 {
                p.v::<1>("granule_pad", (top.div_ceil(8) * 8 - 1) as u8, Live::Whole)?;
            }
            Ok(p)
        }
    }

    fn sub_i32(a: Arch) -> &'static str {
        if a.gfx12() {
            "s_sub_co_i32"
        } else {
            "s_sub_i32"
        }
    }

    /// Lane mask `m` (a pair; the high word is unused in wave32) |= `n`.
    fn or_mask(b: &mut Builder, m: u8, n: u8) -> Result<(), String> {
        sop(b, format!("s_or_b32 s{m}, s{m}, s{n}"), &[m], &[m, n])
    }

    /// Words 0..3 of both activation descriptors (epoch 0, then epoch 1)
    /// from the kernel arguments; leaves `s4 = x_src_rows * 72`.
    fn x_descriptors(b: &mut Builder, gpos: bool) -> Result<(), String> {
        // Epoch plane records: x_src_rows (flat slots), or m_total (positions).
        let records = if gpos { 21 } else { 22 };
        sop(
            b,
            format!("s_mul_i32 s4, s{records}, 0x48"),
            &[4],
            &[records],
        )?;
        sop(
            b,
            format!("s_mov_b32 s{}, s14", SRD_X[0]),
            &[SRD_X[0]],
            &[14],
        )?;
        sop(
            b,
            format!("s_mov_b32 s{}, s15", SRD_X[0] + 1),
            &[SRD_X[0] + 1],
            &[15],
        )?;
        srd_tail(b, SRD_X[0], Some(4))?;
        add64(b, SRD_X[1], 14, 4)?;
        srd_tail(b, SRD_X[1], Some(4))
    }

    /// The weight descriptor at rows `row_base..` of the expert; leaves `s4 = row_base * row_bytes`.
    fn w_descriptor(b: &mut Builder) -> Result<(), String> {
        sop(
            b,
            format!("s_mul_i32 s4, s{RBASE}, s{ROWB}"),
            &[4],
            &[RBASE, ROWB],
        )?;
        add64(b, SRD_W, WPTR, 4)?;
        srd_tail(b, SRD_W, None)
    }

    /// Two-epoch trips before the tail: (K/128 - 1) / 2.
    fn set_trips(b: &mut Builder, a: Arch) -> Result<(), String> {
        let k = 19;
        sop(b, format!("s_lshr_b32 s{TRIPS}, s{k}, 7"), &[TRIPS], &[k])?;
        sop(
            b,
            format!("{} s{TRIPS}, s{TRIPS}, -1", s_add_i32(a)),
            &[TRIPS],
            &[TRIPS],
        )?;
        sop(
            b,
            format!("s_lshr_b32 s{TRIPS}, s{TRIPS}, 1"),
            &[TRIPS],
            &[TRIPS],
        )
    }

    /// Position-indexed activation row offset of tile `j`: `dst = live ? (16 * (tile + j) + lr) * 72 : GATHER_OOB`
    /// with `live = slot >= 0` (the sorted entry still marks padding and dead tiles). `v3` is the lane's row
    /// `lr`, `TILE16` the leader's first position; a padding lane reads zeros exactly like the flat-slot gather.
    fn position_offset(b: &mut Builder, dst: u8, slot: u8, j: u8, live: u8) -> Result<(), String> {
        let q = 14u8;
        op(
            b,
            format!("v_cmp_le_i32_e64 s{live}, 0, v{slot}"),
            &[s(live)],
            &[v(slot)],
        )?;
        op(
            b,
            format!("v_add_nc_u32_e32 v{q}, s{TILE16}, v3"),
            &[v(q)],
            &[s(TILE16), v(3)],
        )?;
        if j > 0 {
            op(
                b,
                format!("v_add_nc_u32_e32 v{q}, {}, v{q}", lit(16 * u32::from(j))),
                &[v(q)],
                &[v(q)],
            )?;
        }
        op(
            b,
            format!(
                "v_mul_u32_u24_e32 v{q}, {}, v{q}",
                lit(iu4_fold::XBLK_BYTES)
            ),
            &[v(q)],
            &[v(q)],
        )?;
        op(
            b,
            format!(
                "v_cndmask_b32_e64 v{dst}, {}, v{q}, s{live}",
                lit(common::GATHER_OOB)
            ),
            &[v(dst)],
            &[v(q), s(live)],
        )
    }

    // ---- prologue pieces: builder-level, control flow stays with the caller ----

    /// Kernel arguments, workgroup ids, lane ids, the wave index, `TILE16 = 16 * tile`.
    fn pro_front(b: &mut Builder, g: &G) -> Result<(), String> {
        let a = g.arch();
        // The LDS address comes first: certification derives it from the
        // work-item id in the entry block. (32*(w&1) + lane) * 32 bytes.
        if g.gate_up() {
            op(
                b,
                format!("v_and_b32_e32 v{LDSA}, 63, v0"),
                &[v(LDSA)],
                &[v(0)],
            )?;
            op(
                b,
                format!("v_lshlrev_b32_e32 v{LDSA}, 5, v{LDSA}"),
                &[v(LDSA)],
                &[v(LDSA)],
            )?;
        }
        smem(b, 8, 8, 0, 0)?;
        smem(b, 16, 2, 0, 0x20)?;
        smem(b, 18, 2, 0, 0x28)?;
        smem(b, 20, 2, 0, 0x30)?;
        smem(b, 22, 1, 0, 0x38)?;
        if a.gfx12() {
            sop(b, format!("s_mov_b32 s{WGX}, ttmp9"), &[WGX], &[])?;
            sop(b, format!("s_and_b32 s{WGY}, ttmp7, 0xffff"), &[WGY], &[])?;
        } else {
            sop(b, format!("s_mov_b32 s{WGX}, s2"), &[WGX], &[2])?;
            sop(b, format!("s_mov_b32 s{WGY}, s3"), &[WGY], &[3])?;
        }
        if g.grouped {
            smem(b, 6, 2, 0, 0x40)?;
            sop(b, format!("s_lshl_b32 s4, s{WGY}, 2"), &[4], &[WGY])?;
            add64(b, 6, 6, 4)?;
            smem(b, WGY, 1, 6, 0)?;
        }
        // v1 = wave, v2 = lane, v3 = lr, v4 = hi.
        op(b, "v_lshrrev_b32_e32 v1, 5, v0", &[v(1)], &[v(0)])?;
        op(b, "v_and_b32_e32 v2, 31, v0", &[v(2)], &[v(0)])?;
        op(
            b,
            format!("v_readfirstlane_b32 s{WAVE}, v1"),
            &[s(WAVE)],
            &[v(1)],
        )?;
        op(b, "v_and_b32_e32 v3, 15, v2", &[v(3)], &[v(2)])?;
        op(b, "v_lshrrev_b32_e32 v4, 4, v2", &[v(4)], &[v(2)])?;
        // A tile at or past m_total, or a sentinel tile, returns.
        sop(
            b,
            format!("s_lshl_b32 s{TILE16}, s{WGY}, 4"),
            &[TILE16],
            &[WGY],
        )?;
        Ok(())
    }

    /// The tile at or past `m_total` returns (`s_cmp_ge_i32 TILE16, m_total`).
    fn outside_insn() -> Instruction {
        Instruction::new(
            format!("s_cmp_ge_i32 s{TILE16}, s21"),
            vec![],
            vec![s(TILE16), s(21)],
        )
    }

    /// The tile's expert id and `n_experts`.
    fn pro_expert(b: &mut Builder, _g: &G) -> Result<(), String> {
        sop(b, format!("s_lshl_b32 s4, s{WGY}, 2"), &[4], &[WGY])?;
        add64(b, 6, 10, 4)?;
        smem(b, 23, 1, 6, 0)?;
        // Fail closed: an expert id outside [0, n_experts) exits before any `expert_weight_ptrs[id]` load (one unsigned compare).
        smem(b, 5, 1, 0, 0x3c)?;
        Ok(())
    }

    /// A sentinel / corrupt expert id returns (one unsigned compare).
    fn sentinel_insn() -> Instruction {
        Instruction::new("s_cmp_ge_u32 s23, s5", vec![], vec![s(23), s(5)])
    }

    /// The expert's weight pointer, the tile-id descriptor, and (flat schedules) the run-walk counter.
    fn pro_run(b: &mut Builder, g: &G) -> Result<(), String> {
        sop(b, "s_lshl_b32 s4, s23, 3", &[4], &[23])?;
        add64(b, 6, 8, 4)?;
        smem(b, WPTR, 2, 6, 0)?;
        // Tile ids through a descriptor of m_total/16 records: an id read at
        // or past the end (or before the start) reads 0, so both bounds are
        // tested explicitly.
        sop(b, format!("s_mov_b32 s{SRD_S}, s10"), &[SRD_S], &[10])?;
        sop(
            b,
            format!("s_mov_b32 s{}, s11", SRD_S + 1),
            &[SRD_S + 1],
            &[11],
        )?;
        sop(b, format!("s_lshr_b32 s5, s21, 2"), &[5], &[21])?;
        srd_tail(b, SRD_S, Some(5))?;
        if !g.grouped {
            // Run position: lanes test tiles tile-1-back-lane; the first that is
            // before the start or another expert ends the run.
            sop(b, format!("s_mov_b32 s{BACK}, 0"), &[BACK], &[])?;
        }
        Ok(())
    }

    /// One iteration of the run walk, up to its lane mask.
    fn walk_head(b: &mut Builder, g: &G) -> Result<(), String> {
        let a = g.arch();
        sop(b, format!("{} s4, s{WGY}, -1", s_add_i32(a)), &[4], &[WGY])?;
        sop(
            b,
            format!("{} s4, s4, s{BACK}", sub_i32(a)),
            &[4],
            &[4, BACK],
        )?;
        op(b, "v_sub_nc_u32_e32 v5, s4, v2", &[v(5)], &[s(4), v(2)])?;
        op(b, "v_lshlrev_b32_e32 v6, 2, v5", &[v(6)], &[v(5)])?;
        bload(b, 1, 7, 6, SRD_S, 0)?;
        op(
            b,
            format!("v_cmp_ne_u32_e64 s{MASKA}, s23, v7"),
            &[s(MASKA)],
            &[s(23), v(7)],
        )?;
        op(
            b,
            format!("v_cmp_gt_i32_e64 s{}, 0, v5", MASKT[0]),
            &[s(MASKT[0])],
            &[v(5)],
        )?;
        or_mask(b, MASKA, MASKT[0])?;
        Ok(())
    }

    /// The walk ends when any lane found the run's start (`s_cmp_lg_u32 MASKA, 0`).
    fn walk_hit_insn() -> Instruction {
        Instruction::new(format!("s_cmp_lg_u32 s{MASKA}, 0"), vec![], vec![s(MASKA)])
    }

    /// The walk's back-edge step: 32 tiles further back.
    fn walk_step(b: &mut Builder, g: &G) -> Result<(), String> {
        let a = g.arch();
        sop(
            b,
            format!("{} s{BACK}, s{BACK}, 32", s_add_i32(a)),
            &[BACK],
            &[BACK],
        )
    }

    /// This tile's position in its run, modulo `nt`.
    fn pro_leader(b: &mut Builder, g: &G) -> Result<(), String> {
        let (a, nt) = (g.arch(), g.nt());
        sop(
            b,
            format!("s_ctz_i32_b32 s{POS}, s{MASKA}"),
            &[POS],
            &[MASKA],
        )?;
        sop(
            b,
            format!("{} s{POS}, s{POS}, s{BACK}", s_add_i32(a)),
            &[POS],
            &[POS, BACK],
        )?;
        // Not a leader: another CTA folds this tile.
        sop(
            b,
            format!("s_and_b32 s{POS}, s{POS}, {}", nt - 1),
            &[POS],
            &[POS],
        )?;
        Ok(())
    }

    /// Not the leader of its `nt` tiles: another CTA folds this tile.
    fn follower_insn() -> Instruction {
        Instruction::new(format!("s_cmp_lg_u32 s{POS}, 0"), vec![], vec![s(POS)])
    }

    /// `cnt` (live tiles of the leader), row bytes, trips, row base, Y offsets, the sorted-slot descriptor.
    fn pro_cnt(b: &mut Builder, g: &G) -> Result<(), String> {
        let (a, kind, nt) = (g.arch(), g.spec.kind, g.nt());
        // cnt: the first lane l < nt whose tile tile+l is past the end or of
        // another expert (lanes >= nt always stop).
        op(
            b,
            format!("v_add_nc_u32_e32 v5, s{WGY}, v2"),
            &[v(5)],
            &[s(WGY), v(2)],
        )?;
        op(b, "v_lshlrev_b32_e32 v6, 2, v5", &[v(6)], &[v(5)])?;
        bload(b, 1, 7, 6, SRD_S, 0)?;
        sop(b, "s_lshr_b32 s5, s21, 4", &[5], &[21])?;
        op(
            b,
            format!("v_cmp_ne_u32_e64 s{MASKA}, s23, v7"),
            &[s(MASKA)],
            &[s(23), v(7)],
        )?;
        op(
            b,
            format!("v_cmp_le_u32_e64 s{}, s5, v5", MASKT[0]),
            &[s(MASKT[0])],
            &[s(5), v(5)],
        )?;
        or_mask(b, MASKA, MASKT[0])?;
        op(
            b,
            format!("v_cmp_le_u32_e64 s{}, {nt}, v2", MASKT[0]),
            &[s(MASKT[0])],
            &[v(2)],
        )?;
        or_mask(b, MASKA, MASKT[0])?;
        sop(
            b,
            format!("s_ctz_i32_b32 s{CNT}, s{MASKA}"),
            &[CNT],
            &[MASKA],
        )?;
        // row_bytes = (K/256)*136 (QT44) or (K/128)*68 (QT53).
        let k = 19;
        match kind {
            Kind::GateUp => {
                sop(b, format!("s_lshr_b32 s{ROWB}, s{k}, 8"), &[ROWB], &[k])?;
                sop(b, format!("s_mulk_i32 s{ROWB}, 0x88"), &[ROWB], &[ROWB])?;
            }
            Kind::Down => {
                sop(b, format!("s_lshr_b32 s{ROWB}, s{k}, 7"), &[ROWB], &[k])?;
                sop(b, format!("s_mulk_i32 s{ROWB}, 0x44"), &[ROWB], &[ROWB])?;
            }
        }
        set_trips(b, a)?;
        // Output column base: gate/up 32x + 16*(w&1), down 64x + 16w. The
        // weight rows of an up wave are M/2 further.
        match kind {
            Kind::GateUp => {
                sop(
                    b,
                    format!("s_lshl_b32 s{RBASE}, s{WGX}, 5"),
                    &[RBASE],
                    &[WGX],
                )?;
                sop(b, format!("s_and_b32 s4, s{WAVE}, 1"), &[4], &[WAVE])?;
            }
            Kind::Down => {
                sop(
                    b,
                    format!(
                        "s_lshl_b32 s{RBASE}, s{WGX}, {}",
                        6 + g.rr().trailing_zeros()
                    ),
                    &[RBASE],
                    &[WGX],
                )?;
                sop(b, format!("s_mov_b32 s4, s{WAVE}"), &[4], &[WAVE])?;
            }
        }
        sop(b, "s_lshl_b32 s4, s4, 4", &[4], &[4])?;
        sop(
            b,
            format!("{} s{RBASE}, s{RBASE}, s4", s_add_i32(a)),
            &[RBASE],
            &[RBASE, 4],
        )?;
        // Y row length and per-tile Y step (16 rows of BF16).
        match kind {
            Kind::GateUp => sop(b, "s_lshr_b32 s5, s18, 1", &[5], &[18])?,
            Kind::Down => sop(b, "s_mov_b32 s5, s18", &[5], &[18])?,
        }
        sop(b, format!("s_lshl_b32 s{YSTEP}, s5, 5"), &[YSTEP], &[5])?;
        // Y offset: ((16*tile + lr)*row_len + col + 8*hi)*2.
        op(
            b,
            format!("v_add_nc_u32_e32 v8, s{TILE16}, v3"),
            &[v(8)],
            &[s(TILE16), v(3)],
        )?;
        op(b, "v_mul_lo_u32 v8, v8, s5", &[v(8)], &[v(8), s(5)])?;
        op(
            b,
            format!("v_lshl_add_u32 v9, v4, 3, s{RBASE}"),
            &[v(9)],
            &[v(4), s(RBASE)],
        )?;
        op(b, "v_add_nc_u32_e32 v8, v8, v9", &[v(8)], &[v(8), v(9)])?;
        op(
            b,
            format!("v_lshlrev_b32_e32 v{YOFF}, 1, v8"),
            &[v(YOFF)],
            &[v(8)],
        )?;
        if kind == Kind::GateUp {
            sop(b, format!("s_lshr_b32 s4, s{WAVE}, 1"), &[4], &[WAVE])?;
            sop(b, "s_mul_i32 s4, s4, s5", &[4], &[4, 5])?;
            sop(
                b,
                format!("{} s{RBASE}, s{RBASE}, s4", s_add_i32(a)),
                &[RBASE],
                &[RBASE, 4],
            )?;
        }
        // Slots of the cnt tiles: sorted[16*(tile+j) + lr] through a
        // descriptor of cnt*64 bytes; dead tiles' slots read -1 below.
        sop(b, format!("s_lshl_b32 s4, s{WGY}, 6"), &[4], &[WGY])?;
        add64(b, SRD_S, 12, 4)?;
        sop(
            b,
            format!("s_lshl_b32 s{}, s{CNT}, 6", SRD_S + 2),
            &[SRD_S + 2],
            &[CNT],
        )?;
        sop(
            b,
            format!("s_mov_b32 s{}, {}", SRD_S + 3, lit(SRD_WORD3)),
            &[SRD_S + 3],
            &[],
        )?;
        op(b, "v_lshlrev_b32_e32 v5, 2, v3", &[v(5)], &[v(3)])?;
        Ok(())
    }

    /// Tile `j`'s sorted slot (one dword per lane).
    fn pro_slot_load(b: &mut Builder, _g: &G, j: u8) -> Result<(), String> {
        bload(b, 1, 6 + j, 5, SRD_S, 64 * u32::from(j))
    }

    /// Activation and weight descriptors, lane offsets, the first epoch's weights.
    fn pro_desc(b: &mut Builder, g: &G) -> Result<(), String> {
        let a = g.arch();
        // Activation descriptors: epoch 0 (X0) and 1 (X1), x_src_rows*72 records.
        x_descriptors(b, g.gpos)?;
        sop(b, format!("s_lshl_b32 s{XS2}, s4, 1"), &[XS2], &[4])?;
        // Weight descriptor: rows rbase.. of the expert.
        w_descriptor(b)?;
        op(
            b,
            format!("v_mul_u32_u24_e32 v{WOFF}, s{ROWB}, v3"),
            &[v(WOFF)],
            &[s(ROWB), v(3)],
        )?;
        if a.gfx12() {
            op(
                b,
                format!("v_lshl_add_u32 v{WOFF}, v4, 3, v{WOFF}"),
                &[v(WOFF)],
                &[v(4), v(WOFF)],
            )?;
        }
        if a.gfx12() {
            op(b, "v_and_b32_e32 v14, 7, v3", &[v(14)], &[v(3)])?;
            op(b, "v_lshl_or_b32 v14, v4, 3, v14", &[v(14)], &[v(4), v(14)])?;
        } else {
            op(b, "v_and_b32_e32 v14, 14, v3", &[v(14)], &[v(3)])?;
            op(b, "v_or_b32_e32 v14, v14, v4", &[v(14)], &[v(14), v(4)])?;
        }
        op(
            b,
            format!("v_mul_u32_u24_e32 v{HOFF}, s{ROWB}, v14"),
            &[v(HOFF)],
            &[s(ROWB), v(14)],
        )?;
        load_w(b, g, 0)?;
        Ok(())
    }

    /// Tile `j`'s activation row offset (dead tiles gather slot -1), before its first-epoch loads.
    fn pro_tile(b: &mut Builder, g: &G, j: u8) -> Result<(), String> {
        let a = g.arch();
        let slot = 6 + j;
        if j > 0 && !g.experimental() {
            let m = MASKT[1];
            sop(b, format!("s_cmp_gt_u32 s{CNT}, {j}"), &[], &[CNT])?;
            sop(b, format!("s_cselect_b32 s{m}, -1, 0"), &[m], &[])?;
            sop(b, format!("s_mov_b32 s{}, 0", m + 1), &[m + 1], &[])?;
            op(
                b,
                format!("v_cndmask_b32_e64 v{slot}, -1, v{slot}, s{m}"),
                &[v(slot)],
                &[v(slot), s(m)],
            )?;
        }
        if g.gpos {
            position_offset(b, XOFF + j, slot, j, g.live(j))?;
        } else {
            common::gather_offset(
                b,
                XOFF + j,
                slot,
                Some(20),
                iu4_fold::XBLK_BYTES,
                g.live(j),
                GatherTemps {
                    v: [14, 15, 1, 2],
                    mask: MASKT[0],
                },
            )?;
        }
        if a.gfx12() {
            op(
                b,
                format!("v_lshl_add_u32 v{}, v4, 3, v{}", XQOFF + j, XOFF + j),
                &[v(XQOFF + j)],
                &[v(4), v(XOFF + j)],
            )?;
        }
        Ok(())
    }

    /// The fold magic and the zeroed sums.
    fn pro_init(b: &mut Builder, g: &G) -> Result<(), String> {
        let nt = g.nt();
        for j in 0..8u8 {
            op(
                b,
                format!("v_mov_b32_e32 v{}, {}", MAGIC8 + j, lit(MAGIC)),
                &[v(MAGIC8 + j)],
                &[],
            )?;
        }
        for t in 0..nt {
            for j in 0..8u8 {
                op(
                    b,
                    format!("v_mov_b32_e32 v{}, 0", g.sum(t) + j),
                    &[v(g.sum(t) + j)],
                    &[],
                )?;
            }
        }
        Ok(())
    }

    fn prologue<T: Target>(wg: &mut Wg<T>, g: &G, end: &End) -> Result<(), String> {
        let nt = g.nt();
        pro_front(wg.isa(), g)?;
        let outside = wg.scmp_wg_uniform(outside_insn())?;
        wg.exit_if(outside, end)?;
        pro_expert(wg.isa(), g)?;
        let sentinel = wg.scmp_wg_uniform(sentinel_insn())?;
        wg.exit_if(sentinel, end)?;
        pro_run(wg.isa(), g)?;
        if !g.grouped {
            let (walk, found) = (g.label("run_walk"), g.label("run_found"));
            // Wave-uniform: every lane mask is a compare over the same loads.
            wg.loop_until(&walk, &found, |w, exit| {
                walk_head(w.isa(), g)?;
                let hit = w.scmp(walk_hit_insn())?;
                w.break_if(hit, exit)?;
                walk_step(w.isa(), g)
            })?;
            pro_leader(wg.isa(), g)?;
            let follower = wg.scmp_wg_uniform(follower_insn())?;
            wg.exit_if(follower, end)?;
        }
        pro_cnt(wg.isa(), g)?;
        for j in 0..nt {
            if g.experimental() && j > 0 {
                // A dead tile issues no sorted-slot load; its slots are -1.
                op(
                    wg.isa(),
                    format!("v_mov_b32_e32 v{}, -1", 6 + j),
                    &[v(6 + j)],
                    &[],
                )?;
                let live = wg.scmp(Instruction::new(
                    format!("s_cmp_gt_u32 s{CNT}, {j}"),
                    vec![],
                    vec![s(CNT)],
                ))?;
                wg.skip_unless(live, &g.label(&format!("pro_s_dead{j}")), (), |w, ()| {
                    pro_slot_load(w.isa(), g, j)
                })?;
            } else {
                pro_slot_load(wg.isa(), g, j)?;
            }
        }
        pro_desc(wg.isa(), g)?;
        // Per tile: dead tiles (j >= cnt) gather slot -1, then the activation
        // row offset (or the out-of-range offset of a padding slot).
        for j in 0..nt {
            pro_tile(wg.isa(), g, j)?;
            if j < g.buffered_tiles() {
                load_x_init(wg, g, j)?;
            }
        }
        pro_init(wg.isa(), g)?;
        if g.streamed() {
            let b = wg.isa();
            // The streamed current-epoch tile loads naturally settle the next
            // weight set. Enter the loop with the same weight-ready ledger,
            // retaining both operand buffers pending (not a full load drain).
            let use_weight = Instruction::new(
                "",
                vec![],
                vec![
                    crate::reg::RegRef {
                        kind: crate::reg::Kind::V,
                        base: g.wa(0),
                        len: g.aw(),
                    },
                    v(g.wh(0)),
                ],
            );
            for (counter, n, _) in b.ledger.required(&use_weight) {
                b.wait(counter, n)?;
            }
        }
        if g.settles_loads() {
            super::super::gemm_uk::Prefetch::<0>::wait_load(wg.isa())?;
        }
        Ok(())
    }

    /// Weight set `p` (epoch parity p of the current trip): header word, then
    /// the nibbles, as one clause.
    fn load_w(b: &mut Builder, g: &G, p: usize) -> Result<(), String> {
        let (width, step) = if g.arch().gfx12() {
            (2u8, 2u8)
        } else {
            (4u8, 4u8)
        };
        b.clause(|b| {
            bload(b, 1, g.wh(p), HOFF, SRD_W, g.header_off(p))?;
            for i in 0..4u8 {
                bload(
                    b,
                    width,
                    g.wa(p) + step * i,
                    WOFF,
                    SRD_W,
                    g.nibble_off(p) + 16 * u32::from(i),
                )?;
            }
            Ok(())
        })
    }

    /// Tile `j`'s activations of the epoch `srd` points at: d, then nibbles.
    fn load_x(b: &mut Builder, g: &G, j: u8, srd: u8) -> Result<(), String> {
        let gfx12 = g.arch().gfx12();
        let (width, step) = if gfx12 { (2u8, 2u8) } else { (4u8, 4u8) };
        let xq = if gfx12 { XQOFF + j } else { XOFF + j };
        b.clause(|b| {
            bload(b, 1, g.d(j), XOFF + j, srd, 0)?;
            for i in 0..4u8 {
                bload(b, width, g.x(j) + step * i, xq, srd, 8 + 16 * u32::from(i))?;
            }
            Ok(())
        })
    }

    /// `load_x` of tile `j`; an experimental schedule issues no load for a
    /// dead whole tile (`j >= cnt`, wave-uniform), tile 0 is always live.
    fn load_x_live<T: Target>(
        w: &mut Wave<'_, T, Builder>,
        g: &G,
        j: u8,
        srd: u8,
        site: &str,
    ) -> Result<(), String> {
        if !g.experimental() || j == 0 {
            return load_x(w.isa(), g, j, srd);
        }
        let live = w.scmp(Instruction::new(
            format!("s_cmp_gt_u32 s{CNT}, {j}"),
            vec![],
            vec![s(CNT)],
        ))?;
        w.skip_unless(live, &g.label(&format!("{site}_dead{j}")), (), |w, ()| {
            load_x(w.isa(), g, j, srd)
        })
    }

    /// Tile `j`'s initial activations (epoch 0). An experimental schedule
    /// defines a dead whole tile's fragments (`d` and `x`) with zeros on a
    /// dead arm instead of loading them. `if_else` branches to the live
    /// else arm when `cnt > j`; tile 0 is always live.
    fn load_x_init<T: Target>(w: &mut Wave<'_, T, Builder>, g: &G, j: u8) -> Result<(), String> {
        if !g.experimental() || j == 0 {
            return load_x(w.isa(), g, j, SRD_X[0]);
        }
        let live = w.scmp(Instruction::new(
            format!("s_cmp_gt_u32 s{CNT}, {j}"),
            vec![],
            vec![s(CNT)],
        ))?;
        let join = g.label(&format!("pro_x_join{j}"));
        w.if_else(
            live,
            &g.label(&format!("pro_x_live{j}")),
            &join,
            |w| {
                let b = w.isa();
                op(
                    b,
                    format!("v_mov_b32_e32 v{}, 0", g.d(j)),
                    &[v(g.d(j))],
                    &[],
                )?;
                for i in 0..g.aw() {
                    op(
                        b,
                        format!("v_mov_b32_e32 v{}, 0", g.x(j) + i),
                        &[v(g.x(j) + i)],
                        &[],
                    )?;
                }
                Ok(())
            },
            |w| load_x(w.isa(), g, j, SRD_X[0]),
        )?;
        w.label(&join)
    }

    /// Fold weight set `p` into every live tile's sums; with `reload`, each
    /// tile's next-epoch activations are issued (through that descriptor)
    /// right after its fold; experimental schedules skip dead whole tiles.
    fn compute<T: Target + peacemaker_author::MmaIu4>(
        w: &mut Wave<'_, T, Builder>,
        g: &G,
        p: usize,
        site: &str,
        reload: Option<u8>,
    ) -> Result<(), String> {
        compute_head(w.isa(), g, p)?;
        let b = w.isa();
        if !g.prefetch {
            super::super::gemm_uk::Prefetch::<0>::wait_load(b)?;
        }
        if g.chains == 1 {
            return serial_tiles(w, g, p, site, reload);
        }
        // The full-live path has no per-step predicates. Ragged runs retain
        // the incumbent's skipped-tile arithmetic and output ownership.
        for j in 0..g.nt() {
            wait_tile(w.isa(), g, j)?;
        }
        let full = w.scmp(Instruction::new(
            format!("s_cmp_lg_u32 s{CNT}, {}", g.nt()),
            vec![],
            vec![s(CNT)],
        ))?;
        w.if_else(
            full,
            &g.label(&format!("{site}_partial")),
            &g.label(&format!("{site}_join")),
            |w| match g.chains {
                2 => full_tiles::<2, T>(w, g, p, reload),
                4 => full_tiles::<4, T>(w, g, p, reload),
                8 => full_tiles::<8, T>(w, g, p, reload),
                _ => unreachable!("validated chain count"),
            },
            |w| serial_tiles(w, g, p, site, reload),
        )?;
        w.label(&g.label(&format!("{site}_join")))
    }

    /// The fold's row scales (header word -> f32 -> the eight rows of the accumulators by
    /// `ds_swizzle` broadcast) and the nibble rebias of weight set `p`.
    fn compute_head(b: &mut Builder, g: &G, p: usize) -> Result<(), String> {
        let a = g.arch();
        op(
            b,
            format!("v_cvt_f32_f16_e64 v{HF}, v{}.l", g.wh(p)),
            &[v(HF)],
            &[v(g.wh(p))],
        )?;
        for j in 0..8u8 {
            let k = if a.gfx12() { j } else { 2 * j };
            let text = format!(
                "ds_swizzle_b32 v{}, v{HF} offset:swizzle(BROADCAST,16,{k})",
                SCF + j
            );
            b.ds_crosslane(
                crate::insn::Instruction::new(text, vec![v(SCF + j)], vec![v(HF)])
                    .memory(MemoryClass::DsLoad),
            )?;
        }
        for i in 0..g.aw() {
            let r = g.wa(p) + i;
            op(
                b,
                format!("v_xor_b32_e32 v{r}, {}, v{r}", lit(REBIAS)),
                &[v(r)],
                &[v(r)],
            )?;
        }
        Ok(())
    }

    /// Wait for the loads tile `j`'s fold reads (its `d` and activation operands).
    fn wait_tile(b: &mut Builder, g: &G, j: u8) -> Result<(), String> {
        let x = Instruction::new(
            "",
            vec![],
            vec![
                v(g.d(j)),
                crate::reg::RegRef {
                    kind: crate::reg::Kind::V,
                    base: g.x(j),
                    len: g.aw(),
                },
            ],
        );
        for (c, n, _) in b.ledger.required(&x) {
            b.wait(c, n)?;
        }
        Ok(())
    }

    /// Tile `j` is live (`cnt > j`, wave-uniform): `s_cmp_gt_u32 CNT, j`.
    fn tile_live_insn(j: u8) -> Instruction {
        Instruction::new(format!("s_cmp_gt_u32 s{CNT}, {j}"), vec![], vec![s(CNT)])
    }

    fn full_tiles<const N: usize, T: Target + peacemaker_author::MmaIu4>(
        w: &mut Wave<'_, T, Builder>,
        g: &G,
        p: usize,
        reload: Option<u8>,
    ) -> Result<(), String> {
        use super::super::gemm_uk::{Chain, Iu4};
        let chain = Chain::<N, Iu4>::new(
            std::array::from_fn(|j| V::<8>(g.cacc(j as u8))),
            V::<8>(MAGIC8),
        )?;
        for first in (0..g.nt()).step_by(N) {
            for i in 0..g.aw() / 2 {
                chain.step(
                    w,
                    V::<2>(g.wa(p) + 2 * i),
                    std::array::from_fn(|j| V::<2>(g.x(first + j as u8) + 2 * i)),
                    i == 0,
                )?;
            }
            for j in first..first + N as u8 {
                let single = Chain::<1, Iu4>::new([V::<8>(g.cacc(j - first))], V::<8>(MAGIC8))?;
                single.fold(w.isa(), [g.sum(j)], SCF, [g.d(j)], TPROD)?;
                if let Some(srd) = reload {
                    load_x(w.isa(), g, j, srd)?;
                }
            }
        }
        Ok(())
    }

    /// Fold weight set `p` into tile `j`'s sums: the K32 WMMA steps over the epoch's nibbles, then
    /// the exact fold pass (`C - magic` scaled by `d * scale`).
    fn fold_tile(b: &mut Builder, g: &G, p: usize, j: u8) -> Result<(), String> {
        let a = g.arch();
        for i in 0..g.aw() / 2 {
            iu4_fold::wmma_step(
                b,
                a,
                V::<8>(CACC),
                V::<2>(g.wa(p) + 2 * i),
                V::<2>(g.x(j) + 2 * i),
                i == 0,
                V::<8>(MAGIC8),
            )?;
        }
        iu4_fold::fold_pass(b, CACC, g.sum(j), SCF, g.d(j), TPROD)
    }

    fn serial_tiles<T: Target>(
        w: &mut Wave<'_, T, Builder>,
        g: &G,
        p: usize,
        site: &str,
        reload: Option<u8>,
    ) -> Result<(), String> {
        // Experimental schedules reload only live tiles, inside the live
        // branch; the original reloads every tile after its (maybe skipped) fold.
        let (inside, outside) = if g.experimental() {
            (reload, None)
        } else {
            (None, reload)
        };
        for j in 0..g.nt() {
            let fold = |w: &mut Wave<'_, T, Builder>, ()| -> Result<(), String> {
                fold_tile(w.isa(), g, p, j)?;
                if let Some(srd) = inside {
                    load_x(w.isa(), g, j, srd)?;
                }
                Ok(())
            };
            if j > 0 {
                wait_tile(w.isa(), g, j)?;
                let live = w.scmp(tile_live_insn(j))?;
                w.skip_unless(live, &g.label(&format!("{site}_dead{j}")), (), fold)?;
            } else {
                fold(w, ())?;
            }
            if g.streamed() {
                // Refill this consumed ring slot with this epoch's tile j+2.
                // Only the final two slots preload the next epoch's tiles 0/1.
                if j + 2 < g.nt() {
                    load_x(w.isa(), g, j + 2, SRD_X[p])?;
                } else if let Some(srd) = outside {
                    load_x(w.isa(), g, j + 2 - g.nt(), srd)?;
                }
            } else if let Some(srd) = outside {
                load_x(w.isa(), g, j, srd)?;
            }
        }
        Ok(())
    }

    fn advance(b: &mut Builder, streamed: bool) -> Result<(), String> {
        add64_imm(b, SRD_W, 136)?;
        for r in SRD_X {
            // Streamed epoch 1 still reads SRD_X[1] while it computes; move
            // only epoch 0 here so epoch 1 can prefetch epoch 2 into its ring.
            if !streamed || r == SRD_X[0] {
                add64(b, r, r, XS2)?;
            }
        }
        Ok(())
    }

    /// One trip done: `s_add_i32 s{TRIPS}, s{TRIPS}, -1`.
    fn trips_dec(b: &mut Builder) -> Result<(), String> {
        op(
            b,
            format!("{} s{TRIPS}, s{TRIPS}, -1", s_add_i32(b.spec.arch)),
            &[s(TRIPS)],
            &[s(TRIPS)],
        )
    }

    /// An odd K/128 leaves one epoch after the trips, not two: `s_bitcmp1_b32 K, 7`.
    fn odd_insn() -> Instruction {
        Instruction::new("s_bitcmp1_b32 s19, 7", vec![], vec![s(19)])
    }

    fn kloop<T: Target + peacemaker_author::MmaIu4>(wg: &mut Wg<T>, g: &G) -> Result<(), String> {
        wg.label(&g.label("k_begin"))?;
        let (head, done) = (g.label("k_loop"), g.label("k_loop_end"));
        let entry = wg.isa().ledger.shape();
        let no_trip = trips_cmp(wg, "s_cmp_eq_u32", TRIPS)?;
        wg.wg_skip_if(no_trip, &done, (), |wg, ()| {
            wg.loop_carried(&head, (), |wg, ()| {
                if g.prefetch {
                    load_w(wg.isa(), g, 1)?;
                    compute(wg, g, 0, "l0", Some(SRD_X[1]))?;
                    advance(wg.isa(), g.streamed())?;
                    load_w(wg.isa(), g, 0)?;
                    compute(wg, g, 1, "l1", Some(SRD_X[0]))?;
                    if g.streamed() {
                        add64(wg.isa(), SRD_X[1], SRD_X[1], XS2)?;
                    }
                } else {
                    compute(wg, g, 0, "l0", None)?;
                    load_epoch(wg, g, 1, "l0x")?;
                    advance(wg.isa(), g.streamed())?;
                    compute(wg, g, 1, "l1", None)?;
                    load_epoch(wg, g, 0, "l1x")?;
                }
                if g.settles_loads() {
                    super::super::gemm_uk::Prefetch::<0>::wait_load(wg.isa())?;
                }
                trips_dec(wg.isa())?;
                Ok(((), trips_cmp(wg, "s_cmp_lg_u32", TRIPS)?))
            })?;
            if wg.isa().ledger.shape() != entry {
                return Err("k loop exit ledger differs from its entry".into());
            }
            Ok(())
        })?;
        match g.spec.kind {
            // K % 256 == 0: an even epoch count, two epochs left.
            Kind::GateUp => tail_pair(wg, g),
            // K % 128 == 0: one epoch left when K/128 is odd, else two.
            Kind::Down => {
                let odd = wg.scmp(odd_insn())?;
                wg.if_else(
                    odd,
                    &g.label("tail_odd"),
                    &g.label("epilogue"),
                    |w| tail_pair(w, g),
                    |w| compute(w, g, 0, "t2", None),
                )
            }
        }
    }

    fn load_epoch<T: Target>(
        w: &mut Wave<'_, T, Builder>,
        g: &G,
        p: usize,
        site: &str,
    ) -> Result<(), String> {
        load_w(w.isa(), g, p)?;
        for j in 0..g.nt() {
            load_x_live(w, g, j, SRD_X[p], site)?;
        }
        Ok(())
    }

    fn tail_pair<T: Target + peacemaker_author::MmaIu4>(
        w: &mut Wave<'_, T, Builder>,
        g: &G,
    ) -> Result<(), String> {
        if g.prefetch {
            load_w(w.isa(), g, 1)?;
            compute(w, g, 0, "t0", Some(SRD_X[1]))?;
        } else {
            compute(w, g, 0, "t0", None)?;
            load_epoch(w, g, 1, "t0x")?;
        }
        compute(w, g, 1, "t1", None)
    }

    /// RNE BF16 bits of `vals` (8 rows), +0 on padding slots, packed to this
    /// lane's 8 consecutive output rows and stored at the current `SRD_Y`.
    fn store_rows(b: &mut Builder, g: &G, vals: u8, live: u8) -> Result<(), String> {
        for j in 0..8u8 {
            Bf16::rne_finite_passthrough(
                b,
                vals + j,
                RT_TMP + j,
                MASKT[usize::from(j % 2)],
                false,
            )?;
        }
        for j in 0..8u8 {
            op(
                b,
                format!("v_cndmask_b32_e64 v{0}, 0, v{0}, s{live}", vals + j),
                &[v(vals + j)],
                &[v(vals + j), s(live)],
            )?;
        }
        let perm =
            |b: &mut Builder, dst: u8, hi_src: u8, lo_src: u8, sel: String| -> Result<(), String> {
                let mut uses = vec![v(hi_src), v(lo_src)];
                if sel.starts_with('v') {
                    uses.push(v(sel[1..].parse::<u8>().map_err(|e| e.to_string())?));
                }
                op(
                    b,
                    format!("v_perm_b32 v{dst}, v{hi_src}, v{lo_src}, {sel}"),
                    &[v(dst)],
                    &uses,
                )
            };
        let out = g.out();
        if g.arch().gfx12() {
            for k in 0..4u8 {
                perm(b, out + k, vals + 2 * k + 1, vals + 2 * k, lit(0x0706_0302))?;
            }
        } else {
            // The 16-slot entry's v_permlanex16 half exchange.
            let (own, send, recv, sel, pack) = (g.own(), g.send(), g.recv(), g.sel(), g.pack());
            for k in 0..4u8 {
                op(
                    b,
                    format!(
                        "v_cndmask_b32_e64 v{}, v{}, v{}, s{MASKA}",
                        own + k,
                        vals + k,
                        vals + 4 + k
                    ),
                    &[v(own + k)],
                    &[v(vals + k), v(vals + 4 + k), s(MASKA)],
                )?;
            }
            for m in 0..2u8 {
                perm(
                    b,
                    pack + 2 * m,
                    vals + 5 + 2 * m,
                    vals + 4 + 2 * m,
                    lit(0x0706_0302),
                )?;
                perm(
                    b,
                    pack + 2 * m + 1,
                    vals + 1 + 2 * m,
                    vals + 2 * m,
                    lit(0x0706_0302),
                )?;
                op(
                    b,
                    format!(
                        "v_cndmask_b32_e64 v{}, v{}, v{}, s{MASKA}",
                        send + m,
                        pack + 2 * m,
                        pack + 2 * m + 1
                    ),
                    &[v(send + m)],
                    &[v(pack + 2 * m), v(pack + 2 * m + 1), s(MASKA)],
                )?;
                op(
                    b,
                    format!("v_mov_b32_e32 v{}, v{}", recv + m, send + m),
                    &[v(recv + m)],
                    &[v(send + m)],
                )?;
                op(
                    b,
                    format!(
                        "v_permlanex16_b32 v{}, v{}, s{PSEL}, 0xfedcba98",
                        recv + m,
                        send + m
                    ),
                    &[v(recv + m)],
                    &[v(recv + m), v(send + m), s(PSEL)],
                )?;
            }
            for (i, (lo, hi)) in [(0x0504_0302u32, 0x0302_0504u32), (0x0706_0302, 0x0302_0706)]
                .into_iter()
                .enumerate()
            {
                let r = sel + i as u8;
                op(b, format!("v_mov_b32_e32 v{r}, {}", lit(hi)), &[v(r)], &[])?;
                op(
                    b,
                    format!("v_cndmask_b32_e64 v{r}, {}, v{r}, s{MASKA}", lit(lo)),
                    &[v(r)],
                    &[v(r), s(MASKA)],
                )?;
            }
            for k in 0..4u8 {
                perm(
                    b,
                    out + k,
                    recv + k / 2,
                    own + k,
                    format!("v{}", sel + k % 2),
                )?;
            }
        }
        bstore_b128(b, out, YOFF, SRD_Y, 128 * u32::from(g.blk))
    }

    /// The output descriptor over `Y` (and, on gfx11, the exchange masks of the row store).
    fn y_descriptor(b: &mut Builder, g: &G) -> Result<(), String> {
        sop(b, format!("s_mov_b32 s{SRD_Y}, s16"), &[SRD_Y], &[16])?;
        sop(
            b,
            format!("s_mov_b32 s{}, s17", SRD_Y + 1),
            &[SRD_Y + 1],
            &[17],
        )?;
        srd_tail(b, SRD_Y, None)?;
        if !g.arch().gfx12() {
            op(
                b,
                format!("s_mov_b32 s{MASKA}, 0xffff0000"),
                &[s(MASKA)],
                &[],
            )?;
            op(
                b,
                format!("s_mov_b32 s{}, 0", MASKA + 1),
                &[s(MASKA + 1)],
                &[],
            )?;
            op(b, format!("s_mov_b32 s{PSEL}, 0x76543210"), &[s(PSEL)], &[])?;
        }
        Ok(())
    }

    /// The output descriptor moves 16 rows (one tile) on.
    fn y_step(b: &mut Builder) -> Result<(), String> {
        add64(b, SRD_Y, SRD_Y, YSTEP)
    }

    /// Stores of every live tile: tile 0 always, tile j while j < cnt; the
    /// descriptor base advances 16 output rows per tile.
    fn store_tiles<T: Target>(
        w: &mut Wave<'_, T, Builder>,
        g: &G,
        end: &End,
        vals: impl Fn(&mut Wave<'_, T, Builder>, u8) -> Result<u8, String>,
    ) -> Result<(), String> {
        y_descriptor(w.isa(), g)?;
        for j in 0..g.nt() {
            if j > 0 {
                let live = w.scmp(tile_live_insn(j))?;
                w.exit_unless(live, end)?;
                y_step(w.isa())?;
            }
            let r = vals(w, j)?;
            store_rows(w.isa(), g, r, g.live(j))?;
        }
        Ok(())
    }

    /// Stores of a non-final row block: `store_tiles` with a dead tile
    /// skipping its store (the wave goes on to the next block) in place of
    /// leaving the kernel. Gfx11 only.
    fn store_tiles_mid<T: Target>(w: &mut Wave<'_, T, Builder>, g: &G) -> Result<(), String> {
        let b = w.isa();
        sop(b, format!("s_mov_b32 s{SRD_Y}, s16"), &[SRD_Y], &[16])?;
        sop(
            b,
            format!("s_mov_b32 s{}, s17", SRD_Y + 1),
            &[SRD_Y + 1],
            &[17],
        )?;
        srd_tail(b, SRD_Y, None)?;
        op(
            b,
            format!("s_mov_b32 s{MASKA}, 0xffff0000"),
            &[s(MASKA)],
            &[],
        )?;
        op(
            b,
            format!("s_mov_b32 s{}, 0", MASKA + 1),
            &[s(MASKA + 1)],
            &[],
        )?;
        op(b, format!("s_mov_b32 s{PSEL}, 0x76543210"), &[s(PSEL)], &[])?;
        for j in 0..g.nt() {
            if j == 0 {
                store_rows(w.isa(), g, g.sum(0), g.live(0))?;
                continue;
            }
            let live = w.scmp(Instruction::new(
                format!("s_cmp_gt_u32 s{CNT}, {j}"),
                vec![],
                vec![s(CNT)],
            ))?;
            w.skip_unless(
                live,
                &g.label(&format!("store_dead{j}")),
                (),
                |w: &mut Wave<'_, T, Builder>, ()| {
                    add64(w.isa(), SRD_Y, SRD_Y, YSTEP)?;
                    store_rows(w.isa(), g, g.sum(j), g.live(j))
                },
            )?;
        }
        Ok(())
    }

    /// Row block `g` to block `g + 1`: reset the descriptors and the trip
    /// count, issue the next block's first-epoch weights, store this
    /// block's tiles, then reload the next block's first-epoch activations
    /// and zero the sums. Pending stores are drained so the next K loop
    /// enters with the ledger shape its back edge leaves.
    fn transition<T: Target>(wg: &mut Wg<T>, g: &G) -> Result<(), String> {
        let nx = g.at(g.blk + 1);
        wg.label(&g.label("epilogue"))?;
        let b = wg.isa();
        x_descriptors(b, g.gpos)?;
        sop(
            b,
            format!("{} s{RBASE}, s{RBASE}, 64", s_add_i32(g.arch())),
            &[RBASE],
            &[RBASE],
        )?;
        w_descriptor(b)?;
        set_trips(b, g.arch())?;
        load_w(b, &nx, 0)?;
        store_tiles_mid(wg, g)?;
        wg.label(&g.label("xreload"))?;
        let b = wg.isa();
        for j in 0..g.nt() {
            load_x(b, g, j, SRD_X[0])?;
        }
        for t in 0..g.nt() {
            for j in 0..8u8 {
                op(
                    b,
                    format!("v_mov_b32_e32 v{}, 0", g.sum(t) + j),
                    &[v(g.sum(t) + j)],
                    &[],
                )?;
            }
        }
        b.wait(crate::ledger::Counter::Vs, 0)
    }

    /// Tile `j`'s gate publication: two `ds_store_b128` per lane of the rounded sums.
    fn publish_insns(g: &G, j: u8) -> [Instruction; 2] {
        std::array::from_fn(|h| {
            let h = h as u8;
            let off = u32::from(j) * NT_GATE_BYTES + 16 * u32::from(h);
            let d = crate::reg::RegRef {
                kind: crate::reg::Kind::V,
                base: g.sum(j) + 4 * h,
                len: 4,
            };
            let text = format!(
                "ds_store_b128 v{LDSA}, {d}{}",
                if off == 0 {
                    String::new()
                } else {
                    format!(" offset:{off}")
                }
            );
            Instruction::new(text, vec![], vec![v(LDSA), d]).memory(MemoryClass::DsStore)
        })
    }

    /// Tile `j`'s gate publication in the linear write token `out`.
    fn publish_tile<T: Target, S: StoreTarget<T::Waits, Out = S>>(
        w: &mut Wave<'_, T, Builder>,
        g: &G,
        j: u8,
        mut out: S,
    ) -> Result<S, String> {
        for insn in publish_insns(g, j) {
            out = w.ds_store(out, insn)?;
        }
        Ok(out)
    }

    /// The up waves' reads of tile `j`'s gate rows, in the same lane layout: two `ds_load_b128`.
    fn gate_load_insns(g: &G, j: u8) -> [Instruction; 2] {
        let gv = g.gval();
        std::array::from_fn(|h| {
            let h = h as u8;
            let off = u32::from(j) * NT_GATE_BYTES + 16 * u32::from(h);
            let d = crate::reg::RegRef {
                kind: crate::reg::Kind::V,
                base: gv + 4 * h,
                len: 4,
            };
            let text = format!(
                "ds_load_b128 {d}, v{LDSA}{}",
                if off == 0 {
                    String::new()
                } else {
                    format!(" offset:{off}")
                }
            );
            Instruction::new(text, vec![d], vec![v(LDSA)]).memory(MemoryClass::DsLoad)
        })
    }

    /// Tile `j`'s SwiGLU: `silu(gate) * up` over the gate rows read back and the up sums, in the
    /// gate registers (the value to store).
    fn silu_tile(b: &mut Builder, g: &G, j: u8) -> Result<(), String> {
        let gv = g.gval();
        for grp in 0..2u8 {
            crate::kernels::gemm_uk::Epilogue::silu_dense(
                b,
                gv + SILU_N * grp,
                g.sum(j) + SILU_N * grp,
                SILU_TMP,
                SILU_MASK,
                SILU_N,
            )?;
        }
        Ok(())
    }

    /// Every wave's BF16 round trip of its sums (gate or up), before the hand-off.
    fn bf16_roundtrip(b: &mut Builder, g: &G) -> Result<(), String> {
        for j in 0..g.nt() {
            for i in 0..8u8 {
                Bf16::rne_finite_passthrough(
                    b,
                    g.sum(j) + i,
                    RT_TMP + i,
                    MASKT[usize::from(i % 2)],
                    true,
                )?;
            }
        }
        Ok(())
    }

    /// The up waves (`wave >= 2`) read the gate rows: `s_cmp_ge_u32 WAVE, 2`.
    fn up_insn() -> Instruction {
        Instruction::new(format!("s_cmp_ge_u32 s{WAVE}, 2"), vec![], vec![s(WAVE)])
    }

    /// The epilogue's label (the join of the tail's `if_else`).
    fn epilogue_label(b: &mut Builder, g: &G) -> Result<(), String> {
        b.label(&g.label("epilogue"))
    }

    fn epilogue<T: Target>(
        wg: &mut Wg<T>,
        g: &G,
        end: End,
        gate: Option<LdsRegion<Gate, Free>>,
    ) -> Result<(), String> {
        let b = wg.isa();
        epilogue_label(b, g)?;
        let Some(gate) = gate else {
            store_tiles(wg, g, &end, |_, j| Ok(g.sum(j)))?;
            return wg.end(end);
        };
        // Every wave: BF16 round trip of its sums (gate or up).
        bf16_roundtrip(b, g)?;
        // Gate waves publish the rounded gate rows and leave; up waves meet
        // them at their own barrier, read the publication and store the
        // tiles (wave scope, up to the kernel exit).
        let up = wg.scmp(up_insn())?;
        let gv = g.gval();
        wg.handoff(
            up,
            &g.label("up"),
            end,
            gate,
            |w, gate| {
                let mut out = w.begin_write(gate);
                for j in 0..g.nt() {
                    if g.experimental() && j > 0 {
                        let live = w.scmp(Instruction::new(
                            format!("s_cmp_gt_u32 s{CNT}, {j}"),
                            vec![],
                            vec![s(CNT)],
                        ))?;
                        out = w.skip_unless(
                            live,
                            &g.label(&format!("pub_dead{j}")),
                            out,
                            |w, out| publish_tile(w, g, j, out),
                        )?;
                    } else {
                        out = publish_tile(w, g, j, out)?;
                    }
                }
                Ok(out)
            },
            |w, gate, end| {
                store_tiles(w, g, end, |w, j| {
                    for insn in gate_load_insns(g, j) {
                        w.ds_load(&gate, insn)?;
                    }
                    silu_tile(w.isa(), g, j)?;
                    Ok(gv)
                })
            },
        )
    }

    pub(super) fn emit(spec: Spec) -> Result<Emitted, String> {
        emit_with(R4Spec {
            base: spec,
            chains: 1,
            prefetch: true,
            grouped: false,
        })
    }

    pub(super) fn emit_with(r4: R4Spec) -> Result<Emitted, String> {
        emit_full(r4, false, r4.base.nt)
    }

    /// The position-indexed twin of `spec` (down only): symbol `..._sym_gp_pm_...`, variant `<tag>_gp`.
    pub(super) fn emit_gp(spec: Spec) -> Result<Emitted, String> {
        emit_full(
            R4Spec {
                base: spec,
                chains: 1,
                prefetch: true,
                grouped: false,
            },
            true,
            spec.nt,
        )
    }

    pub(super) fn emit_buffered(r4: R4Spec, buffered: u8) -> Result<Emitted, String> {
        emit_full(r4, false, buffered)
    }

    fn emit_full(r4: R4Spec, gpos: bool, buffered: u8) -> Result<Emitted, String> {
        let spec = r4.base;
        let g = G {
            spec,
            blk: 0,
            chains: r4.chains,
            prefetch: r4.prefetch,
            grouped: r4.grouped,
            gpos,
            buffered,
        };
        let mut b = Builder::new(kernel_spec_of(&g), g.plan()?);
        b.enable_delay_alu();
        on_target!(spec.arch, &mut b, body(&g))?;
        b.finish()
    }

    /// The entry's launch contract: symbol, variant, kernargs, workgroup and LDS size.
    fn kernel_spec_of(g: &G) -> KernelSpec {
        let spec = g.spec;
        let experimental = g.experimental();
        let suffix = if g.streamed() {
            format!("_stream{}", g.buffered_tiles())
        } else if experimental {
            format!(
                "_r4c{}p{}{}",
                g.chains,
                u8::from(g.prefetch),
                if g.grouped { "g" } else { "" }
            )
        } else {
            String::new()
        };
        let mut kernargs = spec.kernargs();
        if g.grouped {
            kernargs.size = 72;
            kernargs = kernargs.pointer("tile_leaders", 64);
        }
        KernelSpec {
            kernel_id: "qwen4_moe_sym".into(),
            variant: format!(
                "{}{suffix}{}",
                spec.variant(),
                if g.gpos { "_gp" } else { "" }
            ),
            arch: spec.arch,
            symbol: if g.gpos {
                spec.symbol().replacen("_sym_pm_", "_sym_gp_pm_", 1)
            } else {
                format!("{}{suffix}", spec.symbol())
            },
            kernargs,
            user_sgpr_count: 2,
            system_sgpr_workgroup_id_y: true,
            workgroup_size: spec.threads() as u16,
            group_segment_fixed_size: spec.lds_bytes(),
            wave32: true,
            cu_mode: false,
        }
    }

    fn body<T: Target + peacemaker_author::MmaIu4>(wg: &mut Wg<T>, g: &G) -> Result<(), String> {
        let gate = if g.gate_up() {
            Some(wg.lds::<Gate>("swiglu_gate", 0, g.spec.lds_bytes())?)
        } else {
            None
        };
        let end = wg.exit(&g.exit_label())?;
        prologue(wg, g, &end)?;
        for blk in 0..g.rr() - 1 {
            let gb = g.at(blk);
            kloop(wg, &gb)?;
            transition(wg, &gb)?;
        }
        let last = g.at(g.rr() - 1);
        kloop(wg, &last)?;
        epilogue(wg, &last, end, gate)
    }

    /// The gfx1201 MQ4 NT4 / NT8 entries (`gate_up` and `down`, flat-slot and grouped-position) as
    /// builder-level blocks, so a `.rip` script (`qwen4_moe_sym.rip`) owns the kernel's structure:
    /// the exits, the run walk, the per-tile branches of an epoch's fold and where each wait sits,
    /// the K-loop's trip structure and the order of loads inside a trip, the tail, the exit and,
    /// for `gate_up`, the LDS hand-off between the gate and up waves. Every block is a straight run
    /// of instructions (or one `Instruction` for a compare), emitted by the same function the Rust
    /// `emit` / `emit_down_gp` call, so the twin and the generator cannot drift apart. No block
    /// opens a control-flow scope, joins a workgroup barrier or touches the LDS.
    pub mod blocks {
        use super::*;

        /// Register assignment, labels and launch contract of one entry.
        pub struct Gen {
            g: G,
        }

        /// A tile index `j` is below the entry's tile width.
        fn tile(g: &Gen, j: u8) -> Result<u8, String> {
            if j < g.g.nt() {
                Ok(j)
            } else {
                Err(format!("tile {j}: this entry has {} tiles", g.g.nt()))
            }
        }

        /// An epoch parity (weight set) is 0 or 1.
        fn parity(p: usize) -> Result<usize, String> {
            if p < 2 {
                Ok(p)
            } else {
                Err(format!("epoch parity {p}: expected 0 or 1"))
            }
        }

        impl Gen {
            /// The NT4 / NT8 entry `spec` (`gp`: the grouped-position twin of the down entry). The
            /// same refusals as `emit` / `emit_down_gp`.
            pub fn new(spec: Spec, gp: bool) -> Result<Gen, String> {
                spec.validate()?;
                if spec.arch != Arch::Gfx1201
                    || spec.codec != WeightCodec::Mq4Sym
                    || spec.rr != 1
                    || !matches!(spec.nt, 4 | 8)
                {
                    return Err("qwen4_moe blocks: the gfx1201 MQ4 NT4/NT8 entries only".into());
                }
                if gp && spec.kind != Kind::Down {
                    return Err(
                        "qwen4_moe blocks: only the down entry has a grouped-position twin".into(),
                    );
                }
                Ok(Gen {
                    g: G {
                        spec,
                        blk: 0,
                        chains: 1,
                        prefetch: true,
                        grouped: false,
                        gpos: gp,
                        buffered: spec.nt,
                    },
                })
            }
            pub fn spec(&self) -> Spec {
                self.g.spec
            }
            pub fn gp(&self) -> bool {
                self.g.gpos
            }
            /// Tiles per workgroup (4 or 8).
            pub fn nt(&self) -> u8 {
                self.g.nt()
            }
            /// `gate_up` (LDS hand-off, two roles) or `down`.
            pub fn gate_up(&self) -> bool {
                self.g.gate_up()
            }
            /// Bytes of the `swiglu_gate` hand-off region (`gate_up`; 0 for `down`).
            pub fn lds_bytes(&self) -> u32 {
                self.g.spec.lds_bytes()
            }
            pub fn kernel_spec(&self) -> KernelSpec {
                kernel_spec_of(&self.g)
            }
            pub fn plan(&self) -> Result<RegPlan, String> {
                self.g.plan()
            }
            /// A block label of this entry (`k_begin`, `k_loop`, `k_loop_end`, `l0_dead1`,
            /// `tail_odd`, `epilogue`, `up`, `end`, ...).
            pub fn label(&self, name: &str) -> String {
                self.g.label(name)
            }
            /// The kernel exit's label.
            pub fn exit_label(&self) -> String {
                self.g.exit_label()
            }
        }

        /// Activation descriptor SGPR base of epoch parity `p` (the `reload` operand of [`load_x`]).
        pub fn srd_x(p: usize) -> Result<u8, String> {
            Ok(SRD_X[parity(p)?])
        }

        // ---- prologue ----

        /// Kernel arguments, workgroup and lane ids, the wave index, `TILE16`.
        pub fn pro_front(b: &mut Builder, g: &Gen) -> Result<(), String> {
            super::pro_front(b, &g.g)
        }
        /// `s_cmp_ge_i32 TILE16, m_total`: the tile is past the end (the workgroup returns).
        pub fn outside_insn(_g: &Gen) -> Instruction {
            super::outside_insn()
        }
        /// The tile's expert id and `n_experts`.
        pub fn pro_expert(b: &mut Builder, g: &Gen) -> Result<(), String> {
            super::pro_expert(b, &g.g)
        }
        /// `s_cmp_ge_u32 expert, n_experts`: a sentinel or corrupt expert id (the workgroup returns).
        pub fn sentinel_insn(_g: &Gen) -> Instruction {
            super::sentinel_insn()
        }
        /// The expert's weight pointer, the tile-id descriptor and the walk counter.
        pub fn pro_run(b: &mut Builder, g: &Gen) -> Result<(), String> {
            super::pro_run(b, &g.g)
        }
        /// One iteration of the run walk (wave-uniform `loop_until` body), up to its lane mask.
        pub fn walk_head(b: &mut Builder, g: &Gen) -> Result<(), String> {
            super::walk_head(b, &g.g)
        }
        /// `s_cmp_lg_u32 MASKA, 0`: some lane found the run's start (`break_if`).
        pub fn walk_hit_insn(_g: &Gen) -> Instruction {
            super::walk_hit_insn()
        }
        /// The walk's back edge: 32 tiles further back.
        pub fn walk_step(b: &mut Builder, g: &Gen) -> Result<(), String> {
            super::walk_step(b, &g.g)
        }
        /// This tile's position in its run, modulo `nt`.
        pub fn pro_leader(b: &mut Builder, g: &Gen) -> Result<(), String> {
            super::pro_leader(b, &g.g)
        }
        /// `s_cmp_lg_u32 POS, 0`: not the run's leader (the workgroup returns).
        pub fn follower_insn(_g: &Gen) -> Instruction {
            super::follower_insn()
        }
        /// Live tile count, row bytes, trips, row base, output offsets, the slot descriptor.
        pub fn pro_cnt(b: &mut Builder, g: &Gen) -> Result<(), String> {
            super::pro_cnt(b, &g.g)
        }
        /// Tile `j`'s sorted slot load.
        pub fn pro_slot_load(b: &mut Builder, g: &Gen, j: u8) -> Result<(), String> {
            super::pro_slot_load(b, &g.g, tile(g, j)?)
        }
        /// Activation and weight descriptors, lane offsets, and the first epoch's weight loads.
        pub fn pro_desc(b: &mut Builder, g: &Gen) -> Result<(), String> {
            super::pro_desc(b, &g.g)
        }
        /// Tile `j`'s activation row offset (dead tiles gather slot -1).
        pub fn pro_tile(b: &mut Builder, g: &Gen, j: u8) -> Result<(), String> {
            super::pro_tile(b, &g.g, tile(g, j)?)
        }
        /// The fold magic and the zeroed sums.
        pub fn pro_init(b: &mut Builder, g: &Gen) -> Result<(), String> {
            super::pro_init(b, &g.g)
        }

        // ---- loads, fold, K-loop pieces ----

        /// Issue the weight loads of epoch parity `p`.
        pub fn load_w(b: &mut Builder, g: &Gen, p: usize) -> Result<(), String> {
            super::load_w(b, &g.g, parity(p)?)
        }
        /// Tile `j`'s activation loads through the descriptor of epoch parity `p`.
        pub fn load_x(b: &mut Builder, g: &Gen, j: u8, p: usize) -> Result<(), String> {
            super::load_x(b, &g.g, tile(g, j)?, SRD_X[parity(p)?])
        }
        /// Epoch parity `p`'s row scales and nibble rebias, before its tile folds.
        pub fn compute_head(b: &mut Builder, g: &Gen, p: usize) -> Result<(), String> {
            super::compute_head(b, &g.g, parity(p)?)
        }
        /// The counters tile `j`'s fold needs (`d` and activation operands) wait here, nowhere else.
        pub fn wait_tile(b: &mut Builder, g: &Gen, j: u8) -> Result<(), String> {
            super::wait_tile(b, &g.g, tile(g, j)?)
        }
        /// `s_cmp_gt_u32 CNT, j`: tile `j` is live (wave-uniform).
        pub fn tile_live_insn(g: &Gen, j: u8) -> Result<Instruction, String> {
            Ok(super::tile_live_insn(tile(g, j)?))
        }
        /// Fold epoch parity `p` into tile `j`'s sums.
        pub fn fold_tile(b: &mut Builder, g: &Gen, p: usize, j: u8) -> Result<(), String> {
            super::fold_tile(b, &g.g, parity(p)?, tile(g, j)?)
        }
        /// Step the weight descriptor and both activation descriptors one trip's epoch pair on.
        pub fn advance(b: &mut Builder) -> Result<(), String> {
            super::advance(b, false)
        }
        /// `s_cmp_eq_u32 TRIPS, 0`: no full trip (workgroup-uniform).
        pub fn trips_zero_insn() -> Instruction {
            super::trips_insn("s_cmp_eq_u32", TRIPS)
        }
        /// `s_cmp_lg_u32 TRIPS, 0`: trips remain (workgroup-uniform back-edge condition).
        pub fn trips_more_insn() -> Instruction {
            super::trips_insn("s_cmp_lg_u32", TRIPS)
        }
        /// One trip done: `s_add_i32 TRIPS, TRIPS, -1`.
        pub fn trips_dec(b: &mut Builder) -> Result<(), String> {
            super::trips_dec(b)
        }
        /// `s_bitcmp1_b32 K, 7`: K/128 is odd, so one epoch (not two) is left after the trips.
        pub fn odd_insn() -> Instruction {
            super::odd_insn()
        }

        // ---- epilogue ----

        /// The epilogue's label (the join label of the tail's `if_else`).
        pub fn epilogue_label(b: &mut Builder, g: &Gen) -> Result<(), String> {
            super::epilogue_label(b, &g.g)
        }
        /// The output descriptor over `Y`.
        pub fn y_descriptor(b: &mut Builder, g: &Gen) -> Result<(), String> {
            super::y_descriptor(b, &g.g)
        }
        /// The output descriptor moves one tile (16 rows) on.
        pub fn y_step(b: &mut Builder) -> Result<(), String> {
            super::y_step(b)
        }
        /// Tile `j`'s BF16 rows from the sums (`gate = false`) or from the SwiGLU value registers
        /// (`gate = true`), stored at the current output descriptor.
        pub fn store_rows(b: &mut Builder, g: &Gen, j: u8, gate: bool) -> Result<(), String> {
            let j = tile(g, j)?;
            let vals = if gate { g.g.gval() } else { g.g.sum(j) };
            super::store_rows(b, &g.g, vals, g.g.live(j))
        }
        /// Every wave's BF16 round trip of its sums (`gate_up`, before the hand-off).
        pub fn bf16_roundtrip(b: &mut Builder, g: &Gen) -> Result<(), String> {
            if !g.g.gate_up() {
                return Err("bf16_roundtrip: the gate_up entry only".into());
            }
            super::bf16_roundtrip(b, &g.g)
        }
        /// `s_cmp_ge_u32 WAVE, 2`: the up waves (the hand-off readers).
        pub fn up_insn() -> Instruction {
            super::up_insn()
        }
        /// Tile `j`'s gate publication: two `ds_store_b128`, in order.
        pub fn publish_insns(g: &Gen, j: u8) -> Result<Vec<Instruction>, String> {
            if !g.g.gate_up() {
                return Err("publish_insns: the gate_up entry only".into());
            }
            Ok(super::publish_insns(&g.g, tile(g, j)?).to_vec())
        }
        /// Tile `j`'s gate read-back: two `ds_load_b128`, in order.
        pub fn gate_load_insns(g: &Gen, j: u8) -> Result<Vec<Instruction>, String> {
            if !g.g.gate_up() {
                return Err("gate_load_insns: the gate_up entry only".into());
            }
            Ok(super::gate_load_insns(&g.g, tile(g, j)?).to_vec())
        }
        /// Tile `j`'s SwiGLU over the gate rows read back and the up sums.
        pub fn silu_tile(b: &mut Builder, g: &Gen, j: u8) -> Result<(), String> {
            if !g.g.gate_up() {
                return Err("silu_tile: the gate_up entry only".into());
            }
            super::silu_tile(b, &g.g, tile(g, j)?)
        }
    }
}
