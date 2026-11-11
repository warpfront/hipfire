// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Builder-emitted MQ4V2 x block_i4_128 GEMM for gfx1201 (plan §3-§4).
//!
//! Semantics are hipcc K1's `_v3` symbols bit for bit: per 128-K block each
//! (row, token) int32 dot starts from the fold magic, is folded once as
//! `acc = fma(RN(sc*d), float(C) - magic, acc)` in ascending block order,
//! and the epilogue writes `Y = acc`, `Y = RN(Y + acc)` or the imported hipcc
//! SiLU region. The `_b1_bf16` SiLU variant rounds h to packed bf16 (RNE).
//! Everything else (CTA raster, staging map, store shape, waits) is generated here.
//!
//! Launch contract: grid `[ceil(rows / tile_rows), ceil(N / 128)]` with
//! `rows = M` (SET/ADD) or `2M` (gate/up), block `[tile_threads, 1, 1]`,
//! dynamic LDS `Layout::launch` bytes. Preconditions shared with the hipcc
//! wide epilogue: `K % 256 == 0`, `M % 4 == 0`, and `Y`/`H` 4-byte aligned;
//! packed bf16 h uses contiguous rows with `M * 2` byte stride.
pub mod blocks;
pub mod epilogue;
pub mod fold;
pub mod gdn_epilogue;
pub mod kloop;
pub mod prologue;
pub mod publish;
pub mod region;
pub mod ring3;
pub mod sparse24;
pub mod spec;

use crate::kernels::common::op;
use crate::{reg::Live, Builder, Emitted, KernelSpec, RegPlan};
use peacemaker_author::{
    Free, Gfx1201, Gfx12Waits, LdsRegion, LdsWrite, Pending, Published, Ring, Wave, Workgroup,
    Writing,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
pub use spec::{
    ALayout, Cacc, Epi, FetchPlace, Fold, FoldOp, FoldSched, IdleHalf, Mma, QkvStore, RingDepth,
    Spec, Tile, Tune, WaitPlace,
};

/// The program's first label (the kernel entry); `Live` ranges start here.
pub const ENTRY: &str = "entry";
pub const K_BEGIN: &str = ".Liu4_k_begin";
pub const K_LOOP: &str = ".Liu4_k_loop";
pub const K_LOOP_END: &str = ".Liu4_k_loop_end";
pub const EPI: &str = ".Liu4_epilogue";
pub const END: &str = ".Liu4_end";
/// Fused projection only: the QKV tiles' GDN preparation (after the stores).
pub const GDN: &str = ".Liu4_gdn";

/// DS offset fields spelled as objdump prints them (zero fields omitted).
pub(crate) fn ds_offsets(o0: u32, o1: u32) -> String {
    let mut t = String::new();
    if o0 != 0 {
        t.push_str(&format!(" offset0:{o0}"))
    }
    if o1 != 0 {
        t.push_str(&format!(" offset1:{o1}"))
    }
    t
}
pub(crate) fn ds_offset(o: u32) -> String {
    if o == 0 {
        String::new()
    } else {
        format!(" offset:{o}")
    }
}
/// Accumulator range names indexed by `k = nb * 2 + rg`.
pub(crate) const CACC_NAMES: [&str; 8] = [
    "cacc[0][0]",
    "cacc[0][1]",
    "cacc[1][0]",
    "cacc[1][1]",
    "cacc[2][0]",
    "cacc[2][1]",
    "cacc[3][0]",
    "cacc[3][1]",
];
pub(crate) const ACC_NAMES: [&str; 8] = [
    "acc[0][0]",
    "acc[0][1]",
    "acc[1][0]",
    "acc[1][1]",
    "acc[2][0]",
    "acc[2][1]",
    "acc[3][0]",
    "acc[3][1]",
];

/// Kernel-argument SGPRs after the two `s_load_b256` at 0x0 and 0x20.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Args {
    pub a: u8,
    pub u: Option<u8>,
    pub xq: u8,
    pub y: u8,
    pub m: u8,
    pub k: u8,
    pub n: u8,
    /// `Epi::AddSigmoid` only: the `shared_gate` pointer (SGPR pair).
    pub gate: Option<u8>,
    pub bcx: u8,
    pub bcy: u8,
}

/// Physical register assignment. VGPR map (T128, One):
/// `cacc v[0:63]` (prologue/epilogue temporaries alias it), `acc v[64:127]`,
/// `magic8 v[128:135]`, fragment sets F0/F1 `v[136:159]` (fold scale rows and
/// products alias them once their WMMAs have issued), `d v[160:163]`, the
/// prefetch payload, then hoisted addresses: 187 VGPRs (T256: 186). Ring depth
/// 3 (`ring3.rs`) has no `magic8`: its second prefetch set takes `v[128:135]`
/// (each block's first WMMA group seeds from `cacc[3][1]` instead), so the
/// count is the same.
#[derive(Clone, Debug)]
pub struct Gen {
    pub(crate) spec: Spec,
    pub(crate) tile: Tile,
    pub(crate) layout: spec::Layout,
    pub(crate) args: Args,
    pub(crate) cacc: u8,
    pub(crate) acc: u8,
    pub(crate) magic: u8,
    pub(crate) f: [u8; 2],
    pub(crate) d: u8,
    pub(crate) a_pf: Vec<u8>,
    pub(crate) w_pf: Vec<u8>,
    /// Ring depth 3 only: the second prefetch set (a block's slab 1 while set
    /// 0 holds its slab 0), in the eight VGPRs depth 3 frees by seeding the
    /// WMMA accumulators from `cacc[3][1]` instead of `magic8`. Empty at depth 2.
    pub(crate) a_pf1: Vec<u8>,
    pub(crate) w_pf1: Vec<u8>,
    /// `FetchPlace::PairW` only: the third weight set (the next block's slab 1 W
    /// payload, fetched back to back with its slab 0). Empty otherwise.
    pub(crate) w_x: Vec<u8>,
    pub(crate) ds_nx: u8,
    pub(crate) sz_nx: u8,
    pub(crate) fr_a: [u8; 2],
    pub(crate) fr_w: [u8; 2],
    pub(crate) st_lds: u8,
    pub(crate) st_a: u8,
    pub(crate) st_w: Vec<u8>,
    pub(crate) meta_ds: u8,
    pub(crate) meta_sz: u8,
    pub(crate) ds_voff: u8,
    pub(crate) sz_voff: u8,
    pub(crate) sc_addr: u8,
    pub(crate) d_addr: u8,
    // SGPRs
    pub(crate) karg: u8,
    pub(crate) srd_w: u8,
    pub(crate) srd_z: u8,
    pub(crate) srd_a: [u8; 2],
    pub(crate) srd_y: u8,
    pub(crate) step2: u8,
    pub(crate) goff: u8,
    pub(crate) trips: u8,
    pub(crate) rs: u8,
    pub(crate) bs: u8,
    pub(crate) wave: u8,
    pub(crate) gpr136: u8,
    pub(crate) mm1: u8,
    pub(crate) hs: u8,
    pub(crate) tmp: u8,
    pub(crate) epi_s: u8,
    /// `ALayout::Slab` only: the SGPR holding `32 * N`, the byte distance
    /// from a token's slab-0 plane entry to its slab-1 entry.
    pub(crate) a_slab1: Option<u8>,
    /// `IdleHalf::Skip`/`SkipHi` only (else the plan's unused scratch s55): the
    /// wave-uniform "this wave's token half is empty" flag, 1 or 0, set once in the
    /// prologue and read by the K-loop's skip branches and the epilogue's.
    pub(crate) idle: u8,
}

pub(crate) const PRO_TEMPS: u8 = 24;
pub(crate) const S_TEMPS: u8 = 8;
pub(crate) const EPI_S: u8 = 24;

impl Gen {
    /// The generator for `spec`: the register assignment and geometry every
    /// block of `blocks` reads. `Err` when `spec` has no generator
    /// (`Spec::validate`).
    pub fn new(spec: Spec) -> Result<Self, String> {
        spec.validate()?;
        Ok(Self::build(spec))
    }
    /// The spec this generator was built for.
    pub fn spec(&self) -> Spec {
        self.spec
    }
    /// The payload rings are three deep (`Tune::ring`).
    pub fn ring3(&self) -> bool {
        self.spec.tune.ring == RingDepth::Three
    }
    /// The register assignment's VGPR count: one past the highest register any
    /// phase names.
    pub fn vgprs(&self) -> u16 {
        let sets = [&self.a_pf, &self.w_pf, &self.a_pf1, &self.w_pf1, &self.w_x];
        let top_pf = sets
            .iter()
            .filter_map(|v| v.last())
            .map(|&r| u16::from(r) + 2)
            .max()
            .unwrap_or(0);
        top_pf
            .max(u16::from(self.d_addr) + 1)
            .max(u16::from(self.sz_voff) + 1)
    }
    /// The kernel header (symbol, kernargs, workgroup size) of this spec.
    pub fn kernel_spec(&self) -> KernelSpec {
        let spec = self.spec;
        KernelSpec {
            kernel_id: "iu4_gemm".into(),
            variant: spec.variant_name(),
            arch: spec.arch,
            symbol: spec.symbol(),
            kernargs: spec.kernargs(),
            user_sgpr_count: 2,
            system_sgpr_workgroup_id_y: true,
            workgroup_size: spec.tile.threads() as u16,
            group_segment_fixed_size: self.layout.fixed,
            wave32: true,
            cu_mode: false,
        }
    }
    /// A physical register base of this assignment by name (read-only):
    /// `cacc acc magic d f0 f1 ds_nx sz_nx fr_a0 fr_a1 fr_w0 fr_w1 st_lds st_a
    /// meta_ds meta_sz ds_voff sz_voff sc_addr d_addr`, `a_pf<i> w_pf<i>
    /// pf1_a<i> pf1_w<i> (depth 3) st_w<i>`, SGPRs `karg srd_w srd_z srd_a0 srd_a1 srd_y step2 goff trips
    /// rs bs wave gpr136 mm1 hs tmp epi_s a_slab1 idle` and kernel arguments
    /// `arg_a arg_u arg_gate arg_xq arg_y arg_m arg_k arg_n arg_bcx arg_bcy`. `None`
    /// for an unknown name and for an absent optional register (`arg_u`
    /// outside gate/up, `arg_gate` outside the sigmoid ADD, `a_slab1` outside
    /// the slab layout).
    pub fn reg(&self, name: &str) -> Option<u8> {
        let a = &self.args;
        let indexed = |base: &str, v: &[u8]| -> Option<u8> {
            v.get(name.strip_prefix(base)?.parse::<usize>().ok()?)
                .copied()
        };
        Some(match name {
            "cacc" => self.cacc,
            "acc" => self.acc,
            "magic" => self.magic,
            "d" => self.d,
            "f0" => self.f[0],
            "f1" => self.f[1],
            "ds_nx" => self.ds_nx,
            "sz_nx" => self.sz_nx,
            "fr_a0" => self.fr_a[0],
            "fr_a1" => self.fr_a[1],
            "fr_w0" => self.fr_w[0],
            "fr_w1" => self.fr_w[1],
            "st_lds" => self.st_lds,
            "st_a" => self.st_a,
            "meta_ds" => self.meta_ds,
            "meta_sz" => self.meta_sz,
            "ds_voff" => self.ds_voff,
            "sz_voff" => self.sz_voff,
            "sc_addr" => self.sc_addr,
            "d_addr" => self.d_addr,
            "karg" => self.karg,
            "srd_w" => self.srd_w,
            "srd_z" => self.srd_z,
            "srd_a0" => self.srd_a[0],
            "srd_a1" => self.srd_a[1],
            "srd_y" => self.srd_y,
            "step2" => self.step2,
            "goff" => self.goff,
            "trips" => self.trips,
            "rs" => self.rs,
            "bs" => self.bs,
            "wave" => self.wave,
            "gpr136" => self.gpr136,
            "mm1" => self.mm1,
            "hs" => self.hs,
            "tmp" => self.tmp,
            "epi_s" => self.epi_s,
            "a_slab1" => return self.a_slab1,
            "arg_a" => a.a,
            "idle" => self.idle,
            "arg_u" => return a.u,
            "arg_gate" => return a.gate,
            "arg_xq" => a.xq,
            "arg_y" => a.y,
            "arg_m" => a.m,
            "arg_k" => a.k,
            "arg_n" => a.n,
            "arg_bcx" => a.bcx,
            "arg_bcy" => a.bcy,
            _ => {
                return indexed("pf1_a", &self.a_pf1)
                    .or_else(|| indexed("pf1_w", &self.w_pf1))
                    .or_else(|| indexed("a_pf", &self.a_pf))
                    .or_else(|| indexed("w_pf", &self.w_pf))
                    .or_else(|| indexed("w_x", &self.w_x))
                    .or_else(|| indexed("st_w", &self.st_w))
            }
        })
    }
    /// The LDS byte offset of payload slab `slot` of ring `role` (read-only
    /// geometry: the layout a typed `PayA` / `PayW` ring's buffer `slot` sits
    /// at). Refuses a slot the spec's ring depth does not have.
    pub fn payload_base(&self, role: PayloadRole, slot: usize) -> Result<u32, String> {
        let depth = match self.spec.tune.ring {
            RingDepth::Two => 2,
            RingDepth::Three => 3,
        };
        let slots = match role {
            PayloadRole::A => &self.layout.a,
            PayloadRole::W => &self.layout.w,
        };
        match slots.get(slot) {
            Some(&base) if slot < depth => Ok(base),
            _ => Err(format!(
                "payload_base({role:?}, {slot}): the payload rings are {depth} deep (slots 0..{depth})"
            )),
        }
    }
    fn build(spec: Spec) -> Self {
        let tile = spec.tile;
        let args = if spec.epi.is_silu() {
            Args {
                a: 8,
                u: Some(10),
                xq: 12,
                y: 14,
                m: 16,
                k: 17,
                n: 18,
                gate: None,
                bcx: 20,
                bcy: 21,
            }
        } else if spec.epi == Epi::QkvzaGdn {
            // s[8:15] Wqkv Wz Xq Yz, s[16:23] Ybeta Yalpha M0 Mz K N; the block
            // counts load into s[28:29]; s30 is the selected segment's rows.
            Args {
                a: 8,
                u: None,
                xq: 12,
                y: 14,
                m: 30,
                k: 22,
                n: 23,
                gate: None,
                bcx: 28,
                bcy: 29,
            }
        } else if spec.epi == Epi::AddSigmoid {
            // The ADD arguments, then `shared_gate` at offset 40 (s[18:19]);
            // the hidden block counts follow at offset 48.
            Args {
                a: 8,
                u: None,
                xq: 10,
                y: 12,
                m: 14,
                k: 15,
                n: 16,
                gate: Some(18),
                bcx: 20,
                bcy: 21,
            }
        } else {
            Args {
                a: 8,
                u: None,
                xq: 10,
                y: 12,
                m: 14,
                k: 15,
                n: 16,
                gate: None,
                bcx: 18,
                bcy: 19,
            }
        };
        let mut next = 164u8;
        let mut take = |n: u8| {
            let b = next;
            next += n;
            b
        };
        let a_pf = (0..tile.a_rounds()).map(|_| take(2)).collect::<Vec<_>>();
        let w_pf = (0..tile.w_rounds()).map(|_| take(2)).collect::<Vec<_>>();
        // Depth 3: the second prefetch set takes the `magic8` slot (v128..135);
        // `Spec::validate` refuses a tile whose set is wider than those 8.
        let mut p1 = 128u8;
        let mut take1 = |n: u8| {
            let b = p1;
            p1 += n;
            b
        };
        let ring3 = spec.tune.ring == RingDepth::Three;
        let a_pf1 = (0..if ring3 { tile.a_rounds() } else { 0 })
            .map(|_| take1(2))
            .collect::<Vec<_>>();
        let w_pf1 = (0..if ring3 { tile.w_rounds() } else { 0 })
            .map(|_| take1(2))
            .collect::<Vec<_>>();
        let ds_nx = take(1);
        let sz_nx = take(1);
        let fr_a = [take(1), take(1)];
        let fr_w = [take(1), take(1)];
        let st_lds = take(1);
        let st_a = take(1);
        let st_w = (0..tile.w_rounds()).map(|_| take(1)).collect::<Vec<_>>();
        let meta_ds = take(1);
        let meta_sz = if tile == Tile::T128x128x8 {
            meta_ds
        } else {
            take(1)
        };
        let ds_voff = take(1);
        let sz_voff = take(1);
        let sc_addr = take(1);
        let d_addr = take(1);
        let w_x = (0..if spec.tune.fetch == FetchPlace::PairW {
            tile.w_rounds()
        } else {
            0
        })
            .map(|_| take(2))
            .collect::<Vec<_>>();
        let silu = spec.epi.is_silu();
        Self {
            spec,
            tile,
            layout: tile.layout_for(spec.tune.ring),
            args,
            cacc: 0,
            acc: 64,
            magic: 128,
            f: [136, 148],
            d: 160,
            a_pf,
            w_pf,
            a_pf1,
            w_pf1,
            w_x,
            ds_nx,
            sz_nx,
            fr_a,
            fr_w,
            st_lds,
            st_a,
            st_w,
            meta_ds,
            meta_sz,
            ds_voff,
            sz_voff,
            sc_addr,
            d_addr,
            karg: 0,
            srd_w: 24,
            srd_z: if silu { 28 } else { 24 },
            srd_a: [32, 36],
            srd_y: 40,
            step2: 44,
            goff: 46,
            trips: 47,
            rs: 48,
            bs: 49,
            wave: 50,
            gpr136: 51,
            mm1: 52,
            hs: 53,
            tmp: 56,
            epi_s: 64,
            idle: 55,
            a_slab1: (spec.act == ALayout::Slab).then_some(2),
        }
    }

    /// Whole-kernel and phase-scoped ranges. Per-block fragment/fold aliases
    /// are declared by the K-loop generator with label-delimited lifetimes.
    /// The fused projection ends the K-loop/store registers at its GDN label,
    /// where `gdn_epilogue` reuses them.
    pub fn plan(&self) -> Result<RegPlan, String> {
        // The fused projection's segment arguments (`fused`) and its GDN preparation
        // (`gdn`, absent when the QKV tiles store like SET: `QkvStore::Set`).
        let fused = self.spec.epi == Epi::QkvzaGdn;
        let gdn = self.spec.gdn_prep();
        let mut p = RegPlan::new(spec::VGPR_CEILING, 104)?;
        let pro = || Live::Between(ENTRY.into(), K_BEGIN.into());
        let kl = || Live::Between(K_BEGIN.into(), EPI.into());
        let kernel = || Live::Between(ENTRY.into(), EPI.into());
        let epi = || {
            if gdn {
                Live::Between(EPI.into(), GDN.into())
            } else {
                Live::Between(EPI.into(), END.into())
            }
        };
        // Kernel-wide SGPRs the GDN path reuses.
        let shared = || {
            if gdn {
                Live::Between(ENTRY.into(), GDN.into())
            } else {
                Live::Whole
            }
        };
        for i in 0..8u8 {
            p.v::<8>(CACC_NAMES[usize::from(i)], self.cacc + 8 * i, kl())?;
        }
        for i in 0..PRO_TEMPS {
            p.v::<1>(if i == 0 { "tid" } else { "prologue_tmp" }, i, pro())?;
        }
        for i in 0..8u8 {
            p.v::<8>(ACC_NAMES[usize::from(i)], self.acc + 8 * i, Live::Whole)?;
        }
        if !self.ring3() {
            p.v::<8>("magic8", self.magic, kernel())?;
        }
        p.v::<4>("d_x", self.d, kl())?;
        if self.spec.tune.idle != IdleHalf::Off {
            // The idle-half skip bodies define and read these in different, correlated
            // branches, which the definedness check cannot pair: `kloop::define_dead_regs`
            // defines them once at the K-loop entry (the fragment sets F0/F1 are
            // block-scoped ranges that start later).
            for i in 0..3u8 {
                p.v::<8>(
                    "idle_init_f",
                    self.f[0] + 8 * i,
                    Live::Between(K_BEGIN.into(), K_LOOP.into()),
                )?;
            }
        }
        for &r in &self.a_pf {
            p.v::<2>("a_pf", r, kernel())?;
        }
        for &r in &self.w_pf {
            p.v::<2>("w_pf", r, kernel())?;
        }
        for &r in &self.a_pf1 {
            p.v::<2>("a_pf1", r, kernel())?;
        }
        for &r in &self.w_x {
            p.v::<2>("w_x", r, kernel())?;
        }
        for &r in &self.w_pf1 {
            p.v::<2>("w_pf1", r, kernel())?;
        }
        p.v::<1>("ds_nx", self.ds_nx, kernel())?;
        p.v::<1>("sz_nx", self.sz_nx, kernel())?;
        for (name, r) in [
            ("fr_a0", self.fr_a[0]),
            ("fr_a1", self.fr_a[1]),
            ("fr_w0", self.fr_w[0]),
            ("fr_w1", self.fr_w[1]),
            ("st_ldsoff", self.st_lds),
            ("st_avoff", self.st_a),
            ("ds_voff", self.ds_voff),
            ("sz_voff", self.sz_voff),
            ("meta_lds_ds", self.meta_ds),
        ] {
            p.v::<1>(name, r, kernel())?;
        }
        if self.meta_sz != self.meta_ds {
            p.v::<1>("meta_lds_sz", self.meta_sz, kernel())?;
        }
        for &r in &self.st_w {
            p.v::<1>("st_wvoff", r, kernel())?;
        }
        p.v::<1>("sc_addr", self.sc_addr, Live::Whole)?;
        p.v::<1>("d_addr", self.d_addr, Live::Whole)?;
        // Epilogue: temporaries over the dead int32 accumulators and F0/F1.
        for i in 0..16u8 {
            p.v::<4>("epi_quad", self.cacc + 4 * i, epi())?;
        }
        for i in 0..24u8 {
            p.v::<1>("epi_tmp", self.f[0] + i, epi())?;
        }
        // SGPRs.
        p.s::<2>("kernarg_ptr", self.karg, Live::Whole)?;
        p.s::<8>("kernargs_0x00", 8, Live::Whole)?;
        p.s::<8>("kernargs_0x20", 16, Live::Whole)?;
        p.s::<4>("srd_w", self.srd_w, shared())?;
        if self.srd_z != self.srd_w {
            p.s::<4>("srd_z", self.srd_z, Live::Whole)?;
        }
        if fused {
            p.s::<4>("block_counts_segment", 28, shared())?;
        }
        p.s::<4>("srd_a0", self.srd_a[0], shared())?;
        p.s::<4>("srd_a1", self.srd_a[1], shared())?;
        if let Some(r) = self.a_slab1 {
            p.s::<1>("a_slab1_soff", r, shared())?;
        }
        p.s::<4>("srd_y", self.srd_y, shared())?;
        p.s::<2>("a_step2", self.step2, shared())?;
        for (name, r) in [
            ("goff", self.goff),
            ("trips", self.trips),
            ("rs", self.rs),
            ("bs", self.bs),
            ("wave", self.wave),
            ("gpr136", self.gpr136),
            ("m_minus_1", self.mm1),
            ("hs", self.hs),
            ("rows", 54),
            (
                if self.spec.tune.idle == IdleHalf::Off {
                    "scratch_s55"
                } else {
                    "idle_half"
                },
                55,
            ),
        ] {
            // rs, bs and the wave index stay live into the GDN path.
            p.s::<1>(
                name,
                r,
                if matches!(name, "rs" | "bs" | "wave") {
                    Live::Whole
                } else {
                    shared()
                },
            )?;
        }
        for i in 0..S_TEMPS {
            p.s::<1>("s_tmp", self.tmp + i, shared())?;
        }
        for i in 0..EPI_S {
            p.s::<1>("epi_s", self.epi_s + i, shared())?;
        }
        if self.spec.epi == Epi::AddSigmoid {
            epilogue::plan_sigmoid(&mut p)?;
        }
        if gdn {
            gdn_epilogue::plan(&mut p)?;
        }
        Ok(p)
    }

    /// Slots A0, W0, A1, W1, DS0, DS1, SZ0, SZ1 (builder ids 0..7) as four
    /// two-buffer rings; buffer 0 of each is written first.
    pub(crate) fn declare_lds(&self, wg: &mut Wg) -> Result<FreeRings, String> {
        let l = self.layout;
        let (a0, w0) = (
            wg.lds("A0", l.a[0], l.a_bytes)?,
            wg.lds("W0", l.w[0], l.w_bytes)?,
        );
        let (a1, w1) = (
            wg.lds("A1", l.a[1], l.a_bytes)?,
            wg.lds("W1", l.w[1], l.w_bytes)?,
        );
        let (d0, d1) = (
            wg.lds("DS0", l.ds[0], l.ds_bytes)?,
            wg.lds("DS1", l.ds[1], l.ds_bytes)?,
        );
        let (s0, s1) = (
            wg.lds("SZ0", l.sz[0], l.sz_bytes)?,
            wg.lds("SZ1", l.sz[1], l.sz_bytes)?,
        );
        if l.end > l.launch {
            return Err("LDS layout exceeds the launch size".into());
        }
        Ok((
            Ring::new(a0, a1),
            Ring::new(w0, w1),
            Ring::new(d0, d1),
            Ring::new(s0, s1),
        ))
    }
}

impl Gen {
    /// Ring depth 3: slots A0 W0 A1 W1 A2 W2 and the metadata planes (builder
    /// ids by declaration order), the payload rings three deep.
    pub(crate) fn declare_lds3(&self, wg: &mut Wg) -> Result<FreeRings3, String> {
        let l = self.layout;
        let (mut a, mut w) = (Vec::new(), Vec::new());
        for i in 0..3 {
            a.push(wg.lds(&format!("A{i}"), l.a[i], l.a_bytes)?);
            w.push(wg.lds(&format!("W{i}"), l.w[i], l.w_bytes)?);
        }
        let (d0, d1) = (
            wg.lds("DS0", l.ds[0], l.ds_bytes)?,
            wg.lds("DS1", l.ds[1], l.ds_bytes)?,
        );
        let (s0, s1) = (
            wg.lds("SZ0", l.sz[0], l.sz_bytes)?,
            wg.lds("SZ1", l.sz[1], l.sz_bytes)?,
        );
        if l.end > l.fixed + l.launch {
            return Err("LDS layout exceeds the launch size".into());
        }
        fn three<R>(v: Vec<LdsRegion<R, Free>>) -> Result<[LdsRegion<R, Free>; 3], String> {
            v.try_into().map_err(|_| "three slabs".to_string())
        }
        Ok((
            Ring::new_n(three(a)?)?,
            Ring::new_n(three(w)?)?,
            Ring::new(d0, d1),
            Ring::new(s0, s1),
        ))
    }
}

/// LDS tags: activation and weight payload of one 64-K slab, the token
/// scale (`d`) plane and the converted row-scale (`sc`) plane.
pub enum PayA {}
pub enum PayW {}
pub enum MetaD {}
pub enum MetaS {}
/// The role of a payload ring (`PayA` or `PayW`) a script-native body names
/// for one of its loads; the native maps it to the sealed tag, so a script
/// cannot make a ring of either kind by naming it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PayloadRole {
    A,
    W,
}
/// The kernel's workgroup scope on gfx1201.
pub type Wg<'b> = Workgroup<'b, Gfx1201, Builder>;
pub(crate) type Wv<'b> = Wave<'b, Gfx1201, Builder>;
/// The four LDS rings as declared: every buffer free.
pub type FreeRings = (
    Ring<PayA, Free, Free>,
    Ring<PayW, Free, Free>,
    Ring<MetaD, Free, Free>,
    Ring<MetaS, Free, Free>,
);
/// Ring depth 3: the payload rings three deep, the metadata rings as declared.
pub type FreeRings3 = (
    Ring<PayA, Free, Free, 3>,
    Ring<PayW, Free, Free, 3>,
    Ring<MetaD, Free, Free>,
    Ring<MetaS, Free, Free>,
);
/// Ring depth 3 steady state at a block boundary: each payload ring's current
/// and ahead buffers published, the next one free; the metadata rings as at
/// depth 2.
pub type Lds3 = (
    Ring<PayA, Published, Free, 3>,
    Ring<PayW, Published, Free, 3>,
    Ring<MetaD, Published, Free>,
    Ring<MetaS, Published, Free>,
);
/// Steady state at a block boundary: each ring's current buffer published,
/// the next one free.
pub type Lds = (
    Ring<PayA, Published, Free>,
    Ring<PayW, Published, Free>,
    Ring<MetaD, Published, Free>,
    Ring<MetaS, Published, Free>,
);
/// A ring's next buffer with its pending stores: what a publish returns and a
/// wait consumes.
pub type Staged<R, C, const D: usize = 2> =
    (Ring<R, C, Writing, D>, Pending<Gfx12Waits, LdsWrite<R>>);

/// Emit one kernel symbol. The result is a complete single-kernel `.s`.
pub fn emit(spec: Spec) -> Result<Emitted, String> {
    let g = Gen::new(spec)?;
    let mut b = Builder::new(g.kernel_spec(), g.plan()?);
    let mut wg = Wg::new(&mut b)?;
    if g.ring3() {
        let lds = g.declare_lds3(&mut wg)?;
        let end = wg.exit(END)?;
        op(wg.isa(), "s_setprio 1", &[], &[])?;
        prologue::setup3(&mut wg, &g, &end)?;
        ring3::emit(&mut wg, &g, lds)?;
        blocks::epilogue(&mut wg, &g, &end)?;
        wg.end(end)?;
        return b.finish();
    }
    let lds = blocks::declare_lds(&mut wg, &g)?;
    let end = wg.exit(END)?;
    // Every symbol runs its K-loop at wave priority 1 and drops to 0 for its
    // epilogue (`epilogue::emit`; the fused projection keeps its stores at 1
    // and drops at the QKV tiles' GDN epilogue, `gdn_epilogue`), so
    // co-resident K-loops win issue arbitration over a finishing tile.
    // Scheduling only; every result bit is unchanged.
    op(wg.isa(), "s_setprio 1", &[], &[])?;
    let lds = blocks::prologue(&mut wg, &g, &end, lds)?;
    let lds = blocks::kloop(&mut wg, &g, lds)?;
    blocks::epilogue(&mut wg, &g, &end)?;
    // Only the fused projection reuses the LDS after the K loop: its QKV
    // tiles continue at the GDN label the epilogue's exit branch skips to.
    if spec.gdn_prep() {
        blocks::gdn_epilogue(&mut wg, &g, lds)?;
    }
    wg.end(end)?;
    b.finish()
}

/// Proof of a multi-kernel module: the per-symbol builder proofs plus the
/// digest of the combined source, bound the way `peacemaker custom build`
/// binds a single-kernel proof.
#[derive(Serialize)]
pub struct ModuleProof {
    pub module: String,
    pub arch: crate::Arch,
    pub builder_crate_version: String,
    pub builder_git_sha: String,
    pub s_text_sha256: String,
    pub kernels: Vec<crate::BuilderProof>,
}

/// Combine several single-kernel sources into one code object source: one
/// target header, every kernel body and descriptor, one metadata document.
pub fn module(emitted: &[Emitted], name: &str) -> Result<(String, ModuleProof), String> {
    let first = emitted.first().ok_or("empty module")?;
    let mut header = String::new();
    let mut bodies = String::new();
    let mut kernels = String::new();
    let mut tail = String::new();
    for (i, e) in emitted.iter().enumerate() {
        let (code, meta) = e
            .s_text
            .split_once(".amdgpu_metadata\n")
            .ok_or("missing metadata")?;
        let (pre, body) = code.split_once(".text\n").ok_or("missing .text")?;
        if i == 0 {
            header = pre.to_owned();
        } else if pre != header {
            return Err("kernels disagree on target header".into());
        }
        bodies.push_str(".text\n");
        // Local labels are per-kernel names; qualify them by epilogue so the
        // kernels can share one assembler unit. The `_v64` exports repeat the shipped
        // exports' epilogues, so they qualify by the whole variant name.
        let owned_tag;
        let tag = if e.proof.variant.contains("-n64") || e.proof.variant.contains("-idle") {
            owned_tag = e.proof.variant.replace('-', "_");
            owned_tag.as_str()
        } else {
            e.proof.variant.rsplit('-').next().unwrap_or("k")
        };
        // A symbol that itself starts with `iu4_` (`iu4_c8_*_pm_gfx1201`) owns an end label `.Liu4_<symbol>_end` that
        // `pm_check` looks up by name: keep it out of the per-kernel qualification.
        let symbol_end = e
            .s_text
            .lines()
            .find_map(|l| l.strip_prefix(".globl "))
            .map(|s| format!(".L{}_end", s.trim()))
            .filter(|end| end.starts_with(".Liu4_"));
        let protected = symbol_end.as_deref().map_or_else(
            || body.to_owned(),
            |end| body.replace(end, "\u{1}SYMBOL_END"),
        );
        let qualified = protected.replace(".Liu4_", &format!(".Liu4_{tag}_"));
        bodies.push_str(&symbol_end.as_deref().map_or(qualified.clone(), |end| {
            qualified.replace("\u{1}SYMBOL_END", end)
        }));
        let (_, list) = meta
            .split_once("amdhsa.kernels:\n")
            .ok_or("missing kernel list")?;
        let (items, rest) = list.split_once("amdhsa.target:").ok_or("missing target")?;
        kernels.push_str(items);
        if i == 0 {
            tail = format!("amdhsa.target:{rest}");
        }
    }
    let text = format!("{header}{bodies}.amdgpu_metadata\n---\namdhsa.kernels:\n{kernels}{tail}");
    let proof = ModuleProof {
        module: name.into(),
        arch: first.proof.arch,
        builder_crate_version: first.proof.builder_crate_version.clone(),
        builder_git_sha: first.proof.builder_git_sha.clone(),
        s_text_sha256: format!("{:x}", Sha256::digest(text.as_bytes())),
        kernels: emitted.iter().map(|e| e.proof.clone()).collect(),
    };
    Ok((text, proof))
}

/// The tune the shipped module [`emit_module`] emits symbol `epi` with: the 27B gate/up
/// export of the `_b1s` module (128x128x8 tile, slab layout, `Epi::GateUpSilu`,
/// `gemm_mq4g256v2_gate_up_silu_mmq_iu4_b1s`) carries [`Tune::shipped_gate_up_slab`]; every
/// other symbol of every module is the default schedule.
pub fn module_tune(tile: Tile, act: ALayout, epi: Epi) -> Tune {
    if tile == Tile::T128x128x8 && act == ALayout::Slab && epi == Epi::GateUpSilu {
        Tune::shipped_gate_up_slab()
    } else {
        Tune::default()
    }
}

/// The schedule of the small-N (`n <= 128` rows: verify / short batch) `_v64` export of
/// `epi` in the slab module (`qkv_set` selects the fused projection's SET-store export). The
/// outputs are byte-identical to the shipped export's; only the schedule differs: the B1
/// rendezvous window, PairW weight fetches (not generated for the fused projection), and
/// the dead token-half skip with the wave remap that puts one dead and one live wave on
/// every SIMD ([`IdleHalf::SkipHi`]; the predicate is false when more than 64 tokens are
/// live, so up to 128 rows the dead waves compute exactly as before). Ring depth 3 is not
/// used: the independent wait replay (`ledger_replay`) rejects its ledger. These values
/// come from the pm_loop N=64 measurements; this is the one place they live.
pub fn verify64_tune(epi: Epi, qkv_set: bool) -> Tune {
    let base = Tune {
        n64: true,
        wait: WaitPlace::Window,
        ..Tune::default()
    };
    match epi {
        Epi::QkvzaGdn => Tune {
            qkv: if qkv_set {
                QkvStore::Set
            } else {
                QkvStore::Gdn
            },
            idle: if qkv_set {
                IdleHalf::SkipHi
            } else {
                IdleHalf::Off
            },
            ..base
        },
        _ => Tune {
            fetch: FetchPlace::PairW,
            idle: IdleHalf::SkipHi,
            ..base
        },
    }
}

/// All original epilogue symbols plus packed bf16 h and the fused GDN input
/// projection for the production tile.
pub fn emit_module(
    fold: Fold,
    tile: Tile,
    cacc: Cacc,
    act: ALayout,
    arch: crate::Arch,
) -> Result<(Vec<Emitted>, String, ModuleProof), String> {
    let mut epis = vec![Epi::Set, Epi::Add, Epi::GateUpSilu];
    if tile == Tile::T128x128x8 {
        epis.push(Epi::GateUpSiluBf16);
        epis.push(Epi::QkvzaGdn);
        // The sigmoid residual ADD reads token-layout activations only.
        if act == ALayout::Token {
            epis.push(Epi::AddSigmoid);
        }
    }
    let mut specs: Vec<(Epi, Tune)> = epis
        .into_iter()
        .map(|epi| (epi, module_tune(tile, act, epi)))
        .collect();
    // The slab module also exports the fused projection with plain QKV stores
    // (`..._qkvzaset_b1s`): one launch over the QKV weight and the Z fold for the verify
    // path, whose consumers read the SET outputs (`QkvStore::Set`).
    if tile == Tile::T128x128x8 && act == ALayout::Slab {
        specs.push((
            Epi::QkvzaGdn,
            Tune {
                qkv: QkvStore::Set,
                ..Tune::default()
            },
        ));
        // The small-N `_v64` exports (see `verify64_tune`).
        specs.push((Epi::GateUpSilu, verify64_tune(Epi::GateUpSilu, false)));
        specs.push((Epi::Add, verify64_tune(Epi::Add, false)));
        specs.push((Epi::Set, verify64_tune(Epi::Set, false)));
        specs.push((Epi::QkvzaGdn, verify64_tune(Epi::QkvzaGdn, true)));
    }
    let emitted = specs
        .into_iter()
        .map(|(epi, tune)| {
            emit(Spec {
                fold,
                tile,
                cacc,
                epi,
                act,
                arch,
                mma: Mma::Dense,
                tune,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (text, proof) = module(
        &emitted,
        &Spec {
            fold,
            tile,
            cacc,
            epi: Epi::Set,
            act,
            arch,
            mma: Mma::Dense,
            tune: Tune::default(),
        }
        .module(),
    )?;
    Ok((emitted, text, proof))
}

/// Instruction census of the hot K-loop (between the loop head and the
/// loop-end label), counted the way the peacemaker shape gate counts.
pub fn hot_loop_census(s_text: &str) -> std::collections::BTreeMap<String, u32> {
    let mut counts = std::collections::BTreeMap::new();
    let mut inside = false;
    for line in s_text.lines() {
        let t = line.trim();
        if t == format!("{K_LOOP}:") {
            inside = true;
            continue;
        }
        if t == format!("{K_LOOP_END}:") {
            break;
        }
        if !inside || t.ends_with(':') || t.is_empty() || t.starts_with('.') {
            continue;
        }
        let name = t.split_whitespace().next().unwrap_or("").to_owned();
        *counts.entry(name.clone()).or_insert(0) += 1;
        if name.starts_with("v_dual_") {
            *counts.entry("vopd_packets".into()).or_insert(0) += 1
        }
        if name.starts_with("v_") && !name.starts_with("v_wmma_") && !name.starts_with("v_swmmac_")
        {
            *counts.entry("valu_slots".into()).or_insert(0) += 1
        }
    }
    counts
}
