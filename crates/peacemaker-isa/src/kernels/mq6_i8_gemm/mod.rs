// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Builder-emitted MQ6G256V2 x `block_i8_128` exact-INT8 GEMM for gfx1201: the
//! PeaceMaker twin of `kernels/src/gemm_mq6g256v2_mmq_i8.gfx12.hip` (bit-identical
//! output). Entries `gemm_mq6g256v2_mmq_i8_pm_set_gfx1201` (F32) and
//! `gemm_mq6g256v2_mmq_i8_pm_set_bf16_gfx1201` (BF16 bits, the incumbent's rounding).
//!
//! # Launch contract
//! ABI `(A, Xq, Y, M, K, N)` (+ K1's hidden block): `A` MQ6G256V2 rows
//! (`M x K/256 x 200 B`), `Xq` `block_i8_128` blocks laid out `[K/128][N]`, `Y` `[N][M]`.
//! Grid `[ceil(M/128), ceil(N/128), 1]`, block `[256, 1, 1]`, dynamic LDS
//! [`spec::LAUNCH_LDS_BYTES`] = 36864. Preconditions: `K % 256 == 0`, `M % 4 == 0`,
//! `Y` 4-byte aligned, `M * K/256 * 200 < 2^32` (the weight descriptor's byte range; rows past
//! `M` and tokens past `N` read zero through descriptor bounds). Ragged `M` and `N` tiles are
//! masked at the store.
//!
//! # Design (what differs from `iu4_gemm`, whose cacc/acc split, ring and fold shape it forks)
//! - Weights are decoded once per CTA at staging: each lane fetches its K16 step's three raw
//!   dwords (both 64-row rounds), spreads the 6-bit fields into bytes (`sp4`, 6 VALU per field)
//!   and gathers `{c0,c2,c4,c6}` / `{c1,c3,c5,c7}` with `v_perm_b32`; the decoded unsigned codes
//!   go to LDS in fragment order, so the K loop is `ds_load_b128` + WMMA only. 36864 B of LDS
//!   (two K64 slots of 16 KiB = 8 KiB activations + 8 KiB codes, and two 2 KiB metadata
//!   planes), 192 VGPRs (the `iu4_gemm` occupancy ceiling), 8 waves per workgroup.
//! - Each K128 block is two K64 slabs of four K16 `v_wmma_i32_16x16x16_iu8` steps (64 WMMAs
//!   per block per wave); the fragment loads of step `s + 2` issue right after step `s`'s
//!   WMMAs (two ping-pong fragment sets), the decode / publication of the next slab is spread
//!   between the WMMAs of the current one, and the two LDS barriers per block are split or
//!   overlapped with VMEM fetches (`s_barrier_signal` / `s_barrier_wait`).
//! - The fold of a block is 160 VOPD packets (mul/add, fmac/fmac, mul/mul, fmac/fmac per
//!   accumulator fragment) interleaved with the next block's slab-1 fetch; barrier B2 is
//!   signalled once the fold's LDS reads are done and waited at the block end.
//!
//! # Arithmetic (why this is bit-exact with the hipcc kernel)
//! Per (row, token, K128 block) the int32 WMMA chain accumulates `Cq = sum q * qx` (codes
//! unsigned `0..63`, activations signed) from the seed `0x4B400000` (the first WMMA of a
//! block reads it from the eight decode-scratch registers, refilled at the end of the previous
//! block), so `as_float(acc) = 12582912.0f + Cq` exactly (`|Cq| < 2^22`). The fold adds
//! `msn = -(12582912.0f + 32 * float(s))` (two exact integer-valued f32 operations), giving
//! `fc = Cq - 32 s` exactly, and then evaluates the hipcc chain verbatim:
//! `acc = fma(RN(sc * d), fc, acc); acc = fma(RN(z' * d), float(s), acc)` with
//! `z' = fma(32, sc, z)` computed once per row header in f32. BF16: `v = 0 + y`,
//! `isfinite(v) ? (u + 0x7FFF + ((u >> 16) & 1)) >> 16 : u >> 16`.
//!
//! # Not covered by the CPU tests
//! The interpreter models the data path, LDS, barriers and the epilogue stores; it has no
//! timing, no bank conflicts, and lacks the SALU float ops of the banded raster, so the
//! banded product symbols differ from the tested `_linear` twins only in that prologue
//! fragment (copied verbatim from `iu4_gemm`) and the omitted `s_setprio`.
pub mod blocks;
pub mod epilogue;
pub mod fold;
pub mod kloop;
pub mod nr_epilogue;
pub mod prologue;
pub mod publish;
pub mod spec;

use crate::kernels::common::op;
use crate::{reg::Live, Builder, Emitted, KernelSpec, RegPlan};
use peacemaker_author::{
    Free, Gfx1201, Gfx12Waits, LdsWrite, Pending, Published, Ring, Wave, Workgroup, Writing,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
pub use spec::{Epi, Raster, RingDepth, Spec, Tile, Tune};

pub const ENTRY: &str = "entry";
pub const K_BEGIN: &str = ".Lmq6i8_k_begin";
pub const K_LOOP: &str = ".Lmq6i8_k_loop";
pub const K_LOOP_END: &str = ".Lmq6i8_k_loop_end";
pub const EPI: &str = ".Lmq6i8_epilogue";
pub const END: &str = ".Lmq6i8_end";

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
    pub xq: u8,
    pub y: u8,
    pub m: u8,
    pub k: u8,
    pub n: u8,
    pub bcx: u8,
    pub bcy: u8,
}

/// Physical register assignment.
///
/// VGPRs (192 total): `cacc v[0:63]` (prologue/epilogue temporaries alias it), `acc v[64:127]`,
/// fragment sets `F0 v[128:139]` / `F1 v[140:151]` (per K16 step: weight pair, token pairs 0-1,
/// token pairs 2-3; at fold time the same 24 registers hold `sc v[128:135]`, `z' v[136:143]`
/// and the products `t v[144:151]`), per-token `(d, float(s))` pairs `v[152:159]`, `msn`, the raw
/// prefetch payload, then hoisted addresses and decode scratch.
#[derive(Clone, Debug)]
pub struct Gen {
    pub(crate) spec: Spec,
    pub(crate) args: Args,
    pub(crate) cacc: u8,
    pub(crate) acc: u8,
    pub(crate) f: [u8; 2],
    /// `(d, float(s))` of token column block `nb` at `tok + 2 * nb` (`d`) and `+ 1` (`s`).
    pub(crate) tok: u8,
    pub(crate) msn: u8,
    /// Raw prefetch payload per staging round: one b64 pair per activation round, one dword
    /// triple per weight round (`Tile::a_rounds` / `w_rounds` of them).
    pub(crate) a_pf: Vec<u8>,
    pub(crate) w_pf: Vec<u8>,
    /// The depth-3 ring's second raw prefetch set (same shapes as `a_pf` / `w_pf`, empty at
    /// depth 2); [`Gen::pf_set`] swaps it in.
    pub(crate) a_pf_b: Vec<u8>,
    pub(crate) w_pf_b: Vec<u8>,
    pub(crate) ds_nx: u8,
    pub(crate) sz_nx: u8,
    pub(crate) fr_a: u8,
    pub(crate) fr_w: u8,
    pub(crate) st_alds: u8,
    pub(crate) st_wlds: u8,
    pub(crate) st_a: u8,
    pub(crate) st_w: u8,
    /// LDS store address of the token-plane word, and of the row-plane word (the same register
    /// on the shipped tile, whose two planes have the same extent).
    pub(crate) meta_lds: u8,
    pub(crate) meta_lds_sz: u8,
    pub(crate) ds_voff: u8,
    pub(crate) sz_voff: u8,
    pub(crate) sc_addr: u8,
    pub(crate) d_addr: u8,
    /// Decode / epilogue scratch (8 VGPRs).
    pub(crate) scratch: u8,
    // SGPRs
    pub(crate) karg: u8,
    pub(crate) srd_w: u8,
    pub(crate) srd_a: [u8; 2],
    pub(crate) srd_y: u8,
    pub(crate) step2: u8,
    pub(crate) goff: u8,
    pub(crate) goff1: u8,
    pub(crate) trips: u8,
    pub(crate) rs: u8,
    pub(crate) bs: u8,
    pub(crate) wave: u8,
    pub(crate) gpr200: u8,
    pub(crate) omask: u8,
    pub(crate) tmp: u8,
    pub(crate) epi_s: u8,
}

pub(crate) const PRO_TEMPS: u8 = 24;
pub(crate) const S_TEMPS: u8 = 8;
pub(crate) const EPI_S: u8 = 24;

impl Gen {
    /// The generator for `spec`: the register assignment and geometry every block of [`blocks`]
    /// reads. `Err` when `spec` has no generator ([`Spec::validate`]).
    pub fn new(spec: Spec) -> Result<Self, String> {
        spec.validate()?;
        Ok(Self::build(spec))
    }
    /// The spec this generator was built for.
    pub fn spec(&self) -> Spec {
        self.spec
    }
    /// The A/W ring depth this generator was built for (`2` or `3`).
    pub fn depth(&self) -> usize {
        match self.spec.tune.ring {
            RingDepth::Two => 2,
            RingDepth::Three => 3,
        }
    }
    /// This generator with raw prefetch set `set` in the `a_pf` / `w_pf` slots every fetch, decode
    /// and store block reads (`0`: the base set; `1`: the depth-3 ring's second set). Everything
    /// else of the register map is shared.
    pub fn pf_set(&self, set: usize) -> Result<Gen, String> {
        match set {
            0 => Ok(self.clone()),
            1 if self.depth() == 3 => {
                let mut g = self.clone();
                std::mem::swap(&mut g.a_pf, &mut g.a_pf_b);
                std::mem::swap(&mut g.w_pf, &mut g.w_pf_b);
                Ok(g)
            }
            1 => Err("prefetch set 1 exists at ring depth 3 only".into()),
            _ => Err(format!("prefetch set {set} out of range (0, 1)")),
        }
    }
    /// The kernel header (symbol, kernargs, workgroup size) of this spec.
    pub fn kernel_spec(&self) -> KernelSpec {
        let spec = self.spec;
        KernelSpec {
            kernel_id: "mq6_i8_gemm".into(),
            variant: spec.variant_name(),
            arch: spec.arch,
            symbol: spec.symbol(),
            kernargs: spec.kernargs(),
            user_sgpr_count: 2,
            system_sgpr_workgroup_id_y: true,
            workgroup_size: self.spec.tile.threads() as u16,
            group_segment_fixed_size: spec.static_lds(),
            wave32: true,
            cu_mode: false,
        }
    }
    /// A physical register base of this assignment by name (read-only): VGPRs `cacc acc f0 f1 tok
    /// msn a_pf0 a_pf1 w_pf0 w_pf1 ds_nx sz_nx fr_a fr_w st_alds st_wlds st_a st_w meta_lds meta_lds_sz ds_voff
    /// sz_voff sc_addr d_addr scratch` and the fold octets `fold_sc fold_zp fold_t`; SGPRs `karg
    /// srd_w srd_a0 srd_a1 srd_y step2 goff goff1 trips rs bs wave gpr200 omask tmp epi_s`; kernel
    /// arguments `arg_a arg_xq arg_y arg_m arg_k arg_n arg_bcx arg_bcy`. `None` for an unknown name.
    pub fn reg(&self, name: &str) -> Option<u8> {
        let a = &self.args;
        Some(match name {
            "cacc" => self.cacc,
            "acc" => self.acc,
            "f0" => self.f[0],
            "f1" => self.f[1],
            "tok" => self.tok,
            "msn" => self.msn,
            "a_pf0" => self.a_pf[0],
            "a_pf1" => return self.a_pf.get(1).copied(),
            "w_pf0" => self.w_pf[0],
            "w_pf1" => return self.w_pf.get(1).copied(),
            "a_pf_b0" => return self.a_pf_b.first().copied(),
            "a_pf_b1" => return self.a_pf_b.get(1).copied(),
            "w_pf_b0" => return self.w_pf_b.first().copied(),
            "w_pf_b1" => return self.w_pf_b.get(1).copied(),
            "ds_nx" => self.ds_nx,
            "sz_nx" => self.sz_nx,
            "fr_a" => self.fr_a,
            "fr_w" => self.fr_w,
            "st_alds" => self.st_alds,
            "st_wlds" => self.st_wlds,
            "st_a" => self.st_a,
            "st_w" => self.st_w,
            "meta_lds" => self.meta_lds,
            "meta_lds_sz" => self.meta_lds_sz,
            "ds_voff" => self.ds_voff,
            "sz_voff" => self.sz_voff,
            "sc_addr" => self.sc_addr,
            "d_addr" => self.d_addr,
            "scratch" => self.scratch,
            "fold_sc" => fold::SC,
            "fold_zp" => fold::ZP,
            "fold_t" => fold::T,
            "karg" => self.karg,
            "srd_w" => self.srd_w,
            "srd_a0" => self.srd_a[0],
            "srd_a1" => self.srd_a[1],
            "srd_y" => self.srd_y,
            "step2" => self.step2,
            "goff" => self.goff,
            "goff1" => self.goff1,
            "trips" => self.trips,
            "rs" => self.rs,
            "bs" => self.bs,
            "wave" => self.wave,
            "gpr200" => self.gpr200,
            "omask" => self.omask,
            "tmp" => self.tmp,
            "epi_s" => self.epi_s,
            "arg_a" => a.a,
            "arg_xq" => a.xq,
            "arg_y" => a.y,
            "arg_m" => a.m,
            "arg_k" => a.k,
            "arg_n" => a.n,
            "arg_bcx" => a.bcx,
            "arg_bcy" => a.bcy,
            _ => return None,
        })
    }
    /// An LDS slot of the tile's [`spec::Tile::layout`] by name (`A0 W0 A1 W1 DS0 DS1 SZ0 SZ1`) as `(base, bytes)`.
    pub fn slot(&self, name: &str) -> Option<(u32, u32)> {
        let l = self.spec.tile.layout();
        Some(match name {
            "A0" => (l.a[0], l.a_bytes),
            "A1" => (l.a[1], l.a_bytes),
            "W0" => (l.w[0], l.w_bytes),
            "W1" => (l.w[1], l.w_bytes),
            "DS0" => (l.ds[0], l.ds_bytes),
            "DS1" => (l.ds[1], l.ds_bytes),
            "SZ0" => (l.sz[0], l.sz_bytes),
            "SZ1" => (l.sz[1], l.sz_bytes),
            // The third A/W slab buffers exist at depth 3 only.
            "A2" if self.depth() == 3 => (l.a_slot(2).ok()?, l.a_bytes),
            "W2" if self.depth() == 3 => (l.w_slot(2).ok()?, l.w_bytes),
            _ => return None,
        })
    }

    fn build(spec: Spec) -> Self {
        Self {
            spec,
            args: Args {
                a: 8,
                xq: 10,
                y: 12,
                m: 14,
                k: 15,
                n: 16,
                bcx: 18,
                bcy: 19,
            },
            cacc: 0,
            acc: 64,
            f: [128, 140],
            tok: 152,
            msn: 161,
            a_pf: [164, 166][..spec.tile.a_rounds() as usize].to_vec(),
            w_pf: [168, 171][..spec.tile.w_rounds() as usize].to_vec(),
            // The depth-3 ring's second raw prefetch set, past the depth-2 map (192 VGPRs): the
            // activation pairs, then the weight triples.
            a_pf_b: if spec.tune.ring == RingDepth::Three {
                (0..spec.tile.a_rounds() as u8)
                    .map(|r| 192 + 2 * r)
                    .collect()
            } else {
                Vec::new()
            },
            w_pf_b: if spec.tune.ring == RingDepth::Three {
                let base = 192 + 2 * spec.tile.a_rounds() as u8;
                (0..spec.tile.w_rounds() as u8)
                    .map(|r| base + 3 * r)
                    .collect()
            } else {
                Vec::new()
            },
            ds_nx: 160,
            sz_nx: 162,
            fr_a: 163,
            fr_w: 174,
            st_alds: 175,
            st_wlds: 176,
            st_a: 177,
            st_w: 178,
            meta_lds: 179,
            // The shipped tile's planes have the same extent, so one address serves both stores;
            // the others take a prefetch register their single staging round leaves free.
            meta_lds_sz: match spec.tile {
                spec::Tile::T128x128 => 179,
                spec::Tile::T256x128 => 166,
                spec::Tile::T128x256 => 171,
            },
            ds_voff: 180,
            sz_voff: 181,
            sc_addr: 182,
            d_addr: 183,
            scratch: 184,
            karg: 0,
            srd_w: 24,
            srd_a: [32, 36],
            srd_y: 40,
            step2: 44,
            goff: 46,
            goff1: 52,
            trips: 47,
            rs: 48,
            bs: 49,
            wave: 50,
            gpr200: 51,
            omask: 53,
            tmp: 56,
            epi_s: 64,
        }
    }

    /// The register plan of this assignment (a script may add ranges before building).
    pub fn plan(&self) -> Result<RegPlan, String> {
        let mut p = RegPlan::new(self.spec.vgpr_budget(), 104)?;
        let pro = || Live::Between(ENTRY.into(), K_BEGIN.into());
        let kl = || Live::Between(K_BEGIN.into(), EPI.into());
        let kernel = || Live::Between(ENTRY.into(), EPI.into());
        let epi = || Live::Between(EPI.into(), END.into());
        for i in 0..8u8 {
            p.v::<8>(CACC_NAMES[usize::from(i)], self.cacc + 8 * i, kl())?;
        }
        for i in 0..PRO_TEMPS {
            p.v::<1>(if i == 0 { "tid" } else { "prologue_tmp" }, i, pro())?;
        }
        for i in 0..8u8 {
            p.v::<8>(ACC_NAMES[usize::from(i)], self.acc + 8 * i, Live::Whole)?;
        }
        for nb in 0..4u8 {
            p.v::<2>("tok_d_sp", self.tok + 2 * nb, kl())?;
        }
        p.v::<1>("msn", self.msn, kl())?;
        for &r in &self.a_pf {
            p.v::<2>("a_pf", r, kernel())?;
        }
        for &r in &self.w_pf {
            for i in 0..3u8 {
                p.v::<1>("w_pf", r + i, kernel())?;
            }
        }
        // The depth-3 ring's second raw prefetch set.
        for &r in &self.a_pf_b {
            p.v::<2>("a_pf_b", r, kernel())?;
        }
        for &r in &self.w_pf_b {
            for i in 0..3u8 {
                p.v::<1>("w_pf_b", r + i, kernel())?;
            }
        }
        for (name, r) in [
            ("ds_nx", self.ds_nx),
            ("sz_nx", self.sz_nx),
            ("fr_a", self.fr_a),
            ("fr_w", self.fr_w),
            ("st_alds", self.st_alds),
            ("st_wlds", self.st_wlds),
            ("st_a", self.st_a),
            ("st_w", self.st_w),
            ("meta_lds", self.meta_lds),
            ("ds_voff", self.ds_voff),
            ("sz_voff", self.sz_voff),
        ] {
            p.v::<1>(name, r, kernel())?;
        }
        if self.meta_lds_sz != self.meta_lds {
            p.v::<1>("meta_lds_sz", self.meta_lds_sz, kernel())?;
        }
        // The epilogue derives the lane's row and token from these.
        p.v::<1>("sc_addr", self.sc_addr, Live::Whole)?;
        p.v::<1>("d_addr", self.d_addr, Live::Whole)?;
        // Decode / meta scratch: y, z singles, the (p0, p1) pair, the o0 pair, x, w singles.
        p.v::<8>("scratch", self.scratch, kernel())?;
        // Epilogue: temporaries over the dead int32 accumulators and F0/F1.
        if self.spec.epi == Epi::SetNr {
            nr_epilogue::plan(&mut p, self)?;
        } else {
            for i in 0..16u8 {
                p.v::<4>("epi_quad", self.cacc + 4 * i, epi())?;
            }
            for i in 0..24u8 {
                p.v::<1>("epi_tmp", self.f[0] + i, epi())?;
            }
        }
        // SGPRs.
        p.s::<2>("kernarg_ptr", self.karg, Live::Whole)?;
        p.s::<8>("kernargs_0x00", 8, Live::Whole)?;
        p.s::<8>("kernargs_0x20", 16, Live::Whole)?;
        p.s::<4>("srd_w", self.srd_w, Live::Whole)?;
        p.s::<4>("srd_a0", self.srd_a[0], Live::Whole)?;
        p.s::<4>("srd_a1", self.srd_a[1], Live::Whole)?;
        p.s::<4>("srd_y", self.srd_y, Live::Whole)?;
        p.s::<2>("a_step2", self.step2, Live::Whole)?;
        for (name, r) in [
            ("goff", self.goff),
            ("trips", self.trips),
            ("rs", self.rs),
            ("bs", self.bs),
            ("wave", self.wave),
            ("gpr200", self.gpr200),
            ("goff1", self.goff1),
            ("odd_lane_mask", self.omask),
            ("rows", 54),
            ("scratch_s55", 55),
        ] {
            p.s::<1>(name, r, Live::Whole)?;
        }
        for i in 0..S_TEMPS {
            p.s::<1>("s_tmp", self.tmp + i, Live::Whole)?;
        }
        for i in 0..EPI_S {
            p.s::<1>("epi_s", self.epi_s + i, Live::Whole)?;
        }
        Ok(p)
    }

    fn declare_lds(
        &self,
        wg: &mut Wg,
    ) -> Result<
        (
            Ring<PayA, Free, Free>,
            Ring<PayW, Free, Free>,
            Ring<MetaD, Free, Free>,
            Ring<MetaS, Free, Free>,
        ),
        String,
    > {
        let l = self.spec.tile.layout();
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
        Ok((
            Ring::new(a0, a1),
            Ring::new(w0, w1),
            Ring::new(d0, d1),
            Ring::new(s0, s1),
        ))
    }

    /// The depth-3 rings: A and W over three slab buffers each (`A0 W0 A1 W1 A2 W2`), DS/SZ over
    /// their two planes; every buffer `Free`.
    #[allow(clippy::type_complexity)]
    fn declare_lds3(
        &self,
        wg: &mut Wg,
    ) -> Result<
        (
            Ring<PayA, Free, Free, 3>,
            Ring<PayW, Free, Free, 3>,
            Ring<MetaD, Free, Free>,
            Ring<MetaS, Free, Free>,
        ),
        String,
    > {
        let l = self.spec.tile.layout();
        let slab = |wg: &mut Wg, s: usize, an: &str, wn: &str| {
            Ok::<_, String>((
                wg.lds(an, l.a_slot(s)?, l.a_bytes)?,
                wg.lds(wn, l.w_slot(s)?, l.w_bytes)?,
            ))
        };
        let (a0, w0) = slab(wg, 0, "A0", "W0")?;
        let (a1, w1) = slab(wg, 1, "A1", "W1")?;
        let (a2, w2) = slab(wg, 2, "A2", "W2")?;
        let (d0, d1) = (
            wg.lds("DS0", l.ds[0], l.ds_bytes)?,
            wg.lds("DS1", l.ds[1], l.ds_bytes)?,
        );
        let (s0, s1) = (
            wg.lds("SZ0", l.sz[0], l.sz_bytes)?,
            wg.lds("SZ1", l.sz[1], l.sz_bytes)?,
        );
        Ok((
            Ring::new_n([a0, a1, a2])?,
            Ring::new_n([w0, w1, w2])?,
            Ring::new(d0, d1),
            Ring::new(s0, s1),
        ))
    }
}

pub enum PayA {}
pub enum PayW {}
pub enum MetaD {}
pub enum MetaS {}
pub(crate) type Wg<'b> = Workgroup<'b, Gfx1201, Builder>;
pub(crate) type Wv<'b> = Wave<'b, Gfx1201, Builder>;
/// The four rings of the K-loop: the A/W payload rings `D` deep (`2`, or `3` for [`RingDepth::Three`]),
/// the DS/SZ metadata planes always two deep.
pub(crate) type Lds<const D: usize = 2> = (
    Ring<PayA, Published, Free, D>,
    Ring<PayW, Published, Free, D>,
    Ring<MetaD, Published, Free>,
    Ring<MetaS, Published, Free>,
);
pub(crate) type Staged<R, C, const D: usize = 2> =
    (Ring<R, C, Writing, D>, Pending<Gfx12Waits, LdsWrite<R>>);

/// Emit one kernel symbol. The result is a complete single-kernel `.s`.
pub fn emit(spec: Spec) -> Result<Emitted, String> {
    spec.validate()?;
    let g = Gen::new(spec)?;
    let mut b = Builder::new(g.kernel_spec(), g.plan()?);
    if spec.stacked() {
        b.enable_delay_alu();
    }
    let mut wg = Wg::new(&mut b)?;
    if spec.tune.ring == RingDepth::Three {
        // Validate refused `SetNr`: no epilogue reads the rings.
        let lds = g.declare_lds3(&mut wg)?;
        let end = wg.exit(END)?;
        if spec.raster == Raster::Banded {
            op(wg.isa(), "s_setprio 1", &[], &[])?;
        }
        kloop::ring3::emit(&mut wg, &g, &end, lds)?;
        epilogue::emit(&mut wg, &g, &end)?;
        wg.end(end)?;
        return b.finish();
    }
    let lds = g.declare_lds(&mut wg)?;
    let end = wg.exit(END)?;
    if spec.raster == Raster::Banded {
        op(wg.isa(), "s_setprio 1", &[], &[])?;
    }
    let lds = prologue::emit(&mut wg, &g, &end, lds)?;
    let lds = kloop::emit(&mut wg, &g, lds)?;
    if spec.epi == Epi::SetNr {
        nr_epilogue::emit(&mut wg, &g, lds)?;
    } else {
        epilogue::emit(&mut wg, &g, &end)?;
    }
    wg.end(end)?;
    b.finish()
}

#[derive(Serialize)]
pub struct ModuleProof {
    pub module: String,
    pub arch: crate::Arch,
    pub builder_crate_version: String,
    pub builder_git_sha: String,
    pub s_text_sha256: String,
    pub kernels: Vec<crate::BuilderProof>,
}

/// Combine single-kernel sources into one code object source (see `iu4_gemm::module`).
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
        // The shipped tile keeps its historical label tag (`set` / `set_bf16`), so its bytes do
        // not move; other tiles are tagged with the whole variant name.
        let tag = if e.proof.variant.starts_with("128x128x8-") {
            e.proof.variant.rsplit('-').next().unwrap_or("k").to_owned()
        } else {
            e.proof.variant.replace('-', "_")
        };
        bodies.push_str(&body.replace(".Lmq6i8_", &format!(".Lmq6i8_{tag}_")));
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

/// The product module: both epilogue symbols (banded raster) on the shipped 128x128 tile (the
/// unsuffixed symbols) and on the 128x256 tile (`_t128x256`, the host's default for wide N), then
/// the QSA norm+RoPE entry (`set_nr`, 256x128 tile).
pub fn emit_module(arch: crate::Arch) -> Result<(Vec<Emitted>, String, ModuleProof), String> {
    let emitted = [spec::Tile::T128x128, spec::Tile::T128x256]
        .into_iter()
        .flat_map(|tile| [Epi::Set, Epi::SetBf16].map(|epi| (tile, epi)))
        .chain([(spec::Tile::T256x128, Epi::SetNr)])
        .map(|(tile, epi)| {
            emit(Spec {
                epi,
                raster: Raster::Banded,
                tile,
                arch,
                tune: Tune::default(),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (text, proof) = module(&emitted, Spec::module())?;
    Ok((emitted, text, proof))
}

/// Instruction census of the hot K-loop (one trip = two K128 blocks, between the loop head and the
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
        if name.starts_with('v') && !name.starts_with("v_wmma_") {
            *counts.entry("valu_slots".into()).or_insert(0) += 1
        }
    }
    counts
}
