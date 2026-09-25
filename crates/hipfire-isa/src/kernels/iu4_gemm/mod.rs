//! Builder-emitted MQ4V2 x block_i4_128 GEMM for gfx1201 (plan §3-§4).
//!
//! Semantics are hipcc K1's `_v3` symbols bit for bit: per 128-K block each
//! (row, token) int32 dot starts from the fold magic, is folded once as
//! `acc = fma(RN(sc*d), float(C) - magic, acc)` in ascending block order,
//! and the epilogue writes `Y = acc`, `Y = RN(Y + acc)` or the imported hipcc
//! SiLU region. Everything else (CTA raster, staging map, store shape, waits)
//! is a free schedule choice and is generated here.
//!
//! Launch contract: grid `[ceil(rows / tile_rows), ceil(N / 128)]` with
//! `rows = M` (SET/ADD) or `2M` (gate/up), block `[tile_threads, 1, 1]`,
//! dynamic LDS `Layout::launch` bytes. Preconditions shared with the hipcc
//! wide epilogue: `K % 256 == 0`, `M % 4 == 0`, and `Y`/`H` 4-byte aligned.
pub mod spec;
pub mod prologue;
pub mod kloop;
pub mod fold;
pub mod publish;
pub mod epilogue;
pub mod region;

pub use spec::{Cacc, Epi, Fold, Spec, Tile};
use crate::{Builder, Emitted, KernelSpec, RegPlan, insn::{Instruction, MemoryClass, Sop}, reg::{Kind, Live, RegRef}};
use serde::Serialize;
use sha2::{Digest, Sha256};

pub(crate) const ENTRY: &str = "entry";
pub(crate) const K_BEGIN: &str = ".Liu4_k_begin";
pub(crate) const K_LOOP: &str = ".Liu4_k_loop";
pub(crate) const K_LOOP_END: &str = ".Liu4_k_loop_end";
pub(crate) const EPI: &str = ".Liu4_epilogue";
pub(crate) const END: &str = ".Liu4_end";

pub(crate) fn v(n: u8) -> RegRef { RegRef { kind: Kind::V, base: n, len: 1 } }
pub(crate) fn vr(n: u8, len: u8) -> RegRef { RegRef { kind: Kind::V, base: n, len } }
pub(crate) fn s(n: u8) -> RegRef { RegRef { kind: Kind::S, base: n, len: 1 } }
pub(crate) fn sr(n: u8, len: u8) -> RegRef { RegRef { kind: Kind::S, base: n, len } }
/// Literal spelling as llvm-objdump prints it: inline integers in decimal,
/// everything else in hex, so parse-back compares canonical text.
pub(crate) fn lit(x: u32) -> String { if x <= 64 { x.to_string() } else if x as i32 >= -16 && (x as i32) < 0 { (x as i32).to_string() } else { format!("{x:#x}") } }
pub(crate) fn op(b: &mut Builder, text: impl Into<String>, defs: &[RegRef], uses: &[RegRef]) -> Result<(), String> {
    b.push(Instruction::new(text, defs.to_vec(), uses.to_vec()))
}
pub(crate) fn mem(b: &mut Builder, text: impl Into<String>, defs: &[RegRef], uses: &[RegRef], class: MemoryClass) -> Result<(), String> {
    b.push(Instruction::new(text, defs.to_vec(), uses.to_vec()).memory(class))
}
/// DS offset fields spelled as objdump prints them (zero fields omitted).
pub(crate) fn ds_offsets(o0: u32, o1: u32) -> String {
    let mut t = String::new();
    if o0 != 0 { t.push_str(&format!(" offset0:{o0}")) }
    if o1 != 0 { t.push_str(&format!(" offset1:{o1}")) }
    t
}
pub(crate) fn ds_offset(o: u32) -> String { if o == 0 { String::new() } else { format!(" offset:{o}") } }
/// Accumulator range names indexed by `k = nb * 2 + rg`.
pub(crate) const CACC_NAMES: [&str; 8] = ["cacc[0][0]", "cacc[0][1]", "cacc[1][0]", "cacc[1][1]", "cacc[2][0]", "cacc[2][1]", "cacc[3][0]", "cacc[3][1]"];
pub(crate) const ACC_NAMES: [&str; 8] = ["acc[0][0]", "acc[0][1]", "acc[1][0]", "acc[1][1]", "acc[2][0]", "acc[2][1]", "acc[3][0]", "acc[3][1]"];

/// Kernel-argument SGPRs after the two `s_load_b256` at 0x0 and 0x20.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Args { pub a: u8, pub u: Option<u8>, pub xq: u8, pub y: u8, pub m: u8, pub k: u8, pub n: u8, pub bcx: u8, pub bcy: u8 }

/// Physical register assignment. VGPR map (T128, One):
/// `cacc v[0:63]` (prologue/epilogue temporaries alias it), `acc v[64:127]`,
/// `magic8 v[128:135]`, fragment sets F0/F1 `v[136:159]` (fold scale rows and
/// products alias them once their WMMAs have issued), `d v[160:163]`, the
/// prefetch payload, then hoisted addresses: 187 VGPRs (T256: 186).
#[derive(Clone, Debug)]
pub(crate) struct Gen {
    pub spec: Spec,
    pub tile: Tile,
    pub layout: spec::Layout,
    pub args: Args,
    pub cacc: u8, pub acc: u8, pub magic: u8, pub f: [u8; 2], pub d: u8,
    pub a_pf: Vec<u8>, pub w_pf: Vec<u8>, pub ds_nx: u8, pub sz_nx: u8,
    pub fr_a: [u8; 2], pub fr_w: [u8; 2], pub st_lds: u8, pub st_a: u8, pub st_w: Vec<u8>,
    pub meta_ds: u8, pub meta_sz: u8, pub ds_voff: u8, pub sz_voff: u8, pub sc_addr: u8, pub d_addr: u8,
    // SGPRs
    pub karg: u8, pub srd_w: u8, pub srd_z: u8, pub srd_a: [u8; 2], pub srd_y: u8, pub step2: u8,
    pub goff: u8, pub trips: u8, pub rs: u8, pub bs: u8, pub wave: u8, pub gpr136: u8, pub mm1: u8, pub hs: u8,
    pub tmp: u8, pub epi_s: u8,
    // LDS slot ids
    pub slot_a: [usize; 2], pub slot_w: [usize; 2], pub slot_ds: [usize; 2], pub slot_sz: [usize; 2],
}

pub(crate) const PRO_TEMPS: u8 = 24;
pub(crate) const S_TEMPS: u8 = 8;
pub(crate) const EPI_S: u8 = 24;

impl Gen {
    fn new(spec: Spec) -> Self {
        let tile = spec.tile;
        let args = if spec.epi == Epi::GateUpSilu {
            Args { a: 8, u: Some(10), xq: 12, y: 14, m: 16, k: 17, n: 18, bcx: 20, bcy: 21 }
        } else {
            Args { a: 8, u: None, xq: 10, y: 12, m: 14, k: 15, n: 16, bcx: 18, bcy: 19 }
        };
        let mut next = 164u8;
        let mut take = |n: u8| { let b = next; next += n; b };
        let a_pf = (0..tile.a_rounds()).map(|_| take(2)).collect::<Vec<_>>();
        let w_pf = (0..tile.w_rounds()).map(|_| take(2)).collect::<Vec<_>>();
        let ds_nx = take(1); let sz_nx = take(1);
        let fr_a = [take(1), take(1)]; let fr_w = [take(1), take(1)];
        let st_lds = take(1); let st_a = take(1);
        let st_w = (0..tile.w_rounds()).map(|_| take(1)).collect::<Vec<_>>();
        let meta_ds = take(1);
        let meta_sz = if tile == Tile::T128x128x8 { meta_ds } else { take(1) };
        let ds_voff = take(1); let sz_voff = take(1); let sc_addr = take(1); let d_addr = take(1);
        let silu = spec.epi == Epi::GateUpSilu;
        Self {
            spec, tile, layout: tile.layout(), args,
            cacc: 0, acc: 64, magic: 128, f: [136, 148], d: 160,
            a_pf, w_pf, ds_nx, sz_nx, fr_a, fr_w, st_lds, st_a, st_w, meta_ds, meta_sz, ds_voff, sz_voff, sc_addr, d_addr,
            karg: 0, srd_w: 24, srd_z: if silu { 28 } else { 24 }, srd_a: [32, 36], srd_y: 40, step2: 44,
            goff: 46, trips: 47, rs: 48, bs: 49, wave: 50, gpr136: 51, mm1: 52, hs: 53,
            tmp: 56, epi_s: 64,
            slot_a: [0, 2], slot_w: [1, 3], slot_ds: [4, 5], slot_sz: [6, 7],
        }
    }

    /// Whole-kernel and phase-scoped ranges. Per-block fragment/fold aliases
    /// are declared by the K-loop generator with label-delimited lifetimes.
    fn plan(&self) -> Result<RegPlan, String> {
        let mut p = RegPlan::new(spec::VGPR_CEILING, 104)?;
        let pro = || Live::Between(ENTRY.into(), K_BEGIN.into());
        let kl = || Live::Between(K_BEGIN.into(), EPI.into());
        let kernel = || Live::Between(ENTRY.into(), EPI.into());
        let epi = || Live::Between(EPI.into(), END.into());
        for i in 0..8u8 { p.v::<8>(CACC_NAMES[usize::from(i)], self.cacc + 8 * i, kl())?; }
        for i in 0..PRO_TEMPS { p.v::<1>(if i == 0 { "tid" } else { "prologue_tmp" }, i, pro())?; }
        for i in 0..8u8 { p.v::<8>(ACC_NAMES[usize::from(i)], self.acc + 8 * i, Live::Whole)?; }
        p.v::<8>("magic8", self.magic, kernel())?;
        p.v::<4>("d_x", self.d, kl())?;
        for &r in &self.a_pf { p.v::<2>("a_pf", r, kernel())?; }
        for &r in &self.w_pf { p.v::<2>("w_pf", r, kernel())?; }
        p.v::<1>("ds_nx", self.ds_nx, kernel())?;
        p.v::<1>("sz_nx", self.sz_nx, kernel())?;
        for (name, r) in [("fr_a0", self.fr_a[0]), ("fr_a1", self.fr_a[1]), ("fr_w0", self.fr_w[0]), ("fr_w1", self.fr_w[1]),
            ("st_ldsoff", self.st_lds), ("st_avoff", self.st_a), ("ds_voff", self.ds_voff), ("sz_voff", self.sz_voff), ("meta_lds_ds", self.meta_ds)] {
            p.v::<1>(name, r, kernel())?;
        }
        if self.meta_sz != self.meta_ds { p.v::<1>("meta_lds_sz", self.meta_sz, kernel())?; }
        for &r in &self.st_w { p.v::<1>("st_wvoff", r, kernel())?; }
        p.v::<1>("sc_addr", self.sc_addr, Live::Whole)?;
        p.v::<1>("d_addr", self.d_addr, Live::Whole)?;
        // Epilogue: temporaries over the dead int32 accumulators and F0/F1.
        for i in 0..16u8 { p.v::<4>("epi_quad", self.cacc + 4 * i, epi())?; }
        for i in 0..24u8 { p.v::<1>("epi_tmp", self.f[0] + i, epi())?; }
        // SGPRs.
        p.s::<2>("kernarg_ptr", self.karg, Live::Whole)?;
        p.s::<8>("kernargs_0x00", 8, Live::Whole)?;
        p.s::<8>("kernargs_0x20", 16, Live::Whole)?;
        p.s::<4>("srd_w", self.srd_w, Live::Whole)?;
        if self.srd_z != self.srd_w { p.s::<4>("srd_z", self.srd_z, Live::Whole)?; }
        p.s::<4>("srd_a0", self.srd_a[0], Live::Whole)?;
        p.s::<4>("srd_a1", self.srd_a[1], Live::Whole)?;
        p.s::<4>("srd_y", self.srd_y, Live::Whole)?;
        p.s::<2>("a_step2", self.step2, Live::Whole)?;
        for (name, r) in [("goff", self.goff), ("trips", self.trips), ("rs", self.rs), ("bs", self.bs), ("wave", self.wave),
            ("gpr136", self.gpr136), ("m_minus_1", self.mm1), ("hs", self.hs), ("rows", 54), ("scratch_s55", 55)] {
            p.s::<1>(name, r, Live::Whole)?;
        }
        for i in 0..S_TEMPS { p.s::<1>("s_tmp", self.tmp + i, Live::Whole)?; }
        for i in 0..EPI_S { p.s::<1>("epi_s", self.epi_s + i, Live::Whole)?; }
        Ok(p)
    }

    fn declare_lds(&self, b: &mut Builder) -> Result<(), String> {
        let l = self.layout;
        let slots = [("A0", l.a[0], l.a_bytes), ("W0", l.w[0], l.w_bytes), ("A1", l.a[1], l.a_bytes), ("W1", l.w[1], l.w_bytes),
            ("DS0", l.ds[0], l.ds_bytes), ("DS1", l.ds[1], l.ds_bytes), ("SZ0", l.sz[0], l.sz_bytes), ("SZ1", l.sz[1], l.sz_bytes)];
        for (i, (name, base, len)) in slots.into_iter().enumerate() {
            if b.lds.add(name, base, len)? != i { return Err("LDS slot order".into()) }
        }
        if l.end > l.launch { return Err("LDS layout exceeds the launch size".into()) }
        Ok(())
    }
}

/// Emit one kernel symbol. The result is a complete single-kernel `.s`.
pub fn emit(spec: Spec) -> Result<Emitted, String> {
    spec.validate()?;
    let g = Gen::new(spec);
    let kspec = KernelSpec {
        kernel_id: "iu4_gemm".into(), variant: spec.variant_name(), arch: spec.arch, symbol: spec.symbol(),
        kernargs: spec.kernargs(), user_sgpr_count: 2, system_sgpr_workgroup_id_y: true,
        workgroup_size: spec.tile.threads() as u16, group_segment_fixed_size: 0, wave32: true, cu_mode: false,
    };
    let mut b = Builder::new(kspec, g.plan()?);
    g.declare_lds(&mut b)?;
    prologue::emit(&mut b, &g)?;
    kloop::emit(&mut b, &g)?;
    epilogue::emit(&mut b, &g)?;
    b.label(END)?;
    b.push(Sop::End.encode(spec.arch)?)?;
    b.finish()
}

/// Proof of a multi-kernel module: the per-symbol builder proofs plus the
/// digest of the combined source, bound the way `peacemaker custom build`
/// binds a single-kernel proof.
#[derive(Serialize)]
pub struct ModuleProof { pub module: String, pub arch: crate::Arch, pub builder_crate_version: String, pub builder_git_sha: String,
    pub s_text_sha256: String, pub kernels: Vec<crate::BuilderProof> }

/// Combine several single-kernel sources into one code object source: one
/// target header, every kernel body and descriptor, one metadata document.
pub fn module(emitted: &[Emitted], name: &str) -> Result<(String, ModuleProof), String> {
    let first = emitted.first().ok_or("empty module")?;
    let mut header = String::new();
    let mut bodies = String::new();
    let mut kernels = String::new();
    let mut tail = String::new();
    for (i, e) in emitted.iter().enumerate() {
        let (code, meta) = e.s_text.split_once(".amdgpu_metadata\n").ok_or("missing metadata")?;
        let (pre, body) = code.split_once(".text\n").ok_or("missing .text")?;
        if i == 0 { header = pre.to_owned(); }
        else if pre != header { return Err("kernels disagree on target header".into()) }
        bodies.push_str(".text\n");
        // Local labels are per-kernel names; qualify them by epilogue so the
        // kernels can share one assembler unit.
        let tag = e.proof.variant.rsplit('-').next().unwrap_or("k");
        bodies.push_str(&body.replace(".Liu4_", &format!(".Liu4_{tag}_")));
        let (_, list) = meta.split_once("amdhsa.kernels:\n").ok_or("missing kernel list")?;
        let (items, rest) = list.split_once("amdhsa.target:").ok_or("missing target")?;
        kernels.push_str(items);
        if i == 0 { tail = format!("amdhsa.target:{rest}"); }
    }
    let text = format!("{header}{bodies}.amdgpu_metadata\n---\namdhsa.kernels:\n{kernels}{tail}");
    let proof = ModuleProof {
        module: name.into(), arch: first.proof.arch, builder_crate_version: first.proof.builder_crate_version.clone(),
        builder_git_sha: first.proof.builder_git_sha.clone(), s_text_sha256: format!("{:x}", Sha256::digest(text.as_bytes())),
        kernels: emitted.iter().map(|e| e.proof.clone()).collect(),
    };
    Ok((text, proof))
}

/// All three epilogue symbols of one (fold, tile, cacc) point as a module.
pub fn emit_module(fold: Fold, tile: Tile, cacc: Cacc, arch: crate::Arch) -> Result<(Vec<Emitted>, String, ModuleProof), String> {
    let specs = [Epi::Set, Epi::Add, Epi::GateUpSilu].map(|epi| Spec { fold, tile, cacc, epi, arch });
    let emitted = specs.iter().map(|s| emit(*s)).collect::<Result<Vec<_>, _>>()?;
    let (text, proof) = module(&emitted, &specs[0].module())?;
    Ok((emitted, text, proof))
}

/// Instruction census of the hot K-loop (between the loop head and the
/// loop-end label), counted the way the peacemaker shape gate counts.
pub fn hot_loop_census(s_text: &str) -> std::collections::BTreeMap<String, u32> {
    let mut counts = std::collections::BTreeMap::new();
    let mut inside = false;
    for line in s_text.lines() {
        let t = line.trim();
        if t == format!("{K_LOOP}:") { inside = true; continue }
        if t == format!("{K_LOOP_END}:") { break }
        if !inside || t.ends_with(':') || t.is_empty() || t.starts_with('.') { continue }
        let name = t.split_whitespace().next().unwrap_or("").to_owned();
        *counts.entry(name.clone()).or_insert(0) += 1;
        if name.starts_with("v_dual_") { *counts.entry("vopd_packets".into()).or_insert(0) += 1 }
        if name.starts_with("v_") && !name.starts_with("v_wmma_") { *counts.entry("valu_slots".into()).or_insert(0) += 1 }
    }
    counts
}
