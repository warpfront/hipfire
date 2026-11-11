// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Region conversion: a lifted gfx1201 HSA kernel (a hipcc object, unmodified)
//! rewritten into a task-body region for a PeaceMaker persistent dispatcher,
//! with the semantics of `peacemaker_isa::Builder::region`.
//!
//! The rewrite kinds, each with the proof obligation it records in the
//! [`RegionReceipt`]:
//!
//! 1. **exit**: every `s_endpgm` becomes `s_branch .L{p}_quiesce`; the quiesce
//!    block drains LOADcnt/STOREcnt/DScnt/KMcnt, so the region exits with no
//!    memory operation outstanding. The `s_sendmsg(MSG_DEALLOC_VGPRS)` hipcc
//!    places before an `s_endpgm` is dropped (the wave lives on). Obligation:
//!    no `s_endpgm` and no VGPR deallocation remain.
//! 2. **inputs**: the hardware workgroup ids `ttmp9` (X) and `ttmp7` (Y | Z<<16)
//!    become declared SGPR inputs; the kernarg segment pointer stays the body's
//!    `s[0:1]`, which the dispatcher loads with the task's kernarg block (its
//!    hidden arguments included, listed in the receipt). Obligation: the
//!    descriptor enables only the kernarg-pointer user SGPRs, no private
//!    segment, no preloaded kernargs and one workitem-id axis, and the
//!    entry live-in set (liveness) is within `{s0, s1, v0, ttmp7, ttmp9, exec}`.
//! 3. **registers**: hipcc's registers are kept; the region reports the SGPR and
//!    VGPR extent it uses, which the dispatcher's register plan declares as
//!    ranges live only inside the region (disjoint from every dispatcher range
//!    there). LDS stays at the body's own `[0, lds_bytes)` (regions run one at
//!    a time per workgroup); the dispatcher's LDS starts above the largest
//!    hosted body.
//! 4. **geometry**: the dispatcher's workgroup is the body's `blockDim`. With
//!    more than one wave, every wave must execute the same barriers
//!    ([`super::uniformity`]): every branch that is not workgroup-uniform reaches
//!    its immediate post-dominator with the same barrier count on all paths of the
//!    (block, EXEC) product graph, and no barrier lies on a cycle under it; in
//!    particular no barrier may follow a per-wave early exit.
//!
//! Numerics: every other instruction is carried as its canonical text (only
//! `ttmp7`/`ttmp9` substituted, branches retargeted to labels), so its
//! arithmetic is the original's; [`check_carried`] re-lifts the emitted module
//! and compares instruction by instruction.

use peacemaker_ir::cfg::{BlockId, Terminator};
use peacemaker_ir::effects::Control;
use peacemaker_ir::inst::{Abi, Arch, Frontend, Kernel};
use peacemaker_ir::passes::liveness::{Liveness, Loc};
use peacemaker_ir::reg::Kind;

use crate::text::print::canonical;
use crate::{lift_object, sha256, Options};

/// The declared SGPR inputs replacing the hardware workgroup ids.
#[derive(Clone, Debug)]
pub struct RegionInputs {
    /// Label prefix, e.g. `.Lt16`.
    pub prefix: String,
    /// The SGPR carrying the virtual workgroup X (`ttmp9`).
    pub workgroup_x: u16,
    /// The SGPR carrying `vy | vz << 16` (`ttmp7`).
    pub workgroup_yz: u16,
    /// The body's `blockDim.x` (the launch geometry; metadata only bounds it).
    pub block: u32,
    /// Dynamic LDS bytes the eager launch passes (`extern __shared__`), above the
    /// descriptor's static group segment.
    pub dynamic_lds: u32,
}

/// One hidden (implicit) kernarg the task's kernarg block must supply.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HiddenArg {
    pub kind: String,
    pub offset: u32,
    pub size: u32,
}

/// The proof obligations of one conversion and how each was discharged.
#[derive(Clone, Debug)]
pub struct RegionReceipt {
    pub base_object_sha256: [u8; 32],
    pub symbol: String,
    pub block: u32,
    pub waves: u32,
    /// Registers live at the body's entry (all within the declared inputs).
    pub live_in: Vec<String>,
    /// May-live registers the hardware leaves undefined at kernel entry.
    pub undefined_live_in: Vec<String>,
    pub exits_converted: usize,
    pub dealloc_removed: usize,
    /// `(original, substitute, count)`.
    pub input_rewrites: Vec<(String, String, usize)>,
    pub barriers: usize,
    /// Branches proven workgroup-uniform / per-wave (`divergent`), and the
    /// barrier-participation verdict.
    pub uniform_branches: usize,
    pub divergent_branches: usize,
    pub barrier_participation: String,
    pub hidden_args: Vec<HiddenArg>,
    /// Descriptor `COMPUTE_PGM_RSRC1` without its VGPR granule (float modes, WGP
    /// mode...): the dispatcher's descriptor must carry the same bits.
    pub rsrc1_modes: u32,
    /// Instructions carried verbatim (canonical text identical to the original).
    pub carried_verbatim: usize,
    /// Instructions carried with an input substitution only.
    pub carried_substituted: usize,
    pub branches_retargeted: usize,
}

/// A converted body: its region text (labels `name:` and instructions) and extents.
#[derive(Clone, Debug)]
pub struct LiftedRegion {
    pub symbol: String,
    /// The inputs it was converted with (a re-derivation repeats the conversion).
    pub inputs: RegionInputs,
    pub lines: Vec<String>,
    pub quiesce_label: String,
    pub block: u32,
    pub lds_bytes: u32,
    pub kernarg_bytes: u32,
    /// One past the highest SGPR / VGPR the body names (VCC excluded).
    pub sgprs: u16,
    pub vgprs: u16,
    /// The original instruction texts in order, with the region line each became
    /// (`None`: dropped dealloc / converted exit).
    pub carried: Vec<(String, Option<usize>)>,
    pub receipt: RegionReceipt,
}

/// Conversion refused, with the obligation that failed.
#[derive(Debug, thiserror::Error)]
#[error("{symbol}: {reason}")]
pub struct RegionError {
    pub symbol: String,
    pub reason: String,
}

const RSRC1_VGPR_MASK: u32 = 0x3f;
const RSRC1_SGPR_MASK: u32 = 0xf << 6;

fn refuse<T>(symbol: &str, reason: impl Into<String>) -> Result<T, RegionError> {
    Err(RegionError {
        symbol: symbol.into(),
        reason: reason.into(),
    })
}

fn substitute(text: &str, from: &str, to: &str) -> (String, usize) {
    // whole-token replacement (`ttmp9` but not `ttmp91`)
    let mut out = String::with_capacity(text.len());
    let mut count = 0;
    let mut rest = text;
    while let Some(at) = rest.find(from) {
        let before_ok = rest[..at]
            .chars()
            .last()
            .is_none_or(|c| !c.is_ascii_alphanumeric() && c != '_');
        let after = &rest[at + from.len()..];
        let after_ok = after
            .chars()
            .next()
            .is_none_or(|c| !c.is_ascii_alphanumeric() && c != '_');
        out.push_str(&rest[..at]);
        if before_ok && after_ok {
            out.push_str(to);
            count += 1;
        } else {
            out.push_str(from);
        }
        rest = after;
    }
    out.push_str(rest);
    (out, count)
}

/// A hipcc object lifted once (Hipcc frontend) with the digest the receipt records, so the
/// kernels of one object convert without re-lifting and re-emitting the whole object each time.
pub struct HipccObject {
    program: peacemaker_ir::inst::Program,
    sha256: [u8; 32],
}

impl HipccObject {
    /// Lift `object`; the error is the `reason` [`lift_region`] reports for an object that
    /// does not lift.
    pub fn lift(object: &[u8]) -> Result<Self, String> {
        let lifted = lift_object(
            object,
            Options {
                frontend: Frontend::Hipcc,
            },
        )
        .map_err(|e| format!("lift: {e}"))?;
        Ok(Self {
            program: lifted.program,
            sha256: sha256(object),
        })
    }

    /// [`lift_region_in`] on this object.
    pub fn region(&self, symbol: &str, inputs: &RegionInputs) -> Result<LiftedRegion, RegionError> {
        lift_region_in(&self.program, self.sha256, symbol, inputs)
    }
}

/// Convert kernel `symbol` of the hipcc object `object` into a region.
pub fn lift_region(
    object: &[u8],
    symbol: &str,
    inputs: &RegionInputs,
) -> Result<LiftedRegion, RegionError> {
    HipccObject::lift(object)
        .map_err(|reason| RegionError {
            symbol: symbol.into(),
            reason,
        })?
        .region(symbol, inputs)
}

/// [`lift_region`] for an object already lifted: `program` is the Hipcc-frontend lift of
/// the object whose SHA-256 is `object_sha256` (the digest the receipt records). Callers
/// converting several kernels of one object lift it once.
pub fn lift_region_in(
    program: &peacemaker_ir::inst::Program,
    object_sha256: [u8; 32],
    symbol: &str,
    inputs: &RegionInputs,
) -> Result<LiftedRegion, RegionError> {
    let arch = program.target.arch;
    if !arch.is_gfx12() {
        return refuse(symbol, format!("{arch:?}: regions are gfx1201 only"));
    }
    let kernel = program
        .kernels
        .iter()
        .find(|k| k.symbol.0 == symbol)
        .ok_or_else(|| RegionError {
            symbol: symbol.into(),
            reason: "no such kernel in the object".into(),
        })?;
    convert(kernel, arch, object_sha256, inputs)
}

fn convert(
    kernel: &Kernel,
    arch: Arch,
    sha: [u8; 32],
    inputs: &RegionInputs,
) -> Result<LiftedRegion, RegionError> {
    let symbol = kernel.symbol.0.as_str();
    let Abi::Hsa {
        descriptor,
        metadata,
    } = &kernel.abi
    else {
        return refuse(symbol, "not an HSA kernel");
    };
    let meta = &metadata.parsed;
    // ---- (2) entry state the descriptor sets up
    let rsrc2 = descriptor.compute_pgm_rsrc2.0;
    let user_sgprs = (rsrc2 >> 1) & 0x1f;
    let props = descriptor.kernel_code_properties.0;
    if props & 0x3f != 0x08 || user_sgprs != 2 {
        return refuse(
            symbol,
            format!("user SGPRs: kernel_code_properties {props:#x}, count {user_sgprs} (only the kernarg segment pointer is supported)"),
        );
    }
    if rsrc2 & 1 != 0
        || descriptor.private_segment_fixed_size != 0
        || meta.private_segment_fixed_size != 0
    {
        return refuse(symbol, "uses a private (scratch) segment");
    }
    if descriptor.kernarg_preload.0 != 0 {
        return refuse(symbol, "preloads kernargs into SGPRs");
    }
    if (rsrc2 >> 11) & 3 != 0 {
        return refuse(symbol, "enables more than one workitem-id axis (packed v0)");
    }
    if (rsrc2 >> 10) & 1 != 0 {
        return refuse(symbol, "enables the workgroup-info SGPR");
    }
    if meta.uses_dynamic_stack {
        return refuse(symbol, "uses a dynamic stack");
    }
    if inputs.block == 0
        || inputs.block % 32 != 0
        || inputs.block > meta.max_flat_workgroup_size.max(1)
    {
        return refuse(
            symbol,
            format!(
                "blockDim {} is not a multiple of 32 within max_flat_workgroup_size {}",
                inputs.block, meta.max_flat_workgroup_size
            ),
        );
    }
    let body = &kernel.body;
    let entry = BlockId(0);
    let live = Liveness::analyze(body, arch, kernel.wave);
    let declared = |l: &Loc| {
        matches!(
            l,
            Loc::S(0)
                | Loc::S(1)
                | Loc::V(0)
                | Loc::Ttmp(7)
                | Loc::Ttmp(9)
                | Loc::ExecLo
                | Loc::ExecHi
        )
    };
    // Registers the hardware leaves undefined at kernel entry (every VGPR but the
    // workitem id v0, the SGPRs past the user SGPRs, VCC, M0, SCC): a may-live read of
    // one (an exec-masked partial definition) observes an undefined value in the
    // original, so whatever the dispatcher leaves there is one of its behaviours.
    // MODE (`Other(7)`) is the descriptor's: the dispatcher descriptor must carry the
    // same RSRC1 modes (`rsrc1_modes`), and no body may write it (checked below).
    let undefined = |l: &Loc| match l {
        Loc::V(n) => *n > 0,
        Loc::S(n) => *n >= 2,
        Loc::VccLo | Loc::VccHi | Loc::M0 | Loc::Scc | Loc::Other(7) => true,
        _ => false,
    };
    let live_in: Vec<Loc> = live.live_in(entry).iter().copied().collect();
    if let Some(bad) = live_in.iter().find(|l| !declared(l) && !undefined(l)) {
        return refuse(symbol, format!("entry reads {bad:?}, which is neither a declared region input nor undefined at kernel entry"));
    }
    let undefined_live_in: Vec<String> = live_in
        .iter()
        .filter(|l| !declared(l))
        .map(|l| format!("{l:?}"))
        .collect();
    let live_in: Vec<Loc> = live_in.into_iter().filter(|l| declared(l)).collect();
    // ---- (4) geometry: barrier participation
    let waves = inputs.block / 32;
    let (uniform, divergent, participation) =
        super::uniformity::barrier_participation(body, inputs.block).map_err(|reason| {
            RegionError {
                symbol: symbol.into(),
                reason,
            }
        })?;
    // ---- (1)(2) text
    let p = &inputs.prefix;
    let label = |b: BlockId| format!("{p}_{}", b.0);
    let quiesce = format!("{p}_quiesce");
    let (wx, wyz) = (
        format!("s{}", inputs.workgroup_x),
        format!("s{}", inputs.workgroup_yz),
    );
    let mut lines = Vec::new();
    let mut carried = Vec::new();
    let (mut exits, mut dealloc, mut branches, mut verbatim, mut substituted) = (0, 0, 0, 0, 0);
    let (mut n9, mut n7, mut barriers) = (0, 0, 0);
    let (mut max_s, mut max_v) = (0u16, 0u16);
    let mut blocks: Vec<&peacemaker_ir::cfg::Block> = body.blocks.iter().collect();
    blocks.sort_by_key(|b| b.range.0);
    for block in blocks {
        lines.push(format!("{}:", label(block.id)));
        for &id in &body.layout[block.range.0..block.range.1] {
            let inst = body.insts.get(id).expect("laid out");
            let text = canonical(inst, arch).map_err(|e| RegionError {
                symbol: symbol.into(),
                reason: format!("print: {e}"),
            })?;
            let mnemonic = text.split_whitespace().next().unwrap_or("").to_owned();
            // wave-width lane masks, or a wave32 compare into `s5` would count `s6`
            let fx = peacemaker_ir::effects::wave_effects(arch, kernel.wave, inst);
            for r in fx.defs.iter().chain(&fx.uses) {
                let end = r.base + u16::from(r.len);
                match r.kind {
                    Kind::S if r.base < 106 => max_s = max_s.max(end),
                    Kind::V => max_v = max_v.max(end),
                    Kind::Ttmp if !(r.base == 7 || r.base == 9) || r.len != 1 => {
                        return refuse(
                            symbol,
                            format!("`{text}` names a trap temporary other than ttmp7/ttmp9"),
                        );
                    }
                    _ => {}
                }
            }
            // `flat_*` stays: the region resolves flat addresses in the same global and LDS
            // apertures (its LDS is the body's own `[0, lds_bytes)`), and it has no private
            // segment for the scratch aperture to name (the descriptor check above).
            if mnemonic.starts_with("scratch_")
                || matches!(
                    mnemonic.as_str(),
                    "s_getpc_b64"
                        | "s_setpc_b64"
                        | "s_swappc_b64"
                        | "s_trap"
                        | "s_sethalt"
                        | "s_rfe_b64"
                )
            {
                return refuse(
                    symbol,
                    format!("`{text}`: not convertible (PC-relative, scratch or trap)"),
                );
            }
            if mnemonic.starts_with("s_setreg")
                || matches!(mnemonic.as_str(), "s_round_mode" | "s_denorm_mode")
            {
                return refuse(
                    symbol,
                    format!("`{text}` writes MODE, which the dispatcher and later regions share"),
                );
            }
            if matches!(inst.effects.control, Control::Barrier(_)) {
                barriers += 1;
            }
            match inst.effects.control {
                Control::EndPgm => {
                    exits += 1;
                    carried.push((text, None));
                    lines.push(format!("s_branch {quiesce}"));
                    continue;
                }
                Control::Halt | Control::Trap => {
                    return refuse(symbol, format!("`{text}` halts or traps"));
                }
                Control::Branch { .. } | Control::Jump => {
                    let target = match block.term {
                        Terminator::Jump(t) => t,
                        Terminator::Branch { taken, .. } => taken,
                        _ => return refuse(symbol, format!("`{text}` has no CFG target")),
                    };
                    branches += 1;
                    carried.push((text, Some(lines.len())));
                    lines.push(format!("{mnemonic} {}", label(target)));
                    continue;
                }
                _ => {}
            }
            if mnemonic == "s_sendmsg" {
                if text.contains("MSG_DEALLOC_VGPRS") {
                    dealloc += 1;
                    carried.push((text, None));
                    continue;
                }
                return refuse(
                    symbol,
                    format!("`{text}`: messages other than the VGPR deallocation"),
                );
            }
            // The printed text must re-encode to the instruction's own words: the region
            // carries text, so a lossy spelling would silently change the arithmetic.
            let original = peacemaker_ir::codec::gfx12::encode(inst).map_err(|e| RegionError {
                symbol: symbol.into(),
                reason: format!("`{text}`: re-encode: {e}"),
            })?;
            let reparsed = crate::text::parse_line(&text, arch)
                .map_err(|e| format!("{e}"))
                .and_then(|i| peacemaker_ir::codec::gfx12::encode(&i).map_err(|e| format!("{e}")));
            if reparsed.as_deref() != Ok(&original[..]) {
                return refuse(
                    symbol,
                    format!("`{text}` does not re-encode to its words {original:08x?} (got {reparsed:08x?}): the printer loses a field"),
                );
            }
            let (t, a) = substitute(&text, "ttmp9", &wx);
            let (t, b) = substitute(&t, "ttmp7", &wyz);
            n9 += a;
            n7 += b;
            if a + b == 0 {
                verbatim += 1;
            } else {
                substituted += 1;
            }
            carried.push((text, Some(lines.len())));
            lines.push(t);
        }
    }
    if text_has_endpgm(&lines) {
        return refuse(symbol, "an s_endpgm survived the exit conversion");
    }
    if u32::from(inputs.workgroup_x) < u32::from(max_s)
        || u32::from(inputs.workgroup_yz) < u32::from(max_s)
    {
        return refuse(
            symbol,
            format!(
                "input SGPRs s{}/s{} collide with the body's s0..s{}",
                inputs.workgroup_x, inputs.workgroup_yz, max_s
            ),
        );
    }
    lines.push(format!("{quiesce}:"));
    for w in [
        "s_wait_loadcnt 0x0",
        "s_wait_storecnt 0x0",
        "s_wait_dscnt 0x0",
        "s_wait_kmcnt 0x0",
    ] {
        lines.push(w.into());
    }
    let hidden_args = meta
        .args
        .iter()
        .filter(|a| a.value_kind.starts_with("hidden_"))
        .map(|a| HiddenArg {
            kind: a.value_kind.clone(),
            offset: a.offset,
            size: a.size,
        })
        .collect();
    let receipt = RegionReceipt {
        base_object_sha256: sha,
        symbol: symbol.into(),
        block: inputs.block,
        waves,
        live_in: live_in.iter().map(|l| format!("{l:?}")).collect(),
        undefined_live_in,
        exits_converted: exits,
        dealloc_removed: dealloc,
        input_rewrites: vec![
            ("ttmp9".into(), wx.clone(), n9),
            ("ttmp7".into(), wyz.clone(), n7),
        ],
        barriers,
        uniform_branches: uniform,
        divergent_branches: divergent,
        barrier_participation: participation,
        hidden_args,
        rsrc1_modes: descriptor.compute_pgm_rsrc1.0 & !(RSRC1_VGPR_MASK | RSRC1_SGPR_MASK),
        carried_verbatim: verbatim,
        carried_substituted: substituted,
        branches_retargeted: branches,
    };
    Ok(LiftedRegion {
        symbol: symbol.into(),
        inputs: inputs.clone(),
        lines,
        quiesce_label: quiesce,
        block: inputs.block,
        lds_bytes: descriptor
            .group_segment_fixed_size
            .max(meta.group_segment_fixed_size)
            + inputs.dynamic_lds,
        kernarg_bytes: meta.kernarg_segment_size,
        sgprs: max_s,
        vgprs: max_v,
        carried,
        receipt,
    })
}

fn text_has_endpgm(lines: &[String]) -> bool {
    lines.iter().any(|l| l.trim_start().starts_with("s_endpgm"))
}

/// Re-lift `module` and check that kernel `symbol` carries every instruction of
/// `region` unchanged: for each original instruction mapped to a region line, the
/// emitted kernel holds that line's text in order (branches compared by mnemonic,
/// their offsets being re-encoded), and its descriptor carries the body's RSRC1
/// modes. Returns the number of instructions compared.
pub fn check_carried(module: &[u8], symbol: &str, region: &LiftedRegion) -> Result<usize, String> {
    let lifted = lift_object(
        module,
        Options {
            frontend: Frontend::Builder,
        },
    )
    .map_err(|e| format!("re-lift: {e}"))?;
    let program = lifted.program;
    let kernel = program
        .kernels
        .iter()
        .find(|k| k.symbol.0 == symbol)
        .ok_or_else(|| format!("{symbol} not in the module"))?;
    if let Abi::Hsa { descriptor, .. } = &kernel.abi {
        let modes = descriptor.compute_pgm_rsrc1.0 & !(RSRC1_VGPR_MASK | RSRC1_SGPR_MASK);
        if modes != region.receipt.rsrc1_modes {
            return Err(format!(
                "{symbol}: descriptor RSRC1 modes {modes:#x} != the body's {:#x} (float/denorm/WGP modes would differ)",
                region.receipt.rsrc1_modes
            ));
        }
    }
    let texts = crate::emit::text(kernel, program.target.arch).map_err(|e| e.to_string())?;
    let want: Vec<&str> = region
        .lines
        .iter()
        .map(String::as_str)
        .filter(|l| !l.ends_with(':'))
        .collect();
    let key = |t: &str| -> String {
        let m = t.split_whitespace().next().unwrap_or("");
        if m == "s_branch" || m.starts_with("s_cbranch") {
            m.to_owned()
        } else {
            t.to_owned()
        }
    };
    let first = key(want.first().ok_or("empty region")?);
    for start in 0..texts.len() {
        if key(&texts[start]) != first || start + want.len() > texts.len() {
            continue;
        }
        if want
            .iter()
            .zip(&texts[start..])
            .all(|(w, t)| key(w) == key(t))
        {
            return Ok(want.len());
        }
    }
    Err(format!(
        "{symbol}: the region's instruction sequence is not in the emitted kernel"
    ))
}
