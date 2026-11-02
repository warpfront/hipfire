//! Validate script-authored instruction text without changing its emission.
//!
//! Effects are checked at operand-range granularity: a declared register covers
//! an actual effect if their register kinds match and their half-open dword
//! ranges intersect. Even partial overlap counts; this is not a per-dword
//! ownership proof. Extra declarations are allowed. Special registers and TTMPs
//! are deliberately outside this V/S declaration check.

use peacemaker_ir::{
    effects::Effects,
    inst::{Arch as IrArch, Form, FormFields, Inst},
    isa,
};
use peacemaker_isa::{
    reg::{Kind, RegRef},
    Arch,
};
use peacemaker_lift::text::{parse::ParseError, parse_line};
use std::cell::RefCell;

thread_local! {
    static TEXTS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

const CDNA_UNSUPPORTED: &str =
    "the .rip front end lowers onto the RDNA builder; gfx942 kernels use peacemaker_isa::cdna";

/// The RDNA IR architecture for `arch`; CDNA has no `.rip` lowering, so it is refused.
fn ir_arch(arch: Arch) -> Result<IrArch, String> {
    match arch {
        Arch::Gfx1100 => Ok(IrArch::Gfx1100),
        Arch::Gfx1151 => Ok(IrArch::Gfx1151),
        Arch::Gfx1201 => Ok(IrArch::Gfx1201),
        Arch::Gfx942 => Err(CDNA_UNSUPPORTED.into()),
    }
}

fn mnemonic(text: &str) -> &str {
    text.split_whitespace().next().unwrap_or("")
}

fn malformed(arch: Arch, text: &str, reason: impl std::fmt::Display) -> String {
    format!(
        "malformed operands for `{}` on {}: {reason}",
        mnemonic(text),
        arch.name()
    )
}

// Only used on an error path; the successful validation path does not search
// or copy the mnemonic table.
fn distance(a: &str, b: &str) -> usize {
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.bytes().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, y) in b.bytes().enumerate() {
            let previous = row[j + 1];
            row[j + 1] = (previous + 1)
                .min(row[j] + 1)
                .min(diagonal + usize::from(x != y));
            diagonal = previous;
        }
    }
    row[b.len()]
}

fn normalized(name: &str) -> String {
    name.trim_end_matches("_e32")
        .trim_end_matches("_e64")
        .replace("_co_", "_")
        .replace("_num_", "_")
}

fn unknown(arch: Arch, ir: IrArch, name: &str) -> String {
    let target = isa::table(ir);
    let normalized_name = normalized(name);
    let suggestion = target.iter().min_by_key(|row| {
        (
            normalized(row.name) != normalized_name,
            distance(name, row.name),
            row.name,
        )
    });
    let mut message = format!("unknown mnemonic `{name}` on {}", arch.name());
    let others: &[IrArch] = if arch.gfx12() {
        &[IrArch::Gfx1100, IrArch::Gfx1151]
    } else {
        &[IrArch::Gfx1201]
    };
    if others.iter().any(|&other| {
        isa::table(other).iter().any(|row| {
            row.name == name || row.name.trim_end_matches("_e32").trim_end_matches("_e64") == name
        })
    }) {
        message.push_str(if arch.gfx12() {
            "; this mnemonic is known on gfx11, not gfx12"
        } else {
            "; this mnemonic is known on gfx12, not this gfx11 target"
        });
    }
    if let Some(row) = suggestion {
        message.push_str(&format!("; did you mean `{}`?", row.name));
    }
    message
}

fn parse(arch: Arch, ir: IrArch, text: &str) -> Result<Inst, String> {
    parse_line(text, ir).map_err(|error| match error {
        ParseError::UnknownMnemonic(name) => unknown(arch, ir, &name),
        other => malformed(arch, text, other),
    })
}

fn table_effects(arch: IrArch, inst: &Inst) -> Result<Effects, peacemaker_ir::inst::ValidateError> {
    if let FormFields::Vopd { y_op, x_operands } = inst.fields {
        let split = usize::from(x_operands);
        let mut effects = Effects::from_table(
            arch,
            inst.op,
            inst.form,
            &inst.operands[..split],
            &inst.mods.cpol,
        )?;
        let y = Effects::from_table(
            arch,
            y_op,
            Form::Vopd,
            &inst.operands[split..],
            &inst.mods.cpol,
        )?;
        effects.defs.extend(y.defs);
        effects.uses.extend(y.uses);
        Ok(effects)
    } else {
        Effects::from_table(arch, inst.op, inst.form, &inst.operands, &inst.mods.cpol)
    }
}

fn builder_reg(reg: &peacemaker_ir::reg::RegRef) -> Option<RegRef> {
    let kind = match reg.kind {
        peacemaker_ir::reg::Kind::V => Kind::V,
        peacemaker_ir::reg::Kind::S => Kind::S,
        peacemaker_ir::reg::Kind::Ttmp => return None,
    };
    // Parsed IR registers have already passed their architecture's bounds check.
    Some(RegRef {
        kind,
        base: u8::try_from(reg.base).expect("validated V/S base fits u8"),
        len: reg.len,
    })
}

/// Register effects from the target opcode table, excluding special registers
/// and TTMPs. Returns (definitions, uses), retaining operand order and widths.
pub fn effects(arch: Arch, text: &str) -> Result<(Vec<RegRef>, Vec<RegRef>), String> {
    let ir = ir_arch(arch)?;
    let inst = parse(arch, ir, text)?;
    let effects = table_effects(ir, &inst).map_err(|e| malformed(arch, text, e))?;
    Ok((
        effects.defs.iter().filter_map(builder_reg).collect(),
        effects.uses.iter().filter_map(builder_reg).collect(),
    ))
}

fn names(regs: impl IntoIterator<Item = RegRef>) -> String {
    regs.into_iter()
        .map(|reg| reg.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn source_list(declared: &[RegRef], actual: &[RegRef]) -> String {
    let mut corrected = declared.to_vec();
    for &reg in actual {
        if !corrected.iter().any(|&known| known.overlaps(reg)) {
            corrected.push(reg);
        }
    }
    let entries = corrected
        .iter()
        .map(|reg| {
            let prefix = if reg.kind == Kind::V { "v" } else { "s" };
            if reg.len == 1 {
                format!("{prefix}({})", reg.base)
            } else {
                format!("{prefix}r({}, {})", reg.base, reg.len)
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("&[{entries}]")
}

/// Parse and encode exactly as the native writer does, then reject declared
/// effects that omit any table effect under the overlap rule above.
/// Labels and typed branches belong to the driver, not this entry point.
pub fn instruction(arch: Arch, text: &str, defs: &[RegRef], uses: &[RegRef]) -> Result<(), String> {
    let ir = ir_arch(arch)?;
    let inst = parse(arch, ir, text)?;
    let encoded = if arch.gfx12() {
        peacemaker_ir::codec::gfx12::encode(&inst)
    } else {
        peacemaker_ir::codec::gfx11::encode(ir, &inst)
    };
    encoded.map_err(|e| malformed(arch, text, e))?;
    // parse_line already constructs Effects::from_table (including both VOPD
    // halves). Reuse that result instead of recomputing effects on every op.
    let effects = &inst.effects;
    let missing_defs: Vec<_> = effects
        .defs
        .iter()
        .filter_map(builder_reg)
        .filter(|&reg| !defs.iter().any(|&declared| declared.overlaps(reg)))
        .collect();
    let missing_uses: Vec<_> = effects
        .uses
        .iter()
        .filter_map(builder_reg)
        .filter(|&reg| !uses.iter().any(|&declared| declared.overlaps(reg)))
        .collect();
    if !missing_defs.is_empty() || !missing_uses.is_empty() {
        let mut details = Vec::new();
        if !missing_defs.is_empty() {
            details.push(format!("writes {}", names(missing_defs.iter().copied())));
        }
        if !missing_uses.is_empty() {
            details.push(format!("reads {}", names(missing_uses.iter().copied())));
        }
        let message = format!(
            "op declares defs [{}] and uses [{}] but `{text}` also {}: write {}, {}",
            names(defs.iter().copied()),
            names(uses.iter().copied()),
            details.join(" and "),
            source_list(defs, &missing_defs),
            source_list(uses, &missing_uses)
        );
        return Err(message);
    }
    TEXTS.with(|texts| texts.borrow_mut().push(text.to_owned()));
    Ok(())
}

/// Drain validated instruction texts for call-site tagging by the evaluator.
pub(crate) fn take_texts() -> Vec<String> {
    TEXTS.with(|texts| std::mem::take(&mut *texts.borrow_mut()))
}

const S_BASE: usize = 256;
const TTMP_BASE: usize = S_BASE + 106;
const VCC_LO: usize = TTMP_BASE + 16;
const EXEC_LO: usize = VCC_LO + 2;
const SCC: usize = VCC_LO + 4;
const M0: usize = VCC_LO + 5;
const MODE: usize = VCC_LO + 6;
const LOCATIONS: usize = MODE + 1;

fn effect_locations(
    regs: &[peacemaker_ir::reg::RegRef],
    bits: u8,
    wave: peacemaker_ir::inst::Wave,
) -> impl Iterator<Item = usize> + '_ {
    use peacemaker_ir::{effects::ImplicitSet as I, inst::Wave, reg::Kind as K};
    let registers = regs.iter().flat_map(|reg| {
        let file = match reg.kind {
            K::V => 0,
            K::S => S_BASE,
            K::Ttmp => TTMP_BASE,
        };
        let start = file + usize::from(reg.base);
        start..start + usize::from(reg.len)
    });
    let wide = wave == Wave::Wave64;
    registers.chain(
        [
            (VCC_LO, I::VCC),
            (VCC_LO + 1, if wide { I::VCC } else { 0 }),
            (EXEC_LO, I::EXEC),
            (EXEC_LO + 1, if wide { I::EXEC } else { 0 }),
            (SCC, I::SCC),
            (M0, I::M0),
            (MODE, I::MODE),
        ]
        .into_iter()
        .filter_map(move |(location, bit)| (bits & bit != 0).then_some(location)),
    )
}

fn location_name(location: usize) -> String {
    match location {
        n if n < S_BASE => format!("v{n}"),
        n if n < TTMP_BASE => format!("s{}", n - S_BASE),
        n if n < VCC_LO => format!("ttmp{}", n - TTMP_BASE),
        VCC_LO => "vcc_lo".into(),
        n if n == VCC_LO + 1 => "vcc_hi".into(),
        EXEC_LO => "exec_lo".into(),
        n if n == EXEC_LO + 1 => "exec_hi".into(),
        SCC => "scc".into(),
        M0 => "m0".into(),
        _ => "mode".into(),
    }
}

fn entry_state(kernel: &peacemaker_ir::inst::Kernel, arch: Arch) -> ([bool; LOCATIONS], String) {
    use peacemaker_ir::inst::{Abi, EntryCell, Wave};
    let mut defined = [false; LOCATIONS];
    let Abi::Hsa { descriptor, .. } = &kernel.abi else {
        if let Abi::Raw(raw) = &kernel.abi {
            for (cell, _) in raw.entry_cells().expect("validated raw entry facts") {
                let location = match cell {
                    EntryCell::Vgpr(n) => Some(usize::from(n)),
                    EntryCell::Sgpr(n) => Some(S_BASE + usize::from(n)),
                    EntryCell::Ttmp(n) => Some(TTMP_BASE + usize::from(n)),
                    EntryCell::VccLo => Some(VCC_LO),
                    EntryCell::VccHi => Some(VCC_LO + 1),
                    EntryCell::ExecLo => Some(EXEC_LO),
                    EntryCell::ExecHi => Some(EXEC_LO + 1),
                    EntryCell::Scc => Some(SCC),
                    EntryCell::M0 => Some(M0),
                    _ => None,
                };
                if let Some(location) = location {
                    defined[location] = true;
                }
            }
        }
        return (defined, "explicit raw entry facts".into());
    };
    let rsrc2 = descriptor.compute_pgm_rsrc2.0;
    let user = ((rsrc2 >> 1) & 0x1f) as usize;
    defined[S_BASE..S_BASE + user].fill(true);
    let mut description = format!("{} entry state: s[0:1] kernarg pointer", arch.name());
    if arch.gfx12() {
        if rsrc2 >> 7 & 1 != 0 {
            defined[TTMP_BASE + 9] = true;
        }
        if rsrc2 >> 8 & 3 != 0 {
            defined[TTMP_BASE + 7] = true;
        }
        description.push_str(", ttmp9 workgroup id x, ttmp7 = y | z<<16");
    } else {
        let mut next = user;
        for (bit, label) in [(7, "x"), (8, "y"), (9, "z"), (10, "workgroup size")] {
            if rsrc2 >> bit & 1 != 0 {
                defined[S_BASE + next] = true;
                description.push_str(&format!(", s{next} workgroup id {label}"));
                next += 1;
            }
        }
    }
    defined[0] = true;
    defined[EXEC_LO] = true;
    if kernel.wave == Wave::Wave64 {
        defined[EXEC_LO + 1] = true;
    }
    defined[MODE] = true;
    description.push_str(", v0 thread id, exec initialized");
    (defined, description)
}

/// The SGPR reads of `inst` as a definedness check sees them. A wave32 kernel reads only the low
/// register of a lane-mask source: the table models the select / carry-in slot (`SRC2` of
/// `v_cndmask_*` and of the `v_*_co_ci_*` / `v_addc_*` / `v_subb_*` family) as a 64-bit pair and the
/// parser widens `s5` to `s[5:6]`, exactly as the codec decodes it, so a check over the widened pair
/// would demand a write of a half the hardware never reads (`s_mov_b32 s5, mask` defines the whole
/// wave32 mask).
fn definedness_uses(
    arch: IrArch,
    inst: &Inst,
    wave: peacemaker_ir::inst::Wave,
) -> Vec<peacemaker_ir::reg::RegRef> {
    use peacemaker_ir::{operand::Operand, reg::Kind as K};
    let mut uses = inst.effects.uses.to_vec();
    if wave != peacemaker_ir::inst::Wave::Wave32 {
        return uses;
    }
    let Some(row) = isa::lookup(arch, inst.op, inst.form) else {
        return uses;
    };
    let name = row.name;
    let carry = name.starts_with("v_cndmask_")
        || name.contains("_co_ci_")
        || name.starts_with("v_addc_")
        || name.starts_with("v_subb_");
    if !carry {
        return uses;
    }
    let mask = row
        .slots(false)
        .zip(inst.operands.iter())
        .find_map(|((slot, _), operand)| match (slot, operand) {
            ("SRC2", Operand::Reg(reg)) if reg.kind == K::S && reg.len == 2 => Some(*reg),
            _ => None,
        });
    if let Some(mask) = mask {
        if let Some(read) = uses.iter_mut().find(|r| **r == mask) {
            read.len = 1;
        }
    }
    uses
}

/// Validate the final bytes, not just declarations. Used after every successful
/// emit by both compilation and check; all returned messages are errors.
/// R0604 is deliberately path-free: a read is refused only if the launch does
/// not initialize it and no instruction anywhere writes it. Any half write
/// counts as defining its dword. Ordering, paths, and half coverage stay with
/// M7 certification; this inexpensive check does not claim to prove them.
pub(crate) fn emitted(
    arch: Arch,
    text: &str,
    sites: Vec<(String, String)>,
) -> Result<(), Vec<String>> {
    let arch_ir = ir_arch(arch).map_err(|reason| vec![reason])?;
    let bytes = peacemaker_isa::native::assemble(text, arch).map_err(|reason| {
        let at = sites
            .iter()
            .find(|(text, _)| reason.contains(text))
            .map(|(_, at)| format!("{at}: "))
            .unwrap_or_default();
        vec![format!("{at}assemble: {reason}")]
    })?;
    let lifted = peacemaker_lift::lift_object(
        &bytes,
        peacemaker_lift::Options {
            frontend: peacemaker_ir::inst::Frontend::Builder,
        },
    )
    .map_err(|reason| vec![format!("certification: lift: {reason}")])?;
    let mut findings = Vec::new();
    for (index, kernel) in lifted.program.kernels.iter().enumerate() {
        let (mut defined, entry) = entry_state(kernel, arch);
        for &id in &kernel.body.layout {
            let inst = kernel.body.insts.get(id).expect("laid-out instruction");
            for location in effect_locations(
                &inst.effects.defs,
                inst.effects.implicit.writes,
                kernel.wave,
            ) {
                defined[location] = true;
            }
        }
        let mut undefined = Vec::new();
        for (position, &id) in kernel.body.layout.iter().enumerate() {
            let inst = kernel.body.insts.get(id).expect("laid-out instruction");
            let uses = definedness_uses(arch_ir, inst, kernel.wave);
            let missing: std::collections::BTreeSet<_> =
                effect_locations(&uses, inst.effects.implicit.reads, kernel.wave)
                    .filter(|&location| !defined[location])
                    .collect();
            if !missing.is_empty() {
                undefined.push((position, missing));
            }
        }
        if !undefined.is_empty() {
            findings.push((index, entry, undefined));
        }
    }
    if findings.is_empty() {
        return Ok(());
    }
    // Only errors need source matching. Canonicalize texts the same way as
    // lifted instructions, and consume repeated occurrences in emission order.
    // Generated instructions without a matching script text stay unlocated.
    let mut by_text: std::collections::HashMap<String, std::collections::VecDeque<String>> =
        std::collections::HashMap::new();
    for (text, site) in sites {
        if let Ok(inst) = parse_line(&text, arch_ir) {
            if let Ok(text) = inst.text(arch_ir) {
                by_text.entry(text).or_default().push_back(site);
            }
        }
    }
    let mut errors = Vec::new();
    for (kernel_index, entry, undefined) in findings {
        let kernel = &lifted.program.kernels[kernel_index];
        let mut undefined = undefined.into_iter().peekable();
        for (index, &id) in kernel.body.layout.iter().enumerate() {
            let inst = kernel.body.insts.get(id).expect("laid-out instruction");
            let text = inst
                .text(arch_ir)
                .map_err(|reason| vec![format!("certification: instruction text: {reason}")])?;
            let site = by_text
                .get_mut(&text)
                .and_then(std::collections::VecDeque::pop_front);
            if undefined
                .peek()
                .is_some_and(|(position, _)| *position == index)
            {
                let (_, missing) = undefined.next().expect("peeked undefined read");
                let at = site.map(|at| format!("{at}: ")).unwrap_or_default();
                let regs = missing
                    .into_iter()
                    .map(location_name)
                    .collect::<Vec<_>>()
                    .join(", ");
                errors.push(format!("{at}register read before it is defined: {regs} in `{text}`; nothing in the kernel writes it and the launch does not initialize it ({entry})"));
            }
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}
