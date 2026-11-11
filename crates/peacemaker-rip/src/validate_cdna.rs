//! CDNA validation uses the same typed footprints and emitted-code checkers as the checked builder.
use peacemaker_cdna::inst::{RegFile, RegSpan};
use peacemaker_isa::reg::{Kind, RegRef};
use std::collections::{HashMap, HashSet, VecDeque};

const ARCH: peacemaker_cdna::Arch = peacemaker_cdna::Arch::Gfx942;

fn reg(span: &RegSpan) -> Option<RegRef> {
    Some(RegRef {
        kind: match span.file {
            RegFile::Vgpr => Kind::V,
            RegFile::Sgpr => Kind::S,
            RegFile::Agpr => Kind::A,
            _ => return None,
        },
        base: span.base,
        len: span.len,
    })
}

fn parse(text: &str) -> Result<peacemaker_cdna::Inst, String> {
    peacemaker_cdna::text::parse_line(text, ARCH).map_err(|e| {
        if e.starts_with("unknown mnemonic:") {
            format!(
                "unknown mnemonic `{}` on gfx942: {e}",
                super::mnemonic(text)
            )
        } else {
            super::malformed(peacemaker_isa::Arch::Gfx942, text, e)
        }
    })
}

pub(super) fn effects(text: &str) -> Result<(Vec<RegRef>, Vec<RegRef>), String> {
    let inst = parse(text)?;
    let footprint = inst.footprint();
    Ok((
        footprint.defs.iter().filter_map(reg).collect(),
        footprint.uses.iter().filter_map(reg).collect(),
    ))
}

pub(super) fn instruction(text: &str, defs: &[RegRef], uses: &[RegRef]) -> Result<(), String> {
    let inst = parse(text)?;
    let footprint = inst.footprint();
    super::declared_effects(
        text,
        defs,
        uses,
        footprint.defs.iter().filter_map(reg),
        footprint.uses.iter().filter_map(reg),
    )
}

fn cells(span: &RegSpan) -> impl Iterator<Item = (RegFile, u16)> {
    let file = span.file;
    (u16::from(span.base)..u16::from(span.base) + u16::from(span.len)).map(move |n| (file, n))
}

fn name((file, n): (RegFile, u16)) -> String {
    let prefix = match file {
        RegFile::Vgpr => "v",
        RegFile::Sgpr => "s",
        RegFile::Agpr => "a",
        RegFile::Ttmp => "ttmp",
        RegFile::Vcc => "vcc",
        RegFile::Exec => "exec",
        RegFile::M0 => "m0",
        RegFile::Scc => "scc",
        RegFile::FlatScratch => "flat_scratch",
        RegFile::XnackMask => "xnack_mask",
    };
    match file {
        RegFile::M0 | RegFile::Scc => prefix.into(),
        RegFile::Vcc | RegFile::Exec | RegFile::FlatScratch | RegFile::XnackMask => {
            format!("{prefix}_{}", if n == 0 { "lo" } else { "hi" })
        }
        _ => format!("{prefix}{n}"),
    }
}

pub(super) fn emitted(bytes: &[u8], sites: Vec<(String, String)>) -> Result<(), Vec<String>> {
    let lifted = peacemaker_lift::cdna::lift_object(bytes)
        .map_err(|e| vec![format!("certification: CDNA lift: {e}")])?;
    let mut by_text: HashMap<String, VecDeque<String>> = HashMap::new();
    for (text, site) in sites {
        if let Ok(inst) = peacemaker_cdna::text::parse_line(&text, ARCH) {
            by_text.entry(inst.to_string()).or_default().push_back(site);
        }
    }
    let mut errors = Vec::new();
    for kernel in &lifted.kernels {
        if let Err(e) = peacemaker_isa::cdna::lifted::certify(kernel) {
            errors.push(format!("certification: {e}"));
        }
        // Match R0604's deliberately path-free guarantee on RDNA: entry or any write
        // defines a dword. CFG memory ordering is independently checked above.
        let rsrc2 = kernel.image.descriptor.compute_pgm_rsrc2.0;
        let user = ((rsrc2 >> 1) & 0x1f) as u16;
        let mut defined: HashSet<_> = (0..user).map(|n| (RegFile::Sgpr, n)).collect();
        let mut next = user;
        for bit in [7, 8, 9, 10] {
            if rsrc2 >> bit & 1 != 0 {
                defined.insert((RegFile::Sgpr, next));
                next += 1;
            }
        }
        defined.extend([(RegFile::Vgpr, 0), (RegFile::Exec, 0), (RegFile::Exec, 1)]);
        for typed in &kernel.instructions {
            for span in &typed.inst.footprint().defs {
                defined.extend(cells(span));
            }
        }
        for typed in &kernel.instructions {
            let text = typed.inst.to_string();
            let site = by_text.get_mut(&text).and_then(VecDeque::pop_front);
            let mut missing: Vec<_> = typed
                .inst
                .footprint()
                .uses
                .iter()
                .flat_map(cells)
                .filter(|cell| !defined.contains(cell))
                .map(name)
                .collect();
            missing.sort();
            missing.dedup();
            if !missing.is_empty() {
                let at = site.map(|s| format!("{s}: ")).unwrap_or_default();
                errors.push(format!("{at}register read before it is defined: {} in `{text}`; nothing in the kernel writes it and the launch does not initialize it (gfx942 wave64 entry state: s[0:1] kernarg pointer, enabled workgroup IDs, v0 thread id, exec initialized)", missing.join(", ")));
            }
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assembly(body: &str) -> String {
        let source = format!(
            r#"
fn emit(spec: Spec) -> Result<Emitted, String> {{
    let k = KernelSpec {{ kernel_id: "negative".into(), variant: "default".into(),
        arch: spec.arch, symbol: "negative".into(), kernargs: KernargLayout::new(8).pointer("a", 0),
        user_sgpr_count: 2, system_sgpr_workgroup_id_y: false, workgroup_size: 64,
        group_segment_fixed_size: 0, wave32: false, cu_mode: false }};
    let mut p = RegPlan::new(256, 100)?;
    p.s::<4>("entry", 0, Live::Whole)?;
    p.s::<4>("args", 4, Live::Whole)?;
    p.v::<8>("v", 0, Live::Whole)?;
    let mut b = Builder::new(k, p);
    let mut w = Workgroup::new(&mut b, spec.arch)?;
    let end = w.exit(".Lend")?;
    {body}
    w.end(end)?;
    b.finish()
}}
"#
        );
        crate::compile(&source, peacemaker_isa::Arch::Gfx942, "default")
            .unwrap()
            .s_text
    }

    #[test]
    fn final_byte_check_refuses_removed_memory_waits() {
        let asm = assembly(
            r#"
            smem(w.isa(), 4, 2, 0, 0)?;
            mem(w.isa(), "global_load_dword v1, v0, s[4:5]", &[v(1)], &[v(0),sr(4,2)], MemoryClass::VmemLoad)?;
            op(w.isa(), "v_mov_b32_e32 v2, v1", &[v(2)], &[v(1)])?;
        "#,
        );
        assert!(asm.contains("s_waitcnt"));
        let without_waits = asm
            .lines()
            .filter(|l| !l.trim_start().starts_with("s_waitcnt "))
            .collect::<Vec<_>>()
            .join("\n");
        let elf =
            peacemaker_isa::native::assemble(&without_waits, peacemaker_isa::Arch::Gfx942).unwrap();
        let errors = emitted(&elf, vec![]).unwrap_err();
        assert!(
            errors.iter().any(|e| e.contains("wait ledger")),
            "{errors:?}"
        );
    }

    #[test]
    fn final_byte_check_refuses_removed_mfma_hazard_guards() {
        let asm = assembly(
            r#"
            for r in 0u8..8 {
                op(w.isa(), format!("v_mov_b32_e32 v{r}, 0"), &[v(r)], &[])?;
            }
            op(w.isa(), "v_mfma_f32_16x16x16bf16_1k v[4:7], v[0:1], v[2:3], v[4:7]",
                &[vr(4,4)], &[vr(0,2),vr(2,2),vr(4,4)])?;
            op(w.isa(), "v_add_f32_e32 v0, v4, v5", &[v(0)], &[v(4),v(5)])?;
        "#,
        );
        assert!(asm.contains("s_nop"));
        let without_guards = asm
            .lines()
            .filter(|l| !l.trim_start().starts_with("s_nop "))
            .collect::<Vec<_>>()
            .join("\n");
        let elf = peacemaker_isa::native::assemble(&without_guards, peacemaker_isa::Arch::Gfx942)
            .unwrap();
        let errors = emitted(&elf, vec![]).unwrap_err();
        assert!(
            errors.iter().any(|e| e.contains("CDNA hazard")),
            "{errors:?}"
        );
    }
}
