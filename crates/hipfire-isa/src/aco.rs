//! Text interchange with the pinned, out-of-tree Mesa ACO post-RA probe.
//! No Mesa code or binary is linked into this crate.
use crate::arch::Arch;
use crate::insn::{Instruction, MemoryClass, Program};
use crate::reg::{Kind, RegRef};

/// Export the physical-register instruction stream without selecting or rewriting opcodes.
/// The shim rejects instructions it cannot model; it must never silently omit one.
pub fn export(program: &Program) -> String {
    let mut text = format!(".arch {}\n", program.arch.name());
    for instruction in &program.instructions {
        text.push_str(&instruction.text);
        text.push('\n');
    }
    text
}

fn register(token: &str) -> Option<RegRef> {
    let token = token.trim();
    let kind = match token.as_bytes().first()? {
        b'v' => Kind::V,
        b's' => Kind::S,
        _ => return None,
    };
    let (base, end) = if token.as_bytes().get(1) == Some(&b'[') {
        let bounds = token.get(2..token.len().checked_sub(1)?)?;
        let (first, last) = bounds.split_once(':')?;
        (first.parse::<u8>().ok()?, last.parse::<u8>().ok()?)
    } else {
        let base = token.get(1..)?.parse::<u8>().ok()?;
        (base, base)
    };
    Some(RegRef { kind, base, len: end.checked_sub(base)?.checked_add(1)? })
}

fn operands(asm: &str) -> Vec<&str> {
    asm.split_once(' ')
        .map(|(_, rest)| rest.split(',').map(str::trim).collect())
        .unwrap_or_default()
}

fn decoded(asm: &str) -> Instruction {
    let name = asm.split_whitespace().next().unwrap_or("");
    let mut defs = Vec::new();
    let mut uses = Vec::new();
    let mut memory = None;
    if asm.contains(" :: ") {
        for half in asm.split(" :: ") {
            let fields = operands(half);
            assert_eq!(fields.len(), 3, "invalid ACO VOPD half: {half}");
            defs.push(register(fields[0]).expect("VOPD destination must be a VGPR"));
            uses.extend(fields[1..].iter().filter_map(|field| register(field)));
            if half.starts_with("v_dual_fmac_") {
                uses.push(*defs.last().unwrap());
            }
        }
    } else if name == "buffer_load_b64" || name == "buffer_load_b32" {
        let fields = operands(asm);
        assert_eq!(fields.len(), 4, "invalid ACO buffer load: {asm}");
        defs.push(register(fields[0]).expect("buffer load destination must be a VGPR"));
        uses.extend(fields[1..].iter().filter_map(|field| register(field.split_whitespace().next().unwrap_or(""))));
        memory = Some(MemoryClass::VmemLoad);
    } else if name == "s_mov_b32" {
        let fields = operands(asm);
        assert_eq!(fields.len(), 2, "invalid ACO scalar move: {asm}");
        defs.push(register(fields[0]).expect("scalar move destination must be an SGPR"));
        uses.extend(fields[1..].iter().filter_map(|field| register(field)));
    } else {
        assert!(matches!(name, "s_endpgm" | "v_nop") || name.starts_with("s_wait_") || name == "s_clause", "unsupported ACO disassembly: {asm}");
    }
    Instruction { text: asm.into(), defs, uses, memory }
}

/// Import the subset disassembled by `aco-shim` as a physical-register program.
/// Fails closed on unsupported opcodes rather than returning unsound def/use metadata.
pub fn import_disasm(disasm: &str) -> Program {
    let mut arch = Arch::Gfx1201;
    let mut instructions = Vec::new();
    // The CLI prints ACO IR before the assembled disassembly.
    let cli_output = disasm.lines().any(|line| line.starts_with("assembled_bytes="));
    let mut in_disassembly = !cli_output;
    for raw in disasm.lines() {
        let line = raw.trim();
        if cli_output && line.starts_with("assembled_bytes=") {
            in_disassembly = true;
            continue;
        }
        if !in_disassembly { continue; }
        if let Some(value) = line.strip_prefix(".arch ") {
            arch = value.parse().expect("unsupported ACO architecture");
            continue;
        }
        if line.is_empty() || line.starts_with('#') { continue; }
        instructions.push(decoded(line));
    }
    Program { arch, instructions }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_import_preserves_physical_dependencies() {
        let source = ".arch gfx1201
buffer_load_b64 v[0:1], v2, s[4:7], s8 offen
buffer_load_b64 v[4:5], v2, s[4:7], s8 offen
v_dual_subrev_f32 v30, s20, v30 :: v_dual_subrev_f32 v31, s20, v31
v_dual_mul_f32 v10, v0, v20 :: v_dual_mul_f32 v11, v5, v20
v_dual_fmac_f32 v12, v10, v30 :: v_dual_fmac_f32 v13, v11, v31
s_endpgm
";
        let program = import_disasm(source);
        assert_eq!(program.arch, Arch::Gfx1201);
        assert_eq!(program.instructions[0].memory, Some(MemoryClass::VmemLoad));
        assert!(program.instructions[3].uses.iter().any(|r| r.overlaps(program.instructions[0].defs[0])));
        assert!(program.instructions[3].uses.contains(&RegRef { kind: Kind::V, base: 5, len: 1 }));
        assert!(program.instructions[4].uses.contains(&RegRef { kind: Kind::V, base: 12, len: 1 }));
        assert_eq!(export(&program), source);
        let cli_output = format!("After Instruction Selection:\nBB0\nassembled_bytes=60\n{}", source.strip_prefix(".arch gfx1201\n").unwrap());
        let imported = import_disasm(&cli_output);
        assert_eq!(imported.instructions.len(), program.instructions.len());
        assert_eq!(imported.instructions[4].uses, program.instructions[4].uses);
    }
}
