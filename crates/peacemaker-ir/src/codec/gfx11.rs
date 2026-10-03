//! RDNA3/RDNA3.5 codec: separate target opcode tables, shared typed field codec.
use smallvec::SmallVec;
use crate::inst::{Arch, Inst};
pub use super::gfx12::DecodeError;

pub fn decode(arch: Arch, words: &[u32]) -> Result<(Inst, usize), DecodeError> {
    assert!(matches!(arch, Arch::Gfx1100 | Arch::Gfx1151));
    super::gfx12::decode_for(arch, words)
}
pub fn encode(arch: Arch, inst: &Inst) -> Result<SmallVec<[u32; 3]>, DecodeError> {
    assert!(matches!(arch, Arch::Gfx1100 | Arch::Gfx1151));
    super::gfx12::encode_for(arch, inst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{isa, wait::Counter};

    #[test]
    fn target_rows_reencode_from_typed_fields() {
        for arch in [Arch::Gfx1100, Arch::Gfx1151] {
            let mut failures = Vec::new();
            for row in isa::table(arch) {
                let words: Vec<u32> = row.encoding.split_whitespace()
                    .map(|word| u32::from_str_radix(word, 16).unwrap()).collect();
                match decode(arch, &words).and_then(|(inst, used)| {
                    if used != words.len() { return Err(super::super::gfx12::DecodeError::Rejected { offset: 0, reason: "width mismatch".into() }); }
                    let output = encode(arch, &inst)?;
                    if output.as_slice() != words { return Err(super::super::gfx12::DecodeError::Rejected { offset: 0, reason: "word mismatch".into() }); }
                    Ok(())
                }) {
                    Ok(()) => {},
                    Err(error) => failures.push(format!("{} {words:08x?}: {error}", row.name)),
                }
            }
            assert!(failures.is_empty(), "{arch:?}:\n{}", failures.join("\n"));
        }
    }

    #[test]
    fn gfx11_waits_and_scalar_float_are_arch_specific() {
        for arch in [Arch::Gfx1100, Arch::Gfx1151] {
            let (inst, _) = decode(arch, &[0xbf89_0432]).unwrap();
            let wait = inst.mods.wait.as_ref().unwrap();
            assert_eq!(wait.per_counter[Counter::Vm as usize], Some(1));
            assert_eq!(wait.per_counter[Counter::Exp as usize], Some(2));
            assert_eq!(wait.per_counter[Counter::Lgkm as usize], Some(3));
            assert_eq!(encode(arch, &inst).unwrap().as_slice(), &[0xbf89_0432]);
            let (mut variant, _) = decode(arch, &[0xbf89_043a]).unwrap();
            assert_eq!(variant.mods.wait, inst.mods.wait);
            variant.prov.bytes = None;
            assert_eq!(encode(arch, &variant).unwrap().as_slice(), &[0xbf89_043a]);
        }
        assert!(decode(Arch::Gfx1100, &[0xa000_0201]).is_err());
        let (add, _) = decode(Arch::Gfx1151, &[0xa000_0201]).unwrap();
        assert_eq!(add.op.name(Arch::Gfx1151), Some("s_add_f32"));
    }

    /// GLOBAL atomics: GLC selects the returning form. Words 1-7 are every
    /// atomic in the DS4 gfx1151 objects; 8-10 are the opposite forms.
    #[test]
    fn global_atomic_return_form_is_the_glc_bit() {
        use std::{io::Write, process::{Command, Stdio}};
        use crate::{effects::MemClass, operand::Operand, reg::{Kind, RegRef}};
        let Some(llvm_mc) = crate::pinned_llvm_mc() else { return };
        let cases: [([u32; 2], &str, bool); 10] = [
            ([0xdcd2_4000, 0x0000_0002], "global_atomic_cmpswap_b32 v0, v2, v[0:1], s[0:1] glc", true),
            ([0xdcd2_4004, 0x0000_0002], "global_atomic_cmpswap_b32 v0, v2, v[0:1], s[0:1] offset:4 glc", true),
            ([0xdcd2_4000, 0x0000_0004], "global_atomic_cmpswap_b32 v0, v4, v[0:1], s[0:1] glc", true),
            ([0xdcd6_4000, 0x0204_020a], "global_atomic_add_u32 v2, v10, v2, s[4:5] glc", true),
            ([0xdcce_0004, 0x0006_0c0d], "global_atomic_swap_b32 v13, v12, s[6:7] offset:4", false),
            ([0xdcce_0004, 0x0000_0000], "global_atomic_swap_b32 v0, v0, s[0:1] offset:4", false),
            ([0xdcce_0000, 0x0000_0000], "global_atomic_swap_b32 v0, v0, s[0:1]", false),
            ([0xdcd2_0000, 0x0000_0002], "global_atomic_cmpswap_b32 v2, v[0:1], s[0:1]", false),
            ([0xdcd6_0000, 0x0004_020a], "global_atomic_add_u32 v10, v2, s[4:5]", false),
            ([0xdcce_4004, 0x0106_0c0d], "global_atomic_swap_b32 v1, v13, v12, s[6:7] offset:4 glc", true),
        ];
        for arch in [Arch::Gfx1100, Arch::Gfx1151] {
            let cpu = if arch == Arch::Gfx1100 { "gfx1100" } else { "gfx1151" };
            for (words, text, returns) in cases {
                let mut mc = Command::new(&llvm_mc)
                    .args(["-triple=amdgcn-amd-amdhsa", &format!("-mcpu={cpu}"), "-show-encoding"])
                    .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().expect("pinned llvm-mc");
                mc.stdin.take().unwrap().write_all(format!("{text}\n").as_bytes()).unwrap();
                let out = mc.wait_with_output().unwrap();
                assert!(out.status.success(), "{text}: {}", String::from_utf8_lossy(&out.stderr));
                let expected: String = words.iter().flat_map(|w| w.to_le_bytes()).map(|b| format!("0x{b:02x}")).collect::<Vec<_>>().join(",");
                assert!(String::from_utf8_lossy(&out.stdout).contains(&format!("encoding: [{expected}]")), "{text}");

                let (mut inst, used) = decode(arch, &words).unwrap_or_else(|e| panic!("{text}: {e}"));
                assert_eq!(used, 2);
                assert_eq!(inst.mods.cpol.glc, returns);
                let mem = inst.effects.mem.expect("atomic memory effect");
                assert_eq!(mem.class, MemClass::VmemAtomic { returns }, "{text}");
                let (counted, idle) = if returns { (Counter::Vm, Counter::Vs) } else { (Counter::Vs, Counter::Vm) };
                assert!(mem.counters.contains(counted) && !mem.counters.contains(idle), "{text}: {:?}", mem.counters);
                let vdst = RegRef { kind: Kind::V, base: (words[1] >> 24) as u16, len: 1 };
                assert_eq!(inst.effects.defs.to_vec(), if returns { vec![vdst] } else { vec![] }, "{text}");
                let data_len = if text.contains("cmpswap") { 2 } else { 1 };
                let data = RegRef { kind: Kind::V, base: (words[1] >> 8 & 0xff) as u16, len: data_len };
                assert!(inst.effects.uses.contains(&data), "{text}: data {data:?} not read");
                inst.prov.bytes = None;
                assert_eq!(encode(arch, &inst).unwrap().as_slice(), &words, "{text}");
            }
            // Without GLC there is no destination, so a nonzero VDST byte is unmodeled.
            assert!(decode(arch, &[0xdcce_0004, 0x0106_0c0d]).is_err());
        }
    }
}
