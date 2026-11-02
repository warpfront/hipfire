// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The gfx11/gfx12 end-of-module `s_code_end` guard LLVM's AMDGPU AsmPrinter appends to `.text`
//! (`AMDGPUTargetAsmStreamer::EmitCodeEnd`: `.p2alignl 7, 0xbf9f0000` then `.fill 96, 4, 0xbf9f0000`
//! on every gfx10+ AMDHSA target with a 128-byte cache line, i.e. gfx11 and gfx12 alike; confirmed
//! with `clang -x ir -S -mcpu=gfx1100|gfx1151|gfx1201`). The builder's source carries the guard,
//! the module combiner keeps it once, and the native writer lays it out after the last kernel
//! without moving any kernel, symbol size or descriptor. CPU only: no ROCm tool is involved.
use peacemaker_isa::emit::CODE_END_TAIL;
use peacemaker_isa::insn::Sop;
use peacemaker_isa::kernels::iu4_gemm;
use peacemaker_isa::reg::Live;
use peacemaker_isa::{native, Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

const ARCHES: [Arch; 3] = [Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx1201];
const S_CODE_END: [u8; 4] = 0xbf9f_0000u32.to_le_bytes();
const S_NOP_0: [u8; 4] = 0xbf80_0000u32.to_le_bytes();
/// gfx11/gfx12 `s_endpgm` (SOPP opcode 48).
const S_ENDPGM: [u8; 4] = 0xbfb0_0000u32.to_le_bytes();

/// A kernel that is only `s_endpgm`.
fn kernel(arch: Arch, symbol: &str) -> Emitted {
    let mut plan = RegPlan::new(16, 16).unwrap();
    plan.v::<1>("a", 0, Live::Whole).unwrap();
    let mut b = Builder::new(
        KernelSpec {
            kernel_id: symbol.into(),
            variant: "default".into(),
            arch,
            symbol: symbol.into(),
            kernargs: KernargLayout::new(8),
            user_sgpr_count: 2,
            system_sgpr_workgroup_id_y: false,
            workgroup_size: 64,
            group_segment_fixed_size: 0,
            wave32: true,
            cu_mode: false,
        },
        plan,
    );
    b.control(Sop::End.encode(arch).unwrap()).unwrap();
    b.finish().unwrap()
}

fn module(arch: Arch) -> String {
    iu4_gemm::module(
        &[kernel(arch, "guard_a"), kernel(arch, "guard_b")],
        "guard_pair",
    )
    .unwrap()
    .0
}

fn u16_at(b: &[u8], at: usize) -> usize {
    u16::from_le_bytes(b[at..at + 2].try_into().unwrap()) as usize
}
fn u32_at(b: &[u8], at: usize) -> usize {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap()) as usize
}
fn u64_at(b: &[u8], at: usize) -> usize {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap()) as usize
}

/// The bytes of the named section.
fn section(elf: &[u8], name: &str) -> Vec<u8> {
    let (shoff, entsize, count, strndx) = (
        u64_at(elf, 40),
        u16_at(elf, 58),
        u16_at(elf, 60),
        u16_at(elf, 62),
    );
    let header = |i: usize| shoff + i * entsize;
    let names = u64_at(elf, header(strndx) + 24);
    for i in 0..count {
        let at = names + u32_at(elf, header(i));
        let end = at + elf[at..].iter().position(|&c| c == 0).unwrap();
        if &elf[at..end] == name.as_bytes() {
            let (offset, size) = (u64_at(elf, header(i) + 24), u64_at(elf, header(i) + 32));
            return elf[offset..offset + size].to_vec();
        }
    }
    panic!("no section {name}")
}

/// `.symtab` kernel (FUNC) symbol sizes in order.
fn function_sizes(elf: &[u8]) -> Vec<usize> {
    section(elf, ".symtab")
        .chunks(24)
        .filter(|s| s[4] & 0xf == 2)
        .map(|s| u64_at(s, 16))
        .collect()
}

fn without_guard(text: &str) -> &str {
    text.strip_suffix(CODE_END_TAIL)
        .expect("text ends with the guard")
}

fn all(bytes: &[u8], word: [u8; 4]) -> bool {
    bytes.chunks(4).all(|w| w == word)
}

#[test]
fn guard_text_is_llvms_spelling() {
    assert_eq!(
        CODE_END_TAIL,
        ".text\n\t.p2alignl 7, 3214868480\n\t.fill 96, 4, 3214868480\n"
    );
    assert_eq!(3214868480u32, 0xbf9f_0000);
}

#[test]
fn every_builder_kernel_ends_with_the_guard_once_after_its_metadata() {
    for arch in ARCHES {
        let text = kernel(arch, "guard_a").s_text;
        assert!(text.ends_with(CODE_END_TAIL));
        assert_eq!(text.matches(".p2alignl").count(), 1);
        assert_eq!(text.matches(".fill").count(), 1);
        // The pre-guard source is exactly the old output: the guard is a pure suffix after the metadata.
        assert!(without_guard(&text).ends_with("\n...\n.end_amdgpu_metadata\n"));
    }
}

#[test]
fn modules_carry_exactly_one_guard_at_the_end() {
    for arch in ARCHES {
        let text = module(arch);
        assert!(text.ends_with(CODE_END_TAIL));
        assert_eq!(text.matches(".p2alignl").count(), 1);
        assert_eq!(text.matches(".fill").count(), 1);
        assert_eq!(text.matches(".amdhsa_kernel guard_").count(), 2);
        // Neither kernel is followed by padding before the next kernel's alignment.
        let first = text.find(".amdhsa_kernel guard_a").unwrap();
        let second = text.find(".protected guard_b").unwrap();
        assert!(first < second && !text[..second].contains(".p2alignl"));
    }
}

#[test]
fn native_single_kernel_pads_to_128_then_96_s_code_end_dwords() {
    for arch in ARCHES {
        let text = kernel(arch, "guard_a").s_text;
        let guarded = native::assemble(&text, arch).unwrap();
        let bare = native::assemble(without_guard(&text), arch).unwrap();
        let (guarded_text, bare_text) = (section(&guarded, ".text"), section(&bare, ".text"));
        let code = bare_text.len();
        assert_eq!(
            &bare_text[..],
            S_ENDPGM.as_slice(),
            "the kernel is exactly s_endpgm, nothing is appended without the guard"
        );
        assert_eq!(guarded_text.len(), 128 + 96 * 4);
        assert_eq!(guarded_text[..code], bare_text[..]);
        assert!(
            all(&guarded_text[code..], S_CODE_END),
            "alignment fill and guard are s_code_end, not s_nop"
        );
        // Symbol sizes and descriptors are untouched.
        assert_eq!(function_sizes(&guarded), vec![code]);
        assert_eq!(function_sizes(&bare), vec![code]);
        assert_eq!(section(&guarded, ".rodata"), section(&bare, ".rodata"));
    }
}

#[test]
fn native_module_guard_follows_the_last_kernel_only() {
    for arch in ARCHES {
        let text = module(arch);
        let guarded = native::assemble(&text, arch).unwrap();
        let bare = native::assemble(without_guard(&text), arch).unwrap();
        let (guarded_text, bare_text) = (section(&guarded, ".text"), section(&bare, ".text"));
        // guard_a at 0, s_nop 0 to the 256-byte boundary (inter-kernel padding is unchanged), guard_b, guard.
        assert_eq!(bare_text.len(), 256 + 4);
        assert_eq!(&bare_text[..4], S_ENDPGM.as_slice());
        assert!(all(&bare_text[4..256], S_NOP_0));
        assert_eq!(&bare_text[256..], S_ENDPGM.as_slice());
        assert_eq!(guarded_text.len(), 256 + 128 + 96 * 4);
        assert_eq!(guarded_text[..bare_text.len()], bare_text[..]);
        assert!(all(&guarded_text[bare_text.len()..], S_CODE_END));
        assert_eq!(function_sizes(&guarded), vec![4, 4]);
        assert_eq!(section(&guarded, ".rodata"), section(&bare, ".rodata"));
        assert_eq!(section(&guarded, ".note"), section(&bare, ".note"));
    }
}

#[test]
fn native_assembler_accepts_only_the_exact_guard_once_after_the_last_kernel() {
    let arch = Arch::Gfx1201;
    let text = kernel(arch, "guard_a").s_text;
    let bare = without_guard(&text);
    let bad = |source: String| native::assemble(&source, arch).unwrap_err();
    // A different alignment, fill value or count is not the LLVM guard.
    assert!(bad(text.replace(".p2alignl 7,", ".p2alignl 6,")).contains("unsupported"));
    assert!(
        bad(text.replace("3214868480\n\t.fill", "3221225472\n\t.fill")).contains("unsupported")
    );
    assert!(bad(text.replace(".fill 96,", ".fill 95,")).contains("unsupported"));
    // The alignment without its fill, a fill without the alignment, and a second guard.
    assert!(bad(format!("{bare}.text\n\t.p2alignl 7, 3214868480\n")).contains("not followed"));
    assert!(bad(format!("{bare}.text\n\t.fill 96, 4, 3214868480\n")).contains("unsupported"));
    assert!(bad(format!("{text}\t.p2alignl 7, 3214868480\n")).contains("unsupported"));
    // Nothing may follow the guard in .text.
    assert!(bad(format!("{text}\ts_endpgm\n")).contains("instruction outside a kernel"));
    // A source with no guard is still assembled, with no guard.
    let plain = native::assemble(bare, arch).unwrap();
    assert_eq!(section(&plain, ".text"), S_ENDPGM.to_vec());
}
