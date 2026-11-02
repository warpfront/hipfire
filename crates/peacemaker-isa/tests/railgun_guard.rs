// SPDX-License-Identifier: Apache-2.0
use peacemaker_isa::{railgun_guard, Arch};

/// The committed object is this crate's emission: the guard prologue that
/// railgun-cert certifies opens both guarded kernels, the unguarded producer
/// has none.
#[test]
fn committed_guard_module_is_the_authored_object() {
    let (es, text, proof) = railgun_guard::emit_module().unwrap();
    assert_eq!(proof.arch, Arch::Gfx1151);
    for (e, symbol) in es.iter().zip(railgun_guard::SYMBOLS) {
        let guarded = e.s_text.contains(&format!("s_cbranch_scc0 .L{symbol}_go"));
        assert_eq!(guarded, symbol != "railgun_flag_set_pm", "{symbol}");
    }
    let elf = peacemaker_isa::native::assemble(&text, Arch::Gfx1151).unwrap();
    let committed = std::fs::read(format!(
        "{}/../../kernels/{}.co",
        env!("CARGO_MANIFEST_DIR"),
        railgun_guard::MODULE
    ))
    .unwrap();
    assert_eq!(elf, committed);
}
