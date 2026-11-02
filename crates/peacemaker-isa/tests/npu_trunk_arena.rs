// SPDX-License-Identifier: Apache-2.0
use peacemaker_isa::{npu_trunk_arena, Arch};
#[test]
fn committed_arena_readers_are_the_authored_wave32_module() {
    let (es, text, proof) = npu_trunk_arena::emit_module().unwrap();
    assert_eq!(es.len(), 3);
    assert_eq!(proof.arch, Arch::Gfx1151);
    for (e, symbol) in es.iter().zip(npu_trunk_arena::SYMBOLS) {
        assert!(e.s_text.contains(&format!("{symbol}:")));
        assert!(e.s_text.contains(".amdhsa_wavefront_size32 1"));
        assert!(e.s_text.contains(".private_segment_fixed_size: 0"));
        assert_eq!(e.shape.wmma, 0);
    }
    assert!(es[0].s_text.contains("buffer_load_u16"));
    assert!(es[0].s_text.contains("v_lshlrev_b32_e32 v3, 16, v3"));
    let elf = peacemaker_isa::native::assemble(&text, Arch::Gfx1151).unwrap();
    let bundle = peacemaker_isa::native::bundle(
        &elf,
        Arch::Gfx1151,
        peacemaker_isa::native::DEFAULT_HOST_TARGET,
    );
    let committed = std::fs::read(format!(
        "{}/../../kernels/npu_trunk_arena_pm_gfx1151.hxaco",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    assert_eq!(bundle, committed);
}
#[cfg(feature = "toolchain")]
#[test]
fn arena_exports_have_independent_m7_and_wait_replay() {
    let (es, _, _) = npu_trunk_arena::emit_module().unwrap();
    let path = std::path::PathBuf::from(format!(
        "{}/../../kernels/npu_trunk_arena_pm_gfx1151.co",
        env!("CARGO_MANIFEST_DIR")
    ));
    for (e, symbol) in es.iter().zip(npu_trunk_arena::SYMBOLS) {
        peacemaker_isa::ledger_replay::replay_waits(&e.s_text, Arch::Gfx1151).unwrap();
        peacemaker_isa::pm_check::m7(&path, "gfx1151", symbol).unwrap();
    }
}
