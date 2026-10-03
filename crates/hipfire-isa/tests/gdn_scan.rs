//! The authoring-only GDN scan must retire loads on both DOP paths.
#[cfg(feature = "toolchain")]
#[test]
fn dop_paths_pass_real_object_wait_and_hazard_replay() {
    use hipfire_isa::{kernels::gdn_scan, pm_check, toolchain::Toolchain, Arch};
    use std::process::Command;
    let emitted = gdn_scan::emit(Arch::Gfx1201).unwrap();
    let dir = std::env::temp_dir().join(format!("hipfire-isa-gdn-scan-cert-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("gdn_scan.s");
    std::fs::write(&source, &emitted.s_text).unwrap();
    // Exercise the byte-exact lifter directly, as the gfx12 text parse-back
    // currently rejects hipcc's f16 high-half op_sel spelling.
    let toolchain = Toolchain::default();
    let object = dir.join("gdn_scan.o");
    let elf = dir.join("gdn_scan.co");
    assert!(Command::new(&toolchain.llvm_mc)
        .args(["-triple=amdgcn-amd-amdhsa", "-mcpu=gfx1201", "-filetype=obj"])
        .arg(&source).arg("-o").arg(&object).status().unwrap().success());
    assert!(Command::new(&toolchain.linker)
        .arg("-shared").arg(&object).arg("-o").arg(&elf).status().unwrap().success());
    let summary = pm_check::m7(&elf, "gfx1201", gdn_scan::SYMBOL).unwrap();
    assert_eq!(summary["lift"], "byte-exact");
    assert_eq!(summary["obligations"], serde_json::json!({}));
    std::fs::remove_dir_all(dir).unwrap();
}
