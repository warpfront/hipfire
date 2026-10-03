use hipfire_isa::{Arch, kernels::qsa_gather::{self, Kind, Spec}, pm_check,
    toolchain::{build, oracle_assemble_link_bundle, Toolchain}};
use std::{fs, path::PathBuf};

#[path = "../../hipfire-isa/tests/support/native_compare.rs"]
mod native_compare;

const SOURCE: &str = include_str!("../../hipfire-isa/src/kernels/qsa_gather.rip");

#[test]
fn standalone_qsa_matches_rust_and_shipped_objects() {
    let artifacts = std::env::var_os("RIP_GATE_DIR").map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join(format!("rip-qsa-{}", std::process::id())));
    fs::create_dir_all(&artifacts).unwrap();
    // Reproduce the shipped recipe, including hipcc's trailing-hyphen host target.
    let tools = Toolchain { host_target: "host-x86_64-unknown-linux-gnu-".into(), ..Toolchain::oracle() };
    for arch in [Arch::Gfx1151, Arch::Gfx1201] {
        let (rust, rust_text, rust_proof) = qsa_gather::emit_module(arch).unwrap();
        let (rip, rip_text, rip_proof) = hipfire_rip::module(SOURCE, arch,
            &["convert", "attend"], &Spec::module(arch)).unwrap();
        assert_eq!(rip_text, rust_text, "{} assembly", arch.name());
        assert_eq!(serde_json::to_value(&rip_proof).unwrap(), serde_json::to_value(&rust_proof).unwrap(),
            "{} ModuleProof", arch.name());
        for (new, old) in rip.iter().zip(&rust) {
            assert_eq!(new.s_text, old.s_text);
            assert_eq!(serde_json::to_value(&new.proof).unwrap(), serde_json::to_value(&old.proof).unwrap());
        }
        let dir = artifacts.join(arch.name());
        fs::create_dir_all(&dir).unwrap();
        let rs = dir.join("rust.s");
        let ds = dir.join("rip.s");
        fs::write(&rs, &rust_text).unwrap();
        fs::write(&ds, &rip_text).unwrap();
        let a = build(&tools, &rs, &dir.join("rust.hxaco"), arch.name()).unwrap();
        let b = build(&tools, &ds, &dir.join("rip.hxaco"), arch.name()).unwrap();
        // The native writer against the ROCm oracle on the `.rip` module.
        let oracle = oracle_assemble_link_bundle(&tools, &ds, &dir.join("rip-oracle.hxaco"), arch.name()).unwrap();
        let difference = native_compare::compare_native_to_oracle(&fs::read(&b.elf).unwrap(), &fs::read(&b.hsaco).unwrap(),
            &fs::read(&oracle.elf).unwrap(), &fs::read(&oracle.hsaco).unwrap());
        assert!(difference.is_none(), "{} native code object/bundle vs llvm-mc + ld.lld + clang-offload-bundler: {}", arch.name(), difference.unwrap_or_default());
        assert_eq!(fs::read(&a.elf).unwrap(), fs::read(&b.elf).unwrap(), "code object");
        assert_eq!(fs::read(&a.hsaco).unwrap(), fs::read(&b.hsaco).unwrap(), "bundle");
        let shipped = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../kernels")
            .join(format!("qsa_gather_pm_{}.hxaco", arch.name()));
        assert_eq!(fs::read(&b.hsaco).unwrap(), fs::read(&shipped).unwrap(), "shipped bundle {}", arch.name());
        for kind in Kind::ALL {
            let symbol = Spec { arch, kind }.symbol();
            let report = pm_check::m7(&b.elf, arch.name(), &symbol).unwrap();
            assert_eq!(report["lift"], "byte-exact");
            assert_eq!(report["obligations"], serde_json::json!({}));
            fs::write(dir.join(format!("{}.cert.json", kind.tag())), serde_json::to_vec_pretty(&report).unwrap()).unwrap();
            println!("PASS {} {}: .s BuilderProof ModuleProof native .co .hxaco = ROCm oracle = shipped bundle; pm_check obligations=0", arch.name(), kind.tag());
        }
    }
}

#[test]
fn race_twins_and_escape_seams_are_rejected() {
    let fixtures = [
        ("halo_verify_attn_race", include_str!("ui/halo_verify_attn_race.rip"), include_str!("ui/halo_verify_attn_fixed.rip")),
        ("fa2_mailbox_race", include_str!("ui/fa2_mailbox_race.rip"), include_str!("ui/fa2_mailbox_fixed.rip")),
        ("raw_escape_barrier", include_str!("ui/raw_escape_barrier.rip"), include_str!("ui/raw_checked_barrier.rip")),
        ("loop_carried_undrained_store", include_str!("ui/loop_carried_undrained_store.rip"), include_str!("ui/loop_carried_drained_store.rip")),
    ];
    for arch in [Arch::Gfx1151, Arch::Gfx1201] {
        for (name, source, fixed) in fixtures {
            let corrected = hipfire_rip::compile(fixed, arch, "negative").unwrap_or_else(|e| panic!("{name} fixed: {e}"));
            assert!(corrected.s_text.contains(if arch.gfx12() { "s_barrier_signal" } else { "s_barrier" }) || name == "loop_carried_undrained_store");
            let error = hipfire_rip::compile(source, arch, "negative").err().expect(name);
            println!("PASS {} {name}: negative rejected ({error}); fixed control emitted successfully", arch.name());
        }
    }
}
