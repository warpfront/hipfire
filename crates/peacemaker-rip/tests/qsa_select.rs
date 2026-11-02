use peacemaker_isa::{
    kernels::{
        iu4_gemm, qsa_score,
        qsa_select::{self, Kind, Spec},
        qsa_topk,
    },
    pm_check,
    toolchain::{build, oracle_assemble_link_bundle, Toolchain},
    Arch,
};
use std::{fs, path::PathBuf};

#[path = "../../peacemaker-isa/tests/support/native_compare.rs"]
mod native_compare;

const SCORE: &str = include_str!("../../peacemaker-isa/src/kernels/qsa_score.rip");
const TOPK: &str = include_str!("../../peacemaker-isa/src/kernels/qsa_topk.rip");

/// The two standalone RIP sources, compiled independently and joined in the
/// fixed module order, reproduce the Rust emitters' assembly, BuilderProofs,
/// ModuleProof, native code object and bundle, the ROCm oracle and the
/// shipped bundle; both symbols lift byte-exactly with no M7 obligation.
#[test]
fn standalone_qsa_select_matches_rust_and_shipped_objects() {
    let artifacts = std::env::var_os("RIP_GATE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("rip-qsa-select-{}", std::process::id()))
        });
    fs::create_dir_all(&artifacts).unwrap();
    let tools = Toolchain {
        host_target: "host-x86_64-unknown-linux-gnu-".into(),
        ..Toolchain::oracle()
    };
    let arch = Arch::Gfx1151;
    for other in [Arch::Gfx1100, Arch::Gfx1201] {
        assert!(
            ["score", "score32"]
                .into_iter()
                .all(|v| peacemaker_rip::compile(SCORE, other, v).is_err())
                && peacemaker_rip::compile(TOPK, other, "select").is_err()
        );
    }
    let rip = vec![
        peacemaker_rip::compile(SCORE, arch, "score").unwrap(),
        peacemaker_rip::compile(SCORE, arch, "score32").unwrap(),
        peacemaker_rip::compile(TOPK, arch, "select").unwrap(),
    ];
    let (rip_text, rip_proof) = iu4_gemm::module(&rip, &qsa_select::module(arch)).unwrap();
    let (rust, rust_text, rust_proof) = qsa_select::emit_module(arch).unwrap();
    assert_eq!(rip_text, rust_text, "assembly");
    assert_eq!(
        serde_json::to_value(&rip_proof).unwrap(),
        serde_json::to_value(&rust_proof).unwrap(),
        "ModuleProof"
    );
    for (new, old) in rip.iter().zip(&rust) {
        assert_eq!(new.s_text, old.s_text);
        assert_eq!(
            serde_json::to_value(&new.proof).unwrap(),
            serde_json::to_value(&old.proof).unwrap()
        );
    }
    let rs = artifacts.join("rust.s");
    let ds = artifacts.join("rip.s");
    fs::write(&rs, &rust_text).unwrap();
    fs::write(&ds, &rip_text).unwrap();
    let a = build(&tools, &rs, &artifacts.join("rust.hxaco"), arch.name()).unwrap();
    let b = build(&tools, &ds, &artifacts.join("rip.hxaco"), arch.name()).unwrap();
    let oracle = oracle_assemble_link_bundle(
        &tools,
        &ds,
        &artifacts.join("rip-oracle.hxaco"),
        arch.name(),
    )
    .unwrap();
    let difference = native_compare::compare_native_to_oracle(
        &fs::read(&b.elf).unwrap(),
        &fs::read(&b.hsaco).unwrap(),
        &fs::read(&oracle.elf).unwrap(),
        &fs::read(&oracle.hsaco).unwrap(),
    );
    assert!(
        difference.is_none(),
        "native code object/bundle vs llvm-mc + ld.lld + clang-offload-bundler: {}",
        difference.unwrap_or_default()
    );
    assert_eq!(
        fs::read(&a.elf).unwrap(),
        fs::read(&b.elf).unwrap(),
        "code object"
    );
    assert_eq!(
        fs::read(&a.hsaco).unwrap(),
        fs::read(&b.hsaco).unwrap(),
        "bundle"
    );
    let shipped = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels")
        .join(format!("{}.hxaco", qsa_select::module(arch)));
    assert_eq!(
        fs::read(&b.hsaco).unwrap(),
        fs::read(&shipped).unwrap(),
        "shipped bundle"
    );
    for kind in Kind::ALL {
        let _ = Spec { arch, kind };
        let (symbol, tag) = match kind {
            Kind::Score => (qsa_score::symbol(arch), "score"),
            Kind::Score32 => (qsa_score::symbol32(arch), "score32"),
            Kind::ScoreBf16 | Kind::ScoreBf16Rows32 => unreachable!("F32 module test"),
            Kind::Select => (qsa_topk::symbol(arch), "select"),
        };
        let report = pm_check::m7(&b.elf, arch.name(), &symbol).unwrap();
        assert_eq!(report["lift"], "byte-exact");
        assert_eq!(report["obligations"], serde_json::json!({}));
        fs::write(
            artifacts.join(format!("{tag}.cert.json")),
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
        println!("PASS gfx1151 {tag}: .s BuilderProof ModuleProof native .co .hxaco = ROCm oracle = shipped bundle; pm_check obligations=0");
    }
}
