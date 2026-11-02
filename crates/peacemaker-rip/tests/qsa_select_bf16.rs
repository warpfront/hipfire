use peacemaker_isa::{
    kernels::{
        iu4_gemm, qsa_score,
        qsa_select::{self, Kind, Spec},
    },
    pm_check,
    toolchain::{build, oracle_assemble_link_bundle, Toolchain},
    Arch,
};
use std::{fs, path::PathBuf};

#[path = "../../peacemaker-isa/tests/support/native_compare.rs"]
mod native_compare;

const SCORE_BF16: &str = include_str!("../../peacemaker-isa/src/kernels/qsa_score_bf16.rip");

/// The standalone BF16-pooled score RIP source reproduces the Rust emitter's
/// assembly, BuilderProofs and two-symbol (rows16, rows32) ModuleProof, native
/// code object and bundle, the ROCm oracle and the shipped bundle; both
/// symbols lift byte-exactly with no M7 obligation.
#[test]
fn standalone_qsa_select_bf16_matches_rust_and_shipped_objects() {
    let artifacts = std::env::var_os("RIP_GATE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("rip-qsa-select-bf16-{}", std::process::id()))
        });
    fs::create_dir_all(&artifacts).unwrap();
    let tools = Toolchain {
        host_target: "host-x86_64-unknown-linux-gnu-".into(),
        ..Toolchain::oracle()
    };
    let arch = Arch::Gfx1151;
    for other in [Arch::Gfx1100, Arch::Gfx1201] {
        assert!(
            ["score-bf16", "score-bf16-rows32"]
                .into_iter()
                .all(|v| peacemaker_rip::compile(SCORE_BF16, other, v).is_err()),
            "{other:?}"
        );
    }
    let rip = vec![
        peacemaker_rip::compile(SCORE_BF16, arch, "score-bf16").unwrap(),
        peacemaker_rip::compile(SCORE_BF16, arch, "score-bf16-rows32").unwrap(),
    ];
    let (rip_text, rip_proof) = iu4_gemm::module(&rip, &qsa_select::module_bf16(arch)).unwrap();
    let (rust, rust_text, rust_proof) = qsa_select::emit_module_bf16(arch).unwrap();
    assert_eq!((rip.len(), rust.len()), (2, 2), "rows16 and rows32 symbols");
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
            serde_json::to_value(&old.proof).unwrap(),
            "BuilderProof"
        );
    }
    let rs = artifacts.join("bf16-rust.s");
    let ds = artifacts.join("bf16-rip.s");
    fs::write(&rs, &rust_text).unwrap();
    fs::write(&ds, &rip_text).unwrap();
    let a = build(&tools, &rs, &artifacts.join("bf16-rust.hxaco"), arch.name()).unwrap();
    let b = build(&tools, &ds, &artifacts.join("bf16-rip.hxaco"), arch.name()).unwrap();
    let oracle = oracle_assemble_link_bundle(
        &tools,
        &ds,
        &artifacts.join("bf16-rip-oracle.hxaco"),
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
        .join(format!("{}.hxaco", qsa_select::module_bf16(arch)));
    assert_eq!(
        fs::read(&b.hsaco).unwrap(),
        fs::read(&shipped).unwrap(),
        "shipped bundle"
    );
    for (kind, symbol, expected, tag) in [
        (
            Kind::ScoreBf16,
            qsa_score::symbol_bf16(arch),
            "indexed_attention_select_scores_rows16_bf16_pm_gfx1151",
            "score_bf16",
        ),
        (
            Kind::ScoreBf16Rows32,
            qsa_score::symbol_bf16_rows32(arch),
            "indexed_attention_select_scores_rows32_bf16_pm_gfx1151",
            "score_bf16_rows32",
        ),
    ] {
        let _ = Spec { arch, kind };
        assert_eq!(symbol, expected);
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
