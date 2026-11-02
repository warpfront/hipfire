// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! PM/railgun bedrock R1: the `hipfire-kernel-abi` contract against railgun.
//!
//! The committed gfx1201 PM objects of the MQ6 SET-F32 GEMM and the grouped
//! MoE down NT4 GEMM are certified offline (`railgun-cert`), their receipt
//! evidence is compared with the contract (`check_export`), and a canonical
//! contract-validated launch is authored into a railgun node and checked
//! against the contract (`railgun::contract::check_node`). The negative
//! fixtures are deliberately broken launches, nodes and evidence the checks
//! must reject. CPU only.

use std::path::Path;
use std::sync::{Arc, LazyLock};

use hipfire_kernel_abi::families::{self, mq6_i8, qwen4_moe_sym};
use hipfire_kernel_abi::{
    check_export, eval, validate_invocation, Arch, Drift, EvidenceSource, ExportV1, FieldKind,
    InvocationV1, KernelAbiV1, ObjectRefV1, OperandBinding, ValidatedInvocationV1,
};
use railgun::contract::{check_node, launch_record, receipt_evidence};
use railgun::kernel::{author, KernelProgram, LaunchRecord, Lowering, NodeFacts};
use railgun::Binding;
use railgun_corpus::{sha256_hex, CorpusRecord, ObjectOrigin, Receipt};

const MQ6_SYMBOL: &str = "gemm_mq6g256v2_mmq_i8_pm_set_gfx1201";
const MQ6_SIBLING_SYMBOL: &str = "gemm_mq6g256v2_mmq_i8_pm_set_gfx1201_t128x256";
const MOE_SYMBOL: &str = "qwen4_moe_down_iu4_sym_pm_gfx1201_nt4";

/// Canonical launches (`M != K` so a swapped pair is visible).
const MQ6_PARAMS: &[(&str, u64)] = &[("M", 512), ("K", 256), ("N", 128)];
const MQ6_OTHER_PARAMS: &[(&str, u64)] = &[("M", 1024), ("K", 256), ("N", 128)];
const MOE_PARAMS: &[(&str, u64)] = &[
    ("M", 128),
    ("K", 256),
    ("x_row_div", 1),
    ("m_total", 32),
    ("x_src_rows", 32),
    ("n_experts", 8),
];
const MOE_OTHER_PARAMS: &[(&str, u64)] = &[
    ("M", 256),
    ("K", 256),
    ("x_row_div", 1),
    ("m_total", 32),
    ("x_src_rows", 32),
    ("n_experts", 8),
];

/// One committed object, certified, with its contract export.
struct Case {
    abi: &'static KernelAbiV1,
    export: &'static ExportV1,
    receipt: Receipt,
    record: CorpusRecord,
}

fn load(abi: &'static KernelAbiV1, symbol: &str) -> Case {
    let (found, export) = families::find_export(Arch::Gfx1201, symbol)
        .unwrap_or_else(|| panic!("no gfx1201 export {symbol} in any contract"));
    assert_eq!(found.id, abi.id, "{symbol} is owned by another contract");
    let ObjectRefV1::Embedded {
        module,
        artifact,
        sha256,
    } = export.object
    else {
        panic!("{symbol}: the contract names no committed object");
    };
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = repo.join(artifact);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(
        sha256_hex(&bytes),
        sha256,
        "{artifact}: the committed object is not the one the contract pins"
    );
    let receipt =
        railgun_cert::certify(&bytes, None).unwrap_or_else(|e| panic!("{artifact}: certify: {e}"));
    assert_eq!(receipt.object_sha256, sha256);
    let receipt_bytes = serde_json::to_vec(&receipt).expect("receipt serializes");
    let record = CorpusRecord {
        arch: "gfx1201".into(),
        artifact_sha256: sha256.into(),
        module: module.into(),
        origin: ObjectOrigin::PmNative,
        symbols: receipt
            .railgun
            .kernels
            .iter()
            .map(|k| k.symbol.clone())
            .collect(),
        source_sha256: String::new(),
        extra_flags: String::new(),
        flags: Vec::new(),
        scheduler_profile: String::new(),
        packaging_key: String::new(),
        jit_cache_key: String::new(),
        argv: Vec::new(),
        object: artifact.into(),
        receipt: format!("receipts/gfx1201/{sha256}.receipt.json"),
        receipt_sha256: sha256_hex(&receipt_bytes),
        kernels: Vec::new(),
    };
    Case {
        abi,
        export,
        receipt,
        record,
    }
}

static MQ6_CASE: LazyLock<Case> = LazyLock::new(|| load(&mq6_i8::MQ6_I8_PM_SET_F32, MQ6_SYMBOL));
static MOE_CASE: LazyLock<Case> =
    LazyLock::new(|| load(&qwen4_moe_sym::QWEN4_MOE_SYM_DOWN, MOE_SYMBOL));

fn mq6() -> &'static Case {
    &MQ6_CASE
}

fn moe() -> &'static Case {
    &MOE_CASE
}

/// A validated canonical launch and the allocation behind each operand.
struct Launch {
    v: ValidatedInvocationV1<'static>,
    /// Operand order.
    allocations: Vec<Binding>,
}

/// Fake but consistent addresses: operand `i` points 256 bytes into its own
/// allocation, which holds its contract extent plus slack.
fn launch(abi: &'static KernelAbiV1, export: &'static ExportV1, named: &[(&str, u64)]) -> Launch {
    let params: Vec<u64> = abi
        .params
        .iter()
        .map(|p| {
            named
                .iter()
                .find(|(name, _)| *name == p.name)
                .unwrap_or_else(|| panic!("{}: no value for param {}", abi.id, p.name))
                .1
        })
        .collect();
    let interior = 256u64;
    let mut allocations = Vec::new();
    let mut bindings = Vec::new();
    for (i, op) in abi.operands.iter().enumerate() {
        let need = op.extent.map_or(Ok(0), |e| eval(&e, abi, export, &params));
        let need = need.unwrap_or_else(|e| panic!("{} extent of {}: {e}", abi.id, op.name));
        let base = 0x10_0000_0000u64 + (i as u64) * 0x1_0000_0000;
        let bytes = need + interior + 4096;
        allocations.push(Binding { base, bytes });
        bindings.push(OperandBinding {
            address: base + interior,
            bytes: Some(bytes - interior),
            format: op.format,
        });
    }
    let grid = export.grid.map(|e| {
        u32::try_from(eval(&e, abi, export, &params).expect("grid expression")).expect("grid fits")
    });
    let v = validate_invocation(
        abi,
        &InvocationV1 {
            arch: export.arch,
            symbol: export.symbol,
            params: &params,
            operands: &bindings,
            grid,
            block: export.block,
            dynamic_lds: export.dynamic_lds,
        },
    )
    .unwrap_or_else(|e| panic!("{}: canonical launch rejected: {e}", export.symbol));
    Launch { v, allocations }
}

/// The launch record of `l`: its pointer fields probed against the operands'
/// own allocations.
fn record_of(l: &Launch, facts: Result<Arc<NodeFacts>, String>) -> LaunchRecord {
    let abi = l.v.abi();
    let allocations: Vec<(u32, Result<Binding, String>)> = abi
        .fields
        .iter()
        .filter_map(|f| match f.kind {
            FieldKind::Pointer { operand } => {
                let i = abi.operand_index(operand).expect("declared operand");
                Some((f.offset, Ok(l.allocations[i])))
            }
            FieldKind::Scalar { .. } => None,
        })
        .collect();
    launch_record(&l.v, facts, allocations, Vec::new())
}

/// The facts exactly as the receipt states them.
fn real_facts(case: &Case) -> Result<Arc<NodeFacts>, String> {
    NodeFacts::from_hit(&case.record, &case.receipt, case.export.symbol).map(Arc::new)
}

/// TEST FIXTURE ONLY: the receipt's facts with the A1 verdict forced to
/// `proven` (and no unresolved/failed-differential refusal), so a kernel the
/// real receipt does not prove still yields a decomposed node whose
/// *contract comparison* the negative fixtures can exercise. Never used for
/// the positive claim about the real receipt.
fn forced_facts(case: &Case) -> Result<Arc<NodeFacts>, String> {
    let mut receipt = case.receipt.clone();
    for k in receipt
        .railgun
        .kernels
        .iter_mut()
        .filter(|k| k.symbol == case.export.symbol)
    {
        k.status = "proven".to_owned();
        k.unresolved.clear();
        k.differential.status = "pass".to_owned();
    }
    NodeFacts::from_hit(&case.record, &receipt, case.export.symbol).map(Arc::new)
}

fn refused(program: &KernelProgram) -> Option<&str> {
    match &program.nodes[0].lowering {
        Lowering::HipDirect { reason } => Some(reason),
        _ => None,
    }
}

/// The canonical launch, authored from the forced facts, must pass
/// `check_node` (the control of every negative fixture).
fn control(case: &Case, params: &[(&str, u64)]) -> (Launch, LaunchRecord) {
    let l = launch(case.abi, case.export, params);
    let rec = record_of(&l, forced_facts(case));
    let program = author("gfx1201", std::slice::from_ref(&rec));
    check_node(&program, 0, &l.v).unwrap_or_else(|e| {
        panic!(
            "{}: control (forced-proven facts, canonical launch) must pass check_node: {e}",
            case.export.symbol
        )
    });
    (l, rec)
}

fn scalar_offset(abi: &KernelAbiV1, name: &str) -> usize {
    abi.fields
        .iter()
        .find(|f| f.name == name && matches!(f.kind, FieldKind::Scalar { .. }))
        .unwrap_or_else(|| panic!("{}: no scalar field {name}", abi.id))
        .offset as usize
}

fn positive(case: &Case, params: &[(&str, u64)]) {
    let sym = case.export.symbol;

    // Receipt evidence agrees with the contract.
    let evidence = receipt_evidence(&case.receipt, sym).expect("receipt evidence");
    assert_eq!(evidence.source, EvidenceSource::RailgunReceipt);
    assert_eq!(evidence.symbol, sym);
    let drift = check_export(case.abi, case.export, &evidence);
    assert!(
        drift.is_empty(),
        "{sym}: receipt evidence drifts from {}: {drift:#?}",
        case.abi.id
    );

    // The canonical launch, authored from the real facts.
    let l = launch(case.abi, case.export, params);
    let rec = record_of(&l, real_facts(case));
    assert_eq!(rec.kernarg.len() % 16, 0, "{sym}: kernarg is not 16-padded");
    assert!(rec.kernarg.len() >= l.v.explicit().len());
    assert_eq!(&rec.kernarg[..l.v.explicit().len()], l.v.explicit());
    assert!(rec.kernarg[l.v.explicit().len()..].iter().all(|b| *b == 0));
    let program = author("gfx1201", std::slice::from_ref(&rec));
    let result = check_node(&program, 0, &l.v);
    match refused(&program) {
        // `author` refused the real receipt's facts: check_node refuses too,
        // naming the refusal. The reason is recorded here and in stderr.
        Some(reason) => {
            eprintln!("kernel_abi_contract: {sym}: author refused the launch: {reason}");
            let err = result.expect_err("a refused node must not pass check_node");
            assert!(
                err.contains(reason),
                "{sym}: author refused ({reason}); check_node reported: {err}"
            );
        }
        None => result.unwrap_or_else(|e| {
            panic!(
                "{sym}: lowering {:?}: check_node rejects the authored node: {e}",
                program.nodes[0].lowering
            )
        }),
    }
}

#[test]
fn mq6_pm_set_f32_receipt_evidence_and_authored_node_agree_with_the_contract() {
    positive(mq6(), MQ6_PARAMS);
}

#[test]
fn moe_down_nt4_receipt_evidence_and_authored_node_agree_with_the_contract() {
    positive(moe(), MOE_PARAMS);
}

/// Negative 1: a launch record whose kernarg has M and K swapped, otherwise
/// identical, is rejected on its constant bytes.
#[test]
fn swapped_m_and_k_in_the_recorded_kernarg_is_rejected() {
    for (case, params) in [(mq6(), MQ6_PARAMS), (moe(), MOE_PARAMS)] {
        let sym = case.export.symbol;
        let (l, rec) = control(case, params);
        let (m, k) = (scalar_offset(case.abi, "M"), scalar_offset(case.abi, "K"));
        let mut bad = rec.clone();
        for i in 0..4 {
            bad.kernarg.swap(m + i, k + i);
        }
        assert_ne!(
            bad.kernarg, rec.kernarg,
            "{sym}: M == K, the swap is a no-op"
        );
        let program = author("gfx1201", &[bad]);
        assert!(
            refused(&program).is_none(),
            "{sym}: the swapped launch must still author, so the contract check is what rejects it: {:?}",
            refused(&program)
        );
        let err = check_node(&program, 0, &l.v).expect_err("swapped M/K must be rejected");
        assert!(
            err.contains("field M") || err.contains("field K"),
            "{sym}: unexpected rejection: {err}"
        );
    }
}

/// Negative 1b: an output pointer moved inside its own allocation is rejected
/// on the binding, not on the bytes alone.
#[test]
fn a_moved_output_pointer_is_rejected() {
    for (case, params) in [(mq6(), MQ6_PARAMS), (moe(), MOE_PARAMS)] {
        let sym = case.export.symbol;
        let (l, rec) = control(case, params);
        let y = case
            .abi
            .fields
            .iter()
            .rfind(|f| matches!(f.kind, FieldKind::Pointer { .. }))
            .expect("a pointer field")
            .offset as usize;
        let mut bad = rec.clone();
        let address = u64::from_le_bytes(bad.kernarg[y..y + 8].try_into().unwrap()) + 16;
        bad.kernarg[y..y + 8].copy_from_slice(&address.to_le_bytes());
        let program = author("gfx1201", &[bad]);
        assert!(
            refused(&program).is_none(),
            "{sym}: the moved pointer must still author: {:?}",
            refused(&program)
        );
        let err = check_node(&program, 0, &l.v).expect_err("a moved pointer must be rejected");
        assert!(
            err.contains("pointer field"),
            "{sym}: unexpected rejection: {err}"
        );
    }
}

/// Negative 2: a validated invocation of another export, another contract or
/// other params checked against the node is rejected.
#[test]
fn an_invocation_of_another_export_contract_or_shape_is_rejected() {
    let (mq6_case, moe_case) = (mq6(), moe());
    let (mq6_launch, mq6_rec) = control(mq6_case, MQ6_PARAMS);
    let (moe_launch, moe_rec) = control(moe_case, MOE_PARAMS);
    let mq6_program = author("gfx1201", &[mq6_rec]);
    let moe_program = author("gfx1201", &[moe_rec]);
    check_node(&mq6_program, 0, &mq6_launch.v).expect("MQ6 control");
    check_node(&moe_program, 0, &moe_launch.v).expect("MoE control");

    // Another contract.
    let err = check_node(&mq6_program, 0, &moe_launch.v).expect_err("MoE launch vs MQ6 node");
    assert!(err.contains("not the contract export"), "{err}");
    let err = check_node(&moe_program, 0, &mq6_launch.v).expect_err("MQ6 launch vs MoE node");
    assert!(err.contains("not the contract export"), "{err}");

    // Another export of the same contract.
    let sibling = mq6_case
        .abi
        .export(Arch::Gfx1201, MQ6_SIBLING_SYMBOL)
        .expect("t128x256 export");
    let other = launch(mq6_case.abi, sibling, MQ6_PARAMS);
    let err = check_node(&mq6_program, 0, &other.v).expect_err("sibling export vs node");
    assert!(err.contains("not the contract export"), "{err}");

    // The same export at another shape.
    let other = launch(mq6_case.abi, mq6_case.export, MQ6_OTHER_PARAMS);
    check_node(&mq6_program, 0, &other.v).expect_err("MQ6 other shape vs node");
    let other = launch(moe_case.abi, moe_case.export, MOE_OTHER_PARAMS);
    check_node(&moe_program, 0, &other.v).expect_err("MoE other shape vs node");
}

/// Negative 3: the MoE object's receipt evidence against the MQ6 contract
/// drifts; so do single deliberate corruptions of the MQ6 evidence.
#[test]
fn receipt_evidence_of_another_contract_and_corrupted_evidence_drift() {
    let (mq6_case, moe_case) = (mq6(), moe());
    let mq6_export = mq6_case
        .abi
        .export(Arch::Gfx1201, MQ6_SYMBOL)
        .expect("MQ6 export");

    let moe_evidence = receipt_evidence(&moe_case.receipt, MOE_SYMBOL).expect("MoE evidence");
    let drift = check_export(&mq6_i8::MQ6_I8_PM_SET_F32, mq6_export, &moe_evidence);
    assert!(
        !drift.is_empty(),
        "MoE object evidence must drift from the MQ6 contract"
    );
    assert!(
        drift.iter().any(|d| matches!(d, Drift::Symbol { .. })),
        "{drift:#?}"
    );

    let good = receipt_evidence(&mq6_case.receipt, MQ6_SYMBOL).expect("MQ6 evidence");
    assert!(check_export(mq6_case.abi, mq6_export, &good).is_empty());

    let mut bad = good.clone();
    bad.args.iter_mut().find(|a| a.offset == 0).unwrap().size = 4;
    assert!(check_export(mq6_case.abi, mq6_export, &bad)
        .iter()
        .any(|d| matches!(d, Drift::ArgWidth { offset: 0, .. })));

    let mut bad = good.clone();
    bad.args
        .iter_mut()
        .find(|a| a.offset == 24)
        .unwrap()
        .value_kind = "global_buffer".into();
    assert!(check_export(mq6_case.abi, mq6_export, &bad)
        .iter()
        .any(|d| matches!(d, Drift::ArgKind { offset: 24, .. })));

    let mut bad = good.clone();
    bad.segment_bytes = Some(good.segment_bytes.expect("segment") + 8);
    assert!(check_export(mq6_case.abi, mq6_export, &bad)
        .iter()
        .any(|d| matches!(d, Drift::Segment { .. })));

    let mut bad = good.clone();
    bad.object_sha256 = Some("0".repeat(64));
    assert!(check_export(mq6_case.abi, mq6_export, &bad)
        .iter()
        .any(|d| matches!(d, Drift::ObjectSha { .. })));

    let mut bad = good;
    bad.args.retain(|a| a.offset != 16);
    assert!(check_export(mq6_case.abi, mq6_export, &bad)
        .iter()
        .any(|d| matches!(d, Drift::MissingArg { offset: 16, .. })));
}
