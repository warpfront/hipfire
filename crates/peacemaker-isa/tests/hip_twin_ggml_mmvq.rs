// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU-free gate of the portable Route-A twin of the gfx1201 GGUF-native decode GEMV shim
//! (`kernels/ggml_mmvq_shim_pm_gfx1201.hxaco`, sealed by
//! `scripts/pm-leviathan-twin.py --family ggml-mmvq`). The anchor is the shipped runtime object
//! `kernels/ggml_mmvq_shim_gfx1201.co` itself (`rdna-compute` `ggml_mmvq::S1_IMAGE`); the twin
//! carries every export of `kernels/src/ggml_mmvq_shim.hip` and the complete device ELF is
//! byte-identical to the anchor's. Proof only: dispatch stays on the HIP object.
use peacemaker_ir::envelope::Bundle;
use peacemaker_ir::inst::{Arch as IrArch, Frontend, Kernel};
use peacemaker_isa::arch::Arch;
use peacemaker_isa::hip_twin::{seal, verify_hxaco_many};
use peacemaker_isa::insn::{Instruction, Program};
use peacemaker_isa::ledger_cfg;
use peacemaker_lift::bundle::BundleCodec;
use peacemaker_lift::{lift_object, Options};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

const ANCHOR: &[u8] = include_bytes!("../../../kernels/ggml_mmvq_shim_gfx1201.co");
const HXACO: &[u8] = include_bytes!("../../../kernels/ggml_mmvq_shim_pm_gfx1201.hxaco");
const MANIFEST: &str = include_str!("../../../kernels/ggml_mmvq_shim_pm_gfx1201.twin.json");
const SHA: &str = include_str!("../../../kernels/ggml_mmvq_shim_pm_gfx1201.hxaco.sha256");
const SOURCE: &[u8] = include_bytes!("../../../kernels/src/ggml_mmvq_shim.hip");
const PIN: &[u8] = include_bytes!("../../../third_party/ggml/PIN");

/// Every `extern "C" __global__` export of `ggml_mmvq_shim.hip`, in source order.
const SYMBOLS: [&str; 5] = [
    "ggml_q8_1_quantize_gfx1201",
    "ggml_mmvq_iq4_xs_gfx1201",
    "ggml_mmvq_q4_k_gfx1201",
    "ggml_mmvq_q6_k_gfx1201",
    "ggml_mmvq_q8_0_gfx1201",
];

fn hex(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

fn device(bytes: &[u8]) -> &[u8] {
    let (bundle, payloads) = Bundle::read(bytes).unwrap();
    payloads[bundle.device_entry(IrArch::Gfx1201).unwrap()]
}

/// `(offset, size, pointer)` of the explicit (non-hidden) kernargs of a seal report.
fn explicit(report: &Value) -> Vec<(u64, u64, bool)> {
    report["abi"]["args"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|a| !a["value_kind"].as_str().unwrap().starts_with("hidden_"))
        .map(|a| {
            (
                a["offset"].as_u64().unwrap(),
                a["size"].as_u64().unwrap(),
                a["pointer"].as_bool().unwrap(),
            )
        })
        .collect()
}

#[test]
fn provenance_and_portability() {
    let m: Value = serde_json::from_str(MANIFEST).unwrap();
    assert_eq!(m["arch"], "gfx1201");
    assert_eq!(m["module"], "ggml_mmvq_shim_pm_gfx1201");
    assert_eq!(m["symbols"], serde_json::json!(SYMBOLS));
    assert_eq!(m["source"]["path"], "kernels/src/ggml_mmvq_shim.hip");
    assert_eq!(m["source"]["sha256"], hex(SOURCE));
    assert_eq!(m["source"]["bytes"], SOURCE.len());
    assert_eq!(m["third_party"][0]["path"], "third_party/ggml/PIN");
    assert_eq!(m["third_party"][0]["sha256"], hex(PIN));
    assert_eq!(m["third_party"][0]["bytes"], PIN.len());
    // The anchor is the shipped runtime object, not a second copy.
    assert_eq!(m["anchor"]["file"], "ggml_mmvq_shim_gfx1201.co");
    assert_eq!(m["anchor"]["sha256"], hex(ANCHOR));
    assert_eq!(m["anchor"]["bytes"], ANCHOR.len());
    assert_eq!(m["hxaco"]["sha256"], hex(HXACO));
    assert_eq!(m["hxaco"]["bytes"], HXACO.len());
    assert_eq!(SHA.trim(), hex(HXACO));
    // Recipe of scripts/build-ggml-mmvq-shim.sh, include paths repo-relative.
    assert_eq!(
        m["compiler"]["flags"],
        serde_json::json!([
            "--genco",
            "--offload-arch=gfx1201",
            "-O3",
            "--no-offload-compress",
            "-fuse-cuid=none",
            "-mcode-object-version=6",
            "-ffp-contract=fast-honor-pragmas",
            "-DGGML_USE_HIP",
            "-Ithird_party/ggml/include",
            "-Ithird_party/ggml/src",
            "-Ithird_party/ggml/src/ggml-cuda"
        ])
    );
}

/// The twin seals every kernel of the anchor: no subset is sealed.
#[test]
fn twin_covers_every_export_of_the_anchor() {
    let lifted = lift_object(
        ANCHOR,
        Options {
            frontend: Frontend::Hipcc,
        },
    )
    .unwrap();
    let mut anchor_symbols: Vec<&str> = lifted
        .program
        .kernels
        .iter()
        .map(|k| k.symbol.0.as_str())
        .collect();
    anchor_symbols.sort_unstable();
    let mut want = SYMBOLS.to_vec();
    want.sort_unstable();
    assert_eq!(anchor_symbols, want);
}

#[test]
fn full_elf_identity_and_explicit_abi() {
    assert_eq!(device(ANCHOR), device(HXACO), "full device ELF identity");
    let m: Value = serde_json::from_str(MANIFEST).unwrap();
    let verified = verify_hxaco_many(HXACO, Arch::Gfx1201, &SYMBOLS).unwrap();
    // producer: x, vy pointers, ne00, s01, ne0 (hidden args follow, segment 288);
    // mmvq: vx, vy, dst pointers, K (no hidden args, segment 28, 7 * 32 * 4 B static LDS).
    let producer = (
        vec![
            (0, 8, true),
            (8, 8, true),
            (16, 4, false),
            (20, 4, false),
            (24, 4, false),
        ],
        288,
        0,
    );
    let mmvq = (
        vec![(0, 8, true), (8, 8, true), (16, 8, true), (24, 4, false)],
        28,
        896,
    );
    let wants = [&producer, &mmvq, &mmvq, &mmvq, &mmvq];
    for (i, (symbol, v)) in SYMBOLS.iter().zip(verified).enumerate() {
        let sealed = seal(ANCHOR, Arch::Gfx1201, symbol, Some((ANCHOR, symbol))).unwrap();
        assert_eq!(sealed.hxaco, HXACO, "{symbol}: resealing moves the twin");
        assert_eq!(v.hxaco, HXACO);
        assert_eq!(v.elf_sha256, hex(device(ANCHOR)));
        assert_eq!(v.scratch, 0, "{symbol}: scratch");
        assert_eq!(m["seal"]["reports"][i]["kernel"], *symbol);
        assert_eq!(m["seal"]["reports"][i]["text_sha256"], v.text_sha256);
        assert_eq!(m["seal"]["reports"][i]["hazards"]["blocking"], 0);
        assert_eq!(m["seal"]["reports"][i]["hazards"]["pm_ir_obligations"], 0);
        let (args, kernarg_segment, lds) = wants[i];
        assert_eq!(&explicit(&v.report), args, "{symbol}: explicit kernargs");
        assert_eq!(v.report["abi"]["kernarg_segment_size"], *kernarg_segment);
        assert_eq!(v.lds, *lds, "{symbol}: static LDS");
    }
}

#[test]
fn wrong_arch_and_symbol_fail_closed() {
    assert!(verify_hxaco_many(HXACO, Arch::Gfx1200, &SYMBOLS).is_err());
    assert!(verify_hxaco_many(HXACO, Arch::Gfx1201, &["ggml_mmvq_no_such_export"]).is_err());
}

/// One kernel as label + canonical instruction lines in layout order (the spelling of
/// `ledger_cfg_pins.rs`'s `kernel_program`).
fn kernel_program(kernel: &Kernel) -> Program {
    let body = &kernel.body;
    let starts: BTreeMap<usize, usize> = body.blocks.iter().map(|b| (b.range.0, b.id.0)).collect();
    let mut instructions = Vec::new();
    for (index, &id) in body.layout.iter().enumerate() {
        if let Some(block) = starts.get(&index) {
            instructions.push(Instruction::new(format!(".LBB{block}:"), vec![], vec![]));
        }
        let inst = body.insts.get(id).unwrap();
        let text = peacemaker_lift::text::canonical(inst, IrArch::Gfx1201).unwrap();
        instructions.push(Instruction::new(text, vec![], vec![]));
    }
    Program {
        arch: Arch::Gfx1201,
        instructions,
    }
}

/// Strict CFG wait verdict of every kernel of `bytes`, by symbol.
fn ledger(bytes: &[u8], frontend: Frontend) -> BTreeMap<String, Result<(), String>> {
    let lifted = lift_object(bytes, Options { frontend }).unwrap();
    lifted
        .program
        .kernels
        .iter()
        .map(|k| {
            let p = kernel_program(k);
            let verdict = ledger_cfg::verify(&p, &[0..p.instructions.len()]).map(|_| ());
            (k.symbol.0.clone(), verdict)
        })
        .collect()
}

/// OPEN, pinned negatively: the strict wait ledger verifies four of the five exports and
/// reports one `d16_hi` load use on `ggml_mmvq_q4_k_gfx1201`, identically in the anchor (hipcc
/// frontend) and the twin (builder frontend). The report is not claimed a race and is not
/// discharged by byte identity; no allowlist row is added for the twin image. This test fails
/// if the verdict changes in either direction, so a verifier change must update the pin.
#[test]
fn ledger_verdict_is_pinned_open_for_q4_k_only() {
    const OPEN_KERNEL: &str = "ggml_mmvq_q4_k_gfx1201";
    const OPEN_FINDING: &str = "`global_load_d16_hi_b16 v10, v[16:17], off offset:12` (after label `.LBB2`): needs Load wait 0 (Use { reg: \"v10\" })";
    let anchor = ledger(ANCHOR, Frontend::Hipcc);
    let twin = ledger(HXACO, Frontend::Builder);
    assert_eq!(anchor, twin, "anchor and twin ledger verdicts differ");
    for symbol in SYMBOLS {
        match (symbol == OPEN_KERNEL, &twin[symbol]) {
            (false, Ok(())) => {}
            (true, Err(e)) => assert!(e.contains(OPEN_FINDING), "{symbol}: {e}"),
            (open, v) => panic!("{symbol}: expected open={open}, got {v:?}"),
        }
    }
}
