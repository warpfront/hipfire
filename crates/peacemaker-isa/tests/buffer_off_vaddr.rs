// SPDX-License-Identifier: Apache-2.0
//! MUBUF `off` form (OFFEN = IDXEN = 0): the unused VADDR byte must lift as `Operand::Vmem(VmemToken::Off)`, not as a
//! VGPR pair. A `v[0:1]` operand makes `wait-raw-vmem-load` see a read of v0/v1 and flag a pending load into v1.
//!
//! The fixture is the frozen linked object of the F01 g32 variant `r4_d2_dx1_b64_anchor` (76,536 bytes, sha256
//! 58b08f06...), which contains both `off`-form loads and a pending load into v1 (`buffer_load_b32 v1, off, ...`
//! followed by `buffer_load_b32 v14, off, ...`).

use peacemaker_ir::inst::{Form, Frontend, VmemForm};
use peacemaker_ir::operand::{Operand, VmemToken};
use peacemaker_ir::Arch;
use sha2::{Digest, Sha256};

const OBJ: &[u8] = include_bytes!("fixtures/buffer_off_vaddr/g32_r4_d2_dx1_b64_anchor.co");
const SYMBOL: &str = "gemm_mq4g256v2p_moe_grouped_top10_g32_r8_pm_r4_d2_dx1_b64_anchor";

#[test]
fn buffer_off_vaddr_is_off_token_and_object_round_trips() {
    let digest: String = Sha256::digest(OBJ)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(
        digest, "58b08f0620ee381c32c3223e1136ef5bb07cc6666d344c3c8275ee3cc7836ec4",
        "fixture changed"
    );

    let lifted = peacemaker_lift::lift_object(
        OBJ,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    )
    .expect("lift");
    assert_eq!(
        peacemaker_lift::emit::module(&lifted.program).expect("emit"),
        OBJ,
        "emit::module must reproduce the object"
    );

    let kernel = lifted
        .program
        .kernels
        .iter()
        .find(|k| k.symbol.0 == SYMBOL)
        .expect("kernel");
    let (mut off_forms, mut bad) = (0usize, Vec::new());
    for (_, inst) in kernel.body.insts.iter() {
        if inst.form != Form::Vmem(VmemForm::Buffer) {
            continue;
        }
        let text = inst.text(Arch::Gfx1201).expect("text");
        if text.contains("offen") || text.contains("idxen") {
            continue;
        }
        off_forms += 1;
        if inst.operands.get(1) != Some(&Operand::Vmem(VmemToken::Off)) || !text.contains(", off,")
        {
            bad.push(format!("{:#x?}: {text}", inst.prov.pc));
        }
    }
    assert!(off_forms > 0, "fixture has no off-form buffer ops");
    assert!(
        bad.is_empty(),
        "off-form buffer ops with a VGPR VADDR: {bad:#?}"
    );

    let analysis =
        peacemaker_ir::edit::analyze(lifted.program.clone(), &kernel.symbol).expect("analyze");
    let raw: Vec<_> = analysis
        .obligations
        .iter()
        .filter(|o| o.rule_id == "wait-raw-vmem-load")
        .collect();
    assert!(
        raw.is_empty(),
        "{} wait-raw-vmem-load obligations, first {:?}",
        raw.len(),
        raw.first().map(|o| &o.text)
    );
}
