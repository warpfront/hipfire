// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Lift-and-repass on a gfx1100 hipcc object (PM-KRSI-1100): the production
//! `gemm_mqv2_wmma_gfx1100_mw_lds` module (16 kernels, the hot MQ6 trunk GEMM among them)
//! lifts byte-exact, an identity rewrite reproduces it, and a size-changing rewrite
//! (strip every `s_delay_alu` of a MIDDLE kernel) re-lays the module out: later kernels keep
//! their exact `.text`, the descriptor keeps its resources, only `INST_PREF_SIZE` and the
//! entry offsets follow the new layout.

use peacemaker_ir::cfg::InstId;
use peacemaker_ir::edit::{Edit, InstRange};
use peacemaker_ir::effects::Control;
use peacemaker_ir::inst::{Abi, Frontend, Kernel, KernelOrigin, SymbolId};
use peacemaker_lift::rewrite::{apply, EquivalenceKind, Rewrite};
use peacemaker_lift::{emit, lift_object, Options};
use sha2::{Digest, Sha256};

const OPTS: Options = Options {
    frontend: Frontend::Hipcc,
};
const SHA: &str = "c81a1c46757a4c9c27ae696a750534d4d70e301d4716be45d5b8491f3b7c27da";
const MIDDLE: &str = "gemm_gate_up_mq4g256v2_wmma_gfx11_mw4_lds";
const HOT: &str = "gemm_mq6g256v2_residual_wmma_gfx11_mw8_lds";

fn fixture() -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/x1/gemm_mqv2_wmma_gfx1100_mw_lds.hsaco");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(digest, SHA, "not the pinned fixture");
    bytes
}

fn delays(k: &Kernel) -> Vec<InstId> {
    k.body
        .layout
        .iter()
        .copied()
        .filter(|id| k.body.insts.get(*id).unwrap().effects.control == Control::Delay)
        .collect()
}

fn entry(k: &Kernel) -> u64 {
    match k.origin {
        KernelOrigin::Frontend { entry_va, .. } => entry_va,
        _ => panic!("authored"),
    }
}

#[test]
fn lifts_every_kernel_and_reemits_byte_identically() {
    let bytes = fixture();
    let lifted = lift_object(&bytes, OPTS).expect("gfx1100 hipcc object lifts");
    assert_eq!(lifted.program.kernels.len(), 16);
    assert!(lifted.program.kernels.iter().any(|k| k.symbol.0 == HOT));
    assert_eq!(emit::module(&lifted.program).unwrap(), bytes);
}

#[test]
fn identity_rewrite_reproduces_the_object() {
    let bytes = fixture();
    let cert = apply(
        &bytes,
        OPTS,
        Rewrite {
            kernel: SymbolId(MIDDLE.into()),
            script: Vec::new(),
            claim: EquivalenceKind::ByteIdentical,
        },
    )
    .unwrap();
    assert_eq!(cert.bytes, bytes);
    assert_eq!(cert.receipt.definedness, Ok(()));
}

#[test]
fn strip_delay_in_a_middle_kernel_relays_out_the_module() {
    let bytes = fixture();
    let base = lift_object(&bytes, OPTS).unwrap().program;
    let sym = SymbolId(MIDDLE.into());
    let k0 = base.kernels.iter().find(|k| k.symbol == sym).unwrap();
    let ids = delays(k0);
    assert!(ids.len() > 10, "hipcc places s_delay_alu in this kernel");
    let script = vec![Edit::Batch(
        ids.iter()
            .map(|id| Edit::Remove {
                range: InstRange::single(*id),
            })
            .collect(),
    )];
    let cert = apply(
        &bytes,
        OPTS,
        Rewrite {
            kernel: sym.clone(),
            script,
            claim: EquivalenceKind::Unproved,
        },
    )
    .unwrap();
    assert_ne!(cert.bytes, bytes);
    assert_eq!(cert.receipt.definedness, Ok(()));
    let back = lift_object(&cert.bytes, OPTS).unwrap().program;
    let arch = base.target.arch;
    for (a, b) in base.kernels.iter().zip(&back.kernels) {
        assert_eq!(a.symbol, b.symbol);
        let (ta, tb) = (emit::bytes(a, arch).unwrap(), emit::bytes(b, arch).unwrap());
        if a.symbol == sym {
            assert_eq!(tb.len() + 4 * ids.len(), ta.len());
            assert!(delays(b).is_empty());
        } else {
            assert_eq!(ta, tb, "{} must keep its exact .text", a.symbol.0);
        }
        let (
            Abi::Hsa {
                descriptor: da,
                metadata: ma,
            },
            Abi::Hsa {
                descriptor: db,
                metadata: mb,
            },
        ) = (&a.abi, &b.abi)
        else {
            panic!("raw ABI")
        };
        assert_eq!(ma.parsed, mb.parsed);
        assert_eq!(da.compute_pgm_rsrc1, db.compute_pgm_rsrc1);
        assert_eq!(da.compute_pgm_rsrc2, db.compute_pgm_rsrc2);
        assert_eq!(da.group_segment_fixed_size, db.group_segment_fixed_size);
        assert_eq!(da.kernarg_size, db.kernarg_size);
    }
    // The kernels after the edited one are 256-aligned: the shrink moves them only when it
    // crosses a 256-byte boundary, never past their original address.
    let (i, _) = base
        .kernels
        .iter()
        .enumerate()
        .find(|(_, k)| k.symbol == sym)
        .unwrap();
    for (a, b) in base.kernels[i + 1..].iter().zip(&back.kernels[i + 1..]) {
        assert!(entry(b) <= entry(a));
        assert_eq!(entry(b) % 256, 0);
    }
}
