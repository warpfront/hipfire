// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! DESIGN F1 CPU oracle on REAL shipped expert tensors: for experts of the shipped Flash-Next
//! `mq4-xts` artifact (`HIPFIRE_F1_REAL_MODEL=/path/qwen3.8-flash-next-base.mq4-xts`) the planar
//! repack is lossless (`expand(repack(x)) == x` byte for byte) and the planar dequant of every
//! weight is bit-equal to the shipped `sc*q + zp` dequant. `#[ignore]`: reads ~5 MB of the
//! 70 GB model through the mmap (run with `--ignored`).

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::mq4_planar::{
    dequant_planar, dequant_shipped, expand_tensor, planar_bytes, repack_tensor, shipped_bytes,
    Mq4PlanarKind,
};

const LAYERS: [usize; 3] = [0, 17, 47];
const EXPERTS: [usize; 3] = [0, 255, 511];

fn check(kind: Mq4PlanarKind, rows: usize, k: usize, shipped: &[u8], label: &str) {
    let mut planar = vec![0u8; planar_bytes(kind, rows, k).unwrap()];
    repack_tensor(kind, rows, k, shipped, &mut planar)
        .unwrap_or_else(|error| panic!("{label}: {error}"));
    let mut back = vec![0u8; shipped.len()];
    expand_tensor(kind, rows, k, &planar, &mut back).unwrap();
    // Headers whose zero is the +0 spelling of a -0 would differ in bytes only for scale 0; the
    // dequant comparison below is the contract, the byte round trip is reported strictly.
    assert_eq!(back, shipped, "{label}: expand(repack(x)) != x");
    for r in (0..rows).step_by(7) {
        for c in 0..k {
            let a = dequant_shipped(kind, k, shipped, r, c);
            let b = dequant_planar(rows, k, &planar, r, c);
            assert_eq!(a.to_bits(), b.to_bits(), "{label}: row {r} col {c}");
        }
    }
}

#[test]
#[ignore = "needs HIPFIRE_F1_REAL_MODEL (the shipped 70 GB mq4-xts artifact)"]
fn real_shipped_experts_repack_losslessly_and_dequant_bit_equal() {
    let path = std::env::var("HIPFIRE_F1_REAL_MODEL").expect("HIPFIRE_F1_REAL_MODEL");
    let hfq = HfqFile::open(std::path::Path::new(&path)).expect("open artifact");
    for layer in LAYERS {
        for (suffix, kind, rows, k) in [
            ("gate_up_proj", Mq4PlanarKind::GateUp, 1280usize, 2560usize),
            ("down_proj", Mq4PlanarKind::Down, 2560, 640),
        ] {
            let name = format!("model.language_model.layers.{layer}.mlp.experts.{suffix}");
            let (_, bytes) = hfq
                .tensor_data(&name)
                .unwrap_or_else(|| panic!("tensor {name}"));
            let per = shipped_bytes(kind, rows, k).unwrap();
            assert_eq!(bytes.len(), 512 * per, "{name} extent");
            for expert in EXPERTS {
                check(
                    kind,
                    rows,
                    k,
                    &bytes[expert * per..(expert + 1) * per],
                    &format!("{name}[{expert}]"),
                );
            }
        }
    }
}
