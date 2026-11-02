// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Golden bytes of the default V9 / V10 deployments: PDI, TXN, argument specs and both pair core programs, for the
//! same-context shape list of `array.rs` plus the expert shapes, Fast and Slow control. The table was generated from
//! npu commit 8261881 (before the levers-joint lever 3 descriptor experiments) and pins every default selection, so
//! opt-in variants cannot silently change the production bytes. Regenerate with `GOLDEN_PRINT=1` only for an
//! intentional default change, and say so in the commit.
use peacemaker_npu::kernels::{
    gemm_array,
    gemm_core::{self, Control, Epilogue, PairRole},
};

/// FNV-1a 64 (stable across toolchains, unlike `DefaultHasher`).
fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}
const SHAPES: [(usize, usize, usize); 10] = [
    (512, 512, 64),
    (512, 512, 128),
    (1536, 512, 64),
    (1024, 1024, 128),
    (1536, 512, 192),
    (1024, 1536, 128),
    (1536, 1536, 64),
    (4096, 1280, 2560),
    (4096, 2560, 640),
    (2048, 2560, 640),
];
const INT8: Epilogue = Epilogue::Int8 { shift: 12 };

/// `variant m n k control pdi txn args lower upper` for every legal (variant, shape, control).
fn rows() -> Vec<String> {
    let mut v = Vec::new();
    for (name, legal) in [("V9", true), ("V10", false)] {
        for &(m, n, k) in &SHAPES {
            let (kc, nw) = (k / 64, n.div_ceil(512));
            if !legal && kc * (2 + nw) > 112 {
                continue;
            }
            for ctl in [Control::Fast, Control::Slow] {
                let d = if legal {
                    gemm_array::design_v9(m, n, k, INT8, ctl)
                } else {
                    gemm_array::design_v10(m, n, k, INT8, ctl)
                };
                let prog =
                    |role| fnv(&gemm_core::program_pair(kc, d.waves(), role, INT8, ctl).finish());
                v.push(format!(
                    "{name} {m} {n} {k} {} {:016x} {:016x} {:016x} {:016x} {:016x}",
                    if ctl.is_slow() { "Slow" } else { "Fast" },
                    fnv(&d.pdi),
                    fnv(&d.insts),
                    fnv(format!("{:?}", d.args).as_bytes()),
                    prog(PairRole::Lower),
                    prog(PairRole::Upper)
                ));
            }
        }
    }
    v
}

const GOLDEN: &str = "
V9 512 512 64 Fast 28cb9957d2c20726 1e85b2654d1cc72c 0779bf39d7caef56 224da15b23c5e750 476dd2697b4896c0
V9 512 512 64 Slow bc7f450eb929c2f5 1e85b2654d1cc72c 0779bf39d7caef56 e4127677be34c09f fc6fd14e5cd37737
V9 512 512 128 Fast 786b4ad6262f8886 5e37ef4325dd832c 9691928a69bb92d6 dfb9b3e343b41cde a6358d6a885b17de
V9 512 512 128 Slow dc288b32eb297cd5 5e37ef4325dd832c 9691928a69bb92d6 49b9467681e596dd 642bff97ea6174bd
V9 1536 512 64 Fast bf96523ec530cd1b a760c4d283a4b201 b3f6b4712874f065 10fe75bf0e354864 3838d7aade632e54
V9 1536 512 64 Slow dca6222df22ad928 a760c4d283a4b201 b3f6b4712874f065 574c3dca23276233 0da631dbc6f55cbb
V9 1024 1024 128 Fast bbe6739ecd50e3f1 24cd0893c12c56ca 5dbf4f840a673805 8df185fec093bac4 123835fb7789c434
V9 1024 1024 128 Slow b33ccdecb6fc6c52 24cd0893c12c56ca 5dbf4f840a673805 d771e83945e00553 1c58960a948e01db
V9 1536 512 192 Fast 3690a9e7936e65fb 8e9d0be7f03c3ea1 96dca148e18c2da4 cf4de74e23aa4870 973a8709fec27060
V9 1536 512 192 Slow c727767c943b9568 8e9d0be7f03c3ea1 96dca148e18c2da4 864f49467ac3c01f 72a0994366d126b7
V9 1024 1536 128 Fast 4e249527363bc4f1 51d41a6da4ca1ae1 ad9e7130044991c8 ba4a2a4d7e30e180 2aa17910d2955430
V9 1024 1536 128 Slow 8c08c19675c03312 51d41a6da4ca1ae1 ad9e7130044991c8 d1b6cded3ae7a6ef 5b8d3036753066c7
V9 1536 1536 64 Fast 5f69864ea0b2fe9b ff04ba2a2f441aa5 0dffcbe635588246 ea327d1a1444efe0 a418d1ad7460a310
V9 1536 1536 64 Slow 3ea4c8f73e8e5dc8 ff04ba2a2f441aa5 0dffcbe635588246 f0d7847ee7af2f6f cc9d244eccf2bf47
V9 4096 1280 2560 Fast 10ca1b90dd840c1b eacf5d13e55ded3a 63e6832177759286 2215fd49ca9498d8 a7eb72cefdd0a828
V9 4096 1280 2560 Slow 1dd6dffc53611854 eacf5d13e55ded3a 63e6832177759286 cd9e547c7500fe67 568a4c703e6f1ebf
V9 4096 2560 640 Fast 56f85892e07d62bb d92ba0fea09e7615 9defb20df9fe4162 b9862203bda7fa8c 114a5e018ff05ebc
V9 4096 2560 640 Slow 81f71cfa486f3b14 d92ba0fea09e7615 9defb20df9fe4162 6d449f28e71079bb c587e0e550e5a1e3
V9 2048 2560 640 Fast efe3a0464eba8e49 e2e4d5bc01a8190b 736428f2dd38b3d3 8698c0f9c6059834 9b773279bc4054a4
V9 2048 2560 640 Slow 4c55f5a3c3e08d52 e2e4d5bc01a8190b 736428f2dd38b3d3 31b50e9cb4778963 0c9416b34b1147eb
V10 512 512 64 Fast 2eb8afb67737366d 723302eb7272d84d 0779bf39d7caef56 224da15b23c5e750 476dd2697b4896c0
V10 512 512 64 Slow 21f2bb644e946caa 723302eb7272d84d 0779bf39d7caef56 e4127677be34c09f fc6fd14e5cd37737
V10 512 512 128 Fast 37461a6e83af682d eaa2016f141db08d 9691928a69bb92d6 dfb9b3e343b41cde a6358d6a885b17de
V10 512 512 128 Slow 2610ca26f7c9092a eaa2016f141db08d 9691928a69bb92d6 49b9467681e596dd 642bff97ea6174bd
V10 1536 512 64 Fast b54ec29b0dd49abd 289585735d137e81 b3f6b4712874f065 10fe75bf0e354864 3838d7aade632e54
V10 1536 512 64 Slow 37b3c49bea85a5da 289585735d137e81 b3f6b4712874f065 574c3dca23276233 0da631dbc6f55cbb
V10 1024 1024 128 Fast b0c5ed740f5a3ceb 5676d79a1969cdbb 07d785084711d53a 8df185fec093bac4 123835fb7789c434
V10 1024 1024 128 Slow 5dca4f37c73c7afc 5676d79a1969cdbb 07d785084711d53a d771e83945e00553 1c58960a948e01db
V10 1536 512 192 Fast 12b7fddcab97315d e5e47e0d915a0091 96dca148e18c2da4 cf4de74e23aa4870 973a8709fec27060
V10 1536 512 192 Slow a031de4915a9d95a e5e47e0d915a0091 96dca148e18c2da4 864f49467ac3c01f 72a0994366d126b7
V10 1024 1536 128 Fast 74e5de39ee87a526 72922f6f1928f3db ffaf8f2c2c25d43e ba4a2a4d7e30e180 2aa17910d2955430
V10 1024 1536 128 Slow 235810853579e815 72922f6f1928f3db ffaf8f2c2c25d43e d1b6cded3ae7a6ef 5b8d3036753066c7
V10 1536 1536 64 Fast 36eceda4e04a4df6 4df7cc7373b1640e ad5c8680104646b3 ea327d1a1444efe0 a418d1ad7460a310
V10 1536 1536 64 Slow b294b4bf50d1af05 4df7cc7373b1640e ad5c8680104646b3 f0d7847ee7af2f6f cc9d244eccf2bf47
V10 4096 2560 640 Fast 07db39484a956feb e20cad8ca0a0d0cf 2f5898b2615b11f9 b9862203bda7fa8c 114a5e018ff05ebc
V10 4096 2560 640 Slow 537cf0d83fddea98 e20cad8ca0a0d0cf 2f5898b2615b11f9 6d449f28e71079bb c587e0e550e5a1e3
V10 2048 2560 640 Fast 8595d4879d6dce1b 89c8a5dcab2378f7 5a270e8ee0665efe 8698c0f9c6059834 9b773279bc4054a4
V10 2048 2560 640 Slow 33e7ce5283a03fe8 89c8a5dcab2378f7 5a270e8ee0665efe 31b50e9cb4778963 0c9416b34b1147eb
";

#[test]
fn default_v9_v10_bytes_match_golden() {
    let got = rows();
    if std::env::var_os("GOLDEN_PRINT").is_some() {
        for r in &got {
            println!("{r}");
        }
    }
    let want: Vec<&str> = GOLDEN
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    assert_eq!(got.len(), want.len(), "row count");
    for (g, w) in got.iter().zip(&want) {
        assert_eq!(g, w, "default design bytes changed");
    }
}

#[test]
fn g80_legacy_bytes_match_golden() {
    // Captured before the BF16 changes from npu/trunk-pipeline's I32 cutover.
    let hashes = [
        (0xe4085b5059424fc9, 0xae900bee5da82eeb),
        (0x2dd1f2492aa3df98, 0xae900bee5da82eeb),
        (0x577123b1d2e5ae25, 0x7503151e9bf64fa1),
        (0x10e12b171628595e, 0x7503151e9bf64fa1),
        (0x818eeafa6857be09, 0x87b27dc21c8a5786),
        (0xde100eb37076ca78, 0x87b27dc21c8a5786),
        (0xd38932f7c0a8e865, 0xb7d9b0530c9ed98e),
        (0xebfea186a2ed6ade, 0xb7d9b0530c9ed98e),
    ];
    let mut at = 0;
    for m in [256, 8192] {
        for epi in [Epilogue::Int8 { shift: 12 }, Epilogue::I32] {
            for ctl in [Control::Fast, Control::Slow] {
                let d = peacemaker_npu::kernels::gemm_g80::design_g80(m, 1280, 2560, epi, ctl);
                println!(
                    "G80 {m} {epi:?} {ctl:?} {:016x} {:016x}",
                    fnv(&d.pdi),
                    fnv(&d.insts)
                );
                assert_eq!(
                    (fnv(&d.pdi), fnv(&d.insts)),
                    hashes[at],
                    "G80 {m} {epi:?} {ctl:?} legacy bytes"
                );
                at += 1;
            }
        }
    }
}

#[test]
fn g80_bf16_native_bytes_match_golden() {
    for (m, hashes) in [
        (256, (0x0f0957038d4177ef, 0x2cc33282387023c1)),
        (8192, (0x3140db0e2c87154f, 0x6dc4a91db3ec53db)),
    ] {
        let d = peacemaker_npu::kernels::gemm_g80::design_g80(
            m,
            1280,
            2560,
            Epilogue::Bf16,
            Control::Fast,
        );
        assert_eq!(
            (fnv(&d.pdi), fnv(&d.insts)),
            hashes,
            "G80 BF16 native {m} bytes changed"
        );
    }
}

/// The N512 / K6144 out-projection design (`gemm_g80_n512`), native tiles and row-major (pitch 2560, col0 0), M512 and M8192:
/// `(pdi, txn, lower program, upper program)` FNV-1a hashes, pinned when the design was first committed so the production
/// bytes of the out program cannot drift silently. Tuples are `(m, layout, pdi, txn, lower, upper)`.
#[test]
fn g80_n512_out_bytes_match_golden() {
    use peacemaker_npu::kernels::gemm_g80::Bf16Layout;
    let rowmajor = Bf16Layout::RowMajor {
        pitch_elems: 2560,
        col0_elems: 0,
    };
    let pins: [(usize, Bf16Layout, [u64; 4]); 4] = [
        (
            512,
            Bf16Layout::NativeTiles,
            [
                0xc928b89c58012467,
                0x4291c32493ddc1b0,
                0x67f57222824ad41a,
                0x1ce87df7f313cb6a,
            ],
        ),
        (
            8192,
            Bf16Layout::NativeTiles,
            [
                0x3135f27ae09c3207,
                0xac5209c99a816729,
                0xb1a9d43c99961334,
                0x1910880189b8b7ec,
            ],
        ),
        (
            512,
            rowmajor,
            [
                0x5e6b30e57fdec1d7,
                0x1fd333c421b95278,
                0x67f57222824ad41a,
                0x1ce87df7f313cb6a,
            ],
        ),
        (
            8192,
            rowmajor,
            [
                0x51878e7e9a89df77,
                0x93952abc61528a3b,
                0xb1a9d43c99961334,
                0x1910880189b8b7ec,
            ],
        ),
    ];
    let mut bad = Vec::new();
    for (m, layout, want) in pins {
        let d = peacemaker_npu::kernels::gemm_g80_n512::design_g80_n512_layout(
            m,
            512,
            6144,
            Epilogue::Bf16,
            Control::Fast,
            layout,
        );
        let got = [
            fnv(&d.pdi),
            fnv(&d.insts),
            fnv(d.core_program()),
            fnv(d.core_program_upper().unwrap()),
        ];
        println!(
            "N512 {m} {layout:?} {:016x} {:016x} {:016x} {:016x}",
            got[0], got[1], got[2], got[3]
        );
        println!(
            "N512 {m} sizes: pdi {} B, txn {} B, lean {} B, lower {} B, upper {} B",
            d.pdi.len(),
            d.insts.len(),
            d.lean_insts().unwrap().len(),
            d.core_program().len(),
            d.core_program_upper().unwrap().len()
        );
        bad.extend((got != want).then(|| format!("N512 out {m} {layout:?} bytes changed")));
    }
    assert!(bad.is_empty(), "{bad:?}");
}
