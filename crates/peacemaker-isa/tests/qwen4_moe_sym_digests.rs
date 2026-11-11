// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Every `qwen4_moe_sym` entry and module re-emits the bytes recorded before the building blocks
//! (`qwen4_moe_sym::blocks`) were factored out of the generators: the sha256 of `emit(spec).s_text`
//! per entry and of every module text, computed from the unrefactored code. A change to any digest
//! is a change to a production kernel (or to an experimental schedule) and must be intentional.

use peacemaker_isa::kernels::qwen4_moe_down_gp;
use peacemaker_isa::kernels::qwen4_moe_sym::{self as moe, Kind, R4Spec, Spec, WeightCodec};
use peacemaker_isa::Arch;
use sha2::{Digest, Sha256};

fn sha(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

/// `(name, digest)` of every entry of every arch's MQ4 module (gfx1151 and gfx1201), in module order.
fn entry_digests() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for arch in [Arch::Gfx1151, Arch::Gfx1201] {
        for spec in moe::module_specs(arch) {
            let e = moe::emit(spec).unwrap_or_else(|e| panic!("{}: {e}", spec.symbol()));
            out.push((spec.symbol(), sha(&e.s_text)));
        }
    }
    out
}

/// The grouped-position down entries (gfx1201).
fn gp_digests() -> Vec<(String, String)> {
    [4u8, 8]
        .into_iter()
        .map(|nt| {
            let spec = Spec {
                arch: Arch::Gfx1201,
                kind: Kind::Down,
                nt,
                rr: 1,
                codec: WeightCodec::Mq4Sym,
            };
            let e = moe::emit_down_gp(spec).unwrap();
            (format!("down_gp_nt{nt}"), sha(&e.s_text))
        })
        .collect()
}

/// The default-off schedule experiments (`R4Spec`): chains, no prefetch and the grouped ABI on the
/// gfx1201 NT4 / NT8 entries that build.
fn r4_digests() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for kind in Kind::ALL {
        for nt in [4u8, 8] {
            for chains in [1u8, 2, 4, 8] {
                for prefetch in [true, false] {
                    for grouped in [false, true] {
                        let base = Spec {
                            arch: Arch::Gfx1201,
                            kind,
                            nt,
                            rr: 1,
                            codec: WeightCodec::Mq4Sym,
                        };
                        let r4 = R4Spec {
                            base,
                            chains,
                            prefetch,
                            grouped,
                        };
                        let name = format!(
                            "r4_{}_nt{nt}_c{chains}_p{}_g{}",
                            kind.tag(),
                            u8::from(prefetch),
                            u8::from(grouped)
                        );
                        match moe::emit_r4(r4) {
                            Ok(e) => out.push((name, sha(&e.s_text))),
                            Err(_) => continue,
                        }
                    }
                }
            }
        }
    }
    out
}

/// `(module, sha256 of the module text)`.
fn module_digests() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for arch in [Arch::Gfx1151, Arch::Gfx1201] {
        let (_, text, _) = moe::emit_module(arch).unwrap();
        out.push((Spec::module(arch), sha(&text)));
    }
    for (arch, codec) in [
        (Arch::Gfx1151, WeightCodec::Mq3I4),
        (Arch::Gfx1201, WeightCodec::Mq3I4),
        (Arch::Gfx1201, WeightCodec::E8pI4),
    ] {
        let (_, text, _) = moe::emit_module_codec(arch, codec).unwrap();
        out.push((Spec::module_codec(arch, codec), sha(&text)));
    }
    let (_, text, _) = qwen4_moe_down_gp::emit_module(Arch::Gfx1201).unwrap();
    out.push((qwen4_moe_down_gp::module_name(Arch::Gfx1201), sha(&text)));
    // F1 drop-zero planar twins (gfx1201).
    let (_, text, _) = moe::emit_module_codec(Arch::Gfx1201, WeightCodec::Mq4SymPlanar).unwrap();
    out.push((
        Spec::module_codec(Arch::Gfx1201, WeightCodec::Mq4SymPlanar),
        sha(&text),
    ));
    let (_, text, _) =
        qwen4_moe_down_gp::emit_module_codec(Arch::Gfx1201, WeightCodec::Mq4SymPlanar).unwrap();
    out.push((
        qwen4_moe_down_gp::module_name_codec(Arch::Gfx1201, WeightCodec::Mq4SymPlanar),
        sha(&text),
    ));
    out
}

fn check(name: &str, got: Vec<(String, String)>, pinned: &[(&str, &str)]) {
    if std::env::var_os("MOE_PRINT_DIGESTS").is_some() {
        println!("== {name}");
        for (n, d) in &got {
            println!("    (\"{n}\", \"{d}\"),");
        }
    }
    let got: Vec<(&str, &str)> = got.iter().map(|(n, d)| (n.as_str(), d.as_str())).collect();
    assert_eq!(got, pinned, "{name}: emitted bytes changed");
}

const ENTRIES: &[(&str, &str)] = &[
    (
        "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1151",
        "dca7f5e4960783162ddd239f9daf3e62374b1b9d287e8df3c0969809f5c25291",
    ),
    (
        "qwen4_moe_down_iu4_sym_pm_gfx1151",
        "b8132736129d2b049c63192c7a7a8e2f9070041885a56c7262c4499ed54ab283",
    ),
    (
        "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1151_nt4",
        "f8a7e01f47c1513d31e102223ac34c94d88efa774b8b902c4f236006a1fc96cd",
    ),
    (
        "qwen4_moe_down_iu4_sym_pm_gfx1151_nt4",
        "1870412beb20b40fb8b14a2be37872c4d5e9dda9a2d6b52123ca32cdfd006602",
    ),
    (
        "qwen4_moe_down_iu4_sym_pm_gfx1151_nt4r2",
        "2fe7636b42474697d17222c0cd5f080dde64684a9e993e45c584b625276ecb4f",
    ),
    (
        "qwen4_moe_down_iu4_sym_pm_gfx1151_nt4r4",
        "156c4238688784121ad463edfa9a1863da6ee46ee7cde936bf532fac7ae7ea15",
    ),
    (
        "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201",
        "090ae102c6258886a8beaf61e0cf7e9f93b2a559c6a1492a9f1aa8b248dda397",
    ),
    (
        "qwen4_moe_down_iu4_sym_pm_gfx1201",
        "27abdba1b638cf8662c4c250339181edbb962a271c91d58a213cd5dc536cc737",
    ),
    (
        "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt4",
        "567be4bb29ef23572333fc1ac67f4d1f56c3493682eea927f02538c52242cc3b",
    ),
    (
        "qwen4_moe_down_iu4_sym_pm_gfx1201_nt4",
        "7368e2739e4900dca7121a15da076db47fc9aa88388a95fb8f0f81f7aa1ba98a",
    ),
    (
        "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt8",
        "904f6d5a54d1facb2a96e65095b7729eb764fc68eb8fa4a7a09e140f79c31c40",
    ),
    (
        "qwen4_moe_down_iu4_sym_pm_gfx1201_nt8",
        "4499a948cd8a6e927a5eebd30749c9bcbd5c7eed90769f8f85d76d619c53debb",
    ),
];
const GP: &[(&str, &str)] = &[
    (
        "down_gp_nt4",
        "3e19874a7d3966fc446df6461af181169c1d226834c772eea8c1bee0eb765231",
    ),
    (
        "down_gp_nt8",
        "70b867463468f3629ad463a22670c300e3bda0d6c4d509649764b982e82c832a",
    ),
];
const R4: &[(&str, &str)] = &[
    (
        "r4_gate_up_nt4_c1_p1_g0",
        "567be4bb29ef23572333fc1ac67f4d1f56c3493682eea927f02538c52242cc3b",
    ),
    (
        "r4_gate_up_nt4_c1_p1_g1",
        "83564157f673b3278a6d47e8b55ca0c1102b4b1fb4e1c1cdb2d29ad06579b87a",
    ),
    (
        "r4_gate_up_nt4_c1_p0_g0",
        "d22e241d47ba16a2078851ec64b827860467c5919b4c5390d09622f99234952a",
    ),
    (
        "r4_gate_up_nt4_c1_p0_g1",
        "c85338399af92c4bde43df14d826827ae7a3624112d3c42a6a8bbe9547c52d16",
    ),
    (
        "r4_gate_up_nt4_c2_p1_g0",
        "8a884827240bd9ffd4a0ddf6d2413efddab33b43495362e66edb1b083fe19608",
    ),
    (
        "r4_gate_up_nt4_c2_p1_g1",
        "aee821e016434072a403136868e74f32b520703c5483b1ea855ea45db65d3e7d",
    ),
    (
        "r4_gate_up_nt4_c2_p0_g0",
        "67cb3cb499c3205a8c1f3d623aa2115522f10b2e90f503bd3b16541dc5243bf3",
    ),
    (
        "r4_gate_up_nt4_c2_p0_g1",
        "17c4e81cfbade5d9ca776ab6ea9aca5e22981fa927daabd3840e8d1752830e08",
    ),
    (
        "r4_gate_up_nt4_c4_p1_g0",
        "087016506d130de543240b89ae8a2ddfda3631ec21e8178c356198ef5aff4762",
    ),
    (
        "r4_gate_up_nt4_c4_p1_g1",
        "e1c71bcf330c10f7f540e9a9729c70e23550666861fe6454f99fb800f0568b7e",
    ),
    (
        "r4_gate_up_nt4_c4_p0_g0",
        "546ff40e1622e3151c767641a5360628f1a6c2da32e029129a29a525038ef909",
    ),
    (
        "r4_gate_up_nt4_c4_p0_g1",
        "043dfaabd1264d822347b8cc04a35896b9b5b262833130a81c1936648e67337b",
    ),
    (
        "r4_gate_up_nt8_c1_p1_g0",
        "904f6d5a54d1facb2a96e65095b7729eb764fc68eb8fa4a7a09e140f79c31c40",
    ),
    (
        "r4_gate_up_nt8_c1_p1_g1",
        "9222585c024feee7175b32b2ef71eab805c170bbba0cf881496a09a422fe65c6",
    ),
    (
        "r4_gate_up_nt8_c1_p0_g0",
        "d45a88078fd1a07c83420d50a279d8b35d25a07aa7e8a0cf6cab5bcb3366646d",
    ),
    (
        "r4_gate_up_nt8_c1_p0_g1",
        "67a08fedbdf00dc86eff87aea991478759de50405dc19d2a1c7fadd1f1271c6c",
    ),
    (
        "r4_gate_up_nt8_c2_p1_g0",
        "1d538bba246c6e27c2104be0b8f52438f6f77102e436851637b2330984deadd9",
    ),
    (
        "r4_gate_up_nt8_c2_p1_g1",
        "daef73904ed239acb2fd235740475c07acddf52362bec2226ad7c756f75ddd2f",
    ),
    (
        "r4_gate_up_nt8_c2_p0_g0",
        "94bab04a60fdadb0f7cbfe30f4bd97a0b2e72e42e12f119e29e4d97ef86d71dd",
    ),
    (
        "r4_gate_up_nt8_c2_p0_g1",
        "58aeac8240d3217b0f263081fc8da583d8a048d8261e6bf9511ef404e5ea833a",
    ),
    (
        "r4_gate_up_nt8_c4_p1_g0",
        "0073ba5d22faad63fa0c612339ce7b1cb68df26589b629d1c4f1b27cd6e0354c",
    ),
    (
        "r4_gate_up_nt8_c4_p1_g1",
        "d8d5143f93e79f942ae1d11b4d5c5d13dbaebd3ad09f895f628ec470ccf7b51a",
    ),
    (
        "r4_gate_up_nt8_c4_p0_g0",
        "9a154291c53d6aa1beba1cc50e6bb24b83f9ff3414f3d2e28f0a4f493b337367",
    ),
    (
        "r4_gate_up_nt8_c4_p0_g1",
        "bfaaa4eddc27bb6aa941bb0eaa0b01ba859524f48576e4205ee1369bd150d3ed",
    ),
    (
        "r4_down_nt4_c1_p1_g0",
        "7368e2739e4900dca7121a15da076db47fc9aa88388a95fb8f0f81f7aa1ba98a",
    ),
    (
        "r4_down_nt4_c1_p1_g1",
        "5299023e7630da95ea7625651d0565456f7009dd020259c567a0e13ac2d2afad",
    ),
    (
        "r4_down_nt4_c1_p0_g0",
        "a376023ed0e08e05374d44e1ed69bdc39b6955e582c1b9d65f87d974109521dd",
    ),
    (
        "r4_down_nt4_c1_p0_g1",
        "9e15b492b116c12346547b29d68fa51e72d5465bc327cf27861f9375dba328bc",
    ),
    (
        "r4_down_nt4_c2_p1_g0",
        "5ed902d07045adcac34f5f7fbe5a418a5481fcf4f08caf6a07d523863cd56307",
    ),
    (
        "r4_down_nt4_c2_p1_g1",
        "40e82188fdda9d08741e9f5a5c88f4ef88a66ca1c9a2ac60c89e3d6116a93457",
    ),
    (
        "r4_down_nt4_c2_p0_g0",
        "f10fdbbcf533e0719232f09d0ab507a34321ce951f13fa2c077f4d5b1557626c",
    ),
    (
        "r4_down_nt4_c2_p0_g1",
        "38b43aea7dcde5aa5bd78779c3cb93d970977f06fbf22c82a4c8597ff4800e34",
    ),
    (
        "r4_down_nt4_c4_p1_g0",
        "23eb74d9a3b408038e6c3e0bbbfd9c9ebe06f448989b866dd2f0b5c462aa473c",
    ),
    (
        "r4_down_nt4_c4_p1_g1",
        "5eedf29331b522948dbcaf8469ac7daeafad95d6926e42b24dd6a522f4b89dad",
    ),
    (
        "r4_down_nt4_c4_p0_g0",
        "87b688037951b1c9ed82ec3229f6e21b74a6d207539f3c5bed7416d7223b3ae1",
    ),
    (
        "r4_down_nt4_c4_p0_g1",
        "ab680fcb320add8fd1d723f2b842d75bdf2137a1e1388e7b6e9c9b3dfccec3a6",
    ),
    (
        "r4_down_nt8_c1_p1_g0",
        "4499a948cd8a6e927a5eebd30749c9bcbd5c7eed90769f8f85d76d619c53debb",
    ),
    (
        "r4_down_nt8_c1_p1_g1",
        "f4f2aa6cb7bce2a6e5db1aaec21c01cc3078a8bce4757c547a3d37bafa829b6e",
    ),
    (
        "r4_down_nt8_c1_p0_g0",
        "07fc88b63e959a81c332b3df9ee46cdd6567b111598b297f4a6950213ce40d9d",
    ),
    (
        "r4_down_nt8_c1_p0_g1",
        "e1874985a001ecc4d997a97809ca9cf55250f88cd12607e8e67a6281837797c6",
    ),
    (
        "r4_down_nt8_c2_p1_g0",
        "4d36e46487a85fa0a4701068a9bc3b63fde3610ed7dc3b323afe54a9a0ce1644",
    ),
    (
        "r4_down_nt8_c2_p1_g1",
        "a7cd76d92419ceb17437260e2a0496c5151eef297254cad2ddfa24be8c2ff192",
    ),
    (
        "r4_down_nt8_c2_p0_g0",
        "c0d9b04c849e1ee7a00d1b29f22686dc8161629e22b2de7941fce8114771b610",
    ),
    (
        "r4_down_nt8_c2_p0_g1",
        "da85add0544f680c669148b9c1465732376f22fa2ccd4417df6cc20c627c0ac3",
    ),
    (
        "r4_down_nt8_c4_p1_g0",
        "d81a72f2d225df7e71e6d3585b392701056a4763e05fa3602b2ba5b015a3f360",
    ),
    (
        "r4_down_nt8_c4_p1_g1",
        "1cf6dd006cf62a1d15f0fc6c629f9391623a1bc98005ae857d4dbfde23778429",
    ),
    (
        "r4_down_nt8_c4_p0_g0",
        "64a90f7dc892a0e64a11b449bab5aa30165e3fd29a439441809e2b315972cb42",
    ),
    (
        "r4_down_nt8_c4_p0_g1",
        "afd9740312d477167546b5606f7036c8cf11b59c05857f41caaf3b5b87986aee",
    ),
];
const MODULES: &[(&str, &str)] = &[
    (
        "qwen4_moe_iu4_sym_pm_gfx1151",
        "4ad505274e39a2bb1f973e0b2f828cef8806ca722aa51f572ea8e1f12c524917",
    ),
    (
        "qwen4_moe_iu4_sym_pm_gfx1201",
        "d234c9beb45dabf4a1f6b89a9ba2c289778f48e71efbc659e65e57bf6a9533a5",
    ),
    (
        "qwen4_moe_iu4_sym_w3_pm_gfx1151",
        "dc85a33d369aa439bb7ad4bc963c1610561a5b64bd8debb7d1b8a8984a4982ee",
    ),
    (
        "qwen4_moe_iu4_sym_w3_pm_gfx1201",
        "7fef5eead3e3af7337373e4547b4650601d6f810f211ffcca8500b8dd7c98a83",
    ),
    (
        "qwen4_moe_iu4_sym_e8p_pm_gfx1201",
        "77d59f5cd0287037243094ddb0a8dd096c6c4a68aa6ae43a2fdd0f51aad51948",
    ),
    (
        "qwen4_moe_down_gp_pm_gfx1201",
        "a745708078c40797dc4fae02d8d2162fbf4c339f90644a5dd870c93b4ce70a78",
    ),
    (
        "qwen4_moe_iu4_sym_p_pm_gfx1201",
        "be2cd77565503dac50bf3e1c248203398477ea1484a92ba3a4e36c315e955c9e",
    ),
    (
        "qwen4_moe_down_gp_p_pm_gfx1201",
        "f50bfdda13ecb1648541c32470f93ee32efee9bf1bb481afeac6c94d597c0d71",
    ),
];

#[test]
fn qwen4_moe_sym_entry_digests_pinned() {
    check("entries", entry_digests(), ENTRIES);
}

#[test]
fn qwen4_moe_sym_gp_digests_pinned() {
    check("gp", gp_digests(), GP);
}

#[test]
fn qwen4_moe_sym_r4_experiment_digests_pinned() {
    check("r4", r4_digests(), R4);
}

#[test]
fn qwen4_moe_sym_module_digests_pinned() {
    check("modules", module_digests(), MODULES);
}
