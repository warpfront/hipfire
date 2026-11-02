// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Pinned digests of the PM producers: every export (`s_text` sha256 and the whole builder proof as
//! canonical JSON) and every module (`s_text` sha256), recorded from the emit that existed before
//! the producers were split into public blocks (`blocks` modules) for `.rip` composition. The split
//! is a pure refactor: these digests must not move. `PINNED_PRINT=1 cargo test ... -- --nocapture`
//! prints the current values.

use peacemaker_isa::kernels::{
    gdn_chunk_prep, iu4_c8_producer, moe_rotate_i4_pm, qwen4_moe_down_gp,
};
use peacemaker_isa::{Arch, Emitted};
use sha2::{Digest, Sha256};

const ARCH: Arch = Arch::Gfx1201;

fn hex(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

/// The builder proof as canonical JSON without the build-environment fields. Keys are sorted:
/// `hipfire-engine` enables serde_json/preserve_order (b26ba3b495), and workspace feature
/// unification must not turn object ordering into a moved pin.
fn proof_json(e: &Emitted) -> String {
    let mut v = serde_json::to_value(&e.proof).unwrap();
    let o = v.as_object_mut().unwrap();
    o.remove("builder_git_sha");
    o.remove("builder_crate_version");
    v.sort_all_objects();
    serde_json::to_string(&v).unwrap()
}

/// `(s_text sha256, proof-json sha256)` of one export.
fn export(e: &Emitted) -> (String, String) {
    (hex(e.s_text.as_bytes()), hex(proof_json(e).as_bytes()))
}

fn table() -> Vec<(String, String, String)> {
    let mut rows = Vec::new();
    let mut push = |name: &str, (s, p): (String, String)| rows.push((name.to_owned(), s, p));
    for x in iu4_c8_producer::module_exports() {
        push(
            &format!("c8/{}", x.tag()),
            export(&iu4_c8_producer::emit(ARCH, x).unwrap()),
        );
    }
    let (_, text, proof) = iu4_c8_producer::emit_module(ARCH).unwrap();
    push(
        "c8/module",
        (hex(text.as_bytes()), hex(proof.s_text_sha256.as_bytes())),
    );
    let spec = moe_rotate_i4_pm::Spec { arch: ARCH };
    push(
        "rotate/rows",
        export(&moe_rotate_i4_pm::emit(spec).unwrap()),
    );
    push(
        "rotate/gp",
        export(&moe_rotate_i4_pm::emit_gp(spec).unwrap()),
    );
    let (_, text, proof) = moe_rotate_i4_pm::emit_module(ARCH).unwrap();
    push(
        "rotate/module",
        (hex(text.as_bytes()), hex(proof.s_text_sha256.as_bytes())),
    );
    let (_, text, proof) = qwen4_moe_down_gp::emit_module(ARCH).unwrap();
    push(
        "down_gp/module",
        (hex(text.as_bytes()), hex(proof.s_text_sha256.as_bytes())),
    );
    let a3b = gdn_chunk_prep::GdnChunkPrepSpec::A3B;
    push(
        "gdn/chunk_prep",
        export(&gdn_chunk_prep::emit(ARCH, &a3b).unwrap()),
    );
    let (_, text, proof) = gdn_chunk_prep::emit_module(ARCH, &a3b).unwrap();
    push(
        "gdn/module",
        (hex(text.as_bytes()), hex(proof.s_text_sha256.as_bytes())),
    );
    rows
}

const PINNED: &[(&str, &str, &str)] = &[
    (
        "c8/rotate",
        "ba66776319d2fbfc91b2a6c18fa53372c1f5cec2b680c5ad1b917002fd7d1b03",
        "7e737b140e193450504d8eb1db9a57c74c141ca7ae2958b365e2b62d514cf4b3",
    ),
    (
        "c8/norm",
        "caba4887872a76e918c5de86ce9d7eb3277a616d6477c0c9d24c9e955bb39e2c",
        "e24608de43d8756ee0b2d07bbb4e335b5670428a7732479ef999fdab4d2b2602",
    ),
    (
        "c8/sigmoid_rotate",
        "bf358880815ecb9803f20a49ecd3c16e6eb3a12d363c3f5e7d900d512f57f326",
        "7509b63854911a0914545fa2f4b85df7f81f044d94516bfbe9c4c0ef89b3fe62",
    ),
    (
        "c8/norm_out",
        "450bd735872bf14498eabe6e05fd1b90c207fb9d3ab5448516ff47a9443b2397",
        "ca509cde775169bbf7825d796c7d57e1fbcc47c1efcdce695e5aba500b2da987",
    ),
    (
        "c8/module",
        "94af6ed30f09ace46724be5e275f5e766b512a040ed6cc18c1be2c9b4562a68a",
        "5bc5020f4178f0c35a3866ff743a922ff07937909687b9f29a3fae99352759b4",
    ),
    (
        "rotate/rows",
        "3de8eb3ba18ff53a7e164bd1803de8a8caf030f07d712cc1ece6895b9cff158b",
        "7136b9b322c36bad2ad246a816323836acddf718880363a4589b191885cbee68",
    ),
    (
        "rotate/gp",
        "241c9db3d2e5aae53918588c2fcf6c2dc30e54c2efed4f58dce217501bc5cb17",
        "331f3c2700d06d2bfb5adf1200e5862df1c0f189a58bb3c5dd55ec9dfea3f9e9",
    ),
    (
        "rotate/module",
        "3de8eb3ba18ff53a7e164bd1803de8a8caf030f07d712cc1ece6895b9cff158b",
        "58b5f268b6788367845b62494460497c059bd95828726dfad80412b47a817c8f",
    ),
    (
        "down_gp/module",
        "a745708078c40797dc4fae02d8d2162fbf4c339f90644a5dd870c93b4ce70a78",
        "0b6daf57bc98e72584ac9cd4d3b8a9982c4b48dc8c68158608d07830ab7aca3b",
    ),
    (
        "gdn/chunk_prep",
        "24b70b48c5db7ae8c2175ff49dfc98d1d407544b708891d5bed4bd51454d5cda",
        "c574a1bb3c6bb46119d239a72e1474e2039cde8a416de603eeb3b043fbc8aca5",
    ),
    (
        "gdn/module",
        "24b70b48c5db7ae8c2175ff49dfc98d1d407544b708891d5bed4bd51454d5cda",
        "122b731e47fde0395c24a836cf26fa85d1ba73ab50cc1285fa1a36a672d71708",
    ),
];

#[test]
fn producers_keep_their_pinned_digests() {
    let now = table();
    if std::env::var("PINNED_PRINT").is_ok() {
        for (n, s, p) in &now {
            println!("    (\"{n}\", \"{s}\", \"{p}\"),");
        }
    }
    assert_eq!(now.len(), PINNED.len(), "pin table out of date");
    for ((n, s, p), (pn, ps, pp)) in now.iter().zip(PINNED) {
        assert_eq!(n, pn);
        assert_eq!(s, ps, "{n}: s_text moved");
        assert_eq!(p, pp, "{n}: proof moved");
    }
}
