// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The rip942 gfx942 `gemm_fp8b128_moe_grouped_bf16mfma` bitwise-variant corpus, pinned as
//! immutable hipcc bundles (`tests/fixtures/cdna/rip942/<variant>.co`, copied byte-for-byte
//! from `mi300x/beat/S/rip942/obj/`) and exercised through the production sealing path for
//! **both GEMM entries and the tile-descriptor entry** of every bundle.
//!
//! This is evidence, not a promised discharge. The recorded outcomes below come from the
//! pre-change verifier (`seal/<variant>.err` / `.out`, status file `STATUS-Rip942.md`):
//!
//! * five variants sealed (`default d1 d4 xp2 pf2`) and must keep sealing, bit-identically;
//! * six double-buffer variants (`pp pp_d4 pp_xp2 pp_gm1 xall xall_pp`) were refused with
//!   VMEM->VMEM WAWs. The gfx942 in-order VGPR-write proof now discharges same
//!   non-sampler VMEM_READ_ACCESS pairs, independently of scalar path feasibility.
//!   `pp pp_xp2 pp_gm1` now seal and reverify all three entries without rewriting
//!   device ELF bytes; `pp_d4 xall xall_pp` advance to the unchanged 512-state
//!   cap on both GEMM entries (descriptor still seals). These exact outcomes are
//!   pinned below. Old WAWs remain historical evidence, not an allowlist or a
//!   positive numerical/kernel qualification;
//! * `d8 relay relay_xp1` exceeded the 512-states-per-label cap (unchanged cap);
//! * `xp3_fs` was refused for an `s_barrier` with LGKM not drained.
//!
//! A first refusal may legitimately move once an infeasible prefix is removed. It is printed
//! (stderr) when it does; it is neither a new hazard nor evidence that every hazard is gone.
//! No limit, allowlist or obligation exemption is touched by this target.

use peacemaker_ir::envelope::Bundle;
use peacemaker_isa::arch::Arch;
use peacemaker_isa::hip_twin::{
    ledger_diagnostics, seal, seal_many, verify_hxaco_many, KernelSpec, Sealed,
};
use peacemaker_lift::bundle::BundleCodec;
use sha2::{Digest, Sha256};

macro_rules! fixture {
    ($name:literal) => {
        include_bytes!(concat!("fixtures/cdna/rip942/", $name, ".co")) as &[u8]
    };
}

/// GEMM (f32 in), GEMM (bf16 in), tile descriptor; the order of the recorded seal reports.
const SYMBOLS: [&str; 3] = [
    "gemm_fp8b128_moe_grouped_bf16mfma",
    "gemm_fp8b128_moe_grouped_bf16mfma_xbf16",
    "gemm_fp8b128_moe_tile_desc_pad16",
];

/// Frozen outcome of the pre-change verifier.
enum Recorded {
    /// Sealed. Bundle bytes, per-symbol `.text` hash and the old exploration statistics
    /// (`steps`, `label_states`; these may change and are informational only).
    Sealed {
        hxaco_sha256: &'static str,
        hxaco_bytes: usize,
        text_sha256: [&'static str; 3],
        old_stats: [(u64, u64); 3],
    },
    /// First entry refused: VMEM->VMEM WAW without a sufficient vmcnt wait.
    Waw {
        first_refusal: &'static str,
        earlier_edges: u32,
    },
    /// First entry refused: more than 512 in-flight memory states at one label.
    StateCap { label: &'static str },
    /// First entry refused: `s_barrier` with LGKM not drained.
    Barrier {
        first_refusal: &'static str,
        earlier_edges: u32,
    },
}

struct Case {
    name: &'static str,
    bytes: &'static [u8],
    bundle_sha256: &'static str,
    device_elf_sha256: &'static str,
    recorded: Recorded,
}

const DESC_TEXT_SHA256: &str = "e8542a6796ab50e16dace5f426976a72db240c76e7b71d133c003ef253efdfbe";

fn positives() -> Vec<Case> {
    vec![
        Case {
            name: "default",
            bytes: fixture!("default"),
            bundle_sha256: "08ebe4036da4c71b2c5d685af77a6a3692b975a2a28839c3683200dca2de10c0",
            device_elf_sha256: "8359a935d4dc453a76f4856d8dbba5d2230e4957c3a966de982c6308b549ea65",
            recorded: Recorded::Sealed {
                hxaco_sha256: "fc16c27637517e54f078c25f3f9166d2663df10a840badda4f83ecc69008ea44",
                hxaco_bytes: 22392,
                text_sha256: [
                    "c6b140595c83c0edf58809a78ced70f3f0f57220724d9071f513d05a846de7b1",
                    "8f45ae49d9b5ab89126c4bb800fd8c8a0165185dd18ae0bf03955e3f6d7b024f",
                    DESC_TEXT_SHA256,
                ],
                old_stats: [(10407, 216), (10501, 229), (25, 2)],
            },
        },
        Case {
            name: "d1",
            bytes: fixture!("d1"),
            bundle_sha256: "5a709090b3bc6cb19d9d3929c1dd73c762f402f9df554ef79c4067b89f83042e",
            device_elf_sha256: "9118b1b9c4c4bb44cc8120658efccde41c73093c6cfbc118f15d1688dcbc6d8c",
            recorded: Recorded::Sealed {
                hxaco_sha256: "3a2ba69bb94524a4d9649b64e35f390281db6b37478f171138f8dde2be290a06",
                hxaco_bytes: 17528,
                text_sha256: [
                    "5e78104b3da49affac64417742a408291e97c0959f6f0edc42e5a5b1c722239b",
                    "d59d774133b949a34823e6d5efcd3a7f3ba24ac24a3d1ecbe68e16e10f36640c",
                    DESC_TEXT_SHA256,
                ],
                old_stats: [(1462, 30), (1428, 31), (25, 2)],
            },
        },
        Case {
            name: "d4",
            bytes: fixture!("d4"),
            bundle_sha256: "c50c6de526c50307e9a7965b6f2d641ae60687878d91c267d56e54b0a83f8dd2",
            device_elf_sha256: "210fbd032436928e27108cf6872684c297ca10cf13a4e4c3c412ac4d93863d59",
            recorded: Recorded::Sealed {
                hxaco_sha256: "a76ab787f784e347886065313f52228b7f30dd50426590a288f2b0eacbf6be6e",
                hxaco_bytes: 31096,
                text_sha256: [
                    "22df187a0529dcff2a3d9d1eef8d83cdd4f21e17bbfcf3c56923935d0d9cdea1",
                    "3594df7e64687f00767361a693e5ba8111d4fedfe3e444eb0b6f380b86efbf86",
                    DESC_TEXT_SHA256,
                ],
                old_stats: [(263141, 4740), (266731, 4975), (25, 2)],
            },
        },
        Case {
            name: "xp2",
            bytes: fixture!("xp2"),
            bundle_sha256: "87d5220da1cd9cf334bf806408f88fd896f38c6ae41daf47b42be43b972d10e1",
            device_elf_sha256: "a8202cf17a9758e21c52f6fc6fafb7622472837f9a14a0881b26a494ab59e498",
            recorded: Recorded::Sealed {
                hxaco_sha256: "a8ac8fb4843feb215ff83dec94894e7eeb2ac446e051a6eb9ac5db690b83b8f8",
                hxaco_bytes: 22136,
                text_sha256: [
                    "886f45d1f44d63e0fb50a527a2afc5b3362988a5abc30063a581d493ff0093fd",
                    "2e8e52fdcf3fa4f1989e5b2ec828aea0de7e950700ecac891e8394c993b8c923",
                    DESC_TEXT_SHA256,
                ],
                old_stats: [(10377, 216), (10471, 229), (25, 2)],
            },
        },
        Case {
            name: "pf2",
            bytes: fixture!("pf2"),
            bundle_sha256: "e986557583ad4334af60ce7f827846d220a67aef5e9a139794bae4561133071f",
            device_elf_sha256: "2ff00592747c451d7aa24da2d8f1026ff1be0a0bc2537b4f0df298345434063c",
            recorded: Recorded::Sealed {
                hxaco_sha256: "c978eb3a8dbcef82e5c9695945df73a070fd6de611d3c199a07c7fa0836eeb54",
                hxaco_bytes: 37496,
                text_sha256: [
                    "03bb747f7c9595b4063fa0756d30a276eb0096fbf4572546a5fdad73510069bd",
                    "7801be0c9559ff2a4db756bc28eeb35241c5d8daaa908b8ec7103039fa1437f1",
                    DESC_TEXT_SHA256,
                ],
                old_stats: [(98016, 1563), (98944, 1635), (25, 2)],
            },
        },
    ]
}

/// The six mandatory double-buffer-related refusals.
fn waw_refusals() -> Vec<Case> {
    vec![
        Case {
            name: "pp",
            bytes: fixture!("pp"),
            bundle_sha256: "4614b52587ab7143c5e728bb0b0de372bd63023394bd4cdeea47df1b36646a7a",
            device_elf_sha256: "54286d348f66c5e21f66e0453340aef0359a4194e4e79954374429da23813099",
            recorded: Recorded::Waw {
                first_refusal: "pc 440 `global_load_dwordx4 v[0:3], v[72:73], off offset:384` (after label `L415`): needs Vm wait 9 (Waw { reg: \"v[0:3]\" })",
                earlier_edges: 40,
            },
        },
        Case {
            name: "pp_d4",
            bytes: fixture!("pp_d4"),
            bundle_sha256: "87b336ae060b144c40ecdd85a822065bd137cb415d744f541258d8173b49223f",
            device_elf_sha256: "9ff5a814498cc4244536ca742e878693ac900f439a9ec058f7e78ab3435db49f",
            recorded: Recorded::Waw {
                first_refusal: "pc 619 `global_load_dwordx4 v[0:3], v[142:143], off offset:896` (after label `L587`): needs Vm wait 16 (Waw { reg: \"v[0:3]\" })",
                earlier_edges: 89,
            },
        },
        Case {
            name: "pp_xp2",
            bytes: fixture!("pp_xp2"),
            bundle_sha256: "7c6cbbd2ed940afb1c25af2be517f5502237455d1c8668cec88e996706a3a25e",
            device_elf_sha256: "b9b090a371f3d18a989043aca15102282ca364e054dab92c8890c3d298844b05",
            recorded: Recorded::Waw {
                first_refusal: "pc 437 `global_load_dwordx4 v[0:3], v[72:73], off offset:384` (after label `L412`): needs Vm wait 9 (Waw { reg: \"v[0:3]\" })",
                earlier_edges: 40,
            },
        },
        Case {
            name: "pp_gm1",
            bytes: fixture!("pp_gm1"),
            bundle_sha256: "d63d76906954ce1c4a7fc281da3e935cc7002b168327e650e5e1ca8f47631c03",
            device_elf_sha256: "f4038881f688f263255465c09d0a3539c83f610a0e22e27ccd190e71b90df47b",
            recorded: Recorded::Waw {
                first_refusal: "pc 466 `global_load_dwordx4 v[0:3], v[72:73], off offset:384` (after label `L441`): needs Vm wait 9 (Waw { reg: \"v[0:3]\" })",
                earlier_edges: 40,
            },
        },
        Case {
            name: "xall",
            bytes: fixture!("xall"),
            bundle_sha256: "35dbe93336c89619ea58420a71f7117e47de294031f3dd4aa4b1cc1e50d826bb",
            device_elf_sha256: "979e9cd7245406461463c84eb55997a75840b3ca0378e8cbbacf89df2ca93cb8",
            recorded: Recorded::Waw {
                first_refusal: "pc 311 `global_load_dwordx4 v[80:83], v[38:39], off offset:384` (after label `L290`): needs Vm wait 9 (Waw { reg: \"v[80:83]\" })",
                earlier_edges: 25,
            },
        },
        Case {
            name: "xall_pp",
            bytes: fixture!("xall_pp"),
            bundle_sha256: "c93e640cb333226fea025e449161224db6b3b9e9ddee937b8efcd84e488454a8",
            device_elf_sha256: "2653bc7fbde1d697921e299fa3c843a6826ca640e27292f05b34954459857fb5",
            recorded: Recorded::Waw {
                first_refusal: "pc 410 `global_load_dwordx4 v[30:33], v[10:11], off offset:384` (after label `L388`): needs Vm wait 9 (Waw { reg: \"v[30:33]\" })",
                earlier_edges: 39,
            },
        },
    ]
}

fn cap_cases() -> Vec<Case> {
    vec![
        Case {
            name: "d8",
            bytes: fixture!("d8"),
            bundle_sha256: "2f9b5edc2200254d717f097c499155e768c24376b92b15a8f2e071b68d7dc9ff",
            device_elf_sha256: "f11e4448bfdb1005cbb7581ab40e13c575f9b54a22b9a848fdca25182b12a1ac",
            recorded: Recorded::StateCap { label: "L777" },
        },
        Case {
            name: "relay",
            bytes: fixture!("relay"),
            bundle_sha256: "995a762d5bd64eb6b4cc5033c046427920662fa8b1993a1fd60ecff5872e2b8b",
            device_elf_sha256: "191e90e20accc1348238be2feccd25b869dc1031f6cf98218c3fd6ec6731347e",
            recorded: Recorded::StateCap { label: "L2953" },
        },
        Case {
            name: "relay_xp1",
            bytes: fixture!("relay_xp1"),
            bundle_sha256: "56d76c596dbf991814023ad95ff805b7025b6bce88d0aaafd7068d53e44416ac",
            device_elf_sha256: "a9f46d3c9ccfa089359454a4d4042e1a884e5d8a449ead37b6ee1f02fd88012a",
            recorded: Recorded::StateCap { label: "L3565" },
        },
    ]
}

fn barrier_case() -> Case {
    Case {
        name: "xp3_fs",
        bytes: fixture!("xp3_fs"),
        bundle_sha256: "0211035e0d32aa593ad6decdaf010aa431df1a76a5bd99b64800b780f278dea9",
        device_elf_sha256: "8139145a6461303417d870a1f45f3ed2357e787f75ffa5e98b3da0ebbc754d4a",
        recorded: Recorded::Barrier {
            first_refusal: "pc 263 `s_barrier` (after label `L226`): needs Lgkm wait 0 (Barrier)",
            earlier_edges: 18,
        },
    }
}

fn all_cases() -> Vec<Case> {
    let mut all = positives();
    all.extend(waw_refusals());
    all.extend(cap_cases());
    all.push(barrier_case());
    all
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// The single gfx942 device ELF of a hipcc offload bundle.
fn device_elf(bytes: &[u8]) -> Vec<u8> {
    let (bundle, payloads) = Bundle::read(bytes).expect("offload bundle");
    let entries: Vec<_> = bundle
        .entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e.triple == "hipv4-amdgcn-amd-amdhsa--gfx942")
        .collect();
    assert_eq!(entries.len(), 1, "exactly one gfx942 device entry");
    payloads[entries[0].0].to_vec()
}

/// Seal every entry on its own, in `SYMBOLS` order. Each call is a full independent
/// certification (typed lift, hazard check, ledger replay, ABI, re-bundling).
fn seal_each(case: &Case) -> Vec<Result<Sealed, String>> {
    SYMBOLS
        .iter()
        .map(|symbol| seal(case.bytes, Arch::Gfx942, symbol, None))
        .collect()
}

fn specs() -> Vec<KernelSpec<'static>> {
    SYMBOLS
        .iter()
        .map(|symbol| KernelSpec {
            symbol,
            abi_ref: None,
        })
        .collect()
}

#[test]
fn every_fixture_is_byte_exact_and_its_full_image_decodes_and_re_emits() {
    let all = all_cases();
    // Mandatory six + five positives + three caps + one barrier.
    assert_eq!(all.len(), 15);
    for case in &all {
        assert_eq!(sha(case.bytes), case.bundle_sha256, "{}: bundle", case.name);
        let elf = device_elf(case.bytes);
        assert_eq!(
            sha(&elf),
            case.device_elf_sha256,
            "{}: device ELF",
            case.name
        );
        let lifted = peacemaker_lift::cdna::lift_object(&elf)
            .unwrap_or_else(|e| panic!("{}: fully typed image expected: {e:?}", case.name));
        assert_eq!(
            peacemaker_lift::cdna::checked(&lifted)
                .unwrap_or_else(|e| panic!("{}: typed re-emission failed: {e:?}", case.name)),
            elf.as_slice(),
            "{}: decode/re-emit must reproduce the exact device image",
            case.name
        );
        let mut names: Vec<_> = lifted
            .kernels
            .iter()
            .map(|k| k.image.name.as_str())
            .collect();
        names.sort_unstable();
        let mut expected = SYMBOLS;
        expected.sort_unstable();
        assert_eq!(
            names, expected,
            "{}: both GEMM entries + descriptor",
            case.name
        );
    }
}

#[test]
fn recorded_sealing_positives_still_seal_all_three_entries_bit_identically() {
    for case in positives() {
        let Recorded::Sealed {
            hxaco_sha256,
            hxaco_bytes,
            text_sha256,
            old_stats,
        } = case.recorded
        else {
            unreachable!()
        };
        // Production path: one bundle for the three entries.
        let sealed = seal_many(case.bytes, Arch::Gfx942, &specs())
            .unwrap_or_else(|e| panic!("{}: recorded positive no longer seals: {e}", case.name));
        assert_eq!(sealed.len(), 3, "{}", case.name);
        for (i, s) in sealed.iter().enumerate() {
            let symbol = SYMBOLS[i];
            let h = &s.report["hazards"];
            assert_eq!(s.report["kernel"], symbol, "{}", case.name);
            assert_eq!(h["checked"], true, "{} {symbol}: {h}", case.name);
            assert_eq!(h["blocking"], 0, "{} {symbol}: {h}", case.name);
            assert_eq!(h["findings"], 0, "{} {symbol}: {h}", case.name);
            assert_eq!(h["bypasses"], 0, "{} {symbol}: {h}", case.name);
            assert_eq!(h["hazard"]["inserted_nops"], 0, "{} {symbol}", case.name);
            assert_eq!(
                h["hazard"]["rewritten_instructions"], 0,
                "{} {symbol}",
                case.name
            );
            assert_eq!(
                s.elf_sha256, case.device_elf_sha256,
                "{} {symbol}",
                case.name
            );
            assert_eq!(
                s.text_sha256, text_sha256[i],
                "{} {symbol}: .text",
                case.name
            );
            assert_eq!(
                s.hxaco_sha256, hxaco_sha256,
                "{} {symbol}: bundle",
                case.name
            );
            assert_eq!(s.hxaco.len(), hxaco_bytes, "{} {symbol}", case.name);
            // Informational diagnostics (`hip_twin::LEDGER_DIAGNOSTIC_FIELDS`): never compared
            // as a pin, only reported when they move.
            let d = ledger_diagnostics(&s.report)
                .unwrap_or_else(|| panic!("{} {symbol}: ledger diagnostics missing", case.name));
            if (d.steps, d.label_states) != old_stats[i] {
                eprintln!(
                    "rip942 {} {symbol}: ledger (steps, label_states) {:?} -> {:?}",
                    case.name,
                    old_stats[i],
                    (d.steps, d.label_states)
                );
            }
        }
        // And the sealed bundle round-trips through the verifier for every entry.
        let verified = verify_hxaco_many(&sealed[0].hxaco, Arch::Gfx942, &SYMBOLS)
            .unwrap_or_else(|e| panic!("{}: sealed bundle must re-verify: {e}", case.name));
        for (a, b) in sealed.iter().zip(&verified) {
            assert_eq!(a.hxaco, b.hxaco, "{}", case.name);
            assert_eq!(a.text_sha256, b.text_sha256, "{}", case.name);
        }
        // Each entry certifies independently too (same bytes as the combined seal).
        for (i, each) in seal_each(&case).into_iter().enumerate() {
            let each = each.unwrap_or_else(|e| panic!("{} {}: {e}", case.name, SYMBOLS[i]));
            assert_eq!(
                each.text_sha256, text_sha256[i],
                "{} {}",
                case.name, SYMBOLS[i]
            );
            assert_eq!(
                each.hxaco_sha256, hxaco_sha256,
                "{} {}",
                case.name, SYMBOLS[i]
            );
        }
    }
}

/// The old corpus recorded only its first GEMM refusal. The complete refined run
/// additionally observed the same refusal class for xbf16 and a clean descriptor
/// for every negative variant; pin those actual outcomes rather than accepting either.
fn assert_refused(case: &Case, class: &[&str]) -> String {
    let each = seal_each(case);
    let first = match &each[0] {
        Ok(_) => panic!(
            "{}: recorded refusal now seals `{}`; that is a changed outcome for the integration \
             owner to evaluate against an independent proof, not an automatic pass",
            case.name, SYMBOLS[0]
        ),
        Err(e) => e.clone(),
    };
    assert!(first.contains(SYMBOLS[0]), "{}: {first}", case.name);
    assert!(
        first.contains("wait ledger replay"),
        "{}: {first}",
        case.name
    );
    for needle in class {
        assert!(
            first.contains(needle),
            "{}: recorded refusal class `{needle}` missing: {first}",
            case.name
        );
    }
    let second = match &each[1] {
        Ok(_) => panic!("{}: xbf16 no longer has its observed refusal", case.name),
        Err(error) => error,
    };
    assert!(second.contains(SYMBOLS[1]), "{}: {second}", case.name);
    assert!(
        second.contains("wait ledger replay"),
        "{}: {second}",
        case.name
    );
    for needle in class {
        assert!(second.contains(needle), "{} xbf16: {second}", case.name);
    }
    let descriptor = each[2]
        .as_ref()
        .unwrap_or_else(|error| panic!("{}: descriptor no longer certifies: {error}", case.name));
    assert_eq!(
        descriptor.elf_sha256, case.device_elf_sha256,
        "{}",
        case.name
    );
    assert_eq!(descriptor.text_sha256, DESC_TEXT_SHA256, "{}", case.name);
    let hazards = &descriptor.report["hazards"];
    assert_eq!(hazards["checked"], true);
    assert_eq!(hazards["blocking"], 0);
    assert_eq!(hazards["findings"], 0);
    verify_hxaco_many(&descriptor.hxaco, Arch::Gfx942, &[SYMBOLS[2]])
        .unwrap_or_else(|error| panic!("{}: descriptor re-verification: {error}", case.name));
    // The production multi-entry seal refuses the whole bundle and names the first entry.
    let many = match seal_many(case.bytes, Arch::Gfx942, &specs()) {
        Ok(_) => panic!(
            "{}: seal_many sealed a bundle whose first entry is refused",
            case.name
        ),
        Err(e) => e,
    };
    assert!(many.contains(SYMBOLS[0]), "{}: {many}", case.name);
    first
}

#[test]
fn mandatory_six_double_buffer_variants_have_pinned_ordered_writeback_outcomes() {
    for case in waw_refusals() {
        let Recorded::Waw {
            first_refusal,
            earlier_edges,
        } = case.recorded
        else {
            unreachable!()
        };
        let cap_label = match case.name {
            "pp" | "pp_xp2" | "pp_gm1" => None,
            "pp_d4" => Some("L491"),
            "xall" => Some("L1258"),
            "xall_pp" => Some("L1895"),
            other => panic!("unrecorded double-buffer outcome: {other}"),
        };
        if let Some(label) = cap_label {
            let now = assert_refused(
                &case,
                &[
                    "more than 512 distinct in-flight memory states reach label",
                    "path-sensitive state explosion",
                ],
            );
            assert!(
                now.contains(&format!("label `{label}`")),
                "{}: {now}",
                case.name
            );
            continue;
        }
        let sealed = seal_many(case.bytes, Arch::Gfx942, &specs()).unwrap_or_else(|e| {
            panic!(
                "{}: ordered VMEM proof did not discharge `{first_refusal}` \
                ({earlier_edges} earlier edges): {e}",
                case.name
            )
        });
        assert_eq!(sealed.len(), SYMBOLS.len(), "{}", case.name);
        let elf = device_elf(case.bytes);
        assert_eq!(device_elf(&sealed[0].hxaco), elf, "{}", case.name);
        for (s, symbol) in sealed.iter().zip(SYMBOLS) {
            assert_eq!(s.elf_sha256, case.device_elf_sha256, "{}", case.name);
            assert_eq!(s.report["kernel"], symbol);
            let h = &s.report["hazards"];
            assert_eq!(h["checked"], true);
            for field in ["blocking", "findings", "bypasses"] {
                assert_eq!(h[field], 0, "{} {symbol}: {h}", case.name);
            }
            assert_eq!(h["hazard"]["inserted_nops"], 0);
            assert_eq!(h["hazard"]["rewritten_instructions"], 0);
        }
        assert_eq!(sealed[2].text_sha256, DESC_TEXT_SHA256);
        let verified = verify_hxaco_many(&sealed[0].hxaco, Arch::Gfx942, &SYMBOLS)
            .unwrap_or_else(|e| panic!("{}: three-entry re-verification: {e}", case.name));
        for (a, b) in sealed.iter().zip(verified) {
            assert_eq!(a.hxaco, b.hxaco, "{}", case.name);
            assert_eq!(a.text_sha256, b.text_sha256, "{}", case.name);
        }
        for (s, each) in sealed.iter().zip(seal_each(&case)) {
            let each = each.unwrap_or_else(|e| panic!("{}: independent seal: {e}", case.name));
            assert_eq!(s.text_sha256, each.text_sha256);
        }
    }
}

#[test]
fn state_cap_variants_stay_refused_with_the_unchanged_512_state_cap() {
    for case in cap_cases() {
        let Recorded::StateCap { label } = case.recorded else {
            unreachable!()
        };
        let now = assert_refused(
            &case,
            &[
                "more than 512 distinct in-flight memory states reach label",
                "path-sensitive state explosion",
            ],
        );
        if !now.contains(&format!("label `{label}`")) {
            eprintln!(
                "rip942 {}: cap label moved (recorded `{label}`): {now}",
                case.name
            );
        }
    }
}

#[test]
fn xp3_fs_stays_refused_for_the_undrained_lgkm_barrier() {
    let case = barrier_case();
    let Recorded::Barrier {
        first_refusal,
        earlier_edges,
    } = case.recorded
    else {
        unreachable!()
    };
    let now = assert_refused(&case, &["needs Lgkm wait 0 (Barrier)", "`s_barrier`"]);
    if !now.contains(first_refusal) {
        eprintln!(
            "rip942 xp3_fs: first refusal moved (recorded {earlier_edges} earlier edges): \
             recorded `{first_refusal}`; now: {now}"
        );
    }
}
