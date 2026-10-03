//! QSA gathered F16 WMMA builder family (gfx1151 + gfx1201).
use hipfire_isa::Arch;
use hipfire_isa::kernels::qsa_gather::{self, Kind, Spec};

const ARCHES: [Arch; 2] = [Arch::Gfx1151, Arch::Gfx1201];

#[test]
fn entries_are_deterministic_and_target_only_their_arches() {
    for arch in ARCHES {
        for kind in Kind::ALL {
            let spec = Spec { arch, kind };
            let (a, b) = (qsa_gather::emit(spec).unwrap(), qsa_gather::emit(spec).unwrap());
            assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256);
            // One staging barrier and three per tile; the producer has no LDS.
            assert_eq!(a.shape.barriers, if kind == Kind::Attend { 4 } else { 0 }, "{spec:?}");
        }
    }
    assert!(qsa_gather::emit(Spec { arch: Arch::Gfx1100, kind: Kind::Attend }).is_err());
}

/// The runtime embeds one certified module per arch: it must be exactly
/// what the builder emits today.
#[test]
fn committed_bundles_equal_fresh_emission() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../kernels");
    for arch in ARCHES {
        let module = Spec::module(arch);
        let committed = std::fs::read(format!("{root}/{module}.hxaco")).unwrap();
        let text = qsa_gather::emit_module(arch).unwrap().1;
        assert!(text_section(&link(&text, arch.name(), &module)) == text_section(&committed), "{module}: fresh emission differs from the committed bundle");
    }
}

const LLVM: &str = "/opt/rocm/core-10.0/lib/llvm/bin";

fn link(text: &str, arch: &str, stem: &str) -> Vec<u8> {
    use std::process::Command;
    let dir = std::env::temp_dir().join(format!("hipfire-isa-qsa-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (s, o, co) = (dir.join(format!("{stem}.s")), dir.join(format!("{stem}.o")), dir.join(format!("{stem}.co")));
    std::fs::write(&s, text).unwrap();
    assert!(Command::new(format!("{LLVM}/llvm-mc")).args(["-triple=amdgcn-amd-amdhsa", &format!("-mcpu={arch}"), "-filetype=obj"]).arg(&s).arg("-o").arg(&o).status().unwrap().success());
    assert!(Command::new(format!("{LLVM}/ld.lld")).arg("-shared").arg(&o).arg("-o").arg(&co).status().unwrap().success());
    std::fs::read(co).unwrap()
}

/// `.text` of an ELF, or of the device ELF inside a clang offload bundle.
fn text_section(bytes: &[u8]) -> Vec<u8> {
    let start = bytes.windows(4).position(|w| w == b"\x7fELF").unwrap();
    let e = &bytes[start..];
    let u16_ = |o: usize| u16::from_le_bytes([e[o], e[o + 1]]) as usize;
    let u32_ = |o: usize| u32::from_le_bytes(e[o..o + 4].try_into().unwrap()) as usize;
    let u64_ = |o: usize| u64::from_le_bytes(e[o..o + 8].try_into().unwrap()) as usize;
    let (shoff, shentsize, shnum, shstrndx) = (u64_(0x28), u16_(0x3a), u16_(0x3c), u16_(0x3e));
    let sh = |i: usize| shoff + i * shentsize;
    let names = u64_(sh(shstrndx) + 24);
    (0..shnum).find_map(|i| {
        let name = &e[names + u32_(sh(i))..];
        name.starts_with(b".text\0").then(|| e[u64_(sh(i) + 24)..u64_(sh(i) + 24) + u64_(sh(i) + 32)].to_vec())
    }).unwrap()
}

#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use hipfire_isa::{ledger_replay, pm_check, toolchain::{build, certify, Toolchain}};

    /// Every symbol of both modules certifies: parse-back, independent wait
    /// replay, its committed shape contract, the static LDS bound with the
    /// token list host-bounded, and M7's lift identity plus
    /// wait/hazard/barrier/window analyses of the linked ELF, obligation-free.
    #[test]
    fn modules_pass_certification_for_every_symbol() {
        for arch in ARCHES {
            let (emitted, text, _) = qsa_gather::emit_module(arch).unwrap();
            for e in &emitted { ledger_replay::replay_waits(&e.s_text, arch).unwrap(); }
            let dir = std::env::temp_dir().join(format!("hipfire-isa-qsa-cert-{}-{}", arch.name(), std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let s = dir.join("module.s");
            std::fs::write(&s, &text).unwrap();
            let toolchain = Toolchain::oracle();
            let build = build(&toolchain, &s, &dir.join("module.hsaco"), arch.name()).unwrap();
            for kind in Kind::ALL {
                let symbol = Spec { arch, kind }.symbol();
                let m7 = pm_check::m7(&build.elf, arch.name(), &symbol).unwrap_or_else(|e| panic!("{symbol}: {e}"));
                assert_eq!((m7["lift"].as_str(), &m7["obligations"]), (Some("byte-exact"), &serde_json::json!({})), "{symbol}");
                let path = format!("{}/kernels/qsa_gather.{}.{}.contract.json", env!("CARGO_MANIFEST_DIR"), arch.name(), kind.tag());
                let contract = serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"))).unwrap();
                certify(&toolchain, &build, &s, arch.name(), &dir.join(format!("{}.manifest.json", kind.tag())), Some(&contract), "test", "test")
                    .unwrap_or_else(|e| panic!("{symbol}: {e}"));
            }
        }
    }

    /// The token list is the only LDS the static bound leaves to the launch:
    /// every other access ends inside the 18432 static bytes.
    #[test]
    fn static_lds_accesses_stay_inside_the_static_allocation() {
        for arch in ARCHES {
            let e = qsa_gather::emit(Spec { arch, kind: Kind::Attend }).unwrap();
            let symbol = Spec { arch, kind: Kind::Attend }.symbol();
            let (end, hosted) = pm_check::lds_bounds_host(&e.s_text, &symbol, 8, qsa_gather::STATIC_LDS, &qsa_gather::TOKEN_ADDRESS_VGPRS).unwrap();
            assert!(end <= qsa_gather::STATIC_LDS && hosted > 0, "{symbol}: static end {end}, {hosted} token accesses");
            // Without the token-list exemption the dynamic accesses are refused.
            assert!(pm_check::lds_bounds(&e.s_text, &symbol, 8, qsa_gather::STATIC_LDS + qsa_gather::TOKEN_LDS_MAX).is_err());
        }
    }
}
