// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Standalone emitter + certifier of the gfx1100 MQ6 residual twin
//! (`qwen4_mq6_x4_gfx11` builder, `Kind::Resid`): one object `gemm_mq6g256v2_residual_pm_gfx1100`
//! holding both `wr` entries `gemm_mq6g256v2_residual_pm_gfx1100_w{4,8}` (distinct from hipcc's
//! `gemm_mq6g256v2_residual_wmma_gfx11_mw8_lds`), as `kernels/gemm_mq6g256v2_residual_pm_gfx1100.*`.
//!
//! ```text
//! nice -n 19 cargo run --release -j 8 -p peacemaker-isa --features toolchain --example mq6_resid_gfx1100_emit -- kernels
//! ```
//!
//! Writes `<out-dir>/<module>.{s,proof.json,co,hxaco,hxaco.sha256,m7.json}` and `<out-dir>/emit.json`. Every stage is a hard stop: wait replay, empty hazard
//! replay, native assembly, M7 on the native object, ROCm `llvm-mc` + `ld.lld` oracle build of the
//! same source with M7 on it, and a `.text` byte comparison of native vs oracle.
#![allow(dead_code)]

#[cfg(not(feature = "toolchain"))]
fn main() {
    println!(
        "{}",
        serde_json::json!({"verdict": "kill", "stage": "features", "reason": "build with --features toolchain (M7, ROCm oracle build)"})
    );
    std::process::exit(1)
}

#[cfg(feature = "toolchain")]
use peacemaker_isa::{
    kernels::qwen4_mq6_x4_gfx11::{self as mq6, Cfg, Fmt, Kind},
    ledger_replay, native, pm_check,
    toolchain::{build, Toolchain},
    Arch,
};
#[cfg(feature = "toolchain")]
use serde_json::{json, Value};
#[cfg(feature = "toolchain")]
use sha2::{Digest, Sha256};
#[cfg(feature = "toolchain")]
use std::path::{Path, PathBuf};

#[cfg(feature = "toolchain")]
const ARCH: Arch = Arch::Gfx1100;

#[cfg(feature = "toolchain")]
fn kill(stage: &str, reason: impl std::fmt::Display) -> ! {
    println!(
        "{}",
        json!({"verdict": "kill", "stage": stage, "reason": reason.to_string()})
    );
    std::process::exit(1)
}

#[cfg(feature = "toolchain")]
fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(feature = "toolchain")]
fn put(stage: &str, path: &Path, bytes: impl AsRef<[u8]>) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .unwrap_or_else(|e| kill(stage, format!("{}: {e}", parent.display())))
    }
    std::fs::write(path, bytes).unwrap_or_else(|e| kill(stage, format!("{}: {e}", path.display())));
}

/// `.text` of an ELF, or of the device ELF inside a clang offload bundle.
#[cfg(feature = "toolchain")]
fn text_section(bytes: &[u8]) -> Vec<u8> {
    let start = bytes
        .windows(4)
        .position(|w| w == b"\x7fELF")
        .unwrap_or_else(|| kill("text", "no ELF"));
    let e = &bytes[start..];
    let u16_ = |o: usize| u16::from_le_bytes([e[o], e[o + 1]]) as usize;
    let u32_ = |o: usize| u32::from_le_bytes(e[o..o + 4].try_into().unwrap()) as usize;
    let u64_ = |o: usize| u64::from_le_bytes(e[o..o + 8].try_into().unwrap()) as usize;
    let (shoff, shentsize, shnum, shstrndx) = (u64_(0x28), u16_(0x3a), u16_(0x3c), u16_(0x3e));
    let sh = |i: usize| shoff + i * shentsize;
    let names = u64_(sh(shstrndx) + 24);
    (0..shnum)
        .find_map(|i| {
            let name = &e[names + u32_(sh(i))..];
            name.starts_with(b".text\0")
                .then(|| e[u64_(sh(i) + 24)..u64_(sh(i) + 24) + u64_(sh(i) + 32)].to_vec())
        })
        .unwrap_or_else(|| kill("text", "no .text"))
}

/// First `.amdhsa_<key> N` value of the source.
#[cfg(feature = "toolchain")]
fn directive(s: &str, key: &str) -> u32 {
    s.lines()
        .find_map(|l| l.trim().strip_prefix(key))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or_else(|| kill("resources", format!("missing {key}")))
}

#[cfg(feature = "toolchain")]
fn main() {
    let usage = "usage: mq6_resid_gfx1100_emit <out-dir> [--host-target TRIPLE]";
    let mut args = std::env::args().skip(1);
    let out = PathBuf::from(args.next().unwrap_or_else(|| kill("usage", usage)));
    let host = match args.next().as_deref() {
        None => native::DEFAULT_HOST_TARGET.to_owned(),
        Some("--host-target") => args.next().unwrap_or_else(|| kill("usage", usage)),
        Some(_) => kill("usage", usage),
    };
    let toolchain = Toolchain::oracle();
    let stem = Cfg::resid_module(ARCH);
    let (emitted, text, proof) = mq6::emit_resid_module(ARCH).unwrap_or_else(|e| kill("emit", e));
    // Every stage is a hard stop: wait replay, empty hazard replay, native assembly, M7 on the
    // native object, ROCm oracle build of the same module source with M7, `.text` byte identity.
    ledger_replay::replay_waits(&text, ARCH).unwrap_or_else(|er| kill("wait_replay", er));
    let hz =
        ledger_replay::replay_hazards(&text, ARCH).unwrap_or_else(|er| kill("hazard_replay", er));
    if !hz.is_empty() {
        kill("hazard_replay", format!("{} hazards: {hz:?}", hz.len()))
    }
    put("output", &out.join(format!("{stem}.s")), &text);
    put(
        "output",
        &out.join(format!("{stem}.proof.json")),
        serde_json::to_vec_pretty(&proof).unwrap(),
    );
    let elf = native::assemble(&text, ARCH).unwrap_or_else(|er| kill("native_assemble", er));
    let co = out.join(format!("{stem}.co"));
    put("output", &co, &elf);
    let bundle = native::bundle(&elf, ARCH, &host);
    put("output", &out.join(format!("{stem}.hxaco")), &bundle);
    put(
        "output",
        &out.join(format!("{stem}.hxaco.sha256")),
        sha256(&bundle),
    );
    let odir = out.join("oracle");
    let os = odir.join(format!("{stem}.s"));
    put("output", &os, &text);
    let built = build(
        &toolchain,
        &os,
        &odir.join(format!("{stem}.hsaco")),
        ARCH.name(),
    )
    .unwrap_or_else(|er| kill("oracle_build", er));
    let (tn, to) = (
        text_section(&elf),
        text_section(&std::fs::read(&built.elf).unwrap()),
    );
    if tn != to {
        kill("text_identity", "native and ROCm .text differ")
    }
    let mut m7 = serde_json::Map::new();
    let mut rec = serde_json::Map::new();
    for (wr, e) in Cfg::RESID.into_iter().zip(&emitted) {
        let symbol = Cfg {
            arch: ARCH,
            wr,
            kind: Kind::Resid,
            fmt: Fmt::Mq6,
        }
        .symbol();
        let m7n = pm_check::m7(&co, ARCH.name(), &symbol)
            .unwrap_or_else(|er| kill("m7_native", format!("w{wr} {symbol}: {er}")));
        if m7n["lift"].as_str() != Some("byte-exact") || m7n["obligations"] != json!({}) {
            kill("m7_native", format!("w{wr}: {m7n}"))
        }
        let m7o = pm_check::m7(&built.elf, ARCH.name(), &symbol)
            .unwrap_or_else(|er| kill("m7_oracle", format!("w{wr} {symbol}: {er}")));
        if m7o["lift"].as_str() != Some("byte-exact") || m7o["obligations"] != json!({}) {
            kill("m7_oracle", format!("w{wr}: {m7o}"))
        }
        rec.insert(
            format!("w{wr}"),
            json!({
                "symbol": symbol,
                "s_text_sha256": e.proof.s_text_sha256,
                "vgprs": directive(&e.s_text, ".amdhsa_next_free_vgpr"),
                "sgprs": directive(&e.s_text, ".amdhsa_next_free_sgpr"),
                "lds_bytes": directive(&e.s_text, ".amdhsa_group_segment_fixed_size"),
                "workgroup_size": 32 * u32::from(wr),
                "geometry": format!("tile_m={} tile_n=256 block={}", 16 * u32::from(wr), 32 * u32::from(wr)),
            }),
        );
        m7.insert(symbol, m7n);
    }
    put(
        "output",
        &out.join(format!("{stem}.m7.json")),
        serde_json::to_vec_pretty(&Value::Object(m7)).unwrap(),
    );
    let rec = json!({
        "module": stem, "co_sha256": sha256(&elf), "hxaco_sha256": sha256(&bundle),
        "text_sha256": sha256(&tn), "text_bytes": tn.len(), "m7_oracle_text_identical": true,
        "entries": rec,
    });
    put(
        "output",
        &out.join("emit.json"),
        serde_json::to_vec_pretty(&rec).unwrap(),
    );
    println!("{}", json!({"verdict": "pass", "record": rec}));
}
