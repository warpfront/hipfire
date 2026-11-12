// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU-free gate of the committed chunked-GDN hipcc twins
//! (`kernels/gated_delta_chunk_{,arena_,q8_,q8_arena_}pm_gfx1151.hxaco`,
//! `kernels/gated_delta_chunk_pm_gfx1100.hxaco`, sealed by `scripts/pm-hip-twin.sh`): each hxaco
//! re-verifies (lift identity through `emit::checked`, byte-for-byte re-bundle, static hazards),
//! its kernarg layout is the incumbent's frozen layout, its static LDS is the 31168 bytes the
//! hipcc kernel allocates, and the sealed `.text` sha256 equals the incumbent hipcc `.text`
//! recorded by the seal (`*.twin.json`: the script aborts unless they are equal).

use peacemaker_isa::arch::Arch;
use peacemaker_isa::hip_twin::{verify_hxaco, Sealed};
use sha2::{Digest, Sha256};

struct Module {
    name: &'static str,
    hxaco: &'static [u8],
    sha: &'static str,
    twin_json: &'static str,
    arch: Arch,
    symbol: &'static str,
    incumbent: &'static str,
    /// Explicit kernarg bytes after the 15-argument F32 prefix (96 bytes): `state_q8`, `frame`,
    /// `z_pitch`, `z_col0` as the variant carries them.
    extra_i32: u32,
}

macro_rules! module {
    ($name:literal, $arch:expr, $symbol:literal, $incumbent:literal, $extra:expr) => {
        Module {
            name: $name,
            hxaco: include_bytes!(concat!("../../../kernels/", $name, ".hxaco")),
            sha: include_str!(concat!("../../../kernels/", $name, ".hxaco.sha256")),
            twin_json: include_str!(concat!("../../../kernels/", $name, ".twin.json")),
            arch: $arch,
            symbol: $symbol,
            incumbent: $incumbent,
            extra_i32: $extra,
        }
    };
}

fn modules() -> [Module; 5] {
    [
        module!(
            "gated_delta_chunk_pm_gfx1151",
            Arch::Gfx1151,
            "gated_delta_chunk_gate_wmma_pm_gfx1151",
            "gated_delta_chunk_gate_wmma",
            0
        ),
        module!(
            "gated_delta_chunk_arena_pm_gfx1151",
            Arch::Gfx1151,
            "gated_delta_chunk_gate_wmma_arena_pm_gfx1151",
            "gated_delta_chunk_gate_wmma_arena",
            2
        ),
        module!(
            "gated_delta_chunk_q8_pm_gfx1151",
            Arch::Gfx1151,
            "gated_delta_chunk_gate_q8_wmma_pm_gfx1151",
            "gated_delta_chunk_gate_q8_wmma",
            2
        ),
        module!(
            "gated_delta_chunk_q8_arena_pm_gfx1151",
            Arch::Gfx1151,
            "gated_delta_chunk_gate_q8_wmma_arena_pm_gfx1151",
            "gated_delta_chunk_gate_q8_wmma_arena",
            4
        ),
        module!(
            "gated_delta_chunk_pm_gfx1100",
            Arch::Gfx1100,
            "gated_delta_chunk_gate_wmma_pm_gfx1100",
            "gated_delta_chunk_gate_wmma",
            0
        ),
    ]
}

/// Explicit (non-hidden) kernargs of the sealed report, in metadata order.
fn explicit(s: &Sealed) -> Vec<(u32, u32, bool)> {
    s.report["abi"]["args"]
        .as_array()
        .expect("abi args")
        .iter()
        .filter(|a| !a["value_kind"].as_str().unwrap().starts_with("hidden_"))
        .map(|a| {
            (
                a["offset"].as_u64().unwrap() as u32,
                a["size"].as_u64().unwrap() as u32,
                a["pointer"].as_bool().unwrap(),
            )
        })
        .collect()
}

/// The frozen incumbent layout: nine pointers (projection, qn, kn, gate, beta, state, z, norm,
/// output), then six 4-byte values (rows, qkv_width, key_heads, value_heads, query_scale,
/// output_bf16) and `extra` more 4-byte values (the Q8 `state_q8` / `frame`, the arena
/// `z_pitch` / `z_col0`).
fn want_layout(extra: u32) -> Vec<(u32, u32, bool)> {
    let mut v: Vec<(u32, u32, bool)> = (0..9).map(|i| (8 * i, 8, true)).collect();
    v.extend((0..6 + extra).map(|i| (72 + 4 * i, 4, false)));
    v
}

#[test]
fn every_module_verifies_with_the_incumbent_kernarg_layout_and_text() {
    for m in modules() {
        assert_eq!(
            format!("{:x}", Sha256::digest(m.hxaco)),
            m.sha,
            "{}: pinned .sha256 differs",
            m.name
        );
        assert_eq!(m.sha.len(), 64, "{}: sha file must be bare", m.name);
        let s = verify_hxaco(m.hxaco, m.arch, m.symbol).expect(m.name);
        assert_eq!(s.scratch, 0, "{}: scratch", m.name);
        assert_eq!(s.report["hazards"]["checked"], true);
        assert_eq!(s.report["hazards"]["blocking"], 0);
        assert_eq!(s.report["hazards"]["pm_ir_obligations"], 0);
        assert_eq!(s.hxaco, m.hxaco);
        assert_eq!(explicit(&s), want_layout(m.extra_i32), "{}", m.name);
        let seg = 96 + 4 * m.extra_i32;
        assert_eq!(
            s.report["abi"]["kernarg_segment_size"], seg,
            "{}: kernarg segment",
            m.name
        );
        // k_s, q_s, kb_s (3 * 16 * 136 f16) + kdT (128 * 16 f16) + vb_s (16 * 132 f32) + t_s, p_s
        // (2 * 16 * 16 f16) + g_s, bt_s, inv_s (3 * 16 f32) + out_s (16 * 136 u16).
        let lds = 3 * 16 * 136 * 2
            + 128 * 16 * 2
            + 16 * 132 * 4
            + 2 * 16 * 16 * 2
            + 3 * 16 * 4
            + 16 * 136 * 2;
        assert_eq!(s.lds, lds, "{}: static LDS", m.name);
        // gfx1151 is pinned to six waves per SIMD (<= 240 VGPRs); gfx1100 is not pinned, and its
        // F32 triangular products (see the kernel header) allocate 248.
        let vgprs = if m.arch == Arch::Gfx1100 { 248 } else { 240 };
        assert_eq!(s.vgprs, vgprs, "{}: vgprs", m.name);

        // The seal's judged code is the incumbent hipcc kernel's own `.text`.
        let rec: serde_json::Value = serde_json::from_str(m.twin_json).expect("twin.json");
        assert_eq!(rec["module"], m.name);
        let judged = &rec["judged_text_check"][0];
        assert_eq!(judged["symbol"], m.symbol);
        assert_eq!(judged["judged_candidate"], "incumbent-hipcc");
        assert_eq!(
            judged["equal"], true,
            "{}: twin .text != incumbent .text",
            m.name
        );
        assert_eq!(judged["judged_code_sha256"], judged["twin_text_sha256"]);
        assert_eq!(judged["twin_text_sha256"], s.text_sha256, "{}", m.name);
        assert_eq!(rec["reports"][0]["abi"]["incumbent_symbol"], m.incumbent);
        assert_eq!(rec["reports"][0]["abi"]["compared_to_incumbent"], true);
    }
}

#[test]
fn wrong_arch_and_missing_symbols_are_refused() {
    let ms = modules();
    // A gfx1151 object is not a gfx1100 object and vice versa.
    assert!(verify_hxaco(ms[0].hxaco, Arch::Gfx1100, ms[0].symbol).is_err());
    assert!(verify_hxaco(ms[4].hxaco, Arch::Gfx1151, ms[4].symbol).is_err());
    assert!(verify_hxaco(ms[0].hxaco, Arch::Gfx1201, ms[0].symbol).is_err());
    // Each module carries its own symbol only.
    assert!(verify_hxaco(ms[0].hxaco, Arch::Gfx1151, ms[1].symbol).is_err());
    assert!(verify_hxaco(ms[1].hxaco, Arch::Gfx1151, ms[0].symbol).is_err());
    // Distinct code per module.
    let mut shas: Vec<&str> = ms.iter().map(|m| m.sha).collect();
    shas.sort_unstable();
    shas.dedup();
    assert_eq!(shas.len(), ms.len());
}
