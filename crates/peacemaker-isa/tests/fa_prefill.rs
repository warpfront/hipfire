// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The gfx1201 causal Q8_0 FA2-GQA PeaceMaker twin `fa_prefill_pm_gfx1201` (H16 / KV2 / D256): two exports, the Q
//! preconvert `attention_fa2_q_preconvert_pm_gfx1201` then the direct attention `attention_q8_0_fa2_gqa_pm_gfx1201`
//! (`n_splits = 1`). Exact spec/arch/domain admission, determinism, the 60/40-byte kernarg ABI and resource descriptor,
//! the independently enumerated LDS fragment / output ownership, the causal bound and mask predicates, the real emitted
//! code run on the peacemaker CPU interpreter (preconvert byte-exact against an independent RNE model; attention against
//! exactly representable references with canaries, unchanged-input checks and replay), the committed bundle / contracts /
//! M7 receipt against fresh emission, and (feature `toolchain`) the ROCm-oracle certification.
//!
//! The attention references are exact by construction, not tolerance-based: scores are separated by far more than the
//! f32 `exp` underflow distance (probabilities are exactly 0 or 1) or tie exactly (scale 0), the V operands are small
//! integers, so every expected word is an exactly representable f32 sum times the correctly rounded reciprocal. The
//! interpreter is a CPU model of the WMMA / exp / reciprocal arithmetic: it does not replace the GPU byte oracle.
//!
//! The shared-codec interpreter gate below also feeds independently packed native FP8 rows to the FP8 direct export,
//! using this harness's Q8 reference, Q preconvert, ABI checks and guarded memory rig. The exhaustive FP8 decoder,
//! scale-boundary, causal and NaN classification tests remain in `fa_prefill_fp8.rs`; they are not duplicated here.
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;
use peacemaker_isa::kernels::fa_prefill::{self, FaKvCodec, FaPrefillSpec};
use peacemaker_isa::{native, Arch};
use serde_json::Value;
#[cfg(feature = "toolchain")]
#[path = "support/native_compare.rs"]
mod native_compare;
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCH: Arch = Arch::Gfx1201;
const MODULE: &str = "fa_prefill_pm_gfx1201";
const PRE: &str = "attention_fa2_q_preconvert_pm_gfx1201";
const DIRECT: &str = "attention_q8_0_fa2_gqa_pm_gfx1201";
const SYMBOLS: [&str; 2] = [PRE, DIRECT];
/// `(offset, size, name, value_kind)` in kernarg order: the pinned lab object's ABI (direct size 60).
const DIRECT_ARGS: [(u32, u32, &str, &str); 10] = [
    (0, 8, "q16", "global_buffer"),
    (8, 8, "k_cache", "global_buffer"),
    (16, 8, "v_cache", "global_buffer"),
    (24, 8, "out", "global_buffer"),
    (32, 8, "positions", "global_buffer"),
    (40, 4, "n_heads", "by_value"),
    (44, 4, "n_kv_heads", "by_value"),
    (48, 4, "head_dim", "by_value"),
    (52, 4, "batch_size", "by_value"),
    (56, 4, "scale_attn", "by_value"),
];
/// The preconvert ABI (size 40). The last three fields are historical and never dereferenced / applied.
const PRE_ARGS: [(u32, u32, &str, &str); 6] = [
    (0, 8, "q", "global_buffer"),
    (8, 8, "q16", "global_buffer"),
    (16, 8, "signs1", "global_buffer"),
    (24, 8, "signs2", "global_buffer"),
    (32, 4, "batch_size", "by_value"),
    (36, 4, "do_rotate", "by_value"),
];
const DIRECT_KERNARG: u32 = 60;
const PRE_KERNARG: u32 = 40;
const DIRECT_BLOCK: u32 = 160;
const PRE_BLOCK: u32 = 128;
const DIRECT_LDS: u32 = 65_536;
const FP8_DIRECT_LDS: u32 = 32_768;
/// Bytes per position of one K or V allocation: 2 KV heads x 8 blocks x 34 B.
const POS_STRIDE: usize = 544;
const HEAD_STRIDE: usize = 272;

fn spec() -> FaPrefillSpec {
    FaPrefillSpec {
        heads: 16,
        kv_heads: 2,
        head_dim: 256,
        kv: FaKvCodec::Q8_0,
    }
}

// ------------------------------------------------------------------------------------------------ text census
fn field(text: &str, key: &str) -> Vec<u32> {
    text.lines()
        .filter_map(|l| l.trim().strip_prefix(key))
        .map(|v| v.trim().parse().unwrap())
        .collect()
}
/// The instruction lines of an emitted source (directives, labels and metadata excluded).
fn insns(text: &str) -> Vec<&str> {
    text.lines()
        .filter_map(|l| {
            let t = l.split(';').next()?.split("//").next()?.trim();
            let tok = t.split_whitespace().next()?;
            let first = tok.chars().next()?;
            (first.is_ascii_lowercase()
                && tok.contains('_')
                && !tok.ends_with(':')
                && !tok.contains('.'))
            .then_some(t)
        })
        .collect()
}
fn mnemonic(line: &str) -> &str {
    line.split_whitespace().next().unwrap_or("")
}
fn count(text: &str, name: &str) -> usize {
    insns(text).iter().filter(|l| mnemonic(l) == name).count()
}
fn count_prefix(text: &str, prefix: &str) -> usize {
    insns(text)
        .iter()
        .filter(|l| mnemonic(l).starts_with(prefix))
        .count()
}
fn imm_offset(line: &str) -> u32 {
    line.split("offset:")
        .nth(1)
        .and_then(|v| v.split(|c: char| c.is_whitespace() || c == ',').next())
        .map_or(0, |t| match t.strip_prefix("0x") {
            Some(h) => u32::from_str_radix(h, 16).unwrap(),
            None => t.parse().unwrap(),
        })
}
/// Kernarg `(name, offset, size, value_kind)` records of a source's metadata, in order.
fn metadata_args(text: &str) -> Vec<(String, u32, u32, String)> {
    let lines: Vec<&str> = text.lines().map(str::trim).collect();
    let mut out = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        let (Some(name), Some(off)) = (
            l.strip_prefix(".name:"),
            lines.get(i + 1).and_then(|n| n.strip_prefix(".offset:")),
        ) else {
            continue;
        };
        let size = lines[i + 2]
            .strip_prefix(".size:")
            .expect(".size after .offset");
        let kind = lines[i + 3]
            .strip_prefix(".value_kind:")
            .expect(".value_kind after .size");
        out.push((
            name.trim().to_owned(),
            off.trim().parse().unwrap(),
            size.trim().parse().unwrap(),
            kind.trim().to_owned(),
        ));
    }
    out
}
fn emitted() -> (Vec<peacemaker_isa::plan::Emitted>, String) {
    let (e, text, _) = fa_prefill::emit_module(ARCH, &spec()).unwrap();
    (e, text)
}

// ------------------------------------------------------------------------------------------------ 1. spec / domain
#[test]
fn spec_and_domain_are_exact() {
    assert_eq!(spec().validate(), Ok(()));
    assert!(fa_prefill::emit(ARCH, &spec()).is_ok());
    assert!(fa_prefill::emit_preconvert(ARCH, &spec()).is_ok());
    assert!(fa_prefill::emit_module(ARCH, &spec()).is_ok());
    for heads in [0u32, 1, 2, 8, 15, 16, 17, 24, 32, 64] {
        for kv_heads in [0u32, 1, 2, 3, 4, 8, 16] {
            for head_dim in [0u32, 32, 64, 128, 255, 256, 257, 512] {
                let s = FaPrefillSpec {
                    heads,
                    kv_heads,
                    head_dim,
                    kv: FaKvCodec::Q8_0,
                };
                let exact = (heads, kv_heads, head_dim) == (16, 2, 256);
                let case = format!("{heads}/{kv_heads}/{head_dim}");
                assert_eq!(s.validate().is_ok(), exact, "{case}");
                assert_eq!(fa_prefill::validate(&s, 64, 64).is_ok(), exact, "{case}");
                if !exact {
                    assert!(
                        fa_prefill::emit(ARCH, &s).is_err(),
                        "{case}: direct emitted"
                    );
                    assert!(
                        fa_prefill::emit_preconvert(ARCH, &s).is_err(),
                        "{case}: preconvert emitted"
                    );
                    assert!(
                        fa_prefill::emit_module(ARCH, &s).is_err(),
                        "{case}: module emitted"
                    );
                }
            }
        }
    }
    // Every other architecture is refused by every emission entry (no alias, no builder, no bytes).
    for arch in [Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx942] {
        assert!(fa_prefill::emit(arch, &spec()).is_err(), "{arch:?}: direct");
        assert!(
            fa_prefill::emit_preconvert(arch, &spec()).is_err(),
            "{arch:?}: preconvert"
        );
        assert!(
            fa_prefill::emit_module(arch, &spec()).is_err(),
            "{arch:?}: module"
        );
    }
    // Authoring domain: 1 <= batch <= 8192 with exactly `batch` positions. (Production admission is 64..=8192; rows
    // 1..63 are accepted here only so the emitter's tile and tail boundaries can be exercised.)
    for batch in [1usize, 2, 7, 8, 9, 63, 64, 65, 4096, 8191, 8192] {
        assert!(
            fa_prefill::validate(&spec(), batch, batch).is_ok(),
            "batch {batch}"
        );
        assert!(
            fa_prefill::validate(&spec(), batch, batch - 1).is_err(),
            "batch {batch}: short positions"
        );
        assert!(
            fa_prefill::validate(&spec(), batch, batch + 1).is_err(),
            "batch {batch}: long positions"
        );
        assert!(
            fa_prefill::validate(&spec(), batch, 0).is_err(),
            "batch {batch}: no positions"
        );
    }
    for batch in [0usize, 8193, 8194, 1 << 20, usize::MAX] {
        assert!(
            fa_prefill::validate(&spec(), batch, batch).is_err(),
            "batch {batch}"
        );
    }
    let bad = FaPrefillSpec {
        heads: 24,
        kv_heads: 4,
        head_dim: 256,
        kv: FaKvCodec::Q8_0,
    };
    assert!(fa_prefill::validate(&bad, 64, 64).is_err());
}

// ------------------------------------------------------------------------------------------------ 2. determinism
#[test]
fn emission_is_deterministic() {
    let (da, db) = (
        fa_prefill::emit(ARCH, &spec()).unwrap(),
        fa_prefill::emit(ARCH, &spec()).unwrap(),
    );
    let (pa, pb) = (
        fa_prefill::emit_preconvert(ARCH, &spec()).unwrap(),
        fa_prefill::emit_preconvert(ARCH, &spec()).unwrap(),
    );
    assert_eq!(da.s_text, db.s_text);
    assert_eq!(pa.s_text, pb.s_text);
    for (a, b) in [(&da, &db), (&pa, &pb)] {
        assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256);
        assert_eq!(
            serde_json::to_string(&a.proof).unwrap(),
            serde_json::to_string(&b.proof).unwrap()
        );
    }
    let (ea, ta, ma) = fa_prefill::emit_module(ARCH, &spec()).unwrap();
    let (eb, tb, mb) = fa_prefill::emit_module(ARCH, &spec()).unwrap();
    assert_eq!(ta, tb);
    assert_eq!(
        serde_json::to_string(&ma).unwrap(),
        serde_json::to_string(&mb).unwrap()
    );
    assert_eq!(ma.s_text_sha256, mb.s_text_sha256);
    let (bundle_a, bundle_b) = (
        native::bundle(
            &native::assemble(&ta, ARCH).unwrap(),
            ARCH,
            native::DEFAULT_HOST_TARGET,
        ),
        native::bundle(
            &native::assemble(&tb, ARCH).unwrap(),
            ARCH,
            native::DEFAULT_HOST_TARGET,
        ),
    );
    assert!(
        bundle_a == bundle_b,
        "native bundle bytes differ between two emissions"
    );
    // Identity and stable order (preconvert, then direct) in the text, the entries and the proof.
    assert_eq!((ma.module.as_str(), ma.arch), (MODULE, ARCH));
    assert_eq!(ma.kernels.len(), 2);
    assert_eq!(ea.len(), 2);
    assert_eq!(
        ea[0].s_text, pa.s_text,
        "module entry 0 is the standalone preconvert"
    );
    assert_eq!(
        ea[1].s_text, da.s_text,
        "module entry 1 is the standalone direct attention"
    );
    assert_eq!(eb[0].s_text, ea[0].s_text);
    assert_eq!(ta.matches(".amdhsa_kernel ").count(), 2);
    let mut at = 0;
    for symbol in SYMBOLS {
        let key = format!(".amdhsa_kernel {symbol}\n");
        let found = ta[at..]
            .find(&key)
            .unwrap_or_else(|| panic!("{symbol}: missing or out of order in the module source"));
        at += found + key.len();
        assert!(
            ta.contains(&format!(".globl {symbol}\n")),
            "{symbol}: not global"
        );
        assert!(
            ta.contains(&format!(".name: {symbol}\n")),
            "{symbol}: missing from the module metadata"
        );
    }
    assert!(ta.contains("amdgcn-amd-amdhsa--gfx1201"));
    assert_eq!(
        ta.matches(".symbol: ").count(),
        2,
        "exactly the two exports: no split partial / merge kernel"
    );
}

// ------------------------------------------------------------------------------------------------ 3. ABI / resources
fn check_resources(
    text: &str,
    symbol: &str,
    kernarg: u32,
    block: u32,
    args: &[(u32, u32, &str, &str)],
) {
    let has = |directive: &str| text.contains(&format!("\t.amdhsa_{directive}\n"));
    assert!(
        text.contains(&format!(".amdhsa_kernel {symbol}\n")),
        "{symbol}"
    );
    assert_eq!(
        field(text, ".kernarg_segment_size:"),
        [kernarg],
        "{symbol}: metadata kernarg size"
    );
    assert!(
        has(&format!("kernarg_size {kernarg}")),
        "{symbol}: descriptor kernarg size"
    );
    assert_eq!(
        field(text, ".max_flat_workgroup_size:"),
        [block],
        "{symbol}: block size"
    );
    let found = metadata_args(text);
    assert_eq!(found.len(), args.len(), "{symbol}: argument count");
    for ((name, offset, size, kind), (eo, es, en, ek)) in found.iter().zip(args) {
        assert_eq!(
            (name.as_str(), *offset, *size, kind.as_str()),
            (*en, *eo, *es, *ek),
            "{symbol}: argument {en}"
        );
    }
    // Wave32, WGP mode, fixed LDS 0 (codec-specific LDS is launch dynamic), no scratch / private / spills.
    assert!(has("wavefront_size32 1"), "{symbol}: wave32");
    assert!(
        text.contains(".wavefront_size: 32\n"),
        "{symbol}: metadata wave32"
    );
    assert!(
        has("workgroup_processor_mode 1"),
        "{symbol}: WGP mode (cu_mode false)"
    );
    assert!(
        text.contains(".workgroup_processor_mode: 1\n"),
        "{symbol}: metadata WGP mode"
    );
    assert!(has("group_segment_fixed_size 0"), "{symbol}: fixed LDS");
    assert!(
        text.contains(".group_segment_fixed_size: 0\n"),
        "{symbol}: metadata fixed LDS"
    );
    assert!(
        has("private_segment_fixed_size 0") && text.contains(".private_segment_fixed_size: 0\n"),
        "{symbol}: private segment"
    );
    assert!(
        has("enable_private_segment 0") && has("uses_dynamic_stack 0"),
        "{symbol}: scratch / dynamic stack"
    );
    assert!(
        text.contains(".sgpr_spill_count: 0\n") && text.contains(".vgpr_spill_count: 0\n"),
        "{symbol}: spills"
    );
    // The relevant FP-mode fields: RNE (round modes 0), denormals preserved (3), FP16 overflow to inf (0). The same
    // fields of the pinned HIP object read rsrc1 & 0x000ff000 == 0x000f0000 for both exports.
    for d in [
        "float_round_mode_32 0",
        "float_round_mode_16_64 0",
        "float_denorm_mode_32 3",
        "float_denorm_mode_16_64 3",
        "fp16_overflow 0",
    ] {
        assert!(has(d), "{symbol}: {d}");
    }
    // Grid x always; the direct attention's y is the KV head; never z.
    assert!(has("system_sgpr_workgroup_id_x 1"), "{symbol}");
    assert!(
        has(&format!(
            "system_sgpr_workgroup_id_y {}",
            u32::from(symbol != PRE)
        )),
        "{symbol}: grid y"
    );
    assert!(has("system_sgpr_workgroup_id_z 0"), "{symbol}: no z grid");
    assert!(
        !insns(text).iter().any(|l| {
            let m = mnemonic(l);
            m.starts_with("scratch_")
                || m.starts_with("flat_")
                || m.starts_with("s_setpc")
                || m.starts_with("s_swappc")
                || m.contains("atomic")
        }),
        "{symbol}: scratch / flat / atomic / dynamic call"
    );
}
#[test]
fn kernarg_layout_and_resources_match_contract() {
    let (entries, module_text) = emitted();
    let direct = fa_prefill::emit(ARCH, &spec()).unwrap();
    let pre = fa_prefill::emit_preconvert(ARCH, &spec()).unwrap();
    for (e, symbol, kernarg, block, args) in [
        (&pre, PRE, PRE_KERNARG, PRE_BLOCK, &PRE_ARGS[..]),
        (
            &direct,
            DIRECT,
            DIRECT_KERNARG,
            DIRECT_BLOCK,
            &DIRECT_ARGS[..],
        ),
    ] {
        check_resources(&e.s_text, symbol, kernarg, block, args);
        let vgprs = field(&e.s_text, ".amdhsa_next_free_vgpr");
        assert_eq!(vgprs.len(), 1, "{symbol}");
        assert!(vgprs[0] <= 256, "{symbol}: {} VGPRs", vgprs[0]);
        assert_eq!(
            u32::from(e.shape.next_free_vgpr),
            vgprs[0],
            "{symbol}: shape/descriptor VGPRs"
        );
        assert!(
            e.s_text.contains(&format!(".vgpr_count: {}\n", vgprs[0])),
            "{symbol}: metadata VGPRs"
        );
        let sgprs = field(&e.s_text, ".amdhsa_next_free_sgpr");
        assert_eq!(sgprs.len(), 1, "{symbol}");
        assert!(
            sgprs[0] <= 104 && u32::from(e.shape.next_free_sgpr) == sgprs[0],
            "{symbol}: {} SGPRs",
            sgprs[0]
        );
        // The module carries exactly the same descriptor / metadata of this export.
        assert!(
            module_text.contains(&format!(".kernarg_segment_size: {kernarg}\n")),
            "{symbol}: module metadata"
        );
    }
    // Contract fields restate the same launch resources.
    let (cd, cp) = (
        fa_prefill::contract(ARCH, &direct),
        fa_prefill::preconvert_contract(ARCH, &pre),
    );
    for (c, symbol, dynamic) in [(&cd, DIRECT, DIRECT_LDS), (&cp, PRE, 0)] {
        assert_eq!(c["symbol"], symbol);
        assert_eq!(c["launch_dynamic_lds_bytes"], dynamic, "{symbol}");
        assert_eq!(c["group_segment_fixed_bytes"], 0, "{symbol}");
        assert_eq!(c["require_wave32"], true, "{symbol}");
        assert_eq!(c["require_zero_spills"], true, "{symbol}");
        assert_eq!(c["require_zero_private"], true, "{symbol}");
        assert!(
            c["vgpr_max"].as_u64().unwrap() <= 256 && c["sgpr_max"].as_u64().unwrap() <= 104,
            "{symbol}"
        );
    }
    // Entries of the module are the standalone ones.
    assert_eq!(entries[0].s_text, pre.s_text);
    assert_eq!(entries[1].s_text, direct.s_text);
}

// ------------------------------------------------------------------------------------------------ 3a. Q8 prologue ownership
/// The Q8 direct kernel declares its four common prologue temporaries (`v16..v48`, four `v8` blocks) live
/// from the entry block to the tile loop head: the tile phase reuses the same registers for scores, so an
/// FP8-only declaration leaves `v16` with no unique live owner at entry and `emit_module` validation fails.
/// Pins the per-byte owners, the live window and the Q8 launch resources (65,536 dynamic LDS bytes, 256 VGPRs).
#[test]
fn q8_prologue_temps_own_v16_through_v47_before_tile_loop() {
    use peacemaker_isa::reg::{Kind, Live};
    let direct = fa_prefill::emit(ARCH, &spec()).unwrap();
    const TILE: &str = ".Lfapd_tile";
    assert!(
        direct.s_text.contains(TILE),
        "direct: tile loop head {TILE} present"
    );
    for v in 16..48u8 {
        let owners: Vec<_> = direct
            .proof
            .reg_plan
            .iter()
            .filter(|r| {
                r.kind == Kind::V
                    && u16::from(r.base) <= u16::from(v)
                    && u16::from(v) < u16::from(r.base) + u16::from(r.len)
                    && matches!(&r.live, Live::Between(start, end) if start == "entry" && end == TILE)
            })
            .collect();
        assert_eq!(
            owners.len(),
            1,
            "direct: v{v} must have exactly one live owner at entry"
        );
        assert_eq!(
            owners[0].name.as_ref(),
            "prologue_temp",
            "direct: v{v} owner"
        );
        assert_eq!(
            (owners[0].base, owners[0].len),
            (16 + (v - 16) / 8 * 8, 8),
            "direct: v{v} prologue block"
        );
    }
    // Entry-live per-launch state and the Q8 launch resources.
    let tid: Vec<_> = direct
        .proof
        .reg_plan
        .iter()
        .filter(|r| {
            r.kind == Kind::V
                && r.base == 0
                && r.len == 1
                && matches!(&r.live, Live::Between(start, end) if start == "entry" && end == TILE)
        })
        .collect();
    assert_eq!(tid.len(), 1, "direct: v0 tid live entry -> tile loop");
    assert_eq!(tid[0].name.as_ref(), "tid", "direct: v0 owner");
    assert_eq!(direct.shape.next_free_vgpr, 256, "direct: VGPR budget");
    assert_eq!(direct.proof.next_free_vgpr, 256, "direct: proof VGPRs");
    assert_eq!(
        fa_prefill::contract(ARCH, &direct)["launch_dynamic_lds_bytes"],
        serde_json::json!(DIRECT_LDS),
        "direct: dynamic LDS bytes"
    );
}

// ------------------------------------------------------------------------------------------------ 4. fragment layout
/// The lab's half-element fragment offsets (`fa_a3b_fa2.hip` `fa2_koff` / `fa2_voff`), written here independently.
fn koff(sub: usize, dc: usize, lane: usize, j: usize) -> usize {
    ((dc * 4 + sub) * 32 + lane) * 8 + j
}
fn voff(sub: usize, dc: usize, lane: usize, j: usize) -> usize {
    ((sub * 16 + dc) * 32 + lane) * 8 + j
}
/// Half index -> `(key, dim)` of the K plane, from the lab's fill loops over all 128 fillers; also verifies the plan's
/// byte-address derivation reproduces each store.
fn k_owners() -> Vec<Option<(usize, usize)>> {
    let mut own = vec![None; 16384];
    for tid in 0..128usize {
        for t in 0..4usize {
            let idx = tid + 128 * t;
            let (k, b) = (idx >> 3, idx & 7);
            for w in 0..8usize {
                let d0 = b * 32 + w * 4;
                let ln = 16 * ((d0 >> 3) & 1) + (k & 15);
                let o = koff(k >> 4, d0 >> 4, ln, d0 & 7);
                for i in 0..4 {
                    assert!(
                        own[o + i].replace((k, d0 + i)).is_none(),
                        "K half {} written twice",
                        o + i
                    );
                }
                let t0 = tid & 127;
                let base = 4096 * (t0 & 7) + 16 * (t0 >> 3);
                let imm = 512 * t + 2048 * (w >> 2) + 256 * ((w >> 1) & 1) + 8 * (w & 1);
                assert_eq!(
                    base + imm,
                    2 * o,
                    "K store address of tid {tid} slot {t} quad {w}"
                );
                assert!(base + imm + 8 <= 32768);
            }
        }
    }
    own
}
fn v_owners() -> Vec<Option<(usize, usize)>> {
    let mut own = vec![None; 16384];
    for tid in 0..128usize {
        for t in 0..2usize {
            let pair = tid + 128 * t;
            let (m, b) = (pair >> 3, pair & 7);
            let (k0, sub, kgm, j0) = (2 * m, (2 * m) >> 4, ((2 * m) >> 3) & 1, (2 * m) & 7);
            for w in 0..8usize {
                for c in 0..4usize {
                    let d = b * 32 + w * 4 + c;
                    let o = voff(sub, d >> 4, 16 * kgm + (d & 15), j0);
                    assert!(
                        own[o].replace((k0, d)).is_none(),
                        "V half {o} written twice"
                    );
                    assert!(
                        own[o + 1].replace((k0 + 1, d)).is_none(),
                        "V half {} written twice",
                        o + 1
                    );
                    let t0 = tid & 127;
                    let base = 8192 * (t0 >> 6)
                        + 1024 * (t0 & 7)
                        + 256 * ((t0 >> 5) & 1)
                        + 4 * ((t0 >> 3) & 3);
                    let imm = 16384 * t + 512 * (w >> 2) + 64 * (w & 3) + 16 * c;
                    assert_eq!(
                        base + imm,
                        2 * o,
                        "V store address of tid {tid} pair slot {t} quad {w} c {c}"
                    );
                    assert!(base + imm + 4 <= 32768);
                }
            }
        }
    }
    own
}
#[test]
fn fragment_layout_has_unique_owners_and_bounded_accesses() {
    let (kown, vown) = (k_owners(), v_owners());
    // Every half of both planes is written exactly once, and the (key, dim) pairs are exactly the 64 x 256 grid.
    for (plane, own) in [("K", &kown), ("V", &vown)] {
        let mut seen = vec![false; 64 * 256];
        for (half, owner) in own.iter().enumerate() {
            let (key, dim) =
                owner.unwrap_or_else(|| panic!("{plane} half {half} has no owner (hole)"));
            assert!(key < 64 && dim < 256);
            assert!(
                !std::mem::replace(&mut seen[key * 256 + dim], true),
                "{plane} ({key},{dim}) stored twice"
            );
        }
        assert!(
            seen.iter().all(|s| *s),
            "{plane}: a (key, dim) is never stored"
        );
    }
    // The helper wave (tid 128..159) shares the fill address bases of tid & 127 (immutable proof addresses) but stores
    // nothing: only `tid < 128` owns a fill slot. Its proof addresses remain in range.
    for tid in 128..160usize {
        let t0 = tid & 127;
        assert!(4096 * (t0 & 7) + 16 * (t0 >> 3) + 512 * 3 + 2048 + 256 + 8 + 8 <= 32768);
        assert!(
            8192 * (t0 >> 6)
                + 1024 * (t0 & 7)
                + 256 * ((t0 >> 5) & 1)
                + 4 * ((t0 >> 3) & 3)
                + 16384
                + 512
                + 64 * 3
                + 16 * 3
                + 4
                <= 32768
        );
    }
    // Compute fragments: every K fragment of the four compute waves consumes the intended key (sub*16 + ml) and its
    // eight dims (dc*16 + 8*kg + j); every V fragment the intended dim (dc*16 + ml) and keys (sub*16 + 8*kg + j).
    for wave in 0..4usize {
        for lane in 0..32usize {
            let (ml, kg) = (lane & 15, lane >> 4);
            for sub in 0..4usize {
                for dc in 0..16usize {
                    for j in 0..8usize {
                        assert_eq!(
                            kown[koff(sub, dc, lane, j)],
                            Some((sub * 16 + ml, dc * 16 + 8 * kg + j)),
                            "K wave {wave} lane {lane}"
                        );
                        assert_eq!(
                            vown[voff(sub, dc, lane, j)],
                            Some((sub * 16 + 8 * kg + j, dc * 16 + ml)),
                            "V wave {wave} lane {lane}"
                        );
                    }
                    // Byte addresses: base 16*lane plus the frozen immediates.
                    assert_eq!(2 * koff(sub, dc, lane, 0), (dc * 4 + sub) * 512 + 16 * lane);
                    assert_eq!(
                        2 * voff(sub, dc, lane, 0),
                        (sub * 16 + dc) * 512 + 16 * lane
                    );
                    assert!((dc * 4 + sub) * 512 + 16 * lane + 16 <= 32768);
                    assert!(32768 + (sub * 16 + dc) * 512 + 16 * lane + 16 <= 65536);
                }
            }
        }
    }
    // Output ownership (and Q fragment ownership, the same lane map): both KV heads x every compute wave x lane.
    // Wave 4 (the helper) owns nothing. Tail rows' lanes (query >= batch) are masked. No duplicate, no hole.
    for batch in [1usize, 7, 8, 9, 15, 16, 17, 63, 64, 65, 513] {
        let mut covered = vec![0u8; batch * 16 * 256];
        for kv_h in 0..2usize {
            for wg_x in 0..batch.div_ceil(8) {
                for wave in 0..5usize {
                    if wave == 4 {
                        continue;
                    }
                    for lane in 0..32usize {
                        let (ml, kg) = (lane & 15, lane >> 4);
                        let head = 8 * kv_h + 2 * wave + (ml >> 3);
                        let query = 8 * wg_x + (ml & 7);
                        if query >= batch {
                            continue;
                        }
                        assert!(head < 16);
                        for dc in 0..16usize {
                            for i in 0..8usize {
                                covered[(query * 16 + head) * 256 + dc * 16 + 8 * kg + i] += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(
            covered.iter().all(|&c| c == 1),
            "batch {batch}: output/Q element duplicated or missing"
        );
    }
    // The emitted code uses exactly these immediates (the frozen unrolled address strategy): K fill stores are 8-byte,
    // V fill stores are 4-byte packed pairs, fragment reads are 16-byte.
    let direct = fa_prefill::emit(ARCH, &spec()).unwrap();
    let pre = fa_prefill::emit_preconvert(ARCH, &spec()).unwrap();
    assert_eq!(
        count_prefix(&pre.s_text, "ds_"),
        0,
        "preconvert touches no LDS"
    );
    let offsets = |name: &str| -> std::collections::BTreeSet<u32> {
        insns(&direct.s_text)
            .into_iter()
            .filter(|l| mnemonic(l) == name)
            .map(imm_offset)
            .collect()
    };
    let mut k_store = std::collections::BTreeSet::new();
    let mut v_store = std::collections::BTreeSet::new();
    for t in 0..4u32 {
        for w in 0..8u32 {
            k_store.insert(512 * t + 2048 * (w >> 2) + 256 * ((w >> 1) & 1) + 8 * (w & 1));
        }
    }
    for t in 0..2u32 {
        for w in 0..8u32 {
            for c in 0..4u32 {
                v_store.insert(32768 + 16384 * t + 512 * (w >> 2) + 64 * (w & 3) + 16 * c);
            }
        }
    }
    let mut frag = std::collections::BTreeSet::new();
    for sub in 0..4u32 {
        for dc in 0..16u32 {
            frag.insert((dc * 4 + sub) * 512);
            frag.insert(32768 + (sub * 16 + dc) * 512);
        }
    }
    assert_eq!(offsets("ds_store_b64"), k_store, "K fill store immediates");
    assert_eq!(offsets("ds_store_b32"), v_store, "V fill store immediates");
    assert_eq!(offsets("ds_load_b128"), frag, "fragment read immediates");
    for l in insns(&direct.s_text) {
        let m = mnemonic(l);
        if m.starts_with("ds_store_") {
            assert!(
                m == "ds_store_b64" || m == "ds_store_b32",
                "unplanned LDS store {m}"
            );
        }
        if m.starts_with("ds_load_") {
            assert_eq!(m, "ds_load_b128", "unplanned LDS load {m}");
        }
    }
}

// ------------------------------------------------------------------------------------------------ f16 / Q8_0 model
/// Independent f32 -> f16 round-to-nearest-even (finite inputs; overflow rounds to infinity).
fn f16_rne(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 0xff {
        return sign | 0x7c00 | if m != 0 { 0x200 } else { 0 };
    }
    let exp = e - 127 + 15;
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let mant = u64::from(m | 0x80_0000);
        let shift = (14 - exp) as u32;
        let mut q = mant >> shift;
        let rem = mant & ((1u64 << shift) - 1);
        let half = 1u64 << (shift - 1);
        if rem > half || (rem == half && q & 1 == 1) {
            q += 1;
        }
        return sign | q as u16;
    }
    let mut h = ((exp as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    sign | h as u16
}
fn f16_to_f32(h: u16) -> f32 {
    let sign = (u32::from(h) & 0x8000) << 16;
    let e = u32::from((h >> 10) & 0x1f);
    let m = u32::from(h & 0x3ff);
    let bits = if e == 0 {
        if m == 0 {
            sign
        } else {
            let (mut e2, mut m2) = (113u32, m);
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            sign | (e2 << 23) | ((m2 & 0x3ff) << 13)
        }
    } else if e == 0x1f {
        sign | 0x7f80_0000 | (m << 13)
    } else {
        sign | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}
#[test]
fn the_f16_model_is_pinned_by_known_bit_patterns() {
    for (x, bits) in [
        (0.0f32, 0x0000u16),
        (-0.0, 0x8000),
        (1.0, 0x3c00),
        (-2.0, 0xc000),
        (65504.0, 0x7bff),
        (65519.0, 0x7bff),
        (65520.0, 0x7c00),
        (0.1, 0x2e66),
        (f32::from_bits(0x4049_0fdb), 0x4248),
        (f32::from_bits(0x3f80_1000), 0x3c00), // 1 + 2^-11: tie, even -> 1.0
        (f32::from_bits(0x3f80_3000), 0x3c02), // 1 + 3*2^-11: tie, even -> 1 + 2^-9
        (f32::from_bits(0x3f80_1001), 0x3c01), // just above the tie
        (f32::from_bits(0x3f80_0fff), 0x3c00), // just below the tie
        (f32::from_bits(0x3380_0000), 0x0001), // 2^-24 smallest subnormal
        (f32::from_bits(0x3300_0000), 0x0000), // 2^-25 tie -> 0
        (f32::from_bits(0x3300_0001), 0x0001), // above the tie
        (f32::from_bits(0x3380_0000 | 0x0040_0000), 0x0002), // 1.5 * 2^-24 tie -> 2
        (f32::from_bits(0x3880_0000), 0x0400), // 2^-14 smallest normal
    ] {
        assert_eq!(f16_rne(x), bits, "{x:e} ({:#010x})", x.to_bits());
    }
    for bits in [
        0x0000u16, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0x7bff, 0xfbff, 0x2e66, 0x4248,
    ] {
        assert_eq!(f16_rne(f16_to_f32(bits)), bits, "{bits:#06x} round trip");
    }
}

/// One Q8_0 block's staged f16 element: the f32 product of the f16 scale and the signed code, rounded once to f16.
fn stage(cache: &[u8], pos: usize, g: usize, d: usize) -> u16 {
    let at = pos * POS_STRIDE + g * HEAD_STRIDE + (d / 32) * 34;
    let scale = u16::from_le_bytes([cache[at], cache[at + 1]]);
    let code = cache[at + 2 + d % 32] as i8;
    f16_rne(f16_to_f32(scale) * f32::from(code))
}
fn build_cache(npos: usize, f: impl Fn(usize, usize, usize) -> (u16, [i8; 32])) -> Vec<u8> {
    let mut out = vec![0u8; npos * POS_STRIDE];
    for p in 0..npos {
        for g in 0..2 {
            for b in 0..8 {
                let (scale, codes) = f(p, g, b);
                let at = p * POS_STRIDE + g * HEAD_STRIDE + b * 34;
                out[at..at + 2].copy_from_slice(&scale.to_le_bytes());
                for (i, c) in codes.iter().enumerate() {
                    out[at + 2 + i] = *c as u8;
                }
            }
        }
    }
    out
}
fn le16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn le32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A direct-attention case: Q16 `[batch, 16, 256]`, position-major Q8_0 K / V, per-row positions, `scale_attn`.
#[derive(Clone)]
struct Case {
    batch: usize,
    positions: Vec<i32>,
    q16: Vec<u16>,
    k: Vec<u8>,
    v: Vec<u8>,
    scale: f32,
}
/// Scores are `scale * dot` with separations of at least `8192 * qmag * |scale|` between distinct keys (K scale ramps
/// ascending for KV head 0 and descending for KV head 1, Q sign varies per row / head, so the winner differs by row,
/// head and KV head; an unmasked future key would win), V are full-range signed integer codes at scale 1.0 (exact f16).
/// `salt` changes every operand. With `scale == 0` every live key ties.
fn winner_case(batch: usize, positions: Vec<i32>, scale: f32, qmag: f32, salt: usize) -> Case {
    assert_eq!(positions.len(), batch);
    let npos = *positions.iter().max().unwrap() as usize + 1;
    let q16 = (0..batch * 4096)
        .map(|i| {
            let (r, h, d) = (i / 4096, (i / 256) % 16, i % 256);
            let sign = if (r * 5 + h * 3 + salt) % 4 < 2 {
                1.0
            } else {
                -1.0
            };
            f16_rne(sign * qmag * (1 + d % 3) as f32)
        })
        .collect();
    let k = build_cache(npos, |p, g, b| {
        let s = if g == 0 {
            0.25 * (1 + p) as f32
        } else {
            0.25 * (200 - p) as f32
        };
        let mut codes = [0i8; 32];
        for (i, c) in codes.iter_mut().enumerate() {
            let d = b * 32 + i;
            *c = match d % 61 {
                0 => 127,
                30 => -128,
                _ => (64 + ((d * 37 + p * 11 + salt) % 7) as i32 - 3) as i8,
            };
        }
        (f16_rne(s), codes)
    });
    let v = build_cache(npos, |p, g, b| {
        let mut codes = [0i8; 32];
        for (i, c) in codes.iter_mut().enumerate() {
            let d = b * 32 + i;
            *c = ((p * 29 + d * 7 + g * 61 + salt * 13) % 256) as u8 as i8;
        }
        (0x3c00, codes)
    });
    Case {
        batch,
        positions,
        q16,
        k,
        v,
        scale,
    }
}
/// The exact expected output words. Every case must be well posed: all live keys either tie the row's maximum score
/// exactly or sit at least 200 below it (so `exp` underflows to exactly 0 and the online rescale `alpha` is exactly 0),
/// and the tied V sums are exact in f32.
fn expected(c: &Case) -> Vec<u32> {
    let npos = c.k.len() / POS_STRIDE;
    let kst: Vec<Vec<u16>> = (0..2 * npos)
        .map(|i| {
            (0..256)
                .map(|d| stage(&c.k, i % npos, i / npos, d))
                .collect()
        })
        .collect();
    let vst: Vec<Vec<f32>> = (0..2 * npos)
        .map(|i| {
            (0..256)
                .map(|d| f16_to_f32(stage(&c.v, i % npos, i / npos, d)))
                .collect()
        })
        .collect();
    let mut out = vec![0u32; c.batch * 4096];
    for r in 0..c.batch {
        let pos = c.positions[r] as usize;
        for h in 0..16 {
            let g = h / 8;
            let q: Vec<f64> = (0..256)
                .map(|d| f64::from(f16_to_f32(c.q16[(r * 16 + h) * 256 + d])))
                .collect();
            let scores: Vec<f32> = (0..=pos)
                .map(|j| {
                    let dot: f64 = (0..256)
                        .map(|d| q[d] * f64::from(f16_to_f32(kst[g * npos + j][d])))
                        .sum();
                    (dot as f32) * c.scale
                })
                .collect();
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let ties: Vec<usize> = (0..=pos).filter(|&j| scores[j] == m).collect();
            for j in 0..=pos {
                assert!(
                    scores[j] == m || m - scores[j] >= 200.0,
                    "ill-posed case: row {r} head {h} key {j} gap {}",
                    m - scores[j]
                );
            }
            let inv = 1.0f32 / ties.len() as f32;
            for d in 0..256 {
                let (mut sum, mut exact) = (0.0f32, 0.0f64);
                for &j in &ties {
                    let v = vst[g * npos + j][d];
                    sum += v;
                    exact += f64::from(v);
                }
                assert_eq!(
                    f64::from(sum),
                    exact,
                    "inexact V sum: row {r} head {h} dim {d}"
                );
                out[(r * 16 + h) * 256 + d] = (sum * inv).to_bits();
            }
        }
    }
    out
}

// ------------------------------------------------------------------------------------------------ emulator rig
const E_KARG: u64 = 0x1_0000;
const E_Q: u64 = 0x1000_0000;
const E_K: u64 = 0x2000_0000;
const E_V: u64 = 0x3000_0000;
const E_OUT: u64 = 0x4000_0000;
const E_POS: u64 = 0x5000_0000;
const GUARD: usize = 256;
const SENTINEL: u8 = 0xa5;
const POISON: u32 = 0x7fc0_beef;

struct Rig {
    program: peacemaker_ir::Program,
    symbol: &'static str,
    dynamic_lds: u32,
}
impl Rig {
    fn new() -> Self {
        let (entries, text, _) = fa_prefill::emit_module(ARCH, &spec()).unwrap();
        assert_eq!(entries[1].shape.next_free_vgpr, 256);
        Self::from_text(&text, DIRECT, DIRECT_LDS)
    }

    fn from_text(text: &str, symbol: &'static str, dynamic_lds: u32) -> Self {
        let elf = native::assemble(text, ARCH).unwrap();
        let lifted = peacemaker_lift::lift_object(
            &elf,
            peacemaker_lift::Options {
                frontend: Frontend::Builder,
            },
        )
        .unwrap();
        Rig {
            program: lifted.program,
            symbol,
            dynamic_lds,
        }
    }
}
fn grid(batch: usize) -> Vec<[u32; 3]> {
    (0..2u32)
        .flat_map(|y| (0..batch.div_ceil(8) as u32).map(move |x| [x, y, 0]))
        .collect()
}
fn region<'a>(mem: &'a Memory, name: &str) -> &'a peacemaker_ir::emu::Region {
    mem.regions.iter().find(|r| r.name == name).unwrap()
}
struct Run {
    out: Vec<u8>,
    guards_intact: bool,
    out_written: usize,
    inputs_untouched: bool,
}
enum Initial {
    Poison(u32),
    Bytes(Vec<u8>),
}
/// One launch set on freshly mapped, exactly sized buffers: kernarg exactly 60 bytes, Q16 / K / V / positions exactly
/// their extents (any out-of-range read is an unmapped fault), the output between sentinel guards. `args` overrides
/// `[n_heads, n_kv_heads, head_dim, batch_size]` in the kernarg block.
fn run_direct(
    rig: &Rig,
    c: &Case,
    initial: &Initial,
    args: Option<[i32; 4]>,
    wgs: &[[u32; 3]],
) -> Result<Run, String> {
    let out_bytes = c.batch * 4096 * 4;
    let [nh, nkv, hd, bs] = args.unwrap_or([16, 2, 256, c.batch as i32]);
    let mut karg = Vec::new();
    for ptr in [E_Q, E_K, E_V, E_OUT + GUARD as u64, E_POS] {
        karg.extend(ptr.to_le_bytes());
    }
    for v in [nh, nkv, hd, bs] {
        karg.extend(v.to_le_bytes());
    }
    karg.extend(c.scale.to_le_bytes());
    assert_eq!(karg.len(), DIRECT_KERNARG as usize);
    let (q, pos) = (le16(&c.q16), le32(&c.positions));
    let mut out = vec![SENTINEL; out_bytes + 2 * GUARD];
    match initial {
        Initial::Poison(word) => {
            for w in out[GUARD..GUARD + out_bytes].chunks_exact_mut(4) {
                w.copy_from_slice(&word.to_le_bytes());
            }
        }
        Initial::Bytes(b) => {
            assert_eq!(b.len(), out_bytes);
            out[GUARD..GUARD + out_bytes].copy_from_slice(b);
        }
    }
    let before = out.clone();
    let mut mem = Memory::default();
    mem.map("kernarg".into(), E_KARG, karg.clone())?;
    mem.map("q16".into(), E_Q, q.clone())?;
    mem.map("k".into(), E_K, c.k.clone())?;
    mem.map("v".into(), E_V, c.v.clone())?;
    mem.map("pos".into(), E_POS, pos.clone())?;
    mem.map("out".into(), E_OUT, out)?;
    let emu = Emulator::new(&rig.program, rig.symbol)?;
    for wg in wgs {
        emu.run_workgroup(
            &Launch {
                kernarg_ptr: E_KARG,
                workgroup: *wg,
                block: [DIRECT_BLOCK, 1, 1],
                dynamic_lds: rig.dynamic_lds,
            },
            &mut mem,
        )?;
    }
    let inputs_untouched = [
        ("kernarg", &karg),
        ("q16", &q),
        ("k", &c.k),
        ("v", &c.v),
        ("pos", &pos),
    ]
    .iter()
    .all(|(name, bytes)| {
        let r = region(&mem, name);
        r.bytes == **bytes && r.written.count() == 0
    });
    let o = region(&mem, "out");
    Ok(Run {
        out: o.bytes[GUARD..GUARD + out_bytes].to_vec(),
        guards_intact: o.bytes[..GUARD] == before[..GUARD]
            && o.bytes[GUARD + out_bytes..] == before[GUARD + out_bytes..],
        out_written: o.written.count(),
        inputs_untouched,
    })
}
fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        .collect()
}
fn assert_matches(name: &str, run: &Run, want: &[u32], batch: usize) {
    assert!(
        run.guards_intact,
        "{name}: a store escaped the output (tail-row lane or helper wave wrote)"
    );
    assert!(
        run.inputs_untouched,
        "{name}: Q16 / K / V / positions / kernarg changed"
    );
    assert_eq!(
        run.out_written,
        batch * 4096 * 4,
        "{name}: exactly the live output bytes are written"
    );
    for (i, (g, w)) in words(&run.out).iter().zip(want).enumerate() {
        assert_eq!(
            g,
            w,
            "{name}: row {} head {} dim {}: got {:#010x} ({}), want {:#010x} ({})",
            i / 4096,
            (i / 256) % 16,
            i % 256,
            g,
            f32::from_bits(*g),
            w,
            f32::from_bits(*w)
        );
    }
}
fn check(rig: &Rig, name: &str, c: &Case) {
    let run = run_direct(rig, c, &Initial::Poison(POISON), None, &grid(c.batch))
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_matches(name, &run, &expected(c), c.batch);
}

// ------------------------------------------------------------------------------------------------ 5. causal bounds
/// The workgroup's bound reduction as every wave computes it: lanes 0..7 hold the eight row positions (sentinels
/// elsewhere), XOR 16, 8, 4, 2, 1 min / max butterfly, readfirstlane.
fn wave_bounds(positions: &[i32], q_base: usize, batch: usize) -> (i32, i32) {
    let mut mx = [-1i32; 32];
    let mut mn = [i32::MAX; 32];
    for lane in 0..8 {
        if q_base + lane < batch {
            mx[lane] = positions[q_base + lane];
            mn[lane] = positions[q_base + lane];
        }
    }
    for off in [16usize, 8, 4, 2, 1] {
        let (a, b) = (mx, mn);
        for l in 0..32 {
            mx[l] = a[l].max(a[l ^ off]);
            mn[l] = b[l].min(b[l ^ off]);
        }
    }
    assert!(
        mx.iter().all(|&v| v == mx[0]) && mn.iter().all(|&v| v == mn[0]),
        "reduction is not wave-uniform"
    );
    (mn[0], mx[0])
}
#[derive(Debug, PartialEq, Eq)]
enum Kind {
    Skip,
    Full,
    Partial,
}
fn subtile_kind(tile: usize, sub: usize, gmin: i32, gmax: i32) -> Kind {
    let start = (tile * 64 + 16 * sub) as i32;
    if start > gmax {
        Kind::Skip
    } else if start + 15 <= gmin {
        Kind::Full
    } else {
        Kind::Partial
    }
}
fn position_patterns() -> Vec<(usize, Vec<i32>)> {
    const B: [i32; 10] = [0, 15, 16, 31, 32, 47, 48, 63, 64, 65];
    let mut out = Vec::new();
    for batch in [1usize, 7, 8, 9, 63, 64, 65] {
        out.push((batch, (0..batch as i32).collect()));
        out.push((batch, (0..batch).map(|r| B[(r * 7) % 10]).collect()));
        out.push((batch, (0..batch).map(|r| B[(r / 8) % 10]).collect()));
        out.push((batch, (0..batch).map(|r| 65 - (r as i32 % 66)).collect()));
        out.push((batch, (0..batch).map(|r| (r as i32 * 5) % 66).collect()));
    }
    out
}
#[test]
fn causal_bounds_and_tail_masks() {
    // Host proof of the bound reduction and the full / partial / skip predicates, for every group of every pattern.
    for (batch, positions) in position_patterns() {
        let npos = *positions.iter().max().unwrap() as usize + 1;
        for q_base in (0..batch).step_by(8) {
            let rows: Vec<i32> = positions[q_base..(q_base + 8).min(batch)].to_vec();
            let (gmin, gmax) = wave_bounds(&positions, q_base, batch);
            assert_eq!(
                (gmin, gmax),
                (*rows.iter().min().unwrap(), *rows.iter().max().unwrap()),
                "batch {batch} group {q_base}"
            );
            // All five waves reduce the same eight positions with the same graph: identical scalar bounds.
            for _wave in 0..5 {
                assert_eq!(wave_bounds(&positions, q_base, batch), (gmin, gmax));
            }
            let seq_len = gmax as usize + 1;
            assert!(seq_len <= npos, "reads beyond the cache");
            if rows.len() > 1 && rows.iter().any(|&p| p != rows[0]) {
                // XOR 16 alone is not a uniform reduction of eight lanes.
                let mut mx = [-1i32; 32];
                for (lane, p) in rows.iter().enumerate() {
                    mx[lane] = *p;
                }
                let step: Vec<i32> = (0..32).map(|l| mx[l].max(mx[l ^ 16])).collect();
                assert!(
                    step.iter().any(|&v| v != step[0]),
                    "XOR 16 would already be uniform"
                );
            }
            for tile in 0..seq_len.div_ceil(64) {
                assert!(
                    tile * 64 < seq_len,
                    "a tile entirely past seq_len is entered"
                );
                for sub in 0..4usize {
                    let kind = subtile_kind(tile, sub, gmin, gmax);
                    let start = tile * 64 + 16 * sub;
                    let live = |p: i32, key: usize| key as i32 <= p;
                    let all_live = rows
                        .iter()
                        .all(|&p| (start..start + 16).all(|k| live(p, k)));
                    let any_live = rows
                        .iter()
                        .any(|&p| (start..start + 16).any(|k| live(p, k)));
                    match kind {
                        Kind::Skip => assert!(!any_live, "batch {batch} group {q_base} tile {tile} sub {sub}: skipped a live subtile"),
                        Kind::Full => assert!(all_live, "batch {batch} group {q_base} tile {tile} sub {sub}: unmasked subtile has a dead key"),
                        Kind::Partial => assert!(any_live && !all_live, "batch {batch} group {q_base} tile {tile} sub {sub}: partial without a mix"),
                    }
                }
            }
        }
    }
    // The emitted code: exactly sized caches (any key read at or past `max position + 1` faults), exact-size
    // positions, the output guarded, non-monotone and duplicated groups, tail rows and every subtile class.
    const B: [i32; 10] = [0, 15, 16, 31, 32, 47, 48, 63, 64, 65];
    let rig = Rig::new();
    let cases: Vec<(&str, Vec<i32>)> = vec![
        ("1 row pos 0", vec![0]),
        ("1 row pos 65", vec![65]),
        ("7 rows non-monotone", vec![31, 0, 15, 63, 16, 48, 47]),
        ("8 rows around tiles", vec![64, 0, 65, 32, 15, 16, 31, 48]),
        (
            "9 rows, short first group, lone tail row",
            vec![0, 1, 2, 3, 4, 5, 6, 7, 65],
        ),
        ("63 rows ramp", (0..63).collect()),
        ("64 rows strided", (0..64).map(|r| (r * 5) % 66).collect()),
        (
            "65 rows boundary cycle",
            (0..65).map(|r| B[(r * 7) % 10]).collect(),
        ),
    ];
    for (name, positions) in cases {
        let batch = positions.len();
        check(&rig, name, &winner_case(batch, positions, 1.0, 1.0, 0));
    }
}

// ------------------------------------------------------------------------------------------------ 6. preconvert
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}
fn preconvert_values(n: usize) -> Vec<f32> {
    let special: Vec<u32> = vec![
        0x0000_0000,
        0x8000_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x3f80_1000,
        0xbf80_1000,
        0x3f80_3000,
        0x3f80_1001,
        0x3f80_0fff,
        0x3380_0000,
        0xb380_0000,
        0x3300_0000,
        0xb300_0000,
        0x3300_0001,
        0x33c0_0000,
        0x3880_0000,
        0x387f_ff00,
        0x387f_fe00,
        0x3800_0000,
        0x3100_0000,
        0xb100_0000,
        0x322b_cc77,
        0x477f_e000,
        0xc77f_e000,
        0x477f_df00,
        0x4700_0000,
        0xc700_0000,
        0x3dcc_cccd,
        0x4049_0fdb,
        0x449a_522b,
        0x3eaa_aaab,
        0x0000_0001,
        0x007f_ffff,
        0x8000_0001,
    ];
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    (0..n)
        .map(|i| {
            if i < special.len() {
                f32::from_bits(special[i])
            } else {
                let r = rng.next();
                let exp = (127 - 26 + (r >> 40) % 41) as u32; // 2^-26 ..= 2^14: finite, never overflows f16
                f32::from_bits(
                    ((r as u32) & 0x8000_0000) | (exp << 23) | ((r >> 8) as u32 & 0x7f_ffff),
                )
            }
        })
        .collect()
}
fn run_preconvert(
    rig: &Rig,
    q: &[f32],
    karg_batch: i32,
    wgs: u32,
    do_rotate: i32,
) -> Result<(Vec<u8>, bool, usize), String> {
    let out_bytes = q.len() * 2;
    let mut karg = Vec::new();
    for ptr in [E_Q, E_OUT + GUARD as u64, 0x7000_0000u64, 0x7100_0000] {
        karg.extend(ptr.to_le_bytes());
    }
    karg.extend(karg_batch.to_le_bytes());
    karg.extend(do_rotate.to_le_bytes());
    assert_eq!(karg.len(), PRE_KERNARG as usize);
    let mut mem = Memory::default();
    mem.map("kernarg".into(), E_KARG, karg)?;
    mem.map(
        "q".into(),
        E_Q,
        q.iter().flat_map(|x| x.to_le_bytes()).collect(),
    )?;
    mem.map("q16".into(), E_OUT, vec![SENTINEL; out_bytes + 2 * GUARD])?;
    let emu = Emulator::new(&rig.program, PRE)?;
    for x in 0..wgs {
        emu.run_workgroup(
            &Launch {
                kernarg_ptr: E_KARG,
                workgroup: [x, 0, 0],
                block: [PRE_BLOCK, 1, 1],
                dynamic_lds: 0,
            },
            &mut mem,
        )?;
    }
    let o = mem.regions.iter().find(|r| r.name == "q16").unwrap();
    let guards = o.bytes[..GUARD]
        .iter()
        .chain(&o.bytes[GUARD + out_bytes..])
        .all(|&b| b == SENTINEL);
    let q_written = mem
        .regions
        .iter()
        .find(|r| r.name == "q")
        .unwrap()
        .written
        .count();
    assert_eq!(q_written, 0, "preconvert wrote its f32 input");
    Ok((
        o.bytes[GUARD..GUARD + out_bytes].to_vec(),
        guards,
        o.written.count(),
    ))
}
#[test]
fn emitted_preconvert_matches_rne() {
    let rig = Rig::new();
    for batch in [1usize, 3, 5] {
        let q = preconvert_values(batch * 4096);
        let want = le16(&q.iter().map(|&x| f16_rne(x)).collect::<Vec<_>>());
        let needed = (batch * 16).div_ceil(4) as u32;
        // exact grid, then overlaunched (blocks entirely past the last row must neither read nor write).
        for wgs in [needed, needed + 3] {
            let (got, guards, written) = run_preconvert(&rig, &q, batch as i32, wgs, 1)
                .unwrap_or_else(|e| panic!("batch {batch} grid {wgs}: {e} (an ignored historical argument or an out-of-range row was dereferenced)"));
            assert!(
                guards,
                "batch {batch} grid {wgs}: a store escaped the Q16 scratch"
            );
            assert_eq!(
                written,
                batch * 4096 * 2,
                "batch {batch} grid {wgs}: every live element written once, none else"
            );
            if let Some(at) = got.iter().zip(&want).position(|(a, b)| a != b) {
                let e = at / 2;
                panic!("batch {batch} grid {wgs}: f16 element {e} (row {} head {} dim {}): got {:#06x}, want {:#06x} for f32 {:#010x}", e / 4096, (e / 256) % 16, e % 256, u16::from_le_bytes([got[e * 2], got[e * 2 + 1]]), u16::from_le_bytes([want[e * 2], want[e * 2 + 1]]), q[e].to_bits());
            }
        }
        // do_rotate = 0 gives the same bytes: the signs / rotation arguments are ignored.
        let (again, _, _) = run_preconvert(&rig, &q, batch as i32, needed, 0).unwrap();
        assert!(again == want, "batch {batch}: do_rotate changed the output");
    }
    // A zero or negative launch batch converts nothing, with the input still exactly mapped.
    for karg_batch in [0, -1] {
        let q = preconvert_values(4096);
        let (got, guards, written) = run_preconvert(&rig, &q, karg_batch, 4, 0).unwrap();
        assert!(
            guards && written == 0 && got.iter().all(|&b| b == SENTINEL),
            "batch {karg_batch}"
        );
    }
}

// ------------------------------------------------------------------------------------------------ 7. attention state
fn single_key_boundary_case() -> Case {
    // Single live key: output = V[0] dequantized. V blocks span the f16 staging boundaries.
    let mut c = winner_case(8, vec![0; 8], 1.0, 1.0, 3);
    const SCALES: [u16; 8] = [
        0x3c00, 0x6000, 0x0001, 0x03ff, 0x0400, 0x3c01, 0x7bff, 0x2800,
    ];
    c.v = build_cache(1, |_, g, b| {
        let mut codes = [0i8; 32];
        for (i, code) in codes.iter_mut().enumerate() {
            *code = match SCALES[b] {
                0x7bff => [-1, 0, 1, 1][(i + g) % 4],
                0x6000 => (i as i32 * 8 - 127 + g as i32).clamp(-127, 127) as i8,
                _ => [-128i32, 127, -1, 0, 1, 64, -64, 100][(i + b + g) % 8] as i8,
            };
        }
        (SCALES[b], codes)
    });
    c
}
#[test]
fn emitted_attention_preserves_state_and_layout() {
    let rig = Rig::new();
    // Zero-score / uniform probabilities: every live key ties exactly, probabilities are 1, O = integer V sums.
    check(
        &rig,
        "uniform scale +0",
        &winner_case(9, (0..9).collect(), 0.0, 1.0, 0),
    );
    check(
        &rig,
        "uniform scale -0",
        &winner_case(8, vec![64, 0, 63, 17, 5, 31, 32, 9], -0.0, 1.0, 1),
    );
    // Single live key (every row position 0): the output is that key's dequantized V.
    check(
        &rig,
        "single live key",
        &winner_case(8, vec![0; 8], 1.0, 1.0, 2),
    );
    check(
        &rig,
        "single live key, V scale / code boundaries",
        &single_key_boundary_case(),
    );
    // `scale_attn` is the argument, never a constant: the sign and magnitude select the winner of each row.
    let positions = vec![3, 0, 17, 16, 5, 31, 32, 9];
    for (scale, qmag) in [
        (1.0f32, 1.0f32),
        (-1.0, 1.0),
        (0.0625, 16.0),
        (-0.0625, 16.0),
        (8.0, 0.125),
        (1.0e30, 1.0),
        (-1.0e30, 1.0),
    ] {
        check(
            &rig,
            &format!("scale {scale:e}"),
            &winner_case(8, positions.clone(), scale, qmag, 0),
        );
    }
    // Replay on a previously used output: changed Q / K / V / positions, the previous result left in place, then a
    // NaN-poisoned output: all three agree byte for byte with the reference (no dependence on retained scratch).
    let first = winner_case(8, positions.clone(), 1.0, 1.0, 0);
    let second = winner_case(8, vec![32, 31, 16, 17, 0, 3, 9, 5], -1.0, 1.0, 5);
    assert!(first.q16 != second.q16 || first.k != second.k || first.v != second.v);
    let run1 = run_direct(&rig, &first, &Initial::Poison(POISON), None, &grid(8)).unwrap();
    assert_matches("replay, first launch", &run1, &expected(&first), 8);
    let want = expected(&second);
    let run2 = run_direct(
        &rig,
        &second,
        &Initial::Bytes(run1.out.clone()),
        None,
        &grid(8),
    )
    .unwrap();
    assert_matches("replay on the previous output", &run2, &want, 8);
    let run3 = run_direct(&rig, &second, &Initial::Poison(POISON), None, &grid(8)).unwrap();
    assert_matches("replay on a poisoned output", &run3, &want, 8);
    assert!(run2.out == run3.out && run2.out != run1.out);
    // Workgroup-uniform launch guards: a wrong compiled shape, an empty / negative batch, a KV head or row group past
    // the launch store nothing, read nothing past the exact buffers and leave every input untouched.
    for args in [
        [15, 2, 256, 8],
        [16, 1, 256, 8],
        [16, 3, 256, 8],
        [16, 2, 128, 8],
        [16, 2, 512, 8],
        [16, 2, 256, 0],
        [16, 2, 256, -1],
    ] {
        let run = run_direct(&rig, &first, &Initial::Poison(POISON), Some(args), &grid(8))
            .unwrap_or_else(|e| panic!("{args:?}: {e}"));
        assert!(
            run.guards_intact && run.inputs_untouched && run.out_written == 0,
            "{args:?}"
        );
        assert!(
            words(&run.out).iter().all(|&w| w == POISON),
            "{args:?}: output changed"
        );
    }
    for wg in [[1u32, 0, 0], [0, 2, 0], [7, 1, 0], [0, 255, 0]] {
        let run = run_direct(&rig, &first, &Initial::Poison(POISON), None, &[wg])
            .unwrap_or_else(|e| panic!("{wg:?}: {e}"));
        assert!(
            run.guards_intact && run.inputs_untouched && run.out_written == 0,
            "{wg:?}"
        );
        assert!(
            words(&run.out).iter().all(|&w| w == POISON),
            "{wg:?}: output changed"
        );
    }
}

// ------------------------------------------------------------------------------------------------ shared codecs
#[test]
fn emitted_fp8_and_q8_share_preconvert_abi_and_exact_staged_attention() {
    // Literal OCP encodings, independent of the production packer and the FP8 suite's decoder. Each signed integer
    // is exactly representable in both codecs; scales are per head, repeated across the eight Q8 blocks only.
    const CODES: [(i8, u8); 18] = [
        (-128, 0xf0),
        (-96, 0xec),
        (-64, 0xe8),
        (-16, 0xd8),
        (-8, 0xd0),
        (-2, 0xc0),
        (-1, 0xb8),
        (0, 0x00),
        (1, 0x38),
        (2, 0x40),
        (8, 0x50),
        (16, 0x58),
        (32, 0x60),
        (64, 0x68),
        (72, 0x69),
        (80, 0x6a),
        (96, 0x6c),
        (112, 0x6e),
    ];
    fn native_rows(q8: &[u8]) -> Vec<u8> {
        assert_eq!(q8.len() % POS_STRIDE, 0);
        let npos = q8.len() / POS_STRIDE;
        let mut fp8 = vec![0u8; npos * 516];
        for p in 0..npos {
            for g in 0..2 {
                let first = p * POS_STRIDE + g * HEAD_STRIDE;
                fp8[p * 516 + 512 + g * 2..p * 516 + 514 + g * 2]
                    .copy_from_slice(&q8[first..first + 2]);
                for d in 0..256 {
                    let block = first + (d / 32) * 34;
                    assert_eq!(
                        &q8[block..block + 2],
                        &q8[first..first + 2],
                        "per-head scale p{p} g{g}"
                    );
                    let code = q8[block + 2 + d % 32] as i8;
                    fp8[p * 516 + g * 256 + d] = CODES
                        .iter()
                        .find(|&&(q, _)| q == code)
                        .expect("literal OCP code")
                        .1;
                }
            }
        }
        fp8
    }

    let q8_rig = Rig::new();
    let fp8_spec = FaPrefillSpec {
        kv: FaKvCodec::Fp8E4M3,
        ..spec()
    };
    let (entries, text, _) = fa_prefill::emit_fp8_module(ARCH, &fp8_spec).unwrap();
    assert_eq!(
        entries.len(),
        1,
        "FP8 shares the Q8 preconvert rather than re-exporting it"
    );
    let fp8_symbol = "attention_fp8_e4m3_fa2_gqa_f16_pm_gfx1201";
    assert_eq!(fa_prefill::SYMBOL_FP8_DIRECT, fp8_symbol);
    check_resources(
        &entries[0].s_text,
        fp8_symbol,
        DIRECT_KERNARG,
        DIRECT_BLOCK,
        &DIRECT_ARGS,
    );
    let contract = fa_prefill::fp8_contract(ARCH, &entries[0]);
    assert_eq!(contract["launch_dynamic_lds_bytes"], FP8_DIRECT_LDS);
    assert_eq!(contract["vgpr_max"], 224);
    assert_eq!(entries[0].shape.next_free_vgpr, 224);
    assert_eq!(q8_rig.dynamic_lds, DIRECT_LDS);
    let fp8_rig = Rig::from_text(&text, fp8_symbol, FP8_DIRECT_LDS);
    let positions = vec![0, 15, 16, 31, 32, 63, 64, 65, 1];
    let mut q8 = winner_case(9, positions, 1.0, 1.0, 2);
    let npos = q8.k.len() / POS_STRIDE;
    q8.k = build_cache(npos, |p, g, b| {
        let scale = f16_rne(if g == 0 {
            0.25 * (p + 1) as f32
        } else {
            0.25 * (200 - p) as f32
        });
        let codes = std::array::from_fn(|i| (64 + 8 * ((b * 32 + i + g) % 3)) as i8);
        (scale, codes)
    });
    q8.v = build_cache(npos, |p, g, b| {
        let codes = std::array::from_fn(|i| CODES[(p * 29 + g * 7 + b * 32 + i) % CODES.len()].0);
        (if g == 0 { 0x3c01 } else { 0x3400 }, codes)
    });
    // Use the actual emitted Q8 preconvert's bytes for BOTH direct exports, not a second FP8 qprep implementation.
    let q: Vec<f32> = q8.q16.iter().map(|&h| f16_to_f32(h)).collect();
    let (converted, guards, written) = run_preconvert(
        &q8_rig,
        &q,
        q8.batch as i32,
        (q8.batch * 16).div_ceil(4) as u32,
        0,
    )
    .unwrap();
    assert!(guards);
    assert_eq!(written, q8.batch * 4096 * 2);
    assert_eq!(converted, le16(&q8.q16));
    q8.q16 = converted
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let mut fp8 = q8.clone();
    fp8.k = native_rows(&q8.k);
    fp8.v = native_rows(&q8.v);
    for scale in [1.0f32, 0.0, -1.0] {
        q8.scale = scale;
        fp8.scale = scale;
        let want = expected(&q8);
        let q8_run = run_direct(
            &q8_rig,
            &q8,
            &Initial::Poison(POISON),
            None,
            &grid(q8.batch),
        )
        .unwrap();
        let fp8_run = run_direct(
            &fp8_rig,
            &fp8,
            &Initial::Poison(POISON),
            None,
            &grid(fp8.batch),
        )
        .unwrap();
        assert_matches("shared-codec Q8", &q8_run, &want, q8.batch);
        assert_matches("shared-codec FP8", &fp8_run, &want, fp8.batch);
        assert_eq!(
            fp8_run.out, q8_run.out,
            "scale {scale}: equal staged inputs must have equal output bytes"
        );
    }
}

// ------------------------------------------------------------------------------------------------ 8. bundle
fn kernels_dir() -> String {
    format!("{}/kernels", env!("CARGO_MANIFEST_DIR"))
}
fn top_bundle_path() -> String {
    format!(
        "{}/../../kernels/{MODULE}.hxaco",
        env!("CARGO_MANIFEST_DIR")
    )
}
#[test]
fn committed_bundle_equals_fresh_emission() {
    let path = top_bundle_path();
    let committed = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e} (emit it: peacemaker-isa emit --kernel fa_prefill_pm --arch gfx1201 ... --bundle kernels/{MODULE}.hxaco)"));
    let (_, text) = emitted();
    let elf = native::assemble(&text, ARCH).unwrap();
    let fresh = native::bundle(&elf, ARCH, native::DEFAULT_HOST_TARGET);
    assert!(
        committed == fresh,
        "{MODULE}: fresh native emission differs from the committed bundle {path}"
    );
    let lifted = peacemaker_lift::lift_object(
        &committed,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    )
    .unwrap_or_else(|e| panic!("the committed bundle does not lift: {e}"));
    assert_eq!(format!("{:?}", lifted.program.target.arch), "Gfx1201");
    let symbols: Vec<&str> = lifted
        .program
        .kernels
        .iter()
        .map(|k| k.symbol.0.as_str())
        .collect();
    assert_eq!(
        symbols, SYMBOLS,
        "the committed bundle must hold exactly the two exports, preconvert then direct"
    );
}

// ------------------------------------------------------------------------------------------------ 9. contracts
/// The contract's `counts` / `forbidden`, evaluated against an independent census of an emitted source.
fn check_against_census(contract: &Value, text: &str, symbol: &str) {
    let census = |name: &str| -> usize {
        if let Some(prefix) = name.strip_suffix('*') {
            return count_prefix(text, prefix);
        }
        match name {
            "valu_slots" => insns(text)
                .iter()
                .filter(|l| {
                    mnemonic(l).starts_with("v_")
                        && !mnemonic(l).starts_with("v_wmma_")
                        && !mnemonic(l).starts_with("v_swmmac_")
                })
                .count(),
            "vopd_packets" => count_prefix(text, "v_dual_"),
            _ => count(text, name),
        }
    };
    for (name, bound) in contract["counts"]
        .as_object()
        .unwrap_or_else(|| panic!("{symbol}: no counts"))
    {
        let n = census(name) as u64;
        match bound {
            Value::Number(v) => assert_eq!(v.as_u64(), Some(n), "{symbol}: {name}"),
            Value::Object(o) => {
                let min = o.get("min").and_then(Value::as_u64);
                let max = o.get("max").and_then(Value::as_u64);
                assert!(
                    min.is_some() || max.is_some(),
                    "{symbol}: unbounded count {name}"
                );
                assert!(
                    min.is_none_or(|m| n >= m) && max.is_none_or(|m| n <= m),
                    "{symbol}: {name} observed {n}, contract {bound}"
                );
            }
            _ => panic!("{symbol}: bad bound for {name}: {bound}"),
        }
    }
    for pattern in forbidden(contract) {
        assert_eq!(
            census(&pattern),
            0,
            "{symbol}: a forbidden instruction `{pattern}` is emitted"
        );
    }
}
fn forbidden(contract: &Value) -> Vec<String> {
    contract["forbidden"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect()
}
fn forbids(contract: &Value, mnemonic: &str) -> bool {
    forbidden(contract)
        .iter()
        .any(|p| match p.strip_suffix('*') {
            Some(prefix) => mnemonic.starts_with(prefix),
            None => p == mnemonic,
        })
}
#[test]
fn committed_contracts_equal_fresh_contracts() {
    let (entries, _) = emitted();
    let (fresh_pre, fresh_direct) = (
        fa_prefill::preconvert_contract(ARCH, &entries[0]),
        fa_prefill::contract(ARCH, &entries[1]),
    );
    for (name, fresh) in [("preconvert", &fresh_pre), ("direct", &fresh_direct)] {
        let file = format!(
            "{}/fa_prefill_pm.gfx1201.{name}.contract.json",
            kernels_dir()
        );
        let committed: Value =
            serde_json::from_slice(&std::fs::read(&file).unwrap_or_else(|e| panic!("{file}: {e}")))
                .unwrap();
        assert_eq!(
            &committed, fresh,
            "{file}: committed contract differs from the derived contract"
        );
    }
    // Independent assertions on the contents (not on the emitter's own bookkeeping).
    let (pre, direct) = (&fresh_pre, &fresh_direct);
    assert_eq!(
        (pre["symbol"].as_str(), direct["symbol"].as_str()),
        (Some(PRE), Some(DIRECT))
    );
    for (c, text, symbol) in [
        (pre, &entries[0].s_text, PRE),
        (direct, &entries[1].s_text, DIRECT),
    ] {
        check_against_census(c, text, symbol);
        for m in [
            "scratch_load_b32",
            "scratch_store_b32",
            "global_atomic_add_f32",
            "buffer_atomic_add_u32",
            "s_setpc_b64",
            "s_swappc_b64",
        ] {
            assert!(
                forbids(c, m),
                "{symbol}: `{m}` (scratch / atomics / dynamic calls) must be forbidden"
            );
        }
        assert_eq!(
            (
                c["require_wave32"].as_bool(),
                c["require_zero_spills"].as_bool(),
                c["require_zero_private"].as_bool()
            ),
            (Some(true), Some(true), Some(true)),
            "{symbol}"
        );
        assert!(
            c["vgpr_max"].as_u64().unwrap() <= 256 && c["sgpr_max"].as_u64().unwrap() <= 104,
            "{symbol}"
        );
    }
    assert_eq!(
        (
            direct["launch_dynamic_lds_bytes"].as_u64(),
            direct["group_segment_fixed_bytes"].as_u64()
        ),
        (Some(65536), Some(0))
    );
    assert_eq!(
        (
            pre["launch_dynamic_lds_bytes"].as_u64(),
            pre["group_segment_fixed_bytes"].as_u64()
        ),
        (Some(0), Some(0))
    );
    // The direct attention may use DS / global memory, barriers and WMMA; the preconvert may not touch LDS, barrier or
    // use WMMA.
    for m in [
        "ds_load_b128",
        "ds_store_b64",
        "global_load_b128",
        "global_store_b128",
        "s_barrier_signal",
        "s_barrier_wait",
        "v_wmma_f32_16x16x16_f16",
    ] {
        assert!(!forbids(direct, m), "direct: `{m}` must be allowed");
    }
    for m in [
        "ds_load_b128",
        "ds_store_b32",
        "s_barrier_signal",
        "s_barrier_wait",
        "v_wmma_f32_16x16x16_f16",
    ] {
        assert!(forbids(pre, m), "preconvert: `{m}` must be forbidden");
    }
    let (dt, pt) = (&entries[1].s_text, &entries[0].s_text);
    assert!(count_prefix(dt, "v_wmma_") > 0, "direct: no WMMA");
    assert!(count_prefix(dt, "ds_") > 0 && count_prefix(dt, "global_") > 0);
    assert_eq!(
        (count(dt, "s_barrier_signal"), count(dt, "s_barrier_wait")),
        (2, 2),
        "direct: two tile rendezvous sites (publish, retire)"
    );
    assert_eq!(
        entries[1].proof.barriers.len(),
        2,
        "direct: proof barrier sites"
    );
    assert_eq!(
        count_prefix(pt, "ds_") + count_prefix(pt, "s_barrier") + count_prefix(pt, "v_wmma_"),
        0,
        "preconvert: no DS / barrier / WMMA"
    );
    // The committed M7 receipt map: both symbols, byte-exact lift, no obligations, no ambiguous delays.
    let file = format!("{}/fa_prefill_pm.gfx1201.m7.json", kernels_dir());
    let m7: Value =
        serde_json::from_slice(&std::fs::read(&file).unwrap_or_else(|e| panic!("{file}: {e}")))
            .unwrap();
    let keys: std::collections::BTreeSet<&str> =
        m7.as_object().unwrap().keys().map(String::as_str).collect();
    assert_eq!(keys, SYMBOLS.into_iter().collect(), "{file}: receipt keys");
    for symbol in SYMBOLS {
        let r = &m7[symbol];
        assert_eq!(r["lift"], "byte-exact", "{symbol}");
        assert_eq!(r["obligations"], serde_json::json!({}), "{symbol}");
        assert_eq!(r["ambiguous_delays"], 0, "{symbol}");
        assert!(r["instructions"].as_u64().unwrap() > 0, "{symbol}");
    }
}

// ------------------------------------------------------------------------------------------------ 10. certification
#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use peacemaker_isa::{
        audit, ledger_replay, pm_check,
        toolchain::{
            build, certify, oracle_assemble_link_bundle, read_kd, IsaShapeContract, Toolchain,
        },
    };

    const HOST: &str = "host-x86_64-unknown-linux-gnu-";
    /// The pinned HIP object's `compute_pgm_rsrc1 & 0x000ff000` for both exports (RNE, denormals preserved).
    const PINNED_FP_MASK: u32 = 0x000f_0000;

    fn tmp(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "peacemaker-isa-faprefill-{tag}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Native code object and bundle against the ROCm oracle (`llvm-mc` + `ld.lld -shared`,
    /// `clang-offload-bundler`) for each export alone and for the module: `.text` and everything else equal except the
    /// intentionally different `.comment` stamp.
    #[test]
    fn native_code_objects_and_bundles_equal_the_rocm_oracle() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        let toolchain = Toolchain {
            host_target: HOST.into(),
            ..Toolchain::default()
        };
        let (entries, text) = emitted();
        let mut units: Vec<(String, String)> = entries
            .iter()
            .zip(SYMBOLS)
            .map(|(e, s)| (s.to_owned(), e.s_text.clone()))
            .collect();
        units.push((format!("{MODULE} (module)"), text));
        let dir = tmp("identity");
        let mut failures = Vec::new();
        for (i, (name, text)) in units.iter().enumerate() {
            let s = dir.join(format!("u{i}.s"));
            std::fs::write(&s, text).unwrap();
            let oracle = oracle_assemble_link_bundle(
                &toolchain,
                &s,
                &dir.join(format!("u{i}.hxaco")),
                "gfx1201",
            )
            .unwrap_or_else(|e| panic!("{name}: oracle: {e}"));
            let (oracle_elf, oracle_bundle) = (
                std::fs::read(&oracle.elf).unwrap(),
                std::fs::read(&oracle.hsaco).unwrap(),
            );
            match native::assemble(text, ARCH) {
                Ok(elf) => {
                    let bundle = native::bundle(&elf, ARCH, HOST);
                    if let Some(why) = native_compare::compare_native_to_oracle(
                        &elf,
                        &bundle,
                        &oracle_elf,
                        &oracle_bundle,
                    ) {
                        failures.push(format!("{name}: {why}"))
                    }
                }
                Err(e) => failures.push(format!("{name}: native: {e}")),
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
        assert!(
            failures.is_empty(),
            "{} of {} units differ from the oracle:\n{}",
            failures.len(),
            units.len(),
            failures.join("\n")
        );
    }

    /// Wait replay, descriptor FP mode against the pinned HIP object's, M7 and the all-lane LDS bounds of both
    /// exports on the natively linked object, without ROCm tools.
    #[test]
    fn both_exports_pass_wait_replay_descriptor_lds_bounds_and_m7() {
        let (entries, text) = emitted();
        for e in &entries {
            ledger_replay::replay_waits(&e.s_text, ARCH)
                .unwrap_or_else(|err| panic!("{}: {err}", e.proof.variant));
        }
        ledger_replay::replay_waits(&text, ARCH).unwrap();
        let dir = tmp("m7");
        let co = dir.join("module.co");
        std::fs::write(&co, native::assemble(&text, ARCH).unwrap()).unwrap();
        let lab = std::env::var("HIPFIRE_FA_LAB_ELF")
            .ok()
            .map(std::path::PathBuf::from);
        for (e, symbol, waves, limit, lab_symbol) in [
            (
                &entries[0],
                PRE,
                4u32,
                0u32,
                "attention_fa2_q_preconvert_gfx1201",
            ),
            (
                &entries[1],
                DIRECT,
                5,
                DIRECT_LDS,
                "attention_q8_0_fa2_gqa_gfx1201",
            ),
        ] {
            let m7 = pm_check::m7(&co, "gfx1201", symbol)
                .unwrap_or_else(|err| panic!("{symbol}: {err}"));
            assert_eq!(
                (
                    m7["lift"].as_str(),
                    &m7["obligations"],
                    &m7["ambiguous_delays"]
                ),
                (
                    Some("byte-exact"),
                    &serde_json::json!({}),
                    &serde_json::json!(0)
                ),
                "{symbol}"
            );
            // A complete, tight all-lane LDS bound (not just empty obligations).
            let end = pm_check::lds_bounds(&e.s_text, symbol, waves, limit)
                .unwrap_or_else(|err| panic!("{symbol}: {err}"));
            assert_eq!(
                end, limit,
                "{symbol}: the LDS bound is not the whole allocation"
            );
            if limit > 0 {
                assert!(
                    pm_check::lds_bounds(&e.s_text, symbol, waves, limit - 1).is_err(),
                    "{symbol}: bound is not tight"
                );
            }
            let kd = read_kd(&co, symbol).unwrap();
            assert_eq!(
                (kd.group_segment_size, kd.private_segment_size, kd.wave32),
                (0, 0, true),
                "{symbol}"
            );
            assert_eq!(
                kd.compute_pgm_rsrc1 & 0x000f_f000,
                PINNED_FP_MASK,
                "{symbol}: FP round/denorm mode differs from the pinned HIP object"
            );
            assert_eq!(kd.compute_pgm_rsrc1 & (1 << 26), 0, "{symbol}: FP16_OVFL");
            if let Some(lab) = lab.as_ref().filter(|p| p.is_file()) {
                let theirs = read_kd(lab, lab_symbol).unwrap();
                assert_eq!(
                    kd.compute_pgm_rsrc1 & 0x000f_f000,
                    theirs.compute_pgm_rsrc1 & 0x000f_f000,
                    "{symbol}: FP mode vs {lab_symbol} in {}",
                    lab.display()
                );
                assert_eq!(
                    (kd.group_segment_size, kd.private_segment_size, kd.wave32),
                    (
                        theirs.group_segment_size,
                        theirs.private_segment_size,
                        theirs.wave32
                    ),
                    "{symbol}"
                );
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Every symbol certifies against the ROCm oracle build: native-built object with an independent objdump
    /// parse-back, independent wait replay, its derived shape contract, M7 and the LDS bound of the linked ELF, the
    /// lifted-bundle DS-drain scan, and the committed M7 receipts equal fresh ones.
    #[test]
    fn module_passes_certification() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        let (entries, text) = emitted();
        for e in &entries {
            ledger_replay::replay_waits(&e.s_text, ARCH).unwrap();
        }
        ledger_replay::replay_waits(&text, ARCH).unwrap();
        let dir = tmp("cert");
        let s = dir.join("module.s");
        std::fs::write(&s, &text).unwrap();
        let toolchain = Toolchain::oracle();
        let build = build(&toolchain, &s, &dir.join("module.hsaco"), "gfx1201").unwrap();
        assert!(
            build.parse_back_checked,
            "the objdump parse-back did not run"
        );
        let mut fresh_receipts = serde_json::Map::new();
        for (e, symbol, waves, limit) in [
            (&entries[0], PRE, 4u32, 0u32),
            (&entries[1], DIRECT, 5, DIRECT_LDS),
        ] {
            let m7 = pm_check::m7(&build.elf, "gfx1201", symbol)
                .unwrap_or_else(|err| panic!("{symbol}: {err}"));
            assert_eq!(
                (
                    m7["lift"].as_str(),
                    &m7["obligations"],
                    &m7["ambiguous_delays"]
                ),
                (
                    Some("byte-exact"),
                    &serde_json::json!({}),
                    &serde_json::json!(0)
                ),
                "{symbol}"
            );
            assert_eq!(
                pm_check::lds_bounds(&e.s_text, symbol, waves, limit).unwrap(),
                limit,
                "{symbol}"
            );
            let contract_json = if symbol == PRE {
                fa_prefill::preconvert_contract(ARCH, e)
            } else {
                fa_prefill::contract(ARCH, e)
            };
            let contract: IsaShapeContract = serde_json::from_value(contract_json)
                .unwrap_or_else(|err| panic!("{symbol}: contract does not parse: {err}"));
            certify(
                &toolchain,
                &build,
                &s,
                "gfx1201",
                &dir.join(format!("{symbol}.manifest.json")),
                Some(&contract),
                "test",
                "test",
            )
            .unwrap_or_else(|err| panic!("{symbol}: {err}"));
            fresh_receipts.insert(symbol.to_owned(), m7);
        }
        // Distinguish the static tile-loop body from its dynamic iteration count: the shape counts are static.
        assert!(count_prefix(&entries[1].s_text, "v_wmma_") > 0);
        // DS-drain scan of the final bundle: every barrier is preceded by drained DS stores.
        let bundle = std::fs::read(&build.hsaco).unwrap();
        let scan = audit::lds_barrier_scan(&bundle).unwrap();
        assert!(
            scan.sites.is_empty(),
            "lds_store_unwaited_at_barrier: {:?}",
            scan.sites
        );
        assert_eq!(
            (scan.kernels, scan.barrier_kernels, scan.barriers),
            (2, 1, 2),
            "{scan:?}"
        );
        // The committed bundle and receipts are the ones certified here.
        assert!(
            std::fs::read(top_bundle_path()).unwrap() == bundle,
            "the committed bundle is not the certified build"
        );
        let file = format!("{}/fa_prefill_pm.gfx1201.m7.json", kernels_dir());
        let committed: Value =
            serde_json::from_slice(&std::fs::read(&file).unwrap_or_else(|e| panic!("{file}: {e}")))
                .unwrap();
        assert_eq!(
            committed,
            Value::Object(fresh_receipts),
            "{file}: committed M7 receipts differ from fresh ones"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
// ------------------------------------------------------------------------------------------------ Q8 LDS lifecycle
#[test]
fn q8_two_plane_lds_lifecycle_is_ready_reads_retire() {
    // The Q8 direct keeps two resident 32-KiB planes across the whole tile loop: K publishes, every compute wave
    // reads K (QK) and V (PV), then both planes retire before the next tile. The helper wave (wave 4) skips the
    // compute but joins both rendezvous. This pins the proof slot roles, the exact barrier transitions, the label
    // order and the helper path using only the public emitted IR/text.
    let direct = fa_prefill::emit(ARCH, &spec()).unwrap();
    let text = direct.s_text.clone();
    // Proof slot roles: `k_plane` id 0 base 0 len 32768, `v_plane` id 1 base 32768 len 32768.
    let slots = &direct.proof.lds_slots;
    assert_eq!(slots.len(), 2, "Q8 direct: two LDS planes");
    assert_eq!(
        (slots[0].name.as_str(), slots[0].base, slots[0].len),
        ("k_plane", 0, 32_768),
        "Q8 direct: K plane slot"
    );
    assert_eq!(
        (slots[1].name.as_str(), slots[1].base, slots[1].len),
        ("v_plane", 32_768, 32_768),
        "Q8 direct: V plane slot"
    );
    // Proof barrier lifecycle: Ready(0,1) publishes both planes, Retire(0,1) releases them after the last V read.
    let barriers = &direct.proof.barriers;
    assert_eq!(barriers.len(), 2, "Q8 direct: publish + retire rendezvous");
    assert_eq!(
        barriers[0].transitions,
        vec!["Ready(0)".to_string(), "Ready(1)".to_string()],
        "Q8 direct: publish transitions"
    );
    assert_eq!(
        barriers[1].transitions,
        vec!["Retire(0)".to_string(), "Retire(1)".to_string()],
        "Q8 direct: retire transitions"
    );
    // Label order: fill_done (publish) < qk (compute entry) < turn (retire) < epi < end.
    let labels = [
        ".Lfapd_fill_done",
        ".Lfapd_qk",
        ".Lfapd_turn",
        ".Lfapd_epi",
        ".Lfapd_end",
    ];
    let mut cursor = 0;
    for label in labels {
        let at = text[cursor..]
            .find(label)
            .unwrap_or_else(|| panic!("Q8 direct: missing label {label}"));
        cursor += at + label.len();
    }
    // Label definitions carry the trailing colon; bare finds would catch earlier branch uses.
    let def = |label: &str| {
        text.find(&format!("{label}:"))
            .unwrap_or_else(|| panic!("Q8 direct: missing label {label}"))
    };
    let (fill_done_def, qk_def, turn_def) = (
        def(".Lfapd_fill_done"),
        def(".Lfapd_qk"),
        def(".Lfapd_turn"),
    );
    // Helper path: the wave>=4 guard is the first branch after the QK entry label and skips compute straight to
    // the retire label, so the helper wave joins the retire rendezvous without reading either plane.
    let after_qk: Vec<&str> = text[qk_def..].lines().skip(1).take(3).collect();
    assert!(
        after_qk
            .iter()
            .any(|l| l.contains("s_cbranch_scc1 .Lfapd_turn")),
        "Q8 direct: helper wave at the QK entry must branch to .Lfapd_turn"
    );
    // Reads between publish and retire: 16-byte fragment loads hit BOTH planes — K plane offsets below 32768
    // (QK) and V plane offsets at/above 32768 (PV).
    let region: Vec<&str> = text[qk_def..turn_def]
        .lines()
        .filter_map(|l| {
            let t = l.split(';').next()?.split("//").next()?.trim();
            let tok = t.split_whitespace().next()?;
            let first = tok.chars().next()?;
            (first.is_ascii_lowercase()
                && tok.contains('_')
                && !tok.ends_with(':')
                && !tok.contains('.'))
            .then_some(t)
        })
        .collect();
    let mut saw_k = false;
    let mut saw_v = false;
    for l in &region {
        if mnemonic(l) == "ds_load_b128" {
            let off = imm_offset(l);
            assert!(
                off + 16 <= 65_536,
                "Q8 direct: fragment read past both planes"
            );
            if off < 32_768 {
                saw_k = true;
            } else {
                saw_v = true;
            }
        }
    }
    assert!(
        saw_k,
        "Q8 direct: no K-plane fragment read between publish and retire"
    );
    assert!(
        saw_v,
        "Q8 direct: no V-plane fragment read between publish and retire"
    );
    // Barrier placement in the text: the publish rendezvous sits between the fill_done label and the QK entry,
    // the retire rendezvous after the turn label.
    let mut signals: Vec<usize> = text
        .match_indices("s_barrier_signal")
        .map(|(i, _)| i)
        .collect();
    assert_eq!(signals.len(), 2, "Q8 direct: two text rendezvous sites");
    let retire = signals.remove(1);
    assert!(
        fill_done_def < signals[0] && signals[0] < qk_def,
        "Q8 direct: publish barrier is not between fill_done and the QK entry"
    );
    assert!(
        retire > turn_def,
        "Q8 direct: retire barrier is not after the turn label"
    );
}
