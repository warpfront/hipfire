// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The gfx1201 runtime-K R128 A4 producer `moe_rotate_i4_pm_gfx1201` (module `moe_rotate_i4_pm_gfx1201`): symbol,
//! 56-byte kernarg ABI, exact-arch gating, the instruction census (equal to `mq_i4_a4`'s frozen numerics, whose
//! functions it calls), the old/new identity of everything after the prologue, the row-local straight-line
//! structure, the committed bundle and contract identities, and (feature `toolchain`) wait replay, M7 and the
//! radiowave certification.
//!
//! CPU proofs here are ARITHMETIC / ADDRESS proofs: a Rust reference of the record arithmetic (the FWHT128 and the
//! C2 `{5,7}` quantizer, the CPU twin `fake_quantize_block_i4_128`) and a model of the prologue's guards and
//! addresses over every workgroup of K = 512, rows 512 / 4096 / 8192 x top-8, tied to the emitted text by source
//! assertions. The emitted ISA itself (both the sorted-slot and the grouped-position export) runs to completion on the
//! interpreter in `moe_rotate_i4_pm_emu.rs`, byte for byte against the same reference. The byte oracle of the emitted
//! bytes against the CPU twin on silicon is the GPU harness (K = 512 variant of
//! `crates/peacemaker-isa/tools/mq_i4_a4_oracle.py`, driven by the host runner).
use peacemaker_isa::kernels::moe_rotate_i4_pm::{self, Spec};
use peacemaker_isa::kernels::mq_i4_a4;
use peacemaker_isa::Arch;
#[path = "support/moe_rotate_ref.rs"]
mod reference;
use reference::*;
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCH: Arch = Arch::Gfx1201;
const MODULE: &str = "moe_rotate_i4_pm_gfx1201";
const SYMBOL: &str = "moe_rotate_i4_pm_gfx1201";
/// Contract ABI: `h sorted xq signs1 signs2` pointers, then `K P L` u32.
const ARGS: &[(u32, u32)] = &[
    (0, 8),
    (8, 8),
    (16, 8),
    (24, 8),
    (32, 8),
    (40, 4),
    (44, 4),
    (48, 4),
];

fn field(text: &str, key: &str) -> Vec<u32> {
    text.lines()
        .filter_map(|l| l.trim().strip_prefix(key))
        .map(|v| v.trim().parse().unwrap())
        .collect()
}
fn count(text: &str, mnemonic: &str) -> usize {
    text.lines()
        .filter(|l| l.split_whitespace().next() == Some(mnemonic))
        .count()
}
fn spec() -> Spec {
    Spec { arch: ARCH }
}

#[test]
fn module_export_is_the_frozen_symbol() {
    assert_eq!(moe_rotate_i4_pm::module_specs(ARCH), [spec()]);
    assert_eq!(spec().symbol(), SYMBOL);
    assert_eq!(spec().tag(), "r128");
    assert_eq!(Spec::module(ARCH), MODULE);
    let (emitted, text, proof) = moe_rotate_i4_pm::emit_module(ARCH).unwrap();
    assert_eq!(emitted.len(), 1);
    assert_eq!((proof.module.as_str(), proof.arch), (MODULE, ARCH));
    assert_eq!(proof.kernels.len(), 1);
    assert_eq!(text.matches(".amdhsa_kernel ").count(), 1);
    assert!(
        text.contains(&format!(".amdhsa_kernel {SYMBOL}\n"))
            && text.contains(&format!(".globl {SYMBOL}\n"))
    );
    assert!(text.contains(&format!(".name: {SYMBOL}\n")));
    assert!(text.contains("amdgcn-amd-amdhsa--gfx1201"));
}

#[test]
fn emission_is_deterministic_and_refuses_other_arches() {
    let (a, b) = (
        moe_rotate_i4_pm::emit(spec()).unwrap(),
        moe_rotate_i4_pm::emit(spec()).unwrap(),
    );
    assert_eq!(
        (a.proof.s_text_sha256, a.s_text),
        (b.proof.s_text_sha256, b.s_text)
    );
    let (_, ta, pa) = moe_rotate_i4_pm::emit_module(ARCH).unwrap();
    let (_, tb, pb) = moe_rotate_i4_pm::emit_module(ARCH).unwrap();
    assert_eq!((ta, pa.s_text_sha256), (tb, pb.s_text_sha256));
    for arch in [Arch::Gfx1100, Arch::Gfx1151] {
        assert!(moe_rotate_i4_pm::emit(Spec { arch }).is_err(), "{arch:?}");
        assert!(moe_rotate_i4_pm::emit_module(arch).is_err(), "{arch:?}");
    }
}

/// Wave32, one wave per workgroup, no scratch, spills, LDS or barrier, 72 VGPRs.
#[test]
fn export_is_wave32_without_scratch_or_lds() {
    let e = moe_rotate_i4_pm::emit(spec()).unwrap();
    assert_eq!(e.proof.arch, ARCH);
    assert!(e.s_text.contains(".amdhsa_wavefront_size32 1"), "wave32");
    assert!(
        e.s_text.contains(".private_segment_fixed_size: 0\n"),
        "private segment"
    );
    assert!(
        e.s_text.contains(".sgpr_spill_count: 0\n") && e.s_text.contains(".vgpr_spill_count: 0\n"),
        "spills"
    );
    assert!(
        e.s_text.contains(".group_segment_fixed_size: 0\n"),
        "no LDS"
    );
    assert_eq!(e.shape.barriers, 0);
    assert_eq!(e.shape.ds, 0, "no DS instruction at all");
    assert_eq!(e.shape.next_free_vgpr, moe_rotate_i4_pm::VGPRS);
    for forbidden in [
        "ds_",
        "buffer_",
        "flat_",
        "scratch_",
        "s_barrier",
        "v_wmma",
        "v_swmmac",
        "s_waitcnt",
        "v_readlane",
        "v_writelane",
        "v_fma_mix",
    ] {
        let hit = e.s_text.lines().find(|l| {
            l.split_whitespace()
                .next()
                .is_some_and(|m| m.starts_with(forbidden))
        });
        assert!(hit.is_none(), "emits {hit:?}");
    }
}

#[test]
fn kernarg_abi_matches_the_contract() {
    assert_eq!(moe_rotate_i4_pm::KERNARG_BYTES, 56);
    assert_eq!(moe_rotate_i4_pm::WORKGROUP_SIZE, 32);
    let e = moe_rotate_i4_pm::emit(spec()).unwrap();
    assert!(
        e.s_text.contains(".kernarg_segment_size: 56\n"),
        "kernarg size"
    );
    assert_eq!(
        field(&e.s_text, ".offset:"),
        ARGS.iter().map(|a| a.0).collect::<Vec<_>>(),
        "kernarg offsets"
    );
    assert_eq!(
        field(&e.s_text, ".size:"),
        ARGS.iter().map(|a| a.1).collect::<Vec<_>>(),
        "kernarg sizes"
    );
    assert_eq!(
        field(&e.s_text, ".max_flat_workgroup_size:"),
        [32],
        "block 32"
    );
    let (last_off, last_size) = *ARGS.last().unwrap();
    assert_eq!(
        (last_off + last_size).next_multiple_of(8),
        moe_rotate_i4_pm::KERNARG_BYTES,
        "arguments cover the kernarg block (rounded to the 8-byte alignment of the pointer arguments)"
    );
    for (name, access) in [
        ("h", "read_only"),
        ("sorted", "read_only"),
        ("xq", "write_only"),
        ("signs1", "read_only"),
        ("signs2", "read_only"),
    ] {
        assert!(e.s_text.contains(&format!("      - .actual_access: {access}\n        .address_space: global\n        .name: {name}\n")), "{name} access {access}");
    }
    // Kernargs: h, sorted, xq, signs1 in one 32-byte load; signs2, K, P in one 16-byte load at 0x20; L alone at 0x30.
    assert!(
        e.s_text.contains("s_load_b256 s[8:15], s[0:1], 0x0"),
        "first kernarg load"
    );
    assert!(
        e.s_text.contains("s_load_b128 s[16:19], s[0:1], 0x20"),
        "second kernarg load"
    );
    assert!(e.s_text.contains("s_load_b32 s43, s[0:1], 0x30"), "L load");
    // gfx12 delivers the group in ttmp7[15:0] only when the descriptor enables workgroup id y.
    assert!(
        e.s_text.contains(".amdhsa_system_sgpr_workgroup_id_y 1"),
        "workgroup id y enabled"
    );
}

/// The row-local structure. One straight-line function of one (row, group): the three guard branches to the single
/// exit (no loop, no other branch), and exactly the loads of that row group (the slot, the group's bf16 dwords
/// through the pointer s[30:31], the two 128-entry sign quads through s[14:15] / s[16:17]); the only stores are
/// the record's qs half-words and (d, s) through the record pointer s[32:33]. `P` (s19) reaches the code only
/// through the row guard; `K` (s18) only through the group guard and the row pitch; the workgroup ids come only from
/// `ttmp9` (row) and `ttmp7[15:0]` (group), with no division.
#[test]
fn producer_is_one_straight_line_row_local_function() {
    let t = moe_rotate_i4_pm::emit(spec()).unwrap().s_text;
    assert_eq!(
        count(&t, "s_cbranch_scc1"),
        3,
        "exit guards: row, group, pad slot"
    );
    for m in [
        "s_branch",
        "s_cbranch_scc0",
        "s_cbranch_vccz",
        "s_cbranch_vccnz",
        "s_cbranch_execz",
        "s_cbranch_execnz",
    ] {
        assert_eq!(count(&t, m), 0, "{m}");
    }
    let labels: Vec<&str> = t
        .lines()
        .map(str::trim)
        .filter(|l| l.ends_with(':') && (l.starts_with(".L") || *l == format!("{SYMBOL}:")))
        .collect();
    assert_eq!(
        labels,
        [
            format!("{SYMBOL}:").as_str(),
            ".Lmqi4a4_exit:",
            format!(".L{SYMBOL}_end:").as_str()
        ],
        "symbol, exit and end labels only"
    );
    assert_eq!(
        t.matches(".Lmqi4a4_exit").count(),
        4,
        "three branches to one exit label plus its definition"
    );
    assert_eq!(
        (
            count(&t, "s_load_b256"),
            count(&t, "s_load_b128"),
            count(&t, "s_load_b32")
        ),
        (1, 1, 2),
        "kernargs, L and the slot"
    );
    let loads: Vec<&str> = t
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("global_load"))
        .collect();
    assert_eq!(loads.len(), 3, "{loads:?}");
    assert_eq!(
        loads
            .iter()
            .filter(|l| l.starts_with("global_load_b64 v[6:7], v1, s[30:31]"))
            .count(),
        1,
        "{loads:?}"
    );
    assert_eq!(
        loads
            .iter()
            .filter(|l| l.starts_with("global_load_b128 v[8:11], v2, s[14:15]"))
            .count(),
        1,
        "{loads:?}"
    );
    assert_eq!(
        loads
            .iter()
            .filter(|l| l.starts_with("global_load_b128 v[12:15], v2, s[16:17]"))
            .count(),
        1,
        "{loads:?}"
    );
    let stores: Vec<&str> = t
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("global_store"))
        .collect();
    assert_eq!(
        stores,
        [
            "global_store_b16 v3, v43, s[32:33]",
            "global_store_b64 v4, v[69:70], s[32:33]"
        ]
    );
    let uses = |reg: &str| {
        t.lines()
            .map(str::trim)
            .filter(|l| {
                !l.starts_with('.')
                    && l.split_once(' ').is_some_and(|(_, ops)| {
                        ops.split(|c: char| !c.is_ascii_alphanumeric())
                            .any(|w| w == reg)
                    })
            })
            .count()
    };
    assert_eq!(
        uses("s19"),
        1,
        "P reaches the code only through the row guard"
    );
    assert_eq!(
        uses("s18"),
        2,
        "K reaches the code only through the group guard and the row pitch"
    );
    assert_eq!(count(&t, "s_endpgm"), 1);
    // Workgroup ids: row = ttmp9, group = ttmp7[15:0]; nothing else touches a ttmp register; no division by a constant.
    let ttmp: Vec<&str> = t
        .lines()
        .map(str::trim)
        .filter(|l| l.contains("ttmp"))
        .collect();
    assert_eq!(ttmp.len(), 2, "{ttmp:?}");
    assert!(
        ttmp.iter().any(|l| l.starts_with("s_mov_b32 s22, ttmp9"))
            && ttmp
                .iter()
                .any(|l| l.starts_with("s_and_b32 s23, ttmp7, 0xffff")),
        "{ttmp:?}"
    );
    assert!(
        !t.contains("ttmp10") && !t.contains("0xcccccccd"),
        "no ttmp10, no multiply-high division"
    );
}

/// The prologue's address arithmetic as emitted: guards, pitch, group offset, record index and the two 64-bit
/// multiply-adds (`h` by the runtime pitch in s44, `sorted` / `xq` by the literal 4 / 72 through s42). The `xq` record index
/// lives in s44 (never in the clobbered low-product temp s24), so its high product reads an intact multiplicand.
#[test]
fn prologue_address_code_is_the_modelled_one() {
    let t = moe_rotate_i4_pm::emit(spec()).unwrap().s_text;
    let lines: Vec<&str> = t.lines().map(str::trim).collect();
    let at = |needle: &str| {
        lines
            .iter()
            .position(|l| l.starts_with(needle))
            .unwrap_or_else(|| panic!("missing {needle}"))
    };
    let order = [
        "s_mov_b32 s22, ttmp9",
        "s_and_b32 s23, ttmp7, 0xffff",
        "s_cmp_ge_u32 s22, s19",
        "s_cbranch_scc1",
        "s_lshr_b32 s24, s18, 7",
        "s_cmp_ge_u32 s23, s24",
        "s_cbranch_scc1",
        "s_cmp_lt_i32 s28, 0",
        "s_cbranch_scc1",
        "s_lshl_b32 s44, s18, 1",
        "s_mul_i32 s24, s22, s44",
        "s_mul_hi_u32 s25, s22, s44",
        "s_lshl_b32 s24, s23, 8",
        "s_mul_i32 s44, s23, s43",
        "s_add_co_i32 s44, s44, s28",
        "s_mul_i32 s24, s44, s42",
        "s_mul_hi_u32 s25, s44, s42",
    ];
    let mut prev = 0;
    for needle in order {
        let i = lines[prev..]
            .iter()
            .position(|l| l.starts_with(needle))
            .map(|i| i + prev)
            .unwrap_or_else(|| panic!("missing or out of order: {needle}"));
        prev = i + 1;
    }
    let _ = at("s_load_b32 s43, s[0:1], 0x30");
    // The vector body starts only after both pointers exist.
    assert!(at("v_lshlrev_b32_e32 v1, 3, v0") > at("s_mul_hi_u32 s25, s44, s42"));
}

/// The numerics are `mq_i4_a4`'s: the same census (the quantizer clamp stays `v_maxmin_num_f32`), and every line
/// from the first per-lane vector instruction through `s_endpgm` is identical to the old K640 entry's, so the
/// rotation, quantizer and stores of the two families are one instruction sequence.
#[test]
fn numerics_are_the_old_family_instruction_for_instruction() {
    let new = moe_rotate_i4_pm::emit(spec()).unwrap().s_text;
    let old = mq_i4_a4::emit(mq_i4_a4::Spec { arch: ARCH })
        .unwrap()
        .s_text;
    for m in [
        "v_div_scale_f32",
        "v_div_fmas_f32",
        "v_div_fixup_f32",
        "v_rcp_f32_e32",
        "v_mov_b32_dpp",
        "v_permlanex16_b32",
        "v_fma_f32",
        "v_fmac_f32_e32",
        "v_rndne_f32_e32",
        "v_maxmin_num_f32",
        "v_max_num_f32_e64",
        "v_max_num_f32_e32",
        "v_cvt_i32_f32_e32",
        "v_cndmask_b32_e64",
        "v_cmp_eq_f32_e64",
        "v_cmp_gt_f32_e64",
        "v_cmpx_eq_u32_e32",
        "global_load_b64",
        "global_load_b128",
        "global_store_b16",
        "global_store_b64",
    ] {
        assert_eq!(count(&new, m), count(&old, m), "{m}");
    }
    assert_eq!(count(&new, "v_maxmin_num_f32"), 8 + 4);
    assert_eq!(
        (
            count(&new, "v_max_num_f32_e64"),
            count(&new, "v_max_num_f32_e32")
        ),
        (2, 6)
    );
    assert_eq!(count(&new, "v_xor_b32_e32"), 0);
    let body = |t: &str| -> Vec<String> {
        let lines: Vec<&str> = t.lines().map(str::trim).collect();
        let start = lines
            .iter()
            .position(|l| l.starts_with("v_lshlrev_b32_e32 v1, 3, v0"))
            .unwrap();
        let end = lines
            .iter()
            .position(|l| l.starts_with("s_endpgm"))
            .unwrap();
        lines[start..=end].iter().map(|l| (*l).to_owned()).collect()
    };
    let (bn, bo) = (body(&new), body(&old));
    assert!(bn.len() > 500, "{}", bn.len());
    assert_eq!(bn, bo, "everything after the prologue is the old entry's");
    // Frozen constants: scale, candidate 0 multiplier, 7.0, -8.0, 1e30, the permlane selector.
    for needle in [
        "v_mul_f32_e32 v16, 0x3db504f3, v16",
        "0x3fdb6db7",
        "s_mov_b32 s34, 0x40e00000",
        "s_mov_b32 s35, 0xc1000000",
        "0x7149f2ca",
        "s_mov_b32 s41, 0x76543210",
    ] {
        assert!(new.contains(needle), "missing {needle}");
    }
}

#[test]
fn constants_match_the_profile_and_the_frozen_producer() {
    assert_eq!(moe_rotate_i4_pm::GROUP, 128);
    assert_eq!(moe_rotate_i4_pm::RECORD_BYTES, 72);
    assert_eq!(moe_rotate_i4_pm::SIGN_SEEDS, (43, 1043));
    assert_eq!(moe_rotate_i4_pm::VGPRS, 72);
    assert_eq!(mq_i4_a4::ROTATION_SCALE_BITS, 0.0883883476f32.to_bits());
}

// ---------------------------------------------------------------------------------------------------------------
// CPU arithmetic / address proofs (NOT interpreter execution of the emitted ISA; see the module doc).

/// One (row, group) workgroup of the prologue: `None` = it returns before any store. Returns
/// `(h byte offset, xq record byte offset)`: `h + row*2K + g*256`, `xq + 72*(g*L + slot)`, the 32-bit
/// `s_mul_i32` / `s_add_co_i32` index and the exact 64-bit `s_mul_i32` + `s_mul_hi_u32` multiply-add.
fn workgroup(k: u32, p: u32, l: u32, sorted: &[i32], row: u32, grp: u32) -> Option<(u64, u64)> {
    if row >= p || grp >= k >> 7 {
        return None;
    }
    let slot = sorted[row as usize];
    if slot < 0 {
        return None;
    }
    let h = u64::from(row) * u64::from(k << 1) + u64::from(grp << 8);
    let idx = grp.wrapping_mul(l).wrapping_add(slot as u32);
    Some((h, u64::from(idx) * 72))
}

struct Case {
    name: String,
    k: u32,
    p: u32,
    l: u32,
    sorted: Vec<i32>,
}

/// `p` rows; `pad_every` > 0 makes rows `row % pad_every == pad_every / 2` (and the last row) pad rows; `empty` slots of
/// `l = p + empty` are never named.
fn case(name: &str, k: u32, p: u32, pad_every: u32, empty: u32, seed: u64) -> Case {
    let l = p + empty;
    let mut slots: Vec<i32> = (0..l as i32).collect();
    let mut rng = Rng(seed | 1);
    for i in (1..slots.len()).rev() {
        slots.swap(i, rng.below(i + 1));
    }
    let mut next = 0;
    let sorted = (0..p)
        .map(|row| {
            if pad_every > 0 && (row % pad_every == pad_every / 2 || row == p - 1) {
                return -1;
            }
            next += 1;
            slots[next - 1]
        })
        .collect();
    Case {
        name: name.into(),
        k,
        p,
        l,
        sorted,
    }
}

fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for (rows, seed) in [(512u32, 11u64), (4096, 12), (8192, 13)] {
        let p = rows * 8; // top-8: every token row gathers into eight grouped rows
        out.push(case(&format!("rows{rows}_top8_dense"), 512, p, 0, 0, seed));
        out.push(case(
            &format!("rows{rows}_top8_pad_empty"),
            512,
            p,
            9,
            37,
            seed + 100,
        ));
    }
    out.push(case("k640_pad_empty", 640, 3000, 7, 11, 21));
    out.push(case("k128_one_group", 128, 1000, 5, 3, 22));
    out
}

/// Every workgroup of every case, K = 512 and rows 512 / 4096 / 8192 x top-8 (P up to 65,536, past the 65,535 grid-y
/// limit, which only `K/128 = 4` groups use): pad rows and empty slots write nothing, each valid (row, group) owns
/// exactly one in-range record and its own 256 `h` bytes, no two workgroups share a record, and the out-of-grid
/// workgroups (row >= P, group >= K/128) return.
#[test]
fn address_model_over_every_workgroup() {
    for c in cases() {
        assert!(
            c.k % 128 == 0
                && (c.k >> 7) <= 65_535
                && u64::from(c.k >> 7) * u64::from(c.l) <= 1 << 32,
            "{}: host check",
            c.name
        );
        let groups = c.k >> 7;
        let total = u64::from(groups) * u64::from(c.l);
        let mut written = vec![false; total as usize];
        let mut h_cover = vec![0u8; c.p as usize * (c.k as usize) * 2 / 256];
        let (mut valid, mut pads) = (0u64, 0u64);
        for row in 0..c.p {
            let pad = c.sorted[row as usize] < 0;
            pads += u64::from(pad);
            for g in 0..groups {
                match workgroup(c.k, c.p, c.l, &c.sorted, row, g) {
                    None => assert!(pad, "{}: valid row {row} returned", c.name),
                    Some((h, xq)) => {
                        assert!(!pad, "{}: pad row {row} stored", c.name);
                        assert_eq!(xq % 72, 0);
                        let rec = (xq / 72) as usize;
                        assert!(rec < written.len(), "{}: record {rec} out of range", c.name);
                        assert!(
                            !std::mem::replace(&mut written[rec], true),
                            "{}: record {rec} written twice",
                            c.name
                        );
                        assert_eq!(
                            rec as u64,
                            u64::from(g) * u64::from(c.l) + c.sorted[row as usize] as u64
                        );
                        assert_eq!(h % 256, 0);
                        h_cover[(h / 256) as usize] += 1;
                        valid += 1;
                    }
                }
            }
            // Out-of-grid group of this row.
            assert!(workgroup(c.k, c.p, c.l, &c.sorted, row, groups).is_none());
        }
        assert!(
            workgroup(c.k, c.p, c.l, &c.sorted, c.p, 0).is_none(),
            "{}: row == P",
            c.name
        );
        assert_eq!(
            valid + pads * u64::from(groups),
            u64::from(c.p) * u64::from(groups),
            "{}",
            c.name
        );
        // Exactly the valid rows' groups are written; every empty slot's records (all groups) are untouched.
        assert_eq!(
            written.iter().filter(|w| **w).count() as u64,
            valid,
            "{}",
            c.name
        );
        let named: std::collections::HashSet<i32> =
            c.sorted.iter().copied().filter(|s| *s >= 0).collect();
        for slot in 0..c.l as i32 {
            for g in 0..groups as usize {
                assert_eq!(
                    written[g * c.l as usize + slot as usize],
                    named.contains(&slot),
                    "{} slot {slot} group {g}",
                    c.name
                );
            }
        }
        // `h` is read once per valid (row, group) and never for a pad row.
        for row in 0..c.p as usize {
            for g in 0..groups as usize {
                let n = h_cover[row * groups as usize + g];
                assert_eq!(n, u8::from(c.sorted[row] >= 0), "{} h ({row}, {g})", c.name);
            }
        }
    }
}

/// The reference record bytes, applied through the address model into a 0xA5-filled sidecar for a stride of rows of
/// every case (zero and random rows, pad rows with NaN garbage that must never be read): the written records are
/// exactly the reference ones, and the sidecar is otherwise unchanged (pad rows, empty slots, other rows).
#[test]
fn reference_records_land_only_at_their_slots_and_empty_slots_stay_untouched() {
    let (s1, s2) = (signs(43), signs(1043));
    for c in cases()
        .into_iter()
        .filter(|c| c.k == 512 || c.name.starts_with("k128"))
    {
        let groups = (c.k >> 7) as usize;
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ u64::from(c.p));
        let stride = (c.p as usize / 48).max(1);
        let mut h = vec![0u16; c.p as usize * c.k as usize];
        let mut sampled = Vec::new();
        for row in (0..c.p as usize).step_by(stride) {
            sampled.push(row);
            let r = &mut h[row * c.k as usize..(row + 1) * c.k as usize];
            match row / stride % 4 {
                0 => {} // all-zero row: the unit block
                1 => {
                    for x in r.iter_mut() {
                        *x = bf16_rne(
                            (rng.below(4001) as f32 - 2000.0) / 1000.0
                                * f32::from_bits(((127 + rng.below(17) - 8) as u32) << 23),
                        )
                    }
                }
                2 => {
                    for x in r.iter_mut() {
                        *x = if rng.below(8) == 0 {
                            bf16_rne((rng.below(2001) as f32 - 1000.0) / 100.0)
                        } else {
                            0
                        }
                    }
                }
                _ => {
                    for x in r.iter_mut() {
                        *x = 0x8000
                    }
                } // negative zero
            }
            if c.sorted[row] < 0 {
                for x in r.iter_mut() {
                    *x = 0x7fff
                }
            } // NaN garbage behind a pad slot
        }
        let mut xq = vec![0xa5u8; groups * c.l as usize * 72];
        let mut stored = 0usize;
        for &row in &sampled {
            for g in 0..groups {
                if let Some((hoff, off)) = workgroup(c.k, c.p, c.l, &c.sorted, row as u32, g as u32)
                {
                    let at = hoff as usize / 2;
                    let rec = record(&h[at..at + 128], &s1, &s2);
                    xq[off as usize..off as usize + 72].copy_from_slice(&rec);
                    stored += 1;
                }
            }
        }
        let valid_sampled = sampled.iter().filter(|&&r| c.sorted[r] >= 0).count();
        assert_eq!(stored, valid_sampled * groups, "{}", c.name);
        // Expected sidecar: sentinel everywhere except the sampled valid rows' records.
        let mut want = vec![0xa5u8; xq.len()];
        for &row in &sampled {
            let slot = c.sorted[row];
            if slot < 0 {
                continue;
            }
            for g in 0..groups {
                let rec = record(
                    &h[row * c.k as usize + g * 128..row * c.k as usize + g * 128 + 128],
                    &s1,
                    &s2,
                );
                let off = 72 * (g * c.l as usize + slot as usize);
                want[off..off + 72].copy_from_slice(&rec);
            }
        }
        assert!(
            xq == want,
            "{}: sidecar differs from the reference placement",
            c.name
        );
        let touched = xq
            .chunks(72)
            .filter(|r| r.iter().any(|&b| b != 0xa5))
            .count();
        assert!(
            touched <= stored,
            "{}: {touched} touched records > {stored} stores",
            c.name
        );
    }
}

/// Properties of the reference arithmetic the GPU oracle compares against: zero groups are the unit block, codes are
/// 4-bit signed, `s` is their exact sum, `d` is one of the two candidate scales or 1, the rotation is orthogonal
/// (norm preserved to f32 accuracy) and a group's record is independent of K and of the group's position.
#[test]
fn reference_arithmetic_properties() {
    let (s1, s2) = (signs(43), signs(1043));
    assert_ne!(s1, s2);
    let unit = record(&[0u16; 128], &s1, &s2);
    assert_eq!(&unit[0..4], &1.0f32.to_le_bytes());
    assert!(
        unit[4..].iter().all(|&b| b == 0),
        "zero group: d = 1, s = 0, qs = 0"
    );
    assert_eq!(
        record(&[0x8000u16; 128], &s1, &s2),
        unit,
        "negative zero is the unit block too"
    );
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    for case in 0..64 {
        let h: Vec<u16> = (0..128)
            .map(|_| {
                bf16_rne(
                    (rng.below(20001) as f32 - 10000.0) / 1000.0
                        * f32::from_bits(((127 + (case % 9)) as u32) << 23),
                )
            })
            .collect();
        let x = rotate(&h, &s1, &s2);
        let (n_in, n_out): (f64, f64) = (
            h.iter()
                .map(|&b| f64::from(f32::from_bits(u32::from(b) << 16)).powi(2))
                .sum(),
            x.iter().map(|&v| f64::from(v).powi(2)).sum(),
        );
        assert!(
            ((n_in - n_out) / n_in.max(1e-30)).abs() < 1e-5,
            "case {case}: norm {n_in} -> {n_out}"
        );
        let rec = quantize(&x);
        let d = f32::from_le_bytes(rec[0..4].try_into().unwrap());
        let s = i32::from_le_bytes(rec[4..8].try_into().unwrap());
        let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
        let base = (amax / 7.0f32) * 0.5f32;
        let cands = [
            base * f32::from_bits(mq_i4_a4::CAND0_MULT_BITS),
            base * 2.0,
            1.0,
        ];
        assert!(
            cands.contains(&d) && d.is_finite() && d > 0.0,
            "case {case}: d {d}"
        );
        let codes: Vec<i32> = (0..128)
            .map(|i| (i32::from((rec[8 + i / 2] >> (4 * (i % 2))) & 15) << 28) >> 28)
            .collect();
        assert_eq!(codes.iter().sum::<i32>(), s, "case {case}: s");
        // Same group data at any (row, K): identical record.
        let mut wide = vec![0x1234u16; 640];
        wide[256..384].copy_from_slice(&h);
        assert_eq!(
            record(&wide[256..384], &s1, &s2),
            rec,
            "case {case}: K independence"
        );
    }
}

fn bundle_path() -> String {
    format!(
        "{}/../../kernels/{MODULE}.hxaco",
        env!("CARGO_MANIFEST_DIR")
    )
}

/// The committed `kernels/moe_rotate_i4_pm_gfx1201.hxaco` must be exactly what the builder emits today (native
/// writer, no ROCm needed; `peacemaker-isa emit --kernel moe_rotate_i4_pm --arch gfx1201 ... --bundle`).
#[test]
fn committed_bundle_equals_fresh_emission() {
    let path = bundle_path();
    let committed = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e} (emit it: peacemaker-isa emit --kernel moe_rotate_i4_pm --arch gfx1201 ... --bundle)"));
    let (_, text, _) = moe_rotate_i4_pm::emit_module(ARCH).unwrap();
    let elf = peacemaker_isa::native::assemble(&text, ARCH).unwrap();
    let fresh =
        peacemaker_isa::native::bundle(&elf, ARCH, peacemaker_isa::native::DEFAULT_HOST_TARGET);
    assert!(
        committed == fresh,
        "{MODULE}: fresh emission differs from the committed bundle {path}"
    );
}

/// The same `.text` through the ROCm oracle's `llvm-mc` + `ld.lld`.
#[test]
fn committed_bundle_text_equals_oracle_link() {
    let _ = require_rocm_tool!("llvm-mc");
    let _ = require_rocm_tool!("ld.lld");
    let committed = std::fs::read(bundle_path()).unwrap();
    let text = moe_rotate_i4_pm::emit_module(ARCH).unwrap().1;
    assert!(
        text_section(&link(&text, ARCH.name(), MODULE)) == text_section(&committed),
        "{MODULE}: oracle-linked emission differs from the committed bundle"
    );
}

fn contract_path() -> String {
    format!(
        "{}/kernels/moe_rotate_i4_pm.{}.contract.json",
        env!("CARGO_MANIFEST_DIR"),
        ARCH.name()
    )
}

/// The committed shape contract is exactly what `contract()` derives from fresh emission
/// (`peacemaker-isa emit --kernel moe_rotate_i4_pm ... --co ... --shape crates/peacemaker-isa/kernels`).
#[test]
fn committed_contract_equals_fresh_contract() {
    let path = contract_path();
    let committed: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}")))
            .unwrap();
    let fresh = moe_rotate_i4_pm::contract(spec(), &moe_rotate_i4_pm::emit(spec()).unwrap());
    assert_eq!(committed, fresh, "{path}");
    assert_eq!(committed["symbol"], SYMBOL);
    // It is the old entry's contract except for the symbol and the two extra scalar loads / branch of the prologue.
    let old = mq_i4_a4::contract(
        mq_i4_a4::Spec { arch: ARCH },
        &mq_i4_a4::emit(mq_i4_a4::Spec { arch: ARCH }).unwrap(),
    );
    for key in [
        "forbidden",
        "vgpr_max",
        "sgpr_max",
        "require_wave32",
        "require_zero_spills",
        "require_zero_private",
        "launch_dynamic_lds_bytes",
        "group_segment_fixed_bytes",
    ] {
        assert_eq!(committed[key], old[key], "{key}");
    }
    for (name, n) in committed["counts"].as_object().unwrap() {
        let want = match name.as_str() {
            "s_cbranch_scc1" => 3,
            "s_load_b32" => 2,
            _ => old["counts"][name].as_u64().unwrap(),
        };
        assert_eq!(n.as_u64().unwrap(), want, "{name}");
    }
}

fn link(text: &str, arch: &str, stem: &str) -> Vec<u8> {
    use std::process::Command;
    let dir = std::env::temp_dir().join(format!("peacemaker-isa-moerot-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (s, o, co) = (
        dir.join(format!("{stem}.s")),
        dir.join(format!("{stem}.o")),
        dir.join(format!("{stem}.co")),
    );
    std::fs::write(&s, text).unwrap();
    let llvm_mc = rocm::rocm_tool("llvm-mc").expect("ROCm llvm-mc");
    let linker = rocm::rocm_tool("ld.lld").expect("ROCm ld.lld");
    assert!(Command::new(llvm_mc)
        .args([
            "-triple=amdgcn-amd-amdhsa",
            &format!("-mcpu={arch}"),
            "-filetype=obj"
        ])
        .arg(&s)
        .arg("-o")
        .arg(&o)
        .status()
        .unwrap()
        .success());
    assert!(Command::new(linker)
        .arg("-shared")
        .arg(&o)
        .arg("-o")
        .arg(&co)
        .status()
        .unwrap()
        .success());
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
    (0..shnum)
        .find_map(|i| {
            let name = &e[names + u32_(sh(i))..];
            name.starts_with(b".text\0")
                .then(|| e[u64_(sh(i) + 24)..u64_(sh(i) + 24) + u64_(sh(i) + 32)].to_vec())
        })
        .unwrap()
}

#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use peacemaker_isa::{
        ledger_replay, pm_check,
        toolchain::{build, certify, IsaShapeContract, Toolchain},
    };

    /// The export's waits replay independently (the retained wait ledger and the independent source-text replay
    /// agree), the contract parses as an `IsaShapeContract`, no LDS is touched, and M7 lifts the natively linked
    /// object byte-exactly with no outstanding obligations (no ROCm tools).
    #[test]
    fn export_passes_wait_replay_and_m7() {
        let (emitted, text, _) = moe_rotate_i4_pm::emit_module(ARCH).unwrap();
        for e in &emitted {
            ledger_replay::replay_waits(&e.s_text, ARCH).unwrap();
            serde_json::from_value::<IsaShapeContract>(moe_rotate_i4_pm::contract(spec(), e))
                .unwrap();
        }
        ledger_replay::replay_waits(&text, ARCH).unwrap();
        let dir =
            std::env::temp_dir().join(format!("peacemaker-isa-moerot-m7-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let co = dir.join("module.co");
        std::fs::write(&co, peacemaker_isa::native::assemble(&text, ARCH).unwrap()).unwrap();
        let m7 = pm_check::m7(&co, ARCH.name(), SYMBOL).unwrap_or_else(|e| panic!("{SYMBOL}: {e}"));
        assert_eq!(
            (m7["lift"].as_str(), &m7["obligations"]),
            (Some("byte-exact"), &serde_json::json!({})),
            "{SYMBOL}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The symbol certifies against the ROCm oracle build: parse-back, independent wait replay, its committed shape
    /// contract, and M7 of the linked ELF, obligation-free.
    #[test]
    fn module_passes_certification() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        let (emitted, text, _) = moe_rotate_i4_pm::emit_module(ARCH).unwrap();
        for e in &emitted {
            ledger_replay::replay_waits(&e.s_text, ARCH).unwrap();
        }
        let dir =
            std::env::temp_dir().join(format!("peacemaker-isa-moerot-cert-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let s = dir.join("module.s");
        std::fs::write(&s, &text).unwrap();
        let toolchain = Toolchain::oracle();
        let build = build(&toolchain, &s, &dir.join("module.hsaco"), ARCH.name()).unwrap();
        let m7 = pm_check::m7(&build.elf, ARCH.name(), SYMBOL)
            .unwrap_or_else(|e| panic!("{SYMBOL}: {e}"));
        assert_eq!(
            (m7["lift"].as_str(), &m7["obligations"]),
            (Some("byte-exact"), &serde_json::json!({})),
            "{SYMBOL}"
        );
        let path = contract_path();
        let contract =
            serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}")))
                .unwrap();
        certify(
            &toolchain,
            &build,
            &s,
            ARCH.name(),
            &dir.join("manifest.json"),
            Some(&contract),
            "test",
            "test",
        )
        .unwrap_or_else(|e| panic!("{SYMBOL}: {e}"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
