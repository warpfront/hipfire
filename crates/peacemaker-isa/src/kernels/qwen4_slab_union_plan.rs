// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Integer PM twin of the slab union planner, with one independent eight-row
//! descriptor per workgroup. Expert order is first occurrence, NOT numeric ID.
//! Grid `[ceil(rows/8), 1, 1]`, block128, wave32, LDS1280. Each tile writes
//! exactly 482 words, including deterministic tails. Slots stay absolute in
//! the full window; offsets are relative to the descriptor's bucket.
//! Both eighty-wide scans emit through the builder's hazard fixed-point
//! loop, so back-edge-carried guards hold on every iteration.
//!
//! Kernargs: topk@0, codec@8, out@16, rows@24, uniform_bucket@28,
//! n_experts@32 (logical36, aligned segment40). Uniform bucket -1 reads the
//! codec table; 0 selects W3, 1 selects MQ4 without reading that table.
//! Invalid IDs never read codec. Invalid domains and dead workgroups write
//! nothing. There is no floating-point arithmetic or atomic operation.

use super::common::{add64, lit, mem, op, s, smem, sop, sr, v};
use crate::{
    insn::{Instruction, MemoryClass},
    lds::Transition,
    plan::Access,
    reg::{Live, RegRef},
    Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan,
};

type R = Result<(), String>;
const USER_SGPRS: u8 = 2;
/// The workgroup id X is the only system SGPR.
const SYSTEM_SGPRS: u8 = 1;
/// Kernarg block s[base:base+7]: topk, codec, out pointers, rows, uniform bucket.
#[derive(Clone, Copy)]
struct Args {
    top: u8,
    codec: u8,
    out: u8,
    rows: u8,
    uniform: u8,
}
/// gfx1100 `FeatureUserSGPRInit16Bug` (user + system >= 16) declares 15 user SGPRs, so the workgroup
/// id is s15 and the (8-aligned) kernarg block moves from s[8:15] to the unused s[32:39]; every other
/// arch keeps s[8:15].
const fn args(arch: Arch) -> Args {
    let base = if matches!(arch, Arch::Gfx1100) { 32 } else { 8 };
    Args {
        top: base,
        codec: base + 2,
        out: base + 4,
        rows: base + 6,
        uniform: base + 7,
    }
}
/// First SGPR of the workgroup id X (`s2`, gfx1100 `s15`; gfx12 reads ttmp9).
const fn workgroup_x(arch: Arch) -> u8 {
    arch.declared_user_sgprs(USER_SGPRS, SYSTEM_SGPRS)
}
const EXPERTS: u8 = 16;
const TILE: u8 = 17;
const BASE: u8 = 18;
const LIMIT: u8 = 19;
const TEMP: u8 = 20;
const J: u8 = 21;
const MASK: u8 = 22;
const MASK2: u8 = 24;
const SAVE: u8 = 26;
// v0 is the hardware workitem ID x. No lane/wave reconstruction is needed.
const TID: u8 = 0;
const ID: u8 = 1;
const BUCKET: u8 = 2;
const ABS: u8 = 3;
const ADDR: u8 = 4;
const DATA: u8 = 5;
const RANK: u8 = 6;
const TOTAL: u8 = 7;
const FIRST: u8 = 8;
const HEAD: u8 = 9;
const JID: u8 = 10;
const JB: u8 = 11;
const JHEAD: u8 = 12;
const JT: u8 = 13;
const EIDX: u8 = 14;
const START: u8 = 15;
const UB0: u8 = 16;
const UB1: u8 = 17;
const NB0: u8 = 18;
const NB1: u8 = 19;
const TMP: u8 = 20;
const MATCH: u8 = 21;
const LDS_ADDR: u8 = 22;

pub fn symbol(arch: Arch) -> String {
    format!("qwen4_slab_union_plan_t8_pm_{}", arch.name())
}

/// Emit the architecture-specific integer planner, not an image retag.
pub fn emit(arch: Arch) -> Result<Emitted, String> {
    if !matches!(arch, Arch::Gfx1151 | Arch::Gfx1201 | Arch::Gfx1100) {
        return Err(format!(
            "qwen4_slab_union_plan: unsupported {}",
            arch.name()
        ));
    }
    let Args {
        top,
        codec,
        out,
        rows,
        uniform,
    } = args(arch);
    let wg_x = workgroup_x(arch);
    let spec = KernelSpec {
        kernel_id: "qwen4_slab_union_plan".into(),
        variant: "t8".into(),
        arch,
        symbol: symbol(arch),
        kernargs: KernargLayout::new(40)
            .pointer_access("topk", 0, Access::ReadOnly)
            .pointer_access("codec", 8, Access::ReadOnly)
            .pointer_access("out", 16, Access::WriteOnly)
            .hidden("rows", 24, 4, "by_value")
            .hidden("uniform_bucket", 28, 4, "by_value")
            .hidden("n_experts", 32, 4, "by_value"),
        user_sgpr_count: USER_SGPRS,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: 128,
        group_segment_fixed_size: 1280,
        wave32: true,
        cu_mode: false,
    };
    let mut p = RegPlan::new(32, if arch == Arch::Gfx1100 { 40 } else { 32 })?;
    p.s::<2>("kernarg", 0, Live::Whole)?;
    p.s::<1>("workgroup_x", wg_x, Live::Whole)?;
    p.s::<8>("args_0", top, Live::Whole)?;
    p.s::<1>("n_experts", EXPERTS, Live::Whole)?;
    for (name, r) in [
        ("tile", TILE),
        ("absolute_base", BASE),
        ("limit", LIMIT),
        ("temporary", TEMP),
        ("scan_index", J),
        ("saved_exec", SAVE),
    ] {
        p.s::<1>(name, r, Live::Whole)?;
    }
    p.s::<2>("mask", MASK, Live::Whole)?;
    p.s::<2>("mask2", MASK2, Live::Whole)?;
    p.v::<8>("slot_state", 0, Live::Whole)?;
    p.v::<8>("slot_state", 8, Live::Whole)?;
    p.v::<8>("slot_state", 16, Live::Whole)?;
    let mut b = Builder::new(spec, p);
    let ids = b.lds_slot("ids", 0, 320)?;
    let buckets = b.lds_slot("buckets", 320, 320)?;
    let heads = b.lds_slot("heads", 640, 320)?;
    let totals = b.lds_slot("totals", 960, 320)?;
    smem(&mut b, top, 8, 0, 0)?;
    smem(&mut b, EXPERTS, 1, 0, 32)?;
    if arch.gfx12() {
        sop(&mut b, format!("s_mov_b32 s{TILE}, ttmp9"), &[TILE], &[])?;
    } else {
        sop(
            &mut b,
            format!("s_mov_b32 s{TILE}, s{wg_x}"),
            &[TILE],
            &[wg_x],
        )?;
    }
    b.wait_all()?;
    // All exits here are workgroup-uniform and precede every barrier.
    for (cmp, uses) in [
        (format!("s_cmp_lt_i32 s{rows}, 1"), vec![rows]),
        (format!("s_cmp_gt_i32 s{rows}, 64"), vec![rows]),
        (format!("s_cmp_lt_i32 s{uniform}, -1"), vec![uniform]),
        (format!("s_cmp_gt_i32 s{uniform}, 1"), vec![uniform]),
        (format!("s_cmp_lt_i32 s{EXPERTS}, 0"), vec![EXPERTS]),
    ] {
        sop(&mut b, cmp, &[], &uses)?;
        op(&mut b, "s_cbranch_scc1 .Lunion_end", &[], &[])?;
    }
    sop(
        &mut b,
        format!("s_lshl_b32 s{TEMP}, s{TILE}, 3"),
        &[TEMP],
        &[TILE],
    )?;
    // Compare tile ID before multiplication so even a corrupt grid cannot wrap.
    sop(&mut b, format!("s_cmp_ge_u32 s{TILE}, 8"), &[], &[TILE])?;
    op(&mut b, "s_cbranch_scc1 .Lunion_end", &[], &[])?;
    sop(
        &mut b,
        format!("s_cmp_ge_u32 s{TEMP}, s{rows}"),
        &[],
        &[TEMP, rows],
    )?;
    op(&mut b, "s_cbranch_scc1 .Lunion_end", &[], &[])?;
    sop(
        &mut b,
        format!("s_mul_i32 s{BASE}, s{TILE}, {}", lit(80)),
        &[BASE],
        &[TILE],
    )?;
    sop(
        &mut b,
        format!("s_mul_i32 s{LIMIT}, s{rows}, 10"),
        &[LIMIT],
        &[rows],
    )?;
    sop(
        &mut b,
        format!("s_mul_i32 s{TEMP}, s{TILE}, {}", lit(1928)),
        &[TEMP],
        &[TILE],
    )?;
    add64(&mut b, out, out, TEMP)?;
    vi(
        &mut b,
        ABS,
        format!("v_add_nc_u32_e32 v{ABS}, s{BASE}, v{TID}"),
        &[s(BASE), v(TID)],
    )?;
    mov(&mut b, ID, "-1", &[])?;
    mov(&mut b, BUCKET, "-1", &[])?;
    cmp(
        &mut b,
        MASK,
        "gt_u32",
        &lit(80),
        &format!("v{TID}"),
        &[v(TID)],
    )?;
    cmp(
        &mut b,
        MASK2,
        "gt_u32",
        &format!("s{LIMIT}"),
        &format!("v{ABS}"),
        &[s(LIMIT), v(ABS)],
    )?;
    and_mask(&mut b)?;
    masked(&mut b, |b| {
        vi(
            b,
            ADDR,
            format!("v_lshlrev_b32_e32 v{ADDR}, 2, v{ABS}"),
            &[v(ABS)],
        )?;
        load(b, ID, ADDR, top, false)
    })?;
    b.wait_all()?;
    // Unsigned ID comparison rejects negative IDs as well as >=n_experts.
    cmp(
        &mut b,
        MASK,
        "gt_u32",
        &format!("s{EXPERTS}"),
        &format!("v{ID}"),
        &[s(EXPERTS), v(ID)],
    )?;
    masked(&mut b, |b| {
        sop(b, format!("s_cmp_ge_i32 s{uniform}, 0"), &[], &[uniform])?;
        op(b, "s_cbranch_scc1 .Lunion_uniform", &[], &[])?;
        load(b, DATA, ID, codec, true)?;
        b.wait_all()?;
        cmp(b, MASK2, "eq_u32", "1", &format!("v{DATA}"), &[v(DATA)])?;
        choose(b, BUCKET, "1", "0", MASK2, &[])?;
        op(b, "s_branch .Lunion_bucket_done", &[], &[])?;
        b.label(".Lunion_uniform")?;
        mov(b, BUCKET, &format!("s{uniform}"), &[s(uniform)])?;
        b.label(".Lunion_bucket_done")
    })?;
    // Exactly the first80 threads initialize LDS, including inactive tail slots.
    cmp(
        &mut b,
        MASK,
        "gt_u32",
        &lit(80),
        &format!("v{TID}"),
        &[v(TID)],
    )?;
    vi(
        &mut b,
        LDS_ADDR,
        format!("v_lshlrev_b32_e32 v{LDS_ADDR}, 2, v{TID}"),
        &[v(TID)],
    )?;
    masked(&mut b, |b| {
        ds_store(b, ids, LDS_ADDR, ID, 0)?;
        ds_store(b, buckets, LDS_ADDR, BUCKET, 320)
    })?;
    b.barrier(&[Transition::Ready(ids), Transition::Ready(buckets)])?;
    mov(&mut b, RANK, "0", &[])?;
    mov(&mut b, TOTAL, "0", &[])?;
    mov(&mut b, FIRST, &lit(80), &[])?;
    // Fixed-point loop: the builder joins the entry and back-edge hazard
    // states and re-emits this scan until no new guard is owed.
    sop(&mut b, format!("s_mov_b32 s{J}, 0"), &[J], &[])?;
    b.loop_(".Lunion_matches", |b| {
        vi(
            b,
            ADDR,
            format!("v_lshlrev_b32_e64 v{ADDR}, 2, s{J}"),
            &[s(J)],
        )?;
        ds_load(b, ids, JID, ADDR, 0)?;
        b.wait_all()?;
        cmp(
            b,
            MASK,
            "eq_u32",
            &format!("v{ID}"),
            &format!("v{JID}"),
            &[v(ID), v(JID)],
        )?;
        cmp(
            b,
            MASK2,
            "gt_u32",
            &format!("s{EXPERTS}"),
            &format!("v{ID}"),
            &[s(EXPERTS), v(ID)],
        )?;
        and_mask(b)?;
        choose(b, MATCH, "0", "1", MASK, &[])?;
        add(b, TOTAL, TOTAL, MATCH)?;
        vi(
            b,
            TMP,
            format!("v_min_u32_e64 v{TMP}, s{J}, v{FIRST}"),
            &[s(J), v(FIRST)],
        )?;
        choose(
            b,
            FIRST,
            &format!("v{FIRST}"),
            &format!("v{TMP}"),
            MASK,
            &[v(FIRST), v(TMP)],
        )?;
        cmp(
            b,
            MASK2,
            "lt_u32",
            &format!("s{J}"),
            &format!("v{TID}"),
            &[s(J), v(TID)],
        )?;
        and_mask(b)?;
        // Rank counts earlier matches; TMP still holds min(J, FIRST), not that predicate.
        choose(b, TMP, "0", "1", MASK, &[])?;
        add(b, RANK, RANK, TMP)?;
        scan_end(b, ".Lunion_matches")
    })?;
    cmp(
        &mut b,
        MASK,
        "eq_u32",
        &format!("v{TID}"),
        &format!("v{FIRST}"),
        &[v(TID), v(FIRST)],
    )?;
    choose(&mut b, HEAD, "0", "1", MASK, &[])?;
    cmp(
        &mut b,
        MASK,
        "gt_u32",
        &lit(80),
        &format!("v{TID}"),
        &[v(TID)],
    )?;
    masked(&mut b, |b| {
        ds_store(b, heads, LDS_ADDR, HEAD, 640)?;
        ds_store(b, totals, LDS_ADDR, TOTAL, 960)
    })?;
    b.barrier(&[Transition::Ready(heads), Transition::Ready(totals)])?;
    for r in [EIDX, START, UB0, UB1, NB0, NB1] {
        mov(&mut b, r, "0", &[])?;
    }
    // Same fixed-point loop contract as the match scan above.
    sop(&mut b, format!("s_mov_b32 s{J}, 0"), &[J], &[])?;
    b.loop_(".Lunion_heads", |b| {
        vi(
            b,
            ADDR,
            format!("v_lshlrev_b32_e64 v{ADDR}, 2, s{J}"),
            &[s(J)],
        )?;
        ds_load(b, buckets, JB, ADDR, 320)?;
        ds_load(b, heads, JHEAD, ADDR, 640)?;
        ds_load(b, totals, JT, ADDR, 960)?;
        b.wait_all()?;
        for (bucket, ub, nb) in [(0, UB0, NB0), (1, UB1, NB1)] {
            cmp(
                b,
                MASK,
                "eq_u32",
                &bucket.to_string(),
                &format!("v{JB}"),
                &[v(JB)],
            )?;
            cmp(b, MASK2, "eq_u32", "1", &format!("v{JHEAD}"), &[v(JHEAD)])?;
            and_mask(b)?;
            choose(b, TMP, "0", "1", MASK, &[])?;
            add(b, ub, ub, TMP)?;
            choose(b, TMP, "0", &format!("v{JT}"), MASK, &[v(JT)])?;
            add(b, nb, nb, TMP)?;
        }
        cmp(
            b,
            MASK,
            "eq_u32",
            &format!("v{BUCKET}"),
            &format!("v{JB}"),
            &[v(BUCKET), v(JB)],
        )?;
        cmp(b, MASK2, "eq_u32", "1", &format!("v{JHEAD}"), &[v(JHEAD)])?;
        and_mask(b)?;
        cmp(
            b,
            MASK2,
            "lt_u32",
            &format!("s{J}"),
            &format!("v{FIRST}"),
            &[s(J), v(FIRST)],
        )?;
        and_mask(b)?;
        choose(b, TMP, "0", "1", MASK, &[])?;
        add(b, EIDX, EIDX, TMP)?;
        choose(b, TMP, "0", &format!("v{JT}"), MASK, &[v(JT)])?;
        add(b, START, START, TMP)?;
        scan_end(b, ".Lunion_heads")
    })?;
    // Publish each bucket using the HIP planner's disjoint single-owner map.
    for (bucket, ub, nb) in [(0u32, UB0, NB0), (1u32, UB1, NB1)] {
        cmp(
            &mut b,
            MASK,
            "eq_u32",
            &bucket.to_string(),
            &format!("v{BUCKET}"),
            &[v(BUCKET)],
        )?;
        masked(&mut b, |b| {
            add(b, TMP, START, RANK)?;
            output(b, TMP, 322 + bucket * 80, ABS)?;
            cmp(b, MASK2, "eq_u32", "1", &format!("v{HEAD}"), &[v(HEAD)])?;
            // Nested masks use a separate save register, so preserve outer EXEC.
            op(
                b,
                format!("s_and_saveexec_b32 s{TEMP}, s{MASK2}"),
                &[s(TEMP)],
                &[sr(MASK2, 2)],
            )?;
            output(b, EIDX, bucket * 80, ID)?;
            output(b, EIDX, 160 + bucket * 81, START)?;
            op(b, format!("s_mov_b32 exec_lo, s{TEMP}"), &[], &[s(TEMP)])
        })?;
        // Tail writers own all unused IDs/offsets and slots. The final offset
        // word is always thread0, even when all80 experts are distinct.
        cmp(
            &mut b,
            MASK,
            "ge_u32",
            &format!("v{TID}"),
            &format!("v{ub}"),
            &[v(TID), v(ub)],
        )?;
        cmp(
            &mut b,
            MASK2,
            "gt_u32",
            &lit(80),
            &format!("v{TID}"),
            &[v(TID)],
        )?;
        and_mask(&mut b)?;
        mov(&mut b, DATA, "0", &[])?;
        masked(&mut b, |b| {
            output(b, TID, bucket * 80, DATA)?;
            output(b, TID, 160 + bucket * 81, nb)
        })?;
        cmp(
            &mut b,
            MASK,
            "ge_u32",
            &format!("v{TID}"),
            &format!("v{nb}"),
            &[v(TID), v(nb)],
        )?;
        cmp(
            &mut b,
            MASK2,
            "gt_u32",
            &lit(80),
            &format!("v{TID}"),
            &[v(TID)],
        )?;
        and_mask(&mut b)?;
        masked(&mut b, |b| output(b, TID, 322 + bucket * 80, DATA))?;
        cmp(&mut b, MASK, "eq_u32", "0", &format!("v{TID}"), &[v(TID)])?;
        masked(&mut b, |b| output(b, TID, 240 + bucket * 81, nb))?;
    }
    b.wait_all()?;
    b.label(".Lunion_end")?;
    op(&mut b, "s_endpgm", &[], &[])?;
    b.finish()
}

fn vi(b: &mut Builder, dst: u8, text: String, uses: &[RegRef]) -> R {
    op(b, text, &[v(dst)], uses)
}
fn mov(b: &mut Builder, dst: u8, value: &str, uses: &[RegRef]) -> R {
    vi(b, dst, format!("v_mov_b32_e32 v{dst}, {value}"), uses)
}
fn cmp(b: &mut Builder, mask: u8, kind: &str, a: &str, c: &str, uses: &[RegRef]) -> R {
    op(
        b,
        format!("v_cmp_{kind}_e64 s{mask}, {a}, {c}"),
        &[sr(mask, 2)],
        uses,
    )
}
fn and_mask(b: &mut Builder) -> R {
    sop(
        b,
        format!("s_and_b32 s{MASK}, s{MASK}, s{MASK2}"),
        &[MASK],
        &[MASK, MASK2],
    )
}
fn choose(b: &mut Builder, dst: u8, no: &str, yes: &str, mask: u8, uses: &[RegRef]) -> R {
    let mut uses = uses.to_vec();
    uses.push(sr(mask, 2));
    vi(
        b,
        dst,
        format!("v_cndmask_b32_e64 v{dst}, {no}, {yes}, s{mask}"),
        &uses,
    )
}
fn add(b: &mut Builder, dst: u8, a: u8, c: u8) -> R {
    vi(
        b,
        dst,
        format!("v_add_nc_u32_e32 v{dst}, v{a}, v{c}"),
        &[v(a), v(c)],
    )
}
fn masked(b: &mut Builder, f: impl FnOnce(&mut Builder) -> R) -> R {
    op(
        b,
        format!("s_and_saveexec_b32 s{SAVE}, s{MASK}"),
        &[s(SAVE)],
        &[sr(MASK, 2)],
    )?;
    f(b)?;
    op(b, format!("s_mov_b32 exec_lo, s{SAVE}"), &[], &[s(SAVE)])
}
fn load(b: &mut Builder, dst: u8, addr: u8, base: u8, byte: bool) -> R {
    let mnemonic = if byte {
        "global_load_u8"
    } else {
        "global_load_b32"
    };
    mem(
        b,
        format!("{mnemonic} v{dst}, v{addr}, s[{base}:{}]", base + 1),
        &[v(dst)],
        &[v(addr), sr(base, 2)],
        MemoryClass::VmemLoad,
    )
}
fn output(b: &mut Builder, index: u8, base: u32, data: u8) -> R {
    let out = args(b.spec.arch).out;
    vi(
        b,
        ADDR,
        format!("v_lshlrev_b32_e32 v{ADDR}, 2, v{index}"),
        &[v(index)],
    )?;
    mem(
        b,
        format!(
            "global_store_b32 v{ADDR}, v{data}, s[{out}:{}] offset:{}",
            out + 1,
            base * 4
        ),
        &[],
        &[v(ADDR), v(data), sr(out, 2)],
        MemoryClass::VmemStore,
    )
}
fn ds_store(b: &mut Builder, slot: usize, addr: u8, data: u8, off: u32) -> R {
    b.ds_store(
        slot,
        Instruction::new(
            format!("ds_store_b32 v{addr}, v{data} offset:{off}"),
            vec![],
            vec![v(addr), v(data)],
        )
        .memory(MemoryClass::DsStore),
    )
}
fn ds_load(b: &mut Builder, slot: usize, dst: u8, addr: u8, off: u32) -> R {
    b.ds_load(
        slot,
        Instruction::new(
            format!("ds_load_b32 v{dst}, v{addr} offset:{off}"),
            vec![v(dst)],
            vec![v(addr)],
        )
        .memory(MemoryClass::DsLoad),
    )
}
fn scan_end(b: &mut Builder, label: &str) -> R {
    let add = super::common::s_add_i32(b.spec.arch);
    sop(b, format!("{add} s{J}, s{J}, 1"), &[J], &[J])?;
    sop(b, format!("s_cmp_lt_u32 s{J}, {}", lit(80)), &[], &[J])?;
    op(b, format!("s_cbranch_scc1 {label}"), &[], &[])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reg::Kind;

    /// The v0..v24 file must be three non-overlapping legal v8 granules that
    /// cover exactly 0..24 with no gap, alias, or reassignment.
    #[test]
    fn slot_state_is_three_v8_chunks_covering_24_exact() {
        for arch in [Arch::Gfx1151, Arch::Gfx1201, Arch::Gfx1100] {
            let emitted = emit(arch).expect("slab union plan emits");
            let mut chunks: Vec<(u8, u8)> = emitted
                .proof
                .reg_plan
                .iter()
                .filter(|r| r.name.as_ref() == "slot_state" && r.kind == Kind::V)
                .map(|r| (r.base, r.len))
                .collect();
            chunks.sort();
            assert_eq!(chunks, [(0, 8), (8, 8), (16, 8)], "{arch:?}");
            assert_eq!(emitted.proof.next_free_vgpr, 24, "{arch:?}");
            assert_eq!(emitted.shape.next_free_vgpr, 24, "{arch:?}");
        }
    }
}
