// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Independent tests of the K1 slab MoE decode planner, kernarg ABI and launch
//! planning. The module under test is registered by path, so these tests do not
//! depend on `pipeline/mod.rs` and need no GPU.

#[allow(dead_code)]
#[path = "../src/pipeline/slab_moe_decode.rs"]
mod slab_moe_decode;

use slab_moe_decode::*;
use std::mem::{align_of, offset_of, size_of};

const W3: Option<Codec> = Some(Codec::Mq3I4);
const Q4: Option<Codec> = Some(Codec::Mq4Sym);

/// Even experts MQ4, odd experts W3.
fn alternating() -> Vec<Option<Codec>> {
    (0..MAX_EXPERTS)
        .map(|e| if e % 2 == 0 { Q4 } else { W3 })
        .collect()
}

/// Pseudo-random mix by expert id.
fn scrambled() -> Vec<Option<Codec>> {
    (0..MAX_EXPERTS)
        .map(|e| {
            if (e * 2_654_435_761usize >> 7) & 1 == 0 {
                Q4
            } else {
                W3
            }
        })
        .collect()
}

/// `c` lanes of 10 unique ids in `0..512` (partial Fisher-Yates, xorshift).
fn lanes(c: usize, seed: u32) -> Vec<[u32; TOP_K]> {
    let mut s = seed | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    (0..c)
        .map(|_| {
            let mut pool: Vec<u32> = (0..MAX_EXPERTS as u32).collect();
            let mut ids = [0u32; TOP_K];
            for (i, slot) in ids.iter_mut().enumerate() {
                let j = i + next() as usize % (pool.len() - i);
                pool.swap(i, j);
                *slot = pool[i];
            }
            ids
        })
        .collect()
}

/// Brute-force oracle: `(codec, expert, slots)` in the contract order.
fn oracle(codecs: &[Option<Codec>], top: &[[u32; TOP_K]]) -> Vec<(Codec, u32, Vec<u32>)> {
    let mut out = Vec::new();
    for codec in [Codec::Mq3I4, Codec::Mq4Sym] {
        for e in 0..MAX_EXPERTS as u32 {
            let mut slots = Vec::new();
            for (lane, ids) in top.iter().enumerate() {
                for (rank, &id) in ids.iter().enumerate() {
                    if id == e && codecs[e as usize] == Some(codec) {
                        slots.push((lane * TOP_K + rank) as u32);
                    }
                }
            }
            if !slots.is_empty() {
                out.push((codec, e, slots));
            }
        }
    }
    out
}

/// Flatten a plan back into the oracle's shape, asserting the descriptor
/// contract on the way.
fn flatten(plan: &LaneUnionPlan) -> Vec<(Codec, u32, Vec<u32>)> {
    let mut out = Vec::new();
    for (b, bucket) in plan.layout.buckets.iter().enumerate() {
        let offsets = plan.bucket_offsets(b);
        assert_eq!(offsets.len(), bucket.expert_count + 1);
        assert_eq!(offsets[0], 0, "offsets start at 0 of the own bucket");
        assert_eq!(
            offsets[bucket.expert_count] as usize, bucket.slot_count,
            "offsets end at own slot count"
        );
        assert!(
            offsets.windows(2).all(|w| w[0] < w[1]),
            "no empty expert listed"
        );
        let ids = plan.bucket_experts(b);
        assert!(
            ids.windows(2).all(|w| w[0] < w[1]),
            "expert ids strictly increase in a bucket"
        );
        for (u, &id) in ids.iter().enumerate() {
            out.push((bucket.codec, id, plan.expert_slots(b, u).to_vec()));
        }
    }
    out
}

fn check_plan(codecs: &[Option<Codec>], top: &[[u32; TOP_K]]) -> LaneUnionPlan {
    let c = top.len();
    let plan = LaneUnionPlan::build(codecs, top).unwrap();
    plan.layout.validate().unwrap();
    assert_eq!(flatten(&plan), oracle(codecs, top));
    // Buckets partition the original slots exactly once.
    let mut seen = vec![0u8; TOP_K * c];
    for &s in &plan.slot_ids {
        seen[s as usize] += 1;
    }
    assert!(seen.iter().all(|&n| n == 1));
    assert_eq!(plan.slot_ids.len(), TOP_K * c);
    // U bound, array extents, codec order.
    assert!(plan.layout.u_total <= max_union(c));
    assert_eq!(plan.expert_ids.len(), plan.layout.expert_id_words());
    assert_eq!(plan.expert_offsets.len(), plan.layout.expert_offset_words());
    let order: Vec<usize> = plan
        .layout
        .buckets
        .iter()
        .map(|b| b.codec.order_index())
        .collect();
    assert!(order.windows(2).all(|w| w[0] < w[1]));
    // Inverse permutation.
    for (slot, &pos) in plan.inverse.iter().enumerate() {
        assert_eq!(plan.slot_ids[pos as usize] as usize, slot);
    }
    plan
}

// ---------------------------------------------------------------------------
// ABI
// ---------------------------------------------------------------------------

#[test]
fn kernarg_abi_is_the_frozen_96_bytes() {
    assert_eq!(size_of::<MoeGemvKernargs>(), 96);
    assert_eq!(KERNARG_BYTES, 96);
    assert_eq!(align_of::<MoeGemvKernargs>(), 8);
    let offsets = [
        (offset_of!(MoeGemvKernargs, expert_weight_ptrs), 0),
        (offset_of!(MoeGemvKernargs, expert_ids), 8),
        (offset_of!(MoeGemvKernargs, expert_offsets), 16),
        (offset_of!(MoeGemvKernargs, slot_ids), 24),
        (offset_of!(MoeGemvKernargs, xrot), 32),
        (offset_of!(MoeGemvKernargs, y0), 40),
        (offset_of!(MoeGemvKernargs, y1), 48),
        (offset_of!(MoeGemvKernargs, m), 56),
        (offset_of!(MoeGemvKernargs, k), 60),
        (offset_of!(MoeGemvKernargs, u), 64),
        (offset_of!(MoeGemvKernargs, c), 68),
        (offset_of!(MoeGemvKernargs, x_row_div), 72),
        (offset_of!(MoeGemvKernargs, x_src_rows), 76),
        (offset_of!(MoeGemvKernargs, x_row_stride), 80),
        (offset_of!(MoeGemvKernargs, y_slot_stride), 84),
        (offset_of!(MoeGemvKernargs, n_experts), 88),
        (offset_of!(MoeGemvKernargs, reserved), 92),
    ];
    for (got, want) in offsets {
        assert_eq!(got, want);
    }
}

#[test]
fn kernarg_bytes_are_little_endian_at_the_frozen_offsets() {
    let a = MoeGemvKernargs {
        expert_weight_ptrs: 0x0102_0304_0506_0708,
        expert_ids: 0x1112_1314_1516_1718,
        expert_offsets: 0x2122_2324_2526_2728,
        slot_ids: 0x3132_3334_3536_3738,
        xrot: 0x4142_4344_4546_4748,
        y0: 0x5152_5354_5556_5758,
        y1: 0x6162_6364_6566_6768,
        m: 0xa1a2_a3a4,
        k: 0xb1b2_b3b4,
        u: 0xc1c2_c3c4,
        c: 0xd1d2_d3d4,
        x_row_div: 0xe1e2_e3e4,
        x_src_rows: 0xf1f2_f3f4,
        x_row_stride: 0x0a0b_0c0d,
        y_slot_stride: 0x1a1b_1c1d,
        n_experts: 0x2a2b_2c2d,
        reserved: 0,
    };
    let b = a.to_bytes();
    assert_eq!(b.len(), 96);
    assert_eq!(
        u64::from_le_bytes(b[0..8].try_into().unwrap()),
        a.expert_weight_ptrs
    );
    assert_eq!(
        u64::from_le_bytes(b[8..16].try_into().unwrap()),
        a.expert_ids
    );
    assert_eq!(
        u64::from_le_bytes(b[16..24].try_into().unwrap()),
        a.expert_offsets
    );
    assert_eq!(
        u64::from_le_bytes(b[24..32].try_into().unwrap()),
        a.slot_ids
    );
    assert_eq!(u64::from_le_bytes(b[32..40].try_into().unwrap()), a.xrot);
    assert_eq!(u64::from_le_bytes(b[40..48].try_into().unwrap()), a.y0);
    assert_eq!(u64::from_le_bytes(b[48..56].try_into().unwrap()), a.y1);
    let words = [
        a.m,
        a.k,
        a.u,
        a.c,
        a.x_row_div,
        a.x_src_rows,
        a.x_row_stride,
        a.y_slot_stride,
        a.n_experts,
        a.reserved,
    ];
    for (i, want) in words.iter().enumerate() {
        let o = 56 + 4 * i;
        assert_eq!(
            u32::from_le_bytes(b[o..o + 4].try_into().unwrap()),
            *want,
            "word {i}"
        );
    }
}

#[test]
fn symbols_and_profiles_are_fixed() {
    assert_eq!(
        SYMBOLS,
        [
            "mq_i4_moe_gemv_mq3i4_r256_gate_up",
            "mq_i4_moe_gemv_mq3i4_r128_down",
            "mq_i4_moe_gemv_mq4sym_r256_gate_up",
            "mq_i4_moe_gemv_mq4sym_r128_down",
        ]
    );
    assert_eq!(Kind::GateUp.profile(), Profile::R256);
    assert_eq!(Kind::Down.profile(), Profile::R128);
    assert_eq!((Kind::GateUp.m(), Kind::GateUp.k()), (1280, 2560));
    assert_eq!((Kind::Down.m(), Kind::Down.k()), (2560, 640));
    assert_eq!((Kind::GateUp.x_row_div(), Kind::Down.x_row_div()), (10, 1));
    for (i, key) in BucketKey::ORDER.iter().enumerate() {
        assert_eq!(key.order_index(), i);
        assert_eq!(BucketKey::new(key.codec, key.kind()), *key);
    }
}

// ---------------------------------------------------------------------------
// Invalid codec / profile
// ---------------------------------------------------------------------------

#[test]
fn only_mq3i4_and_mq4sym_qt_types_are_admitted() {
    assert_eq!(Codec::from_qt(Kind::GateUp, 131), Ok(Codec::Mq3I4));
    assert_eq!(Codec::from_qt(Kind::Down, 132), Ok(Codec::Mq3I4));
    assert_eq!(Codec::from_qt(Kind::GateUp, 44), Ok(Codec::Mq4Sym));
    assert_eq!(Codec::from_qt(Kind::Down, 53), Ok(Codec::Mq4Sym));
    assert_eq!(Codec::from_qt_pair(131, 132), Ok(Codec::Mq3I4));
    assert_eq!(Codec::from_qt_pair(44, 53), Ok(Codec::Mq4Sym));
    // W2, the other kind's type, legacy aggregates and unknown types.
    for (kind, qt) in [
        (Kind::GateUp, 133),
        (Kind::Down, 134),
        (Kind::GateUp, 132),
        (Kind::Down, 131),
        (Kind::GateUp, 53),
        (Kind::Down, 44),
        (Kind::GateUp, 52),
        (Kind::GateUp, 0),
        (Kind::Down, 255),
    ] {
        assert_eq!(
            Codec::from_qt(kind, qt),
            Err(SlabMoeError::UnsupportedQt { kind, qt })
        );
    }
    for (g, d) in [(131, 53), (44, 132)] {
        assert_eq!(
            Codec::from_qt_pair(g, d),
            Err(SlabMoeError::QtPairMismatch {
                gate_up_qt: g,
                down_qt: d
            })
        );
    }
    for (g, d) in [
        (133, 134),
        (131, 134),
        (133, 132),
        (132, 131),
        (53, 44),
        (0, 0),
    ] {
        assert!(Codec::from_qt_pair(g, d).is_err(), "({g}, {d})");
    }
}

// ---------------------------------------------------------------------------
// Planner
// ---------------------------------------------------------------------------

#[test]
fn c1_mixed_order_buckets_w3_first_then_mq4_with_stable_slots() {
    let top = [[9, 4, 7, 2, 5, 0, 3, 8, 1, 6]];
    let plan = check_plan(&alternating(), &top);
    assert_eq!(plan.layout.c, 1);
    assert_eq!(plan.layout.u_total, 10);
    assert_eq!(plan.layout.buckets.len(), 2);
    // W3 (odd) bucket first, ids increasing; slot = rank in the original lane.
    assert_eq!(plan.layout.buckets[0].codec, Codec::Mq3I4);
    assert_eq!(plan.bucket_experts(0), &[1, 3, 5, 7, 9]);
    assert_eq!(plan.bucket_slots(0), &[8, 6, 4, 2, 0]);
    assert_eq!(plan.bucket_offsets(0), &[0, 1, 2, 3, 4, 5]);
    assert_eq!(plan.layout.buckets[1].codec, Codec::Mq4Sym);
    assert_eq!(plan.bucket_experts(1), &[0, 2, 4, 6, 8]);
    assert_eq!(plan.bucket_slots(1), &[5, 3, 1, 9, 7]);
    assert_eq!(plan.bucket_offsets(1), &[0, 1, 2, 3, 4, 5]);
    // Descriptor layout: ids concatenated, one (U_b + 1) offsets run per bucket.
    assert_eq!(
        plan.expert_offsets,
        vec![0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
    );
    assert_eq!(plan.layout.buckets[1].expert_start, 5);
    assert_eq!(plan.layout.buckets[1].offsets_start, 6);
    assert_eq!(plan.layout.buckets[1].slot_start, 5);
}

#[test]
fn c1_single_codec_produces_one_bucket_for_either_codec() {
    let top = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]];
    let all_w3 = vec![W3; MAX_EXPERTS];
    let plan = check_plan(&all_w3, &top);
    assert_eq!(plan.layout.buckets.len(), 1);
    assert_eq!(plan.layout.buckets[0].codec, Codec::Mq3I4);
    assert_eq!(plan.slot_ids, (0..10).collect::<Vec<u32>>());
    let all_q4 = vec![Q4; MAX_EXPERTS];
    let plan = check_plan(&all_q4, &top);
    assert_eq!(plan.layout.buckets.len(), 1);
    assert_eq!(plan.layout.buckets[0].codec, Codec::Mq4Sym);
}

#[test]
fn c8_union_of_mixed_codecs_matches_the_oracle_for_many_routes() {
    for codecs in [alternating(), scrambled()] {
        for seed in 1..=64u32 {
            for c in [1, 2, 3, 7, 8] {
                let top = lanes(c, seed.wrapping_mul(0x9e37_79b9));
                let plan = check_plan(&codecs, &top);
                assert!(
                    plan.layout.u_total >= TOP_K,
                    "each lane alone selects 10 distinct experts"
                );
            }
        }
    }
}

#[test]
fn c8_shared_experts_keep_original_slot_order_and_read_once() {
    // All lanes pick the same ten experts in different rank order: U == 10.
    let mut top = Vec::new();
    for lane in 0..8u32 {
        let mut ids = [0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        ids.rotate_left(lane as usize);
        top.push(ids);
    }
    let plan = check_plan(&alternating(), &top);
    assert_eq!(plan.layout.u_total, 10);
    for b in 0..plan.layout.buckets.len() {
        for u in 0..plan.layout.buckets[b].expert_count {
            let slots = plan.expert_slots(b, u);
            assert_eq!(slots.len(), 8);
            assert!(
                slots.windows(2).all(|w| w[0] < w[1]),
                "slots ascend (lane-major)"
            );
            for (lane, &s) in slots.iter().enumerate() {
                assert_eq!(s as usize / TOP_K, lane, "one slot per lane, lane order");
            }
        }
    }
}

#[test]
fn union_bound_is_min_10c_512() {
    assert_eq!(max_union(1), 10);
    assert_eq!(max_union(8), 80);
    assert_eq!(max_union(60), 512);
    // Eight lanes of pairwise disjoint experts reach the bound 80.
    let top: Vec<[u32; TOP_K]> = (0..8u32)
        .map(|l| core::array::from_fn(|r| l * 10 + r as u32))
        .collect();
    let plan = check_plan(&alternating(), &top);
    assert_eq!(plan.layout.u_total, 80);
    // A layout above the bound is refused.
    let mut bad = plan.layout.clone();
    bad.c = 1;
    assert!(bad.validate().is_err());
}

#[test]
fn expert_ids_511_and_0_are_valid_bounds() {
    let top = [[0, 511, 255, 256, 1, 510, 2, 509, 3, 508]];
    let plan = check_plan(&alternating(), &top);
    assert_eq!(plan.layout.u_total, 10);
}

#[test]
fn planner_refuses_out_of_bounds_routes() {
    let codecs = alternating();
    let good = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    assert_eq!(
        LaneUnionPlan::build(&codecs, &[]),
        Err(SlabMoeError::InvalidLaneCount { c: 0 })
    );
    assert_eq!(
        LaneUnionPlan::build(&codecs, &vec![good; 33]),
        Err(SlabMoeError::InvalidLaneCount { c: 33 })
    );
    assert!(LaneUnionPlan::build(&codecs, &vec![good; 32]).is_ok());
    let mut high = good;
    high[3] = 512;
    assert_eq!(
        LaneUnionPlan::build(&codecs, &[good, high]),
        Err(SlabMoeError::ExpertOutOfRange {
            lane: 1,
            rank: 3,
            id: 512
        })
    );
    high[3] = u32::MAX;
    assert!(matches!(
        LaneUnionPlan::build(&codecs, &[high]),
        Err(SlabMoeError::ExpertOutOfRange { id: u32::MAX, .. })
    ));
    // Duplicate within a lane is refused; the same id in two lanes is legal.
    let mut dup = good;
    dup[9] = 4;
    assert_eq!(
        LaneUnionPlan::build(&codecs, &[good, dup]),
        Err(SlabMoeError::DuplicateExpert { lane: 1, id: 4 })
    );
    assert!(LaneUnionPlan::build(&codecs, &[good, good]).is_ok());
}

#[test]
fn planner_refuses_experts_without_a_declared_codec() {
    let mut codecs = alternating();
    codecs[7] = None;
    assert_eq!(
        LaneUnionPlan::build(&codecs, &[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
        Err(SlabMoeError::MissingCodec { expert: 7 })
    );
    // A short table leaves high ids undeclared; unselected experts are free.
    let full = alternating();
    let short = &full[..16];
    assert_eq!(
        LaneUnionPlan::build(short, &[[0, 1, 2, 3, 4, 5, 6, 7, 8, 16]]),
        Err(SlabMoeError::MissingCodec { expert: 16 })
    );
    assert!(LaneUnionPlan::build(short, &[[0, 1, 2, 3, 4, 5, 6, 7, 8, 15]]).is_ok());
    codecs[500] = None;
    assert!(LaneUnionPlan::build(&codecs, &[[0, 1, 2, 3, 4, 5, 6, 8, 9, 10]]).is_ok());
}

#[test]
fn layout_validation_catches_corrupt_descriptors() {
    let plan = check_plan(&alternating(), &lanes(8, 7));
    let ok = plan.layout.clone();
    assert_eq!(ok.buckets.len(), 2, "this route mixes both codecs");
    fn mutate(ok: &UnionLayout, f: impl Fn(&mut UnionLayout)) {
        let mut l = ok.clone();
        f(&mut l);
        l.validate().expect_err("corrupt layout must be refused");
    }
    mutate(&ok, |l| l.buckets.reverse());
    mutate(&ok, |l| l.buckets[0].slot_count += 1);
    mutate(&ok, |l| l.buckets[1].slot_start += 1);
    mutate(&ok, |l| l.buckets[1].offsets_start += 1);
    mutate(&ok, |l| l.buckets[1].expert_start += 1);
    mutate(&ok, |l| l.buckets[0].expert_count = 0);
    mutate(&ok, |l| l.u_total += 1);
    mutate(&ok, |l| l.c = 9);
    mutate(&ok, |l| l.c = 33);
    mutate(&ok, |l| l.c = 0);
    mutate(&ok, |l| l.buckets.clear());
    assert!(ok.validate().is_ok());
}

// ---------------------------------------------------------------------------
// Launch planning
// ---------------------------------------------------------------------------

struct Fixture {
    plan: LaneUnionPlan,
    desc: DescriptorArrays,
    table: DevView,
    x: DevView,
    y0: DevView,
    y1: DevView,
}

const DESC_IDS: u64 = 0x1000_0000;
const DESC_OFFS: u64 = 0x1100_0000;
const DESC_SLOTS: u64 = 0x1200_0000;
const TABLE: u64 = 0x2000_0000;
const X: u64 = 0x3000_0000;
const Y0: u64 = 0x4000_0000;
const Y1: u64 = 0x5000_0000;

fn fixture(kind: Kind, c: usize, seed: u32) -> Fixture {
    let plan = check_plan(&alternating(), &lanes(c, seed));
    let l = &plan.layout;
    let x_bytes = kind.x_src_rows(c) * kind.k() * 4;
    let y_bytes = c * TOP_K * kind.y_slot_stride() * 4;
    Fixture {
        desc: DescriptorArrays {
            expert_ids: DevView::new(DESC_IDS, l.expert_id_words() * 4),
            expert_offsets: DevView::new(DESC_OFFS, l.expert_offset_words() * 4),
            slot_ids: DevView::new(DESC_SLOTS, l.slot_words() * 4),
        },
        table: DevView::new(TABLE, MAX_EXPERTS * 8),
        x: DevView::new(X, x_bytes),
        y0: DevView::new(Y0, y_bytes),
        y1: DevView::new(Y1, y_bytes),
        plan,
    }
}

fn request<'a>(kind: Kind, f: &'a Fixture) -> DecodeRequest<'a> {
    DecodeRequest {
        kind,
        profile: kind.profile(),
        m: kind.m(),
        k: kind.k(),
        layout: &f.plan.layout,
        descriptors: f.desc,
        weight_ptrs: f.table,
        xrot: f.x,
        y0: f.y0,
        y1: kind.has_y1().then_some(f.y1),
    }
}

#[test]
fn gate_up_launch_plan_matches_the_frozen_geometry() {
    for c in [1usize, 8] {
        let f = fixture(Kind::GateUp, c, 11);
        let launches = plan_launches(&request(Kind::GateUp, &f)).unwrap();
        assert_eq!(launches.len(), f.plan.layout.buckets.len());
        for (l, b) in launches.iter().zip(&f.plan.layout.buckets) {
            assert_eq!(l.key, BucketKey::new(b.codec, Kind::GateUp));
            let want = match b.codec {
                Codec::Mq3I4 => "mq_i4_moe_gemv_mq3i4_r256_gate_up",
                Codec::Mq4Sym => "mq_i4_moe_gemv_mq4sym_r256_gate_up",
            };
            assert_eq!(l.symbol, want);
            assert_eq!(l.grid, [1280, b.expert_count as u32, 1]);
            assert_eq!(l.block, [32, 1, 1]);
            let k = &l.kernargs;
            assert_eq!(k.expert_weight_ptrs, TABLE);
            assert_eq!(k.expert_ids, DESC_IDS + 4 * b.expert_start as u64);
            assert_eq!(k.expert_offsets, DESC_OFFS + 4 * b.offsets_start as u64);
            assert_eq!(k.slot_ids, DESC_SLOTS + 4 * b.slot_start as u64);
            assert_eq!((k.xrot, k.y0, k.y1), (X, Y0, Y1));
            assert_eq!(
                (k.m, k.k, k.u, k.c),
                (1280, 2560, b.expert_count as u32, c as u32)
            );
            assert_eq!(
                (k.x_row_div, k.x_src_rows, k.x_row_stride, k.y_slot_stride),
                (10, c as u32, 2560, 640)
            );
        }
    }
}

#[test]
fn down_launch_plan_is_unweighted_original_slot_output() {
    for c in [1usize, 8] {
        let f = fixture(Kind::Down, c, 23);
        let launches = plan_launches(&request(Kind::Down, &f)).unwrap();
        assert_eq!(launches.len(), f.plan.layout.buckets.len());
        for (l, b) in launches.iter().zip(&f.plan.layout.buckets) {
            let want = match b.codec {
                Codec::Mq3I4 => "mq_i4_moe_gemv_mq3i4_r128_down",
                Codec::Mq4Sym => "mq_i4_moe_gemv_mq4sym_r128_down",
            };
            assert_eq!(l.symbol, want);
            assert_eq!(l.grid, [2560, b.expert_count as u32, 1]);
            let k = &l.kernargs;
            assert_eq!(k.y1, 0, "down has no second output");
            assert_eq!(
                (k.m, k.k, k.u, k.c),
                (2560, 640, b.expert_count as u32, c as u32)
            );
            assert_eq!(
                (k.x_row_div, k.x_src_rows, k.x_row_stride, k.y_slot_stride),
                (1, 10 * c as u32, 640, 2560)
            );
        }
    }
}

#[test]
fn launch_plan_buckets_partition_the_slots_and_keep_codec_order() {
    let f = fixture(Kind::GateUp, 8, 99);
    let launches = plan_launches(&request(Kind::GateUp, &f)).unwrap();
    let indices: Vec<usize> = launches.iter().map(|l| l.key.order_index()).collect();
    assert!(indices.windows(2).all(|w| w[0] < w[1]));
    let slots: usize = f.plan.layout.buckets.iter().map(|b| b.slot_count).sum();
    assert_eq!(slots, 80);
    let experts: u32 = launches.iter().map(|l| l.kernargs.u).sum();
    assert_eq!(experts as usize, f.plan.layout.u_total);
}

#[test]
fn launch_refuses_wrong_profile_geometry_and_outputs() {
    let f = fixture(Kind::GateUp, 8, 5);
    let ok = request(Kind::GateUp, &f);
    assert!(plan_launches(&ok).is_ok());
    let mut r = ok;
    r.profile = Profile::R128;
    assert_eq!(
        plan_launches(&r),
        Err(SlabMoeError::ProfileMismatch {
            kind: Kind::GateUp,
            profile: Profile::R128
        })
    );
    let mut r = ok;
    r.m = 640;
    assert_eq!(
        plan_launches(&r),
        Err(SlabMoeError::Geometry {
            what: "M",
            got: 640,
            want: 1280
        })
    );
    let mut r = ok;
    r.k = 640;
    assert_eq!(
        plan_launches(&r),
        Err(SlabMoeError::Geometry {
            what: "K",
            got: 640,
            want: 2560
        })
    );
    let mut r = ok;
    r.y1 = None;
    assert!(matches!(
        plan_launches(&r),
        Err(SlabMoeError::Operand { name: "y1", .. })
    ));
    let d = fixture(Kind::Down, 8, 5);
    let mut r = request(Kind::Down, &d);
    r.y1 = Some(d.y1);
    assert!(matches!(
        plan_launches(&r),
        Err(SlabMoeError::Operand { name: "y1", .. })
    ));
    // Down geometry swapped onto gate/up is refused by the explicit M/K.
    let mut r = request(Kind::Down, &d);
    r.m = 1280;
    r.k = 2560;
    assert!(matches!(
        plan_launches(&r),
        Err(SlabMoeError::Geometry { .. })
    ));
}

#[test]
fn launch_refuses_bad_operands() {
    let f = fixture(Kind::Down, 8, 31);
    let ok = request(Kind::Down, &f);
    assert!(plan_launches(&ok).is_ok());
    let name_of = |r: DecodeRequest<'_>| match plan_launches(&r) {
        Err(SlabMoeError::Operand { name, .. }) => name,
        other => panic!("expected an operand refusal, got {other:?}"),
    };
    let shrink = |v: DevView| DevView::new(v.ptr, v.bytes - 4);
    let mut r = ok;
    r.descriptors.expert_ids = shrink(ok.descriptors.expert_ids);
    assert_eq!(name_of(r), "expert_ids");
    let mut r = ok;
    r.descriptors.expert_offsets = shrink(ok.descriptors.expert_offsets);
    assert_eq!(name_of(r), "expert_offsets");
    let mut r = ok;
    r.descriptors.slot_ids = shrink(ok.descriptors.slot_ids);
    assert_eq!(name_of(r), "slot_ids");
    let mut r = ok;
    r.weight_ptrs = DevView::new(TABLE, MAX_EXPERTS * 8 - 8);
    assert_eq!(name_of(r), "weight_ptrs");
    let mut r = ok;
    r.weight_ptrs = DevView::new(TABLE + 4, MAX_EXPERTS * 8);
    assert_eq!(name_of(r), "weight_ptrs");
    let mut r = ok;
    r.xrot = shrink(ok.xrot);
    assert_eq!(name_of(r), "xrot");
    let mut r = ok;
    r.xrot = DevView::new(0, ok.xrot.bytes);
    assert_eq!(name_of(r), "xrot");
    let mut r = ok;
    r.y0 = shrink(ok.y0);
    assert_eq!(name_of(r), "y0");
    // Output overlapping the input is refused.
    let mut r = ok;
    r.y0 = DevView::new(ok.xrot.ptr + 16, ok.y0.bytes);
    assert_eq!(name_of(r), "y0");
    // Gate/up outputs must not overlap each other.
    let g = fixture(Kind::GateUp, 2, 31);
    let mut r = request(Kind::GateUp, &g);
    r.y1 = Some(DevView::new(g.y0.ptr + 64, g.y1.bytes));
    assert_eq!(name_of(r), "y1");
}

#[test]
fn launch_refuses_a_corrupt_layout() {
    let f = fixture(Kind::GateUp, 8, 3);
    let mut bad = f.plan.layout.clone();
    bad.buckets[0].slot_count += 1;
    let mut r = request(Kind::GateUp, &f);
    r.layout = &bad;
    assert!(matches!(plan_launches(&r), Err(SlabMoeError::Layout(_))));
}

#[test]
fn descriptor_bytes_are_little_endian_u32_arrays() {
    let plan = check_plan(&alternating(), &lanes(2, 4));
    let bytes = u32_le_bytes(&plan.slot_ids);
    assert_eq!(bytes.len(), 80);
    for (i, w) in plan.slot_ids.iter().enumerate() {
        assert_eq!(
            u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap()),
            *w
        );
    }
}

// ---------------------------------------------------------------------------
// Fixed-capacity device planner model (`qwen4_slab_union_plan`), C1..=C32
// ---------------------------------------------------------------------------

fn flat(top: &[[u32; TOP_K]]) -> Vec<u32> {
    top.iter().flatten().copied().collect()
}

/// Independent first-occurrence construction of the `6 S + 2` words (no shared
/// code with the thread model): experts per bucket in the order their first slot
/// appears, slots ascending inside an expert, tails 0 / `n_b`.
fn reference_words(topk: &[u32], bucket_of: &dyn Fn(u32) -> usize) -> Vec<u32> {
    let s = topk.len();
    let mut out = vec![0u32; 6 * s + 2];
    for b in 0..2 {
        let mut experts: Vec<u32> = Vec::new();
        for &id in topk {
            if bucket_of(id) == b && !experts.contains(&id) {
                experts.push(id);
            }
        }
        let mut n = 0usize;
        for (e, &id) in experts.iter().enumerate() {
            out[b * s + e] = id;
            out[2 * s + b * (s + 1) + e] = n as u32;
            for (slot, &x) in topk.iter().enumerate() {
                if x == id {
                    out[2 * s + 2 * (s + 1) + b * s + n] = slot as u32;
                    n += 1;
                }
            }
        }
        for e in experts.len()..=s {
            out[2 * s + b * (s + 1) + e] = n as u32;
        }
    }
    out
}

/// Model the planner, assert single ownership of every word, equality with the
/// independent reference and (canonicalized) with the sorted compact plan.
fn check_fixed(
    codecs: &[Option<Codec>],
    top: &[[u32; TOP_K]],
    uniform: Option<Codec>,
) -> FixedUnion {
    let c = top.len();
    let m = FixedUnion::from_routes(codecs, top, uniform).unwrap();
    assert_eq!(m.s, TOP_K * c);
    assert_eq!(m.words.len(), fixed_union_words(c).unwrap());
    assert!(
        m.writes.iter().all(|&w| w == 1),
        "every one of the 6S+2 words has exactly one owner"
    );
    m.validate().unwrap();
    let bucket_of = |id: u32| match uniform {
        Some(k) => k.order_index(),
        None => codecs[id as usize].unwrap().order_index(),
    };
    assert_eq!(m.words, reference_words(&flat(top), &bucket_of));
    let effective: Vec<Option<Codec>> = match uniform {
        Some(k) => vec![Some(k); MAX_EXPERTS],
        None => codecs.to_vec(),
    };
    let compact = LaneUnionPlan::build(&effective, top).unwrap();
    assert_eq!(m.canonical(), flatten(&compact));
    for bucket in &compact.layout.buckets {
        let idx = bucket.codec.order_index();
        assert_eq!(m.expert_count(idx), bucket.expert_count);
        assert_eq!(m.slot_count(idx), bucket.slot_count);
    }
    assert_eq!(
        m.expert_count(0) + m.expert_count(1),
        compact.layout.u_total,
        "empty buckets contribute no experts"
    );
    m
}

#[test]
fn kernel_source_constants_match_the_cpu_model() {
    let src = include_str!("../../../kernels/src/qwen4_slab_union_plan.hip");
    let define = |name: &str| -> usize {
        let pat = format!("#define {name} ");
        let at = src
            .find(&pat)
            .unwrap_or_else(|| panic!("{name} not defined"))
            + pat.len();
        src[at..]
            .split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap()
    };
    assert_eq!(define("SLAB_PLAN_MAX_SLOTS"), PLAN_MAX_SLOTS);
    assert_eq!(define("SLAB_PLAN_THREADS"), PLAN_THREADS);
    assert!(src.contains("__launch_bounds__(SLAB_PLAN_THREADS)"));
    assert_eq!(PLAN_MAX_SLOTS, 320);
    assert_eq!(PLAN_THREADS, 512);
}

#[test]
fn fixed_descriptor_extents_c16_c32_and_refusals() {
    assert_eq!(fixed_union_words(1), Ok(62));
    assert_eq!(fixed_union_bytes(8), Ok(1928));
    assert_eq!(fixed_union_bytes(16), Ok(3848));
    assert_eq!(fixed_union_bytes(32), Ok(7688));
    assert_eq!(
        fixed_union_words(0),
        Err(SlabMoeError::InvalidLaneCount { c: 0 })
    );
    assert_eq!(
        fixed_union_words(33),
        Err(SlabMoeError::InvalidLaneCount { c: 33 })
    );
    assert_eq!(
        fixed_union_bytes(usize::MAX),
        Err(SlabMoeError::InvalidLaneCount { c: usize::MAX })
    );
    // Extent refusals: exact fits pass, one word short refuses, before any launch.
    assert_eq!(check_plan_extents(32, 320 * 4, 7688), Ok(1922));
    assert!(matches!(
        check_plan_extents(32, 320 * 4 - 4, 7688),
        Err(SlabMoeError::Operand { name: "topk", .. })
    ));
    assert!(matches!(
        check_plan_extents(32, 320 * 4, 7688 - 4),
        Err(SlabMoeError::Operand {
            name: "union scratch",
            ..
        })
    ));
    assert_eq!(
        check_plan_extents(0, 1 << 20, 1 << 20),
        Err(SlabMoeError::InvalidLaneCount { c: 0 })
    );
    assert_eq!(
        check_plan_extents(33, 1 << 20, 1 << 20),
        Err(SlabMoeError::InvalidLaneCount { c: 33 })
    );
    // A C16-sized reservation is too small for C32.
    assert!(check_plan_extents(32, 320 * 4, 3848).is_err());
    assert!(check_plan_extents(16, 160 * 4, 3848).is_ok());
}

#[test]
fn fixed_planner_refuses_invalid_c_and_inputs() {
    let ids = vec![0i32; 400];
    let table = vec![0u8; MAX_EXPERTS];
    assert_eq!(
        FixedUnion::model(&ids, 0, Some(&table), -1),
        Err(SlabMoeError::InvalidLaneCount { c: 0 })
    );
    assert_eq!(
        FixedUnion::model(&ids, 33, Some(&table), -1),
        Err(SlabMoeError::InvalidLaneCount { c: 33 })
    );
    assert!(matches!(
        FixedUnion::model(&ids[..99], 10, Some(&table), -1),
        Err(SlabMoeError::Operand { name: "topk", .. })
    ));
    assert!(matches!(
        FixedUnion::model(&ids, 4, None, -1),
        Err(SlabMoeError::Operand { name: "codec", .. })
    ));
    assert!(matches!(
        FixedUnion::model(&ids, 4, Some(&table[..511]), -1),
        Err(SlabMoeError::Operand { name: "codec", .. })
    ));
    assert!(matches!(
        FixedUnion::model(&ids, 4, Some(&table), 2),
        Err(SlabMoeError::Layout(_))
    ));
    // A uniform bucket needs no codec table.
    assert!(FixedUnion::model(&ids, 4, None, 1).is_ok());
    assert!(FixedUnion::model(&ids, 32, None, 0).is_ok());
    assert!(matches!(
        FixedUnion::from_routes(&alternating(), &[[0, 1, 2, 3, 4, 5, 6, 7, 8, 512]], None),
        Err(SlabMoeError::ExpertOutOfRange {
            lane: 0,
            rank: 9,
            id: 512
        })
    ));
    let mut codecs = alternating();
    codecs[7] = None;
    assert_eq!(
        FixedUnion::from_routes(&codecs, &[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], None),
        Err(SlabMoeError::MissingCodec { expert: 7 })
    );
    // The uniform route never consults the table.
    assert!(FixedUnion::from_routes(
        &codecs,
        &[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
        Some(Codec::Mq4Sym)
    )
    .is_ok());
}

#[test]
fn fixed_planner_matches_reference_for_every_c_1_to_32() {
    for c in 1..=MAX_LANES {
        for (codecs, seed) in [(alternating(), 1u32), (scrambled(), 2), (alternating(), 3)] {
            let top = lanes(c, seed.wrapping_mul(0x9e37_79b9).wrapping_add(c as u32));
            let m = check_fixed(&codecs, &top, None);
            assert!(m.expert_count(0) + m.expert_count(1) >= TOP_K);
            check_fixed(&codecs, &top, Some(Codec::Mq3I4));
            check_fixed(&codecs, &top, Some(Codec::Mq4Sym));
        }
    }
}

#[test]
fn fixed_planner_identical_sets_have_n_equals_c() {
    for c in 1..=MAX_LANES {
        // Every token selects the same ten experts, rank order rotated per lane.
        let top: Vec<[u32; TOP_K]> = (0..c)
            .map(|lane| {
                let mut ids = [3u32, 14, 15, 92, 65, 35, 89, 79, 32, 38];
                ids.rotate_left(lane % TOP_K);
                ids
            })
            .collect();
        let m = check_fixed(&alternating(), &top, None);
        assert_eq!(m.expert_count(0) + m.expert_count(1), TOP_K, "U = 10");
        for b in 0..2 {
            let u = m.expert_count(b);
            for e in 0..u {
                let run = &m.slots(b)[m.offsets(b)[e] as usize..m.offsets(b)[e + 1] as usize];
                assert_eq!(run.len(), c, "n = C slots per expert");
                for (lane, &s) in run.iter().enumerate() {
                    assert_eq!(s as usize / TOP_K, lane, "one slot per token, token order");
                }
            }
        }
        // First-occurrence order is lane 0's rank order inside each bucket.
        let lane0: Vec<u32> = top[0].to_vec();
        for b in 0..2 {
            let want: Vec<u32> = lane0
                .iter()
                .copied()
                .filter(|&id| alternating()[id as usize].unwrap().order_index() == b)
                .collect();
            assert_eq!(&m.ids(b)[..want.len()], &want[..]);
        }
    }
}

#[test]
fn fixed_planner_maximally_disjoint_reaches_u_equals_10c() {
    for c in 1..=MAX_LANES {
        let top: Vec<[u32; TOP_K]> = (0..c as u32)
            .map(|l| core::array::from_fn(|r| l * 10 + r as u32))
            .collect();
        let m = check_fixed(&alternating(), &top, None);
        assert_eq!(m.expert_count(0) + m.expert_count(1), TOP_K * c);
        assert_eq!(m.expert_count(0) + m.expert_count(1), max_union(c));
        for b in 0..2 {
            let n = m.slot_count(b);
            assert_eq!(m.expert_count(b), n, "every expert has exactly one slot");
            assert!(m.offsets(b)[..=n]
                .iter()
                .enumerate()
                .all(|(i, &o)| o as usize == i));
        }
        // The compact plan accepts the same union.
        let plan = check_plan(&alternating(), &top);
        assert_eq!(plan.layout.u_total, TOP_K * c);
        // One-codec twin: a single full bucket with U = S and an empty tail-only twin.
        let q = check_fixed(&alternating(), &top, Some(Codec::Mq4Sym));
        assert_eq!(q.expert_count(1), TOP_K * c);
        assert_eq!(q.slot_count(1), TOP_K * c);
        assert_eq!(
            q.offsets(1)[TOP_K * c],
            (TOP_K * c) as u32,
            "offs[S] = n_b is written by thread 0"
        );
        assert_eq!(q.expert_count(0), 0);
    }
}

#[test]
fn fixed_planner_single_and_empty_buckets_initialize_every_word() {
    let top = lanes(32, 0xC0FFEE);
    for (codec, full, empty) in [(Codec::Mq3I4, 0usize, 1usize), (Codec::Mq4Sym, 1, 0)] {
        let table = vec![Some(codec); MAX_EXPERTS];
        let m = check_fixed(&table, &top, None);
        let s = m.s;
        assert_eq!(m.slot_count(full), s);
        assert_eq!(m.slot_count(empty), 0);
        assert_eq!(m.expert_count(empty), 0);
        assert!(m.ids(empty).iter().all(|&w| w == 0));
        assert!(m.slots(empty).iter().all(|&w| w == 0));
        assert!(
            m.offsets(empty).iter().all(|&w| w == 0),
            "empty bucket: all S+1 offsets are 0"
        );
        // Same descriptor via the uniform-bucket path.
        let u = check_fixed(&alternating(), &top, Some(codec));
        assert_eq!(u.words, m.words);
        // Raw codec bytes: `1` selects bucket 0 (W3), any other byte bucket 1.
        let ids: Vec<i32> = top.iter().flatten().map(|&x| x as i32).collect();
        let bytes = vec![if full == 0 { 1u8 } else { 0u8 }; MAX_EXPERTS];
        let by_table = FixedUnion::model(&ids, 32, Some(&bytes), -1).unwrap();
        assert_eq!(by_table.words, m.words);
    }
}

#[test]
fn fixed_planner_adjacent_expert_boundaries_are_single_owner() {
    // Experts with one slot each in consecutive bucket positions: the end of
    // expert e is the start of expert e + 1 and used to be written twice.
    for c in [1usize, 2, 9, 16, 17, 31, 32] {
        let top: Vec<[u32; TOP_K]> = (0..c as u32)
            .map(|l| core::array::from_fn(|r| l * 10 + r as u32))
            .collect();
        let m = check_fixed(&vec![Q4; MAX_EXPERTS], &top, Some(Codec::Mq4Sym));
        assert_eq!(m.writes.len(), 6 * TOP_K * c + 2);
        assert!(m.writes.iter().all(|&w| w == 1));
        // Offsets words that sit on an expert boundary or the end of the live run.
        let u = m.expert_count(1);
        assert_eq!(m.offsets(1)[u], (TOP_K * c) as u32);
    }
}

#[test]
fn fixed_planner_expert_511_token_31_rank_9_and_first_occurrence_order() {
    // Token 0 holds 511 at rank 0 and expert 5 at rank 1; token 31 repeats 511
    // at rank 9 (slot 319, the last slot) and lists expert ids descending so
    // that first-occurrence order is not ascending-id order.
    let mut top: Vec<[u32; TOP_K]> = lanes(32, 0xABCDEF);
    top[0] = [511, 5, 100, 101, 102, 103, 104, 105, 106, 107];
    top[31] = [9, 8, 7, 6, 4, 3, 2, 1, 0, 511];
    let codecs = alternating();
    let m = check_fixed(&codecs, &top, None);
    let b = codecs[511].unwrap().order_index();
    let ids = m.ids(b);
    let e = ids.iter().position(|&i| i == 511).unwrap();
    assert!(
        e < m.expert_count(b),
        "511 is a live expert, not a tail zero"
    );
    let run = &m.slots(b)[m.offsets(b)[e] as usize..m.offsets(b)[e + 1] as usize];
    assert_eq!(
        run.first(),
        Some(&0),
        "slot 0 (token 0 rank 0) owns 511 first"
    );
    assert_eq!(run.last(), Some(&319), "token 31 rank 9 is slot 319");
    assert!(run.windows(2).all(|w| w[0] < w[1]));
    // Slot 319 is the largest slot id and appears exactly once in the descriptor.
    let all: Vec<u32> = m.slots(0)[..m.slot_count(0)]
        .iter()
        .chain(&m.slots(1)[..m.slot_count(1)])
        .copied()
        .collect();
    assert_eq!(all.iter().filter(|&&s| s == 319).count(), 1);
    assert_eq!(all.len(), 320);
    // Experts of the last token that are new appear in token order, not id order.
    let mut seen_before: Vec<u32> = flat(&top[..31]);
    seen_before.sort_unstable();
    let fresh: Vec<u32> = top[31]
        .iter()
        .copied()
        .filter(|id| seen_before.binary_search(id).is_err())
        .collect();
    for b in 0..2 {
        let order: Vec<u32> = m.ids(b)[..m.expert_count(b)]
            .iter()
            .copied()
            .filter(|id| fresh.contains(id))
            .collect();
        let want: Vec<u32> = fresh
            .iter()
            .copied()
            .filter(|&id| codecs[id as usize].unwrap().order_index() == b)
            .collect();
        assert_eq!(
            order, want,
            "first-occurrence order of the last token's new experts"
        );
    }
    // Pure first-occurrence order test with a deliberately descending route.
    let desc = [[9u32, 8, 7, 6, 5, 4, 3, 2, 1, 0]];
    let d = check_fixed(&vec![W3; MAX_EXPERTS], &desc, None);
    assert_eq!(&d.ids(0)[..10], &[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
    assert_ne!(&d.ids(0)[..10], &{
        let mut s = [9u32, 8, 7, 6, 5, 4, 3, 2, 1, 0];
        s.sort_unstable();
        s
    });
}

#[test]
fn fixed_planner_clamps_out_of_range_ids_like_the_kernel_and_stays_single_owner() {
    // Invalid routes are refused on the host; the kernel still must not write
    // outside the 6S+2 words or leave one unowned. Expert 0 absorbs the clamp.
    let s = 20;
    let mut ids = vec![0i32; s];
    for (i, id) in ids.iter_mut().enumerate() {
        *id = match i % 5 {
            0 => -1,
            1 => 512,
            2 => i32::MAX,
            3 => 7,
            _ => 0,
        };
    }
    let table = vec![1u8; MAX_EXPERTS];
    let m = FixedUnion::model(&ids, 2, Some(&table), -1).unwrap();
    assert!(m.writes.iter().all(|&w| w == 1));
    assert!(!m.words.contains(&0xDEAD_BEEF));
    let clamped: Vec<u32> = ids
        .iter()
        .map(|&i| if (i as u32) < 512 { i as u32 } else { 0 })
        .collect();
    assert_eq!(m.words, reference_words(&clamped, &|_| 0));
}
