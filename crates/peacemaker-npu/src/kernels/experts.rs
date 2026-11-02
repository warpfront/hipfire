// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Grouped V9 experts: ONE TXN that runs `E` independent `m_e x n x k` GEMMs (MoE experts sharing `n`, `k`) on one PDI.
//!
//! # Shape
//! Every expert has `1 <= m_e <= WAVE_M` (512) rows. A V9 design pads M to the wave (`WAVE_M`), so every `m_e` maps to
//! the identical image: one M-wave, `NW = ceil(n / 512)` N-waves, the same core program, memtile descriptors, shim BDs
//! and instruction body. The group therefore builds ONE padded design, `design_v9(WAVE_M, n, k, Int8 { shift: 12 },
//! Control::Fast)`, and `m_e` only changes host packing: `pack_in` zero pads A below row `m_e`, `unpack_out` and
//! `reference` keep the first `m_e` rows of the padded result (rows are independent, so the kept rows equal the CPU
//! GEMM of the unpadded operands). Packed byte counts (`a_bytes` / `b_bytes` / `c_bytes`) are expert independent.
//!
//! # Command stream
//! `[ring poll + drain]` `body_0 (FULL)` `body_1..body_{E-1} (LEAN)` `[ring done]`. Expert `e`'s body is appended with
//! the arena offsets `[a_off, b_off, c_off]`, so every shim DDR patch of args 0 / 1 / 2 is the vendor patch plus the
//! expert's offset (the same mechanism as `ring::SlotPlan`). Body 0 is [`ArrayDesign::append_run_body`] and is valid
//! right after the PDI load and after any completed submit; bodies `1..` are [`ArrayDesign::append_lean_run_body`] with
//! no forced repairs: each directly follows a COMPLETED body of the same design (its C token SYNCs returned) and
//! nothing but the design's own run has touched the array since, which is exactly the lean contract
//! (`gemm_array_lean.rs`). Without a ring, nothing sits between bodies. With a ring, the poll / drain / done tasks
//! surround the WHOLE group: they run once before body 0 and once after body `E-1`, so the lean bodies need none of
//! the forced col-0 S2MM4 / MM2S0 repairs of `ring::persistent_v9_lean`.
//!
//! # Ring
//! The ring protocol is not re-encoded. [`ring::empty_persistent`] with one run encodes prologue + POLL + DRAIN + DONE
//! (the protocol depends only on spare memtile BDs derived at the maximal topology, never on the design that supplies
//! the body). The group stream is that stream with the body payload inserted between the DRAIN `MASKPOLL` and the first
//! op of the DONE block. The insertion offset is derived from the encoder's own patch sites (`PollSeq(0)`,
//! `DoneSeq(0)` and the fixed `MASKPOLL` / `WRITE32` / `MASKWRITE` op sizes), cross-checked by the opcodes and the
//! sentinel value at the cut, and the build panics if `ring.rs` ever changes the protocol shape. The header op count
//! and size are rewritten, `PollSeq(0)` stays and `DoneSeq(0)` moves by the inserted words; both remain valid input of
//! [`ring::patch_seq`]. Exactly one slot line is polled and one done line (`slot`) is written for the whole group, with
//! `seq0`. The ring PDI is the group's own PDI (the prototype's PDI is discarded). `args = [A, B, C, ring]`.
//!
//! # Footprint (bytes of `insts`)
//! With `F` = payload bytes of the standalone full TXN (`design_v9(..).insts.len() - 16`) and `L` = payload bytes of
//! the standalone lean TXN (`lean_insts().len() - 16`):
//! * no ring: `16 + F + (E - 1) * L`;
//! * ring: `1252 + F + (E - 1) * L` (396 B header/prologue + 856 B POLL + DRAIN + DONE, the measured one-run empty ring).
//!
//! `F` and `L` are independent of every `m_e`. `L` follows the lean parity model of `gemm_array_lean.rs` (`MW = 1`,
//! `W = waves = NW`, `J = kc * W`): `7776 + [J odd] * 9920 + [NW odd] * 8 * 212 + [W odd] * 4608`. `F` is the measured
//! length of the encoder; `grouped_v9(..).insts.len()` is the direct measurement.
use super::gemm_array::{self, ArrayDesign, V8_DEFAULT_EPILOGUE, WAVE_M};
use super::gemm_core::Control;
use super::gemm_i8::{self, ArgKind, ArgSpec};
use super::ring::{self, PatchKind, RingLayout, SlotPlan};
use crate::txn::Txn;

/// One expert of a group: `m` rows and the byte offsets of its operands / result inside args 0 / 1 / 2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertPlan {
    pub m: usize,
    pub a_off: u64,
    pub b_off: u64,
    pub c_off: u64,
}

/// Built grouped image + command stream; no NPU access occurs here.
pub struct GroupedDesign {
    pub pdi: Vec<u8>,
    pub insts: Vec<u8>,
    /// `[A arena, B arena, C arena]`, plus the ring BO (`In`) as arg 3 when built with a ring. Arena sizes are
    /// `max offset + one expert's packed bytes`.
    pub args: Vec<ArgSpec>,
    /// `(u32 word index in insts, kind)`: `PollSeq(0)` and `DoneSeq(0)` with a ring, empty without.
    pub patch_sites: Vec<(usize, PatchKind)>,
    design: ArrayDesign,
    /// Rows of the design's one M-wave: every expert's `m` is zero-padded to it.
    wave_m: usize,
    ms: Vec<usize>,
    n: usize,
    k: usize,
}

/// Ops of the ring protocol that the insertion offset is derived from (see the module header).
const MASKWRITE_WORDS: usize = 7;
const MASKPOLL_WORDS: usize = 7;
const WRITE32_WORDS: usize = 6;
const OP_MASKWRITE: u32 = 3;
const OP_MASKPOLL: u32 = 4;
const HEADER_WORDS: usize = Txn::HEADER_WORDS;

/// Group of V9 experts: expert 0 FULL body, experts `1..` LEAN bodies, optional ring `(layout, slot, seq0)` once around
/// the whole group. Panics on an empty group, `m` outside `1..=512`, non word aligned offsets or an invalid ring.
pub fn grouped_v9(
    n: usize,
    k: usize,
    experts: &[ExpertPlan],
    ring: Option<(RingLayout, usize, u32)>,
) -> GroupedDesign {
    grouped(
        gemm_array::design_v9(WAVE_M, n, k, V8_DEFAULT_EPILOGUE, Control::Fast),
        WAVE_M,
        n,
        k,
        experts,
        ring,
    )
}

/// Design-generic group (e.g. G80): `design` must be built for exactly `wave_m` padded rows, shape `wave_m x n x k`
/// (its packed A for `wave_m` rows is the whole arg 0), and provide `append_run_body` / `append_lean_run_body`. That
/// block may be one design M-wave (`wave_m()`) or several: npu-experts passes `design_g80(round_up(max m, 256), ..)`
/// with `wave_m = waves() * wave_m()` (one or two G80 waves; sim gates `g80_grouped_*two_waves*`). Same contract as
/// [`grouped_v9`] otherwise; no variant check is made here.
pub fn grouped(
    design: ArrayDesign,
    wave_m: usize,
    n: usize,
    k: usize,
    experts: &[ExpertPlan],
    ring: Option<(RingLayout, usize, u32)>,
) -> GroupedDesign {
    build(design, wave_m, n, k, experts, ring, true)
}

/// Simulator oracle, diagnostic only: [`grouped_v9`] with the FULL vendor body for every expert. It is valid on any
/// state (no lean argument) and exists so tests can prove the lean bodies leave outputs and retained state unchanged.
/// It makes no size or performance claim and is not a production entry point.
#[doc(hidden)]
pub fn grouped_v9_all_full(
    n: usize,
    k: usize,
    experts: &[ExpertPlan],
    ring: Option<(RingLayout, usize, u32)>,
) -> GroupedDesign {
    build(
        gemm_array::design_v9(WAVE_M, n, k, V8_DEFAULT_EPILOGUE, Control::Fast),
        WAVE_M,
        n,
        k,
        experts,
        ring,
        false,
    )
}

/// Design-generic all-full oracle (see [`grouped_v9_all_full`]).
#[doc(hidden)]
pub fn grouped_all_full(
    design: ArrayDesign,
    wave_m: usize,
    n: usize,
    k: usize,
    experts: &[ExpertPlan],
    ring: Option<(RingLayout, usize, u32)>,
) -> GroupedDesign {
    build(design, wave_m, n, k, experts, ring, false)
}

fn word(bytes: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap())
}

fn build(
    mut design: ArrayDesign,
    wave_m: usize,
    n: usize,
    k: usize,
    experts: &[ExpertPlan],
    ring: Option<(RingLayout, usize, u32)>,
    lean: bool,
) -> GroupedDesign {
    assert!(
        !experts.is_empty(),
        "a grouped submission needs at least one expert"
    );
    for (e, x) in experts.iter().enumerate() {
        assert!(
            (1..=wave_m).contains(&x.m),
            "expert {e}: m {} outside 1..={wave_m}",
            x.m
        );
        assert!(
            x.a_off % 4 == 0 && x.b_off % 4 == 0 && x.c_off % 4 == 0,
            "expert {e}: arena offsets must be word aligned"
        );
    }
    // One M-wave: the design's packed A for `wave_m` rows must be exactly one wave's A (pack_in panics otherwise).
    assert_eq!(
        design.pack_in(&vec![0i8; wave_m * k], &vec![0i8; k * n])[0].len(),
        design.args[0].bytes,
        "design is not a {wave_m}-row single-wave {wave_m}x{n}x{k} design"
    );

    let mut body = Txn::aie2p_8col();
    for (e, x) in experts.iter().enumerate() {
        let arena = [x.a_off, x.b_off, x.c_off];
        if lean && e > 0 {
            design
                .append_lean_run_body(&mut body, arena, &[])
                .unwrap_or_else(|err| panic!("lean body of expert {e}: {err}"));
        } else {
            design.append_run_body(&mut body, arena);
        }
    }
    let body_bytes = body.to_bytes();
    let payload = &body_bytes[4 * HEADER_WORDS..];

    let arena = |bytes: usize, off: fn(&ExpertPlan) -> u64| {
        let hi = experts
            .iter()
            .map(|x| usize::try_from(off(x)).expect("arena offset"))
            .max()
            .unwrap();
        hi.checked_add(bytes).expect("arena size")
    };
    let [a, b, c] = [0, 1, 2].map(|i| design.args[i].bytes);
    let mut args = vec![
        ArgSpec {
            bytes: arena(a, |x| x.a_off),
            kind: ArgKind::In,
        },
        ArgSpec {
            bytes: arena(b, |x| x.b_off),
            kind: ArgKind::In,
        },
        ArgSpec {
            bytes: arena(c, |x| x.c_off),
            kind: ArgKind::Out,
        },
    ];

    let (insts, patch_sites) = match ring {
        None => (body_bytes, Vec::new()),
        Some((layout, slot, seq0)) => {
            args.push(ArgSpec {
                bytes: layout.bytes(),
                kind: ArgKind::In,
            });
            wrap_in_ring(layout, slot, seq0, payload, body.op_count())
        }
    };
    GroupedDesign {
        pdi: std::mem::take(&mut design.pdi),
        insts,
        args,
        patch_sites,
        design,
        wave_m,
        ms: experts.iter().map(|x| x.m).collect(),
        n,
        k,
    }
}

/// The one-run empty ring stream with `payload` (`ops` ops) inserted between its DRAIN and DONE blocks.
fn wrap_in_ring(
    layout: RingLayout,
    slot: usize,
    seq0: u32,
    payload: &[u8],
    ops: usize,
) -> (Vec<u8>, Vec<(usize, PatchKind)>) {
    let proto = ring::empty_persistent(
        layout,
        &[SlotPlan {
            slot,
            a_off: 0,
            b_off: 0,
            c_off: 0,
        }],
        seq0,
    );
    let site = |want: PatchKind| {
        let mut hits = proto.patch_sites.iter().filter(|&&(_, kind)| kind == want);
        let word = hits
            .next()
            .unwrap_or_else(|| panic!("ring protocol lost its {want:?} site"))
            .0;
        assert!(
            hits.next().is_none() && proto.patch_sites.len() == 2,
            "ring protocol patch inventory changed"
        );
        word
    };
    let (poll_value, done_value) = (site(PatchKind::PollSeq(0)), site(PatchKind::DoneSeq(0)));
    let p = &proto.insts;
    assert_eq!(word(p, 3) as usize, p.len(), "ring prototype header size");
    // POLL MASKPOLL, DRAIN WRITE32 + MASKPOLL (sentinel), then DONE: reset_release (2 MASKWRITE), 16 WRITE32 ...
    let poll_op = poll_value - Txn::MASKPOLL_VALUE_WORD;
    let at = poll_op + MASKPOLL_WORDS + WRITE32_WORDS + MASKPOLL_WORDS;
    let done_op = done_value - Txn::WRITE32_VALUE_WORD;
    assert_eq!(
        at + 2 * MASKWRITE_WORDS,
        done_op,
        "ring protocol changed: DONE block does not follow the DRAIN poll"
    );
    let drain_poll = at - MASKPOLL_WORDS;
    assert!(
        word(p, poll_op) == OP_MASKPOLL
            && word(p, poll_op + Txn::MASKPOLL_VALUE_WORD) == seq0
            && word(p, drain_poll) == OP_MASKPOLL
            && word(p, drain_poll + Txn::MASKPOLL_VALUE_WORD) == ring::SENTINEL
            && word(p, at) == OP_MASKWRITE
            && word(p, at + MASKWRITE_WORDS) == OP_MASKWRITE,
        "ring protocol changed: cut point {at} is not between the DRAIN poll and the DONE block"
    );

    let mut out = Vec::with_capacity(p.len() + payload.len());
    out.extend_from_slice(&p[..4 * at]);
    out.extend_from_slice(payload);
    out.extend_from_slice(&p[4 * at..]);
    let (op_count, size) = (word(p, 2) + ops as u32, out.len() as u32);
    out[8..12].copy_from_slice(&op_count.to_le_bytes());
    out[12..16].copy_from_slice(&size.to_le_bytes());
    let sites = vec![
        (poll_value, PatchKind::PollSeq(0)),
        (done_value + payload.len() / 4, PatchKind::DoneSeq(0)),
    ];
    (out, sites)
}

impl GroupedDesign {
    pub fn experts(&self) -> usize {
        self.ms.len()
    }
    fn m(&self, expert: usize) -> usize {
        *self
            .ms
            .get(expert)
            .unwrap_or_else(|| panic!("expert {expert} >= {} experts", self.ms.len()))
    }
    /// Packed `[A, B]` of ONE expert (copy to arg0 at `a_off` / arg1 at `b_off`); `a` is row-major `m*k` (zero padded to
    /// the wave), `b` row-major `k*n`.
    pub fn pack_in(&self, expert: usize, a: &[i8], b: &[i8]) -> [Vec<u8>; 2] {
        let m = self.m(expert);
        assert_eq!(a.len(), m * self.k, "expert {expert}: A must be m*k");
        if m == self.wave_m {
            return self.design.pack_in(a, b);
        }
        let mut padded = vec![0i8; self.wave_m * self.k];
        padded[..a.len()].copy_from_slice(a);
        self.design.pack_in(&padded, b)
    }
    /// Row-major `m*n` result of ONE expert from its `c_bytes(expert)` slice of arg2 (starting at `c_off`).
    pub fn unpack_out(&self, expert: usize, c: &[u8]) -> Vec<i32> {
        let m = self.m(expert);
        let mut out = self.design.unpack_out(c);
        out.truncate(m * self.n);
        out
    }
    /// Exact CPU result in the form [`GroupedDesign::unpack_out`] returns (int8 epilogue applied).
    pub fn reference(&self, expert: usize, a: &[i8], b: &[i8]) -> Vec<i32> {
        let m = self.m(expert);
        let mut c = gemm_i8::cpu_reference(a, b, m, self.n, self.k);
        let epi = self.design.epilogue().unwrap_or(V8_DEFAULT_EPILOGUE);
        for x in &mut c {
            *x = gemm_array::apply_epilogue(*x, epi);
        }
        c
    }
    /// Packed bytes of ONE expert in args 0 / 1 / 2 (identical for every expert: M is padded to one wave).
    pub fn a_bytes(&self, expert: usize) -> usize {
        self.m(expert);
        self.design.args[0].bytes
    }
    pub fn b_bytes(&self, expert: usize) -> usize {
        self.m(expert);
        self.design.args[1].bytes
    }
    pub fn c_bytes(&self, expert: usize) -> usize {
        self.m(expert);
        self.design.args[2].bytes
    }
}
