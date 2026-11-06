// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! §A9.5 host binding/lifecycle tests 6–8, plus the fixture
//! `window::tests` reuses for the submit-side checks.

use std::sync::Arc;

use super::*;
use crate::kernel::{
    author, ArgFact, ArgGuard, ConditionalFacts, Expr, FactStatus, IndexLoad, IndexedArgEffect,
    KernelFacts, KernelRef, LaunchRecord, MemClass, NodeFacts, ReadCache,
};
use crate::window::{CopyStep, FillStep, Region};

pub(crate) const H: u32 = 4;
pub(crate) const ROW: u64 = H as u64 * 258;
pub(crate) const ROWS_IN_SLICE: u64 = 1016;
pub(crate) const SLICE_BYTES: u64 = ROWS_IN_SLICE * ROW;
pub(crate) const DST_K: u64 = 0x7100_0000;
pub(crate) const DST_V: u64 = 0x7110_0000;
pub(crate) const DST_BYTES: u64 = 0x10_0000;
pub(crate) const SRC: u64 = 0x7200_0000;
pub(crate) const SRC_BYTES: u64 = 16 * H as u64 * 256 * 4;
pub(crate) const P_BASE: u64 = 0x7300_0000;
pub(crate) const P_BYTES: u64 = 256;
pub(crate) const P_GEN: u64 = 5;
pub(crate) const DST_GEN: u64 = 3;
pub(crate) const SEGMENT: u32 = 56;

pub(crate) fn p_identity() -> PositionsIdentity {
    PositionsIdentity {
        base: P_BASE,
        bytes: P_BYTES,
        generation: P_GEN,
    }
}

pub(crate) fn contiguous(first: i32) -> [i32; 16] {
    std::array::from_fn(|i| first + i as i32)
}

#[derive(Default)]
pub(crate) struct MockTransport {
    pub uploaded: Vec<[u8; 64]>,
    pub wait_failure: Option<Quiescence>,
    pub begin_failure: Option<Quiescence>,
    pub acquires: usize,
}

impl PositionsTransport for MockTransport {
    fn begin_h2d(&mut self, device: u64, bytes: &[u8; 64]) -> Result<(), TransportFailure> {
        assert_eq!(device, P_BASE);
        if let Some(q) = self.begin_failure {
            return Err(TransportFailure {
                message: "begin".into(),
                quiescence: q,
            });
        }
        self.uploaded.push(*bytes);
        Ok(())
    }

    fn wait_h2d(&mut self) -> Result<(), TransportFailure> {
        match self.wait_failure {
            Some(q) => Err(TransportFailure {
                message: "wait".into(),
                quiescence: q,
            }),
            None => Ok(()),
        }
    }

    fn acquire(&mut self) -> Result<(), TransportFailure> {
        self.acquires += 1;
        Ok(())
    }
}

/// A `Retired` positions allocation through `HostChecked -> UploadInFlight ->
/// SealedReady` for `positions`.
pub(crate) fn ready(positions: &[i32; 16]) -> SealedPositions {
    let sealed = SealedPositions::new(p_identity()).unwrap();
    seal(&sealed, positions);
    sealed
}

pub(crate) fn seal(sealed: &SealedPositions, positions: &[i32; 16]) {
    let mut transport = MockTransport::default();
    let stage = PositionsStage::new();
    stage.fill(positions).unwrap();
    let checked = sealed.host_check(positions, positions).unwrap();
    let token = sealed
        .begin_upload(checked, &stage, &mut transport)
        .unwrap();
    sealed.complete_upload(token, &mut transport).unwrap();
    assert_eq!(sealed.state(), PositionsState::SealedReady);
}

fn arg_facts() -> Vec<ArgFact> {
    let ptr = |offset, classes: &[MemClass]| ArgFact {
        offset,
        size: 8,
        role: ArgRole::Pointer {
            classes: classes.to_vec(),
            read_cache: ReadCache::VmemOnly,
        },
    };
    let int = |offset| ArgFact {
        offset,
        size: 4,
        role: ArgRole::Integer,
    };
    vec![
        ptr(0, &[]),
        ptr(8, &[MemClass::VmemLoad]),
        ptr(16, &[]),
        int(24),
        int(28),
        int(32),
        // Real receipt: guarded-null optional slots are integers, no access.
        ArgFact {
            offset: 40,
            size: 8,
            role: ArgRole::Integer,
        },
        ArgFact {
            offset: 48,
            size: 8,
            role: ArgRole::Integer,
        },
    ]
}

pub(crate) fn writer_facts() -> Arc<NodeFacts> {
    let effect = IndexedArgEffect {
        pc: 0x40,
        base_arg_offset: 0,
        mode: "write".into(),
        class: MemClass::VmemStore,
        path: "vmem".into(),
        cache: "th=0".into(),
        width: 4,
        address_expr: "dst + zext64(P_load)*R + head*256 + lane*8".into(),
        index_loads: vec![IndexLoad {
            pc: 0x20,
            source_arg_offset: 16,
            element: "i32".into(),
            offset_expr: "4*workgroup_id_y".into(),
            extension: "sext32".into(),
            class: MemClass::VmemLoad,
            path: "vmem".into(),
            cache: "th=0".into(),
            width: 4,
        }],
        guards: vec!["kernarg_eq@28:4=256".into()],
    };
    let guard = |kind: &str, arg_offset, size, value: &str| ArgGuard {
        kind: kind.into(),
        arg_offset,
        size,
        value: value.into(),
    };
    Arc::new(NodeFacts {
        kref: KernelRef {
            module: "kv".into(),
            symbol: "kv_cache_write_fp8_e4m3_batched".into(),
            artifact_sha256: "ab".repeat(32),
            kd_sha256: String::new(),
            receipt_sha256: "cd".repeat(32),
            tier: "D".into(),
            obligations: vec!["InBoundsNoAlias".into()],
        },
        facts: KernelFacts {
            status: FactStatus::Conditional(ConditionalFacts {
                guards: vec![
                    guard("kernarg_eq", 28, 4, "256"),
                    guard("kernarg_eq", 32, 4, "16"),
                    guard("kernarg_eq", 40, 8, "0"),
                    guard("kernarg_eq", 48, 8, "0"),
                    guard("block_dim", 0, 0, "32"),
                    guard("grid_dim", 1, 0, "16"),
                    guard("grid_dim", 2, 0, "1"),
                ],
                indexed: vec![effect],
                tables: Vec::new(),
                arenas: Vec::new(),
                task_protocol: None,
                witnesses: vec!["indexed_bind".into()],
                case_digest: "ef".repeat(32),
            }),
            explicit_bytes: SEGMENT,
            segment_bytes: SEGMENT,
            read_cache: ReadCache::VmemOnly,
            args: arg_facts(),
            hidden: Vec::new(),
            hidden_grid_live: Vec::new(),
            differential: "pass".into(),
            guard_flag: None,
        },
    })
}

pub(crate) fn writer_launch(dst: u64, dst_ptr: u64) -> LaunchRecord {
    let mut kernarg = vec![0u8; SEGMENT as usize];
    kernarg[0..8].copy_from_slice(&dst_ptr.to_le_bytes());
    kernarg[8..16].copy_from_slice(&SRC.to_le_bytes());
    kernarg[16..24].copy_from_slice(&P_BASE.to_le_bytes());
    kernarg[24..28].copy_from_slice(&H.to_le_bytes());
    kernarg[28..32].copy_from_slice(&256u32.to_le_bytes());
    kernarg[32..36].copy_from_slice(&16u32.to_le_bytes());
    LaunchRecord {
        symbol: "kv_cache_write_fp8_e4m3_batched".into(),
        grid: [H, 16, 1],
        block: [32, 1, 1],
        dynamic_lds: 0,
        kernarg,
        declared: Vec::new(),
        facts: Ok(writer_facts()),
        allocations: vec![
            (
                0,
                Ok(Binding {
                    base: dst,
                    bytes: DST_BYTES,
                }),
            ),
            (
                8,
                Ok(Binding {
                    base: SRC,
                    bytes: SRC_BYTES,
                }),
            ),
            (
                16,
                Ok(Binding {
                    base: P_BASE,
                    bytes: P_BYTES,
                }),
            ),
        ],
    }
}

/// A certified (`Proven`) node touching `target` of `bytes` with `classes`.
pub(crate) fn extra_launch(
    symbol: &str,
    target: u64,
    binding: Binding,
    classes: &[MemClass],
) -> LaunchRecord {
    let mut kernarg = vec![0u8; 8];
    kernarg.copy_from_slice(&target.to_le_bytes());
    let mut facts =
        (*crate::kernel::tests::facts(vec![crate::kernel::tests::ptr(0, classes)], 8)).clone();
    facts.kref.symbol = symbol.into();
    LaunchRecord {
        symbol: symbol.into(),
        grid: [1, 1, 1],
        block: [64, 1, 1],
        dynamic_lds: 0,
        kernarg,
        declared: Vec::new(),
        facts: Ok(Arc::new(facts)),
        allocations: vec![(0, Ok(binding))],
    }
}

pub(crate) struct Fixture {
    /// The program a plan is laid out against, assuming the witnesses.
    pub program: KernelProgram,
    pub bindings: Vec<Binding>,
    pub allocations: Vec<AllocationIdentity>,
    pub images: Vec<Vec<u8>>,
    pub grids: Vec<[u32; 3]>,
    pub steps: Vec<WindowStep>,
    pub p_res: ResourceId,
    pub extents: Vec<WriterExtent>,
}

impl Fixture {
    /// K and V writers (nodes 0 and 1, resources 0 dst-K, 1 src, 2 P,
    /// 3 dst-V) then `extra` launches.
    pub fn new(extra: Vec<LaunchRecord>) -> Self {
        Self::with_v_dst(DST_V, extra)
    }

    pub fn with_v_dst(v_dst: u64, extra: Vec<LaunchRecord>) -> Self {
        let mut launches = vec![writer_launch(DST_K, DST_K), writer_launch(v_dst, v_dst)];
        launches.extend(extra);
        let authored = author("gfx1201", &launches);
        for n in &authored.nodes[..2] {
            assert!(
                matches!(n.lowering, Lowering::Conditional { .. }),
                "{:?}",
                n.lowering
            );
        }
        let program = authored.assume_indexed_witness();
        let words = program.authored_words();
        let images = (0..program.nodes.len())
            .map(|i| {
                program
                    .kernarg_image(i, program.nodes[i].segment_bytes.unwrap() as usize, &words)
                    .unwrap()
            })
            .collect();
        let grids = program.nodes.iter().map(|n| n.grid).collect();
        let bindings = program.resources.clone();
        let allocations = bindings
            .iter()
            .enumerate()
            .map(|(i, b)| AllocationIdentity {
                resource: ResourceId(i as u32),
                base: b.base,
                bytes: b.bytes,
                generation: match b.base {
                    P_BASE => P_GEN,
                    DST_K | DST_V => DST_GEN,
                    _ => 1,
                },
            })
            .collect();
        let steps = (0..program.nodes.len())
            .map(|node| WindowStep::Kernel { node })
            .collect();
        let p_res = program
            .resources
            .iter()
            .position(|b| b.base == P_BASE && b.bytes == P_BYTES)
            .map(|i| ResourceId(i as u32))
            .unwrap();
        let extent = |node| WriterExtent {
            node,
            dst_slice: 0..SLICE_BYTES,
            mapped: vec![0..DST_BYTES],
            mapping_generation: DST_GEN,
        };
        Self {
            program,
            bindings,
            allocations,
            images,
            grids,
            steps,
            p_res,
            extents: vec![extent(0), extent(1)],
        }
    }

    pub fn bind(&self, sealed: &SealedPositions) -> Result<IndexedBindWitness, IndexBindError> {
        sealed.bind_writers(&BindInputs {
            window: 1,
            shape_generation: 7,
            program: &self.program,
            steps: &self.steps,
            bindings: &self.bindings,
            allocations: &self.allocations,
            images: &self.images,
            grids: &self.grids,
            positions_resource: self.p_res,
            writers: &self.extents,
        })
    }
}

fn expect_writer(r: Result<IndexedBindWitness, IndexBindError>, needle: &str) {
    match r {
        Err(IndexBindError::Writer { detail, .. }) => {
            assert!(detail.contains(needle), "{detail:?} lacks {needle:?}")
        }
        other => panic!("expected a writer refusal containing {needle:?}, got {other:?}"),
    }
}

fn set_u32(image: &mut [u8], at: usize, v: u32) {
    image[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

// ------------------------------------------------------------------ test 6

#[test]
fn rows_0_to_15_and_the_last_16_rows_of_a_mapped_slice_are_accepted() {
    for first in [0, ROWS_IN_SLICE as i32 - 16] {
        let sealed = ready(&contiguous(first));
        let fx = Fixture::new(Vec::new());
        let w = fx
            .bind(&sealed)
            .unwrap_or_else(|e| panic!("first={first}: {e}"));
        assert_eq!(w.writers().len(), 2);
        let last = w.writers()[0].rows[15];
        assert_eq!(last.1, DST_K + (first as u64 + 16) * ROW, "scale row end");
        assert_eq!(w.positions(), &contiguous(first));
    }
}

#[test]
fn host_check_rejects_negative_duplicate_truncated_and_foreign_positions() {
    let sealed = SealedPositions::new(p_identity()).unwrap();
    let mut neg = contiguous(0);
    neg[3] = -1;
    assert!(
        matches!(sealed.host_check(&neg, &neg), Err(IndexBindError::HostCheck(m)) if m.contains("negative"))
    );
    let mut dup = contiguous(0);
    dup[9] = dup[2];
    assert!(
        matches!(sealed.host_check(&dup, &dup), Err(IndexBindError::HostCheck(m)) if m.contains("twice"))
    );
    let short = contiguous(0);
    assert!(
        matches!(sealed.host_check(&short[..15], &short[..15]), Err(IndexBindError::HostCheck(m)) if m.contains("15 positions"))
    );
    // Not the producer's vector (an independently guessed/compacted one).
    let other = contiguous(1);
    assert!(
        matches!(sealed.host_check(&contiguous(0), &other), Err(IndexBindError::HostCheck(m)) if m.contains("producer"))
    );
    assert_eq!(
        sealed.state(),
        PositionsState::Retired,
        "a failed check changes nothing"
    );
    // Non-contiguous but unique is legal: uniqueness is checked, not assumed.
    let mut gap = contiguous(10);
    gap[5] = 400;
    sealed.host_check(&gap, &gap).unwrap();
}

#[test]
fn positions_allocation_needs_64_bytes() {
    let mut id = p_identity();
    id.bytes = 60;
    assert!(matches!(
        SealedPositions::new(id),
        Err(IndexBindError::Allocation(_))
    ));
    id.bytes = 64;
    SealedPositions::new(id).unwrap();
}

#[test]
fn capacity_position_and_scale_overflow_are_rejected() {
    // Row 1016 is one row past the tensor slice.
    let sealed = ready(&contiguous(ROWS_IN_SLICE as i32 - 15));
    expect_writer(
        Fixture::new(Vec::new()).bind(&sealed),
        "leaves the tensor slice",
    );
    // The last row's final scale byte: the slice ends one byte short.
    let sealed = ready(&contiguous(ROWS_IN_SLICE as i32 - 16));
    let mut fx = Fixture::new(Vec::new());
    fx.extents[0].dst_slice = 0..SLICE_BYTES - 1;
    expect_writer(fx.bind(&sealed), "leaves the tensor slice");
    // And the mapped interval ends one byte short.
    let mut fx = Fixture::new(Vec::new());
    fx.extents[1].mapped = vec![0..SLICE_BYTES - 1];
    expect_writer(fx.bind(&sealed), "mapped/writable");
}

#[test]
fn zero_or_overflowing_h_and_wrong_hd_b_and_dimensions_are_rejected() {
    let sealed = ready(&contiguous(0));
    let cases: [(&str, Box<dyn Fn(&mut Fixture)>); 8] = [
        (
            "not positive",
            Box::new(|fx| set_u32(&mut fx.images[0], 24, 0)),
        ),
        (
            "not positive",
            Box::new(|fx| set_u32(&mut fx.images[0], 24, 0x8000_0001)),
        ),
        (
            "exceed signed 32",
            Box::new(|fx| {
                set_u32(&mut fx.images[0], 24, 0x7fff_ffff);
                fx.grids[0] = [0x7fff_ffff, 16, 1];
            }),
        ),
        (
            "head_dim",
            Box::new(|fx| set_u32(&mut fx.images[1], 28, 128)),
        ),
        (
            "batch_size",
            Box::new(|fx| set_u32(&mut fx.images[1], 32, 8)),
        ),
        ("grid", Box::new(|fx| fx.grids[0] = [H, 8, 1])),
        (
            "block",
            Box::new(|fx| fx.program.nodes[1].block = [64, 1, 1]),
        ),
        ("slot_descs", Box::new(|fx| fx.images[0][40] = 1)),
    ];
    for (needle, mutate) in cases {
        let mut fx = Fixture::new(Vec::new());
        mutate(&mut fx);
        expect_writer(fx.bind(&sealed), needle);
    }
}

#[test]
fn wrong_dimensions_of_the_kv_slice_and_mapping_generation_changes_are_rejected() {
    let sealed = ready(&contiguous(0));
    let mut fx = Fixture::new(Vec::new());
    // A slice that reaches past the mapped extent does not make rows
    // writable: reserved address space is not mapped memory.
    fx.extents[0].dst_slice = 0..4 * DST_BYTES;
    fx.bind(&sealed).unwrap();
    let past_mapped = ready(&contiguous(ROWS_IN_SLICE as i32 - 15));
    expect_writer(fx.bind(&past_mapped), "mapped/writable");
    let mut fx = Fixture::new(Vec::new());
    fx.extents[0].dst_slice = 10..10;
    expect_writer(fx.bind(&sealed), "empty");
    let mut fx = Fixture::new(Vec::new());
    fx.extents[1].mapping_generation = DST_GEN + 1;
    expect_writer(fx.bind(&sealed), "mapping generation");
    // An unmapped hole inside rows 0..15.
    let mut fx = Fixture::new(Vec::new());
    fx.extents[0].mapped = vec![0..5 * ROW, 6 * ROW..DST_BYTES];
    expect_writer(fx.bind(&sealed), "mapped/writable");
    // A reserved-but-unmapped tail is not coverage.
    let mut fx = Fixture::new(Vec::new());
    fx.extents[0].mapped = vec![0..4 * ROW];
    expect_writer(fx.bind(&sealed), "mapped/writable");
    // The bound allocation itself changed generation.
    let mut fx = Fixture::new(Vec::new());
    fx.allocations[0].generation += 1;
    expect_writer(fx.bind(&sealed), "mapping generation");
    // A different positions mapping generation than the sealed one.
    let mut fx = Fixture::new(Vec::new());
    fx.allocations[fx.p_res.0 as usize].generation += 1;
    assert!(
        matches!(fx.bind(&sealed), Err(IndexBindError::Closure(m)) if m.contains("sealed positions"))
    );
}

#[test]
fn dst_interiors_and_cross_writer_overlap_are_checked() {
    // A nonzero dst interior shifts every row; the slice is relative to the base.
    let sealed = ready(&contiguous(0));
    let mut fx = Fixture::new(Vec::new());
    let interior = 3 * ROW;
    fx.images[0][0..8].copy_from_slice(&(DST_K + interior).to_le_bytes());
    fx.bind(&sealed).unwrap();
    fx.extents[0].dst_slice = 0..interior + 15 * ROW; // row 15 now ends past the slice
    expect_writer(fx.bind(&sealed), "leaves the tensor slice");
    // A misaligned interior breaks the 4-byte code stores.
    let mut fx = Fixture::new(Vec::new());
    fx.images[0][0..8].copy_from_slice(&(DST_K + 2).to_le_bytes());
    expect_writer(fx.bind(&sealed), "alignment");
    // K and V writing the same rows are refused.
    let fx = Fixture::with_v_dst(DST_K, Vec::new());
    expect_writer(fx.bind(&sealed), "overlap");
    // dst overlapping the source.
    let mut fx = Fixture::new(Vec::new());
    fx.images[0][8..16].copy_from_slice(&DST_K.to_le_bytes());
    expect_writer(fx.bind(&sealed), "outside its allocation");
}

// ------------------------------------------------------------------ test 7

#[test]
fn upload_needs_the_checked_payload_and_the_stage_is_frozen_in_flight() {
    let sealed = SealedPositions::new(p_identity()).unwrap();
    let mut transport = MockTransport::default();
    let checked_positions = contiguous(40);
    let stage = PositionsStage::new();
    // A different payload than the one that was checked.
    stage.fill(&contiguous(41)).unwrap();
    let checked = sealed
        .host_check(&checked_positions, &checked_positions)
        .unwrap();
    assert_eq!(
        sealed
            .begin_upload(checked, &stage, &mut transport)
            .unwrap_err(),
        IndexBindError::PayloadMismatch
    );
    assert_eq!(sealed.state(), PositionsState::Retired);
    assert!(transport.uploaded.is_empty(), "nothing reached the device");

    stage.fill(&checked_positions).unwrap();
    let checked = sealed
        .host_check(&checked_positions, &checked_positions)
        .unwrap();
    let token = sealed
        .begin_upload(checked, &stage, &mut transport)
        .unwrap();
    assert_eq!(sealed.state(), PositionsState::UploadInFlight);
    // The stage may not be reused before the H2D completion fence.
    assert_eq!(
        stage.fill(&contiguous(0)).unwrap_err(),
        IndexBindError::StageInFlight
    );
    assert_eq!(
        stage.write_byte(0, 9).unwrap_err(),
        IndexBindError::StageInFlight
    );
    // Submit-side binding before the upload completed is refused.
    let fx = Fixture::new(Vec::new());
    assert!(matches!(
        fx.bind(&sealed),
        Err(IndexBindError::State {
            found: PositionsState::UploadInFlight,
            ..
        })
    ));
    sealed.complete_upload(token, &mut transport).unwrap();
    assert!(!stage.is_in_flight());
    assert_eq!(transport.acquires, 1, "the acquire follows the fence");
    stage.fill(&contiguous(0)).unwrap();
    let mut expect = [0u8; 64];
    for (i, p) in checked_positions.iter().enumerate() {
        expect[i * 4..i * 4 + 4].copy_from_slice(&p.to_le_bytes());
    }
    assert_eq!(transport.uploaded, vec![expect]);
}

#[test]
fn a_host_check_token_from_another_generation_is_stale() {
    let sealed = SealedPositions::new(p_identity()).unwrap();
    let mut transport = MockTransport::default();
    let stage = PositionsStage::new();
    let a = contiguous(0);
    stage.fill(&a).unwrap();
    let first = sealed.host_check(&a, &a).unwrap();
    sealed.release_unsubmitted().unwrap();
    let _second = sealed.host_check(&a, &a).unwrap();
    assert!(matches!(
        sealed.begin_upload(first, &stage, &mut transport),
        Err(IndexBindError::Stale(_))
    ));
}

#[test]
fn stale_witnesses_and_post_bind_changes_are_rejected() {
    let sealed = ready(&contiguous(8));
    let fx = Fixture::new(Vec::new());
    let witness = fx.bind(&sealed).unwrap();
    witness.matches_window(1, 7, &fx.images, &fx.grids).unwrap();
    sealed.check_witness(&witness).unwrap();

    // Any post-bind kernarg or shape change.
    let mut changed = fx.images.clone();
    set_u32(&mut changed[0], 24, H + 1);
    assert!(matches!(
        witness.matches_window(1, 7, &changed, &fx.grids),
        Err(IndexBindError::Writer { detail, .. }) if detail.contains("changed after bind")
    ));
    let mut grids = fx.grids.clone();
    grids[1] = [H, 16, 2];
    assert!(matches!(
        witness.matches_window(1, 7, &fx.images, &grids),
        Err(IndexBindError::Writer { .. })
    ));
    assert!(matches!(
        witness.matches_window(1, 8, &fx.images, &fx.grids),
        Err(IndexBindError::Stale(_))
    ));
    assert!(matches!(
        witness.matches_window(2, 7, &fx.images, &fx.grids),
        Err(IndexBindError::Stale(_))
    ));

    // Submit consumes the sealed generation; a retired-and-resealed P is a
    // different generation, so last window's witness cannot be reused.
    sealed.mark_submitted(&witness).unwrap();
    assert_eq!(sealed.state(), PositionsState::SubmittedReadOnly);
    assert!(matches!(
        sealed.mark_submitted(&witness),
        Err(IndexBindError::State { .. })
    ));
    sealed.finish(Quiescence::Proven);
    assert_eq!(sealed.state(), PositionsState::Retired);
    seal(&sealed, &contiguous(8)); // the same bytes, a new generation
    assert!(matches!(
        sealed.check_witness(&witness),
        Err(IndexBindError::Stale(_))
    ));
    let fresh = fx.bind(&sealed).unwrap();
    assert_ne!(fresh.p_generation(), witness.p_generation());
    // A witness for another payload at the same generation count also fails.
    let other = ready(&contiguous(9));
    assert!(matches!(
        other.check_witness(&fresh),
        Err(IndexBindError::Stale(_))
    ));
}

#[test]
fn sealed_positions_refuse_overwrite_remap_recycle_and_external_writers() {
    let sealed = SealedPositions::new(p_identity()).unwrap();
    // Retired: those operations are legal.
    sealed.overwrite().unwrap();
    sealed.external_write_enqueued(0x1000, 16).unwrap(); // elsewhere: no effect
    seal(&sealed, &contiguous(0));
    for state_op in [
        sealed.overwrite().unwrap_err(),
        sealed.remap(0x7400_0000, 9).unwrap_err(),
        sealed.recycle().unwrap_err(),
        sealed.external_write_enqueued(P_BASE + 8, 4).unwrap_err(),
    ] {
        assert!(
            matches!(
                state_op,
                IndexBindError::Leased(PositionsState::SealedReady, _)
            ),
            "{state_op:?}"
        );
    }
    assert_eq!(sealed.identity(), p_identity(), "nothing moved");
    sealed.release_unsubmitted().unwrap();
    // Retired again: a remap must raise the generation.
    assert!(sealed.remap(0x7400_0000, P_GEN).is_err());
    sealed.remap(0x7400_0000, P_GEN + 1).unwrap();
    assert_eq!(sealed.identity().generation, P_GEN + 1);
}

// ------------------------------------------------------------------ test 8

fn closure_refused(fx: &Fixture, needle: &str) {
    let sealed = ready(&contiguous(0));
    match fx.bind(&sealed) {
        Err(IndexBindError::Closure(m)) => assert!(m.contains(needle), "{m:?} lacks {needle:?}"),
        other => panic!("expected a closure refusal ({needle}), got {other:?}"),
    }
}

fn p_binding() -> Binding {
    Binding {
        base: P_BASE,
        bytes: P_BYTES,
    }
}

#[test]
fn a_same_ib_store_alias_atomic_copy_or_unknown_node_fails_admission() {
    // Direct store to P.
    let fx = Fixture::new(vec![extra_launch(
        "poke",
        P_BASE,
        p_binding(),
        &[MemClass::VmemStore],
    )]);
    closure_refused(&fx, "writes the positions allocation");
    // Atomics (both kinds) to P.
    let fx = Fixture::new(vec![extra_launch(
        "atomic",
        P_BASE,
        p_binding(),
        &[MemClass::AtomicCommutative],
    )]);
    closure_refused(&fx, "writes the positions allocation");
    // A non-commutative atomic is not even PM4-lowered.
    let fx = Fixture::new(vec![extra_launch(
        "atomic",
        P_BASE,
        p_binding(),
        &[MemClass::Atomic],
    )]);
    closure_refused(&fx, "not PM4-lowered");
    // A store through another resource that overlaps P (an alias by address).
    let fx = Fixture::new(vec![extra_launch(
        "alias",
        P_BASE + 64,
        Binding {
            base: P_BASE + 64,
            bytes: 128,
        },
        &[MemClass::VmemStore],
    )]);
    closure_refused(&fx, "aliases");
    // Copy and fill destinations.
    let mut fx = Fixture::new(Vec::new());
    fx.steps.push(WindowStep::Copy(CopyStep {
        src: Region {
            resource: ResourceId(1),
            byte_offset: Expr::Const(0),
        },
        dst: Region {
            resource: fx.p_res,
            byte_offset: Expr::Const(0),
        },
        bytes: Expr::Const(64),
    }));
    closure_refused(&fx, "step 2 writes the positions");
    let mut fx = Fixture::new(Vec::new());
    fx.steps.push(WindowStep::Fill(FillStep {
        dst: Region {
            resource: fx.p_res,
            byte_offset: Expr::Const(0),
        },
        value: 0,
        words: Expr::Const(16),
    }));
    closure_refused(&fx, "step 2 writes the positions");
    // An unknown-effect node anywhere in the IB.
    let mut fx = Fixture::new(vec![extra_launch(
        "opaque",
        SRC,
        Binding {
            base: SRC,
            bytes: SRC_BYTES,
        },
        &[MemClass::VmemLoad],
    )]);
    fx.program.nodes[2].lowering = Lowering::HipDirect {
        reason: "A1 Unknown".into(),
    };
    closure_refused(&fx, "unknown-effect node");
    let mut fx = Fixture::new(vec![extra_launch(
        "opaque",
        SRC,
        Binding {
            base: SRC,
            bytes: SRC_BYTES,
        },
        &[MemClass::VmemLoad],
    )]);
    fx.program.nodes[2].kref = None;
    closure_refused(&fx, "no certified facts");
    // An unbound conditional node (witness not consumed) is not PM4.
    let mut fx = Fixture::new(Vec::new());
    fx.program.nodes[1].lowering = Lowering::Conditional { reason: "x".into() };
    closure_refused(&fx, "not PM4-lowered");
}

#[test]
fn a_pure_read_consumer_of_p_is_allowed() {
    let sealed = ready(&contiguous(0));
    let fx = Fixture::new(vec![extra_launch(
        "reader",
        P_BASE,
        p_binding(),
        &[MemClass::VmemLoad, MemClass::SmemLoad],
    )]);
    let w = fx.bind(&sealed).unwrap();
    assert_eq!(w.writers().len(), 2, "the reader is not a conditional node");
}

#[test]
fn a_separately_queued_writer_blocks_upload_and_admission() {
    let sealed = SealedPositions::new(p_identity()).unwrap();
    sealed.external_write_enqueued(P_BASE, 16).unwrap();
    let mut transport = MockTransport::default();
    let stage = PositionsStage::new();
    let p = contiguous(0);
    stage.fill(&p).unwrap();
    let checked = sealed.host_check(&p, &p).unwrap();
    assert!(matches!(
        sealed.begin_upload(checked, &stage, &mut transport),
        Err(IndexBindError::Closure(m)) if m.contains("separately queued")
    ));
    assert_eq!(sealed.state(), PositionsState::Retired);
    assert!(transport.uploaded.is_empty());
    sealed.external_write_completed();
    seal(&sealed, &p);
}

#[test]
fn completion_and_unknown_quiescence_release_or_quarantine_the_positions() {
    // Proven completion: SubmittedReadOnly -> Retired, reusable.
    let sealed = ready(&contiguous(0));
    let w = Fixture::new(Vec::new()).bind(&sealed).unwrap();
    sealed.mark_submitted(&w).unwrap();
    sealed.finish(Quiescence::Proven);
    assert_eq!(sealed.state(), PositionsState::Retired);
    sealed.recycle().unwrap();

    // Unknown quiescence after submit: quarantined, never writable.
    let sealed = ready(&contiguous(0));
    let w = Fixture::new(Vec::new()).bind(&sealed).unwrap();
    sealed.mark_submitted(&w).unwrap();
    sealed.finish(Quiescence::Unknown);
    assert_eq!(sealed.state(), PositionsState::Quarantined);
    assert!(matches!(
        sealed.recycle(),
        Err(IndexBindError::Leased(PositionsState::Quarantined, _))
    ));
    assert!(matches!(
        sealed.overwrite(),
        Err(IndexBindError::Leased(PositionsState::Quarantined, _))
    ));
    assert!(matches!(
        sealed.remap(0x7500_0000, 99),
        Err(IndexBindError::Leased(PositionsState::Quarantined, _))
    ));
    let p = contiguous(0);
    assert!(matches!(
        sealed.host_check(&p, &p),
        Err(IndexBindError::State {
            found: PositionsState::Quarantined,
            ..
        })
    ));
    assert!(matches!(
        sealed.release_unsubmitted(),
        Err(IndexBindError::State { .. })
    ));

    // A submit that did not queue: proven -> Retired, unknown -> Quarantined.
    let sealed = ready(&contiguous(0));
    sealed.submit_failed(Quiescence::Proven);
    assert_eq!(sealed.state(), PositionsState::Retired);
    let sealed = ready(&contiguous(0));
    sealed.submit_failed(Quiescence::Unknown);
    assert_eq!(sealed.state(), PositionsState::Quarantined);

    // H2D faults: proven retires and frees the stage; unknown quarantines.
    for (q, state, stage_free) in [
        (Quiescence::Proven, PositionsState::Retired, true),
        (Quiescence::Unknown, PositionsState::Quarantined, false),
    ] {
        let sealed = SealedPositions::new(p_identity()).unwrap();
        let mut transport = MockTransport {
            wait_failure: Some(q),
            ..Default::default()
        };
        let stage = PositionsStage::new();
        let p = contiguous(3);
        stage.fill(&p).unwrap();
        let checked = sealed.host_check(&p, &p).unwrap();
        let token = sealed
            .begin_upload(checked, &stage, &mut transport)
            .unwrap();
        assert!(matches!(
            sealed.complete_upload(token, &mut transport),
            Err(IndexBindError::Transport { quiescence, .. }) if quiescence == q
        ));
        assert_eq!(sealed.state(), state);
        assert_eq!(!stage.is_in_flight(), stage_free);
    }
    // A begin fault leaves nothing in flight.
    let sealed = SealedPositions::new(p_identity()).unwrap();
    let mut transport = MockTransport {
        begin_failure: Some(Quiescence::Unknown),
        ..Default::default()
    };
    let stage = PositionsStage::new();
    let p = contiguous(3);
    stage.fill(&p).unwrap();
    let checked = sealed.host_check(&p, &p).unwrap();
    assert!(sealed
        .begin_upload(checked, &stage, &mut transport)
        .is_err());
    assert_eq!(sealed.state(), PositionsState::Quarantined);
    assert!(!stage.is_in_flight());
}

#[test]
fn guarded_null_slots_are_checked_through_the_guard_not_pointer_args() {
    let sealed = ready(&contiguous(0));
    // Real-receipt shape (integer +40/+48, kernarg_eq guards) binds.
    Fixture::new(Vec::new()).bind(&sealed).unwrap();
    // Non-null still refuses.
    let mut fx = Fixture::new(Vec::new());
    fx.images[1][48] = 1;
    expect_writer(fx.bind(&sealed), "not null");
    // A receipt without the null guard cannot consume the case.
    let mut fx = Fixture::new(Vec::new());
    for n in &mut fx.program.nodes[..1] {
        n.indexed
            .as_mut()
            .unwrap()
            .guards
            .retain(|g| g.arg_offset != 40);
    }
    expect_writer(fx.bind(&sealed), "no null guard on +40");
    let mut fx = Fixture::new(Vec::new());
    fx.program.nodes[0]
        .indexed
        .as_mut()
        .unwrap()
        .guards
        .retain(|g| g.arg_offset != 32);
    expect_writer(fx.bind(&sealed), "no `== 16` guard");
}
