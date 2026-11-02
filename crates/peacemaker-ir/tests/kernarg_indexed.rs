// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Amendment 2026-10-25 §A9.5 fixtures 1 and 4 on the real gfx1201 writer
//! object `kv_cache_write_fp8_e4m3_batched` (an unmodified hipcc offload
//! bundle copied from the corpus pack): the guarded analysis of the original
//! object, and the guard sets that must not inherit its case.
use peacemaker_ir::inst::{Arch, Frontend, Kernel};
use peacemaker_ir::kernarg::{
    AccessClass, AccessMode, CacheBits, Guard, IndexElem, IndexExt, KernargFacts, KernelStatus,
    MemPath, ReadCacheClass,
};
use peacemaker_ir::passes::kernargs::{analyze, analyze_guarded};
use peacemaker_lift::{lift_object, Options};

const BATCHED: &[u8] = include_bytes!("fixtures/kv_cache_write_fp8_e4m3_batched.gfx1201.hsaco");
const SINGLE: &[u8] = include_bytes!("fixtures/kv_cache_write_fp8_e4m3.gfx1201.hsaco");

fn kernel(object: &[u8], symbol: &str) -> Kernel {
    let lifted = lift_object(
        object,
        Options {
            frontend: Frontend::Hipcc,
        },
    )
    .expect("lifts");
    lifted
        .program
        .kernels
        .into_iter()
        .find(|k| k.symbol.0 == symbol)
        .expect("kernel present")
}
fn writer() -> Kernel {
    kernel(BATCHED, "kv_cache_write_fp8_e4m3_batched")
}

/// The §A9.1 case: HD256, B16, null slot tables, grid [n_kv_heads,16,1], block [32,1,1].
fn case() -> Vec<Guard> {
    vec![
        Guard::KernargEq {
            offset: 28,
            size: 4,
            value: 256,
        },
        Guard::KernargEq {
            offset: 32,
            size: 4,
            value: 16,
        },
        Guard::KernargEq {
            offset: 40,
            size: 8,
            value: 0,
        },
        Guard::KernargEq {
            offset: 48,
            size: 8,
            value: 0,
        },
        Guard::GridDim { axis: 1, value: 16 },
        Guard::GridDim { axis: 2, value: 1 },
        Guard::BlockDim { axis: 0, value: 32 },
        Guard::BlockDim { axis: 1, value: 1 },
        Guard::BlockDim { axis: 2, value: 1 },
    ]
}
fn sorted(mut g: Vec<Guard>) -> Vec<Guard> {
    g.sort();
    g.dedup();
    g
}
fn rules(f: &KernargFacts) -> Vec<&'static str> {
    f.unresolved.iter().map(|u| u.rule).collect()
}
fn mode(f: &KernargFacts, offset: u32) -> Option<AccessMode> {
    f.arg_at(offset).and_then(|a| a.mode)
}

#[test]
fn the_unguarded_writer_is_still_unknown() {
    let f = analyze(&writer(), Arch::Gfx1201);
    assert_eq!(f.status, KernelStatus::Unknown);
    assert_eq!(rules(&f), ["base-memory-derived", "base-memory-derived"]);
    assert!(f.indexed.is_empty() && f.guards.is_empty());
    // Both optional tables are dereferenced somewhere on the unguarded CFG.
    assert!(!f.arg_at(40).unwrap().accesses.is_empty(), "slot_descs");
    assert!(!f.arg_at(48).unwrap().accesses.is_empty(), "row_slot");
}

#[test]
fn the_writer_object_is_conditional_with_exact_effects() {
    let k = writer();
    let f = analyze_guarded(&k, Arch::Gfx1201, &case());
    assert_eq!(f.status, KernelStatus::Conditional, "{:?}", f.unresolved);
    assert!(f.unresolved.is_empty());
    assert_eq!(f.guards, sorted(case()));

    // Two stores: the codes (eight code bytes per lane as one b64) and the
    // per-head f16 scale; both write `dst` allocation-wide.
    assert_eq!(f.indexed.len(), 2);
    let (codes, scale) = (&f.indexed[0], &f.indexed[1]);
    for e in [codes, scale] {
        assert_eq!(
            (e.base_arg_offset, e.mode, e.class, e.path),
            (
                0,
                AccessMode::Write,
                AccessClass::VmemStore,
                MemPath::Global
            )
        );
        assert_eq!(e.cache, CacheBits::default());
        assert_eq!(e.guards, sorted(case()));
        assert_eq!(e.index_loads.len(), 1, "one P load feeds both stores");
        let il = &e.index_loads[0];
        assert_eq!(
            (il.src_arg_offset, il.elem, il.class, il.path, il.cache),
            (
                16,
                IndexElem::I32,
                AccessClass::SmemLoad,
                MemPath::Smem,
                CacheBits::default()
            )
        );
        assert_eq!(
            il.byte_offset.to_string(),
            "(shl (sext wg_yz) 2)",
            "4 * bid"
        );
        assert_eq!(il.ext, [IndexExt::Sext64]);
        assert_eq!(e.index_loads[0], codes.index_loads[0]);
    }
    assert_eq!((codes.width_bytes, scale.width_bytes), (8, 2));
    assert!(codes.pc < scale.pc);
    // dst + sext(P)*R + head*256 + lane*8 with R = H*256 + H*2, as observed.
    assert_eq!(
        codes.displacement.to_string(),
        "(add (add (mul c64:256 (sext wg_x)) (mul (sext load:0) (sext (add (mul karg32:24 c32:256) (shl karg32:24 1))))) (zext (shl wi 3)))"
    );
    println!("codes : {}", codes.displacement);
    println!("scale : {}", scale.displacement);

    // Effects: dst write, src read, P read (scalar cache); no other pointer.
    assert_eq!(mode(&f, 0), Some(AccessMode::Write));
    assert_eq!(mode(&f, 8), Some(AccessMode::Read));
    assert_eq!(mode(&f, 16), Some(AccessMode::Read));
    assert_eq!(f.arg_at(16).unwrap().read_cache, ReadCacheClass::ScalarOnly);
    assert_eq!(f.arg_at(8).unwrap().read_cache, ReadCacheClass::VmemOnly);
    // Branch analysis: under the null guards neither optional table is ever
    // dereferenced (the excluded loads are unreachable), and a guarded null
    // is a constant, not an allocation at address 0.
    for table in [40, 48] {
        let a = f.arg_at(table).unwrap();
        assert!(
            a.accesses.is_empty() && a.mode.is_none(),
            "table {table} untouched"
        );
    }
    assert!(f.side.kernarg_segment == 0 && !f.dynamic_kernarg_reads);
}

#[test]
fn changed_or_missing_guards_cannot_inherit_the_case() {
    let k = writer();
    let with = |edit: &dyn Fn(&mut Vec<Guard>)| {
        let mut g = case();
        edit(&mut g);
        analyze_guarded(&k, Arch::Gfx1201, &g)
    };
    // A non-null slot table or row map makes the descriptor path live: the
    // loaded descriptor base reaches the store address and stays Unknown.
    let f = with(&|g| {
        g[2] = Guard::KernargEq {
            offset: 40,
            size: 8,
            value: 0x1000,
        }
    });
    assert_eq!(f.status, KernelStatus::Unknown, "{:?}", f.unresolved);
    let f = with(&|g| {
        g[3] = Guard::KernargEq {
            offset: 48,
            size: 8,
            value: 0x1000,
        }
    });
    assert_eq!(f.status, KernelStatus::Unknown, "{:?}", f.unresolved);
    // Dropping the slot_descs null guard leaves the descriptor load reachable.
    let f = with(&|g| g.retain(|x| !matches!(x, Guard::KernargEq { offset: 40, .. })));
    assert_eq!(f.status, KernelStatus::Unknown, "{:?}", f.unresolved);
    // Dropping only the row_slot guard is not a way to keep the case: with
    // slot_descs null the slot is never used, but `row_slot[bid]` is now an
    // ordinary read the facts must (and do) report, under a different guard set.
    let f = with(&|g| g.retain(|x| !matches!(x, Guard::KernargEq { offset: 48, .. })));
    assert_eq!(f.status, KernelStatus::Conditional, "{:?}", f.unresolved);
    assert_eq!(mode(&f, 48), Some(AccessMode::Read));
    assert_ne!(f.guards, sorted(case()));
    // Wrong grid/block: the packed entry registers are not pinned.
    let f = with(&|g| g[7] = Guard::BlockDim { axis: 1, value: 2 });
    assert_eq!(f.status, KernelStatus::Unknown);
    assert!(
        rules(&f).contains(&"indexed-entry-guard"),
        "{:?}",
        f.unresolved
    );
    let f = with(&|g| g[5] = Guard::GridDim { axis: 2, value: 2 });
    assert_eq!(f.status, KernelStatus::Unknown);
    assert!(
        rules(&f).contains(&"indexed-entry-guard"),
        "{:?}",
        f.unresolved
    );
    let f = with(&|g| g.retain(|x| !matches!(x, Guard::BlockDim { .. } | Guard::GridDim { .. })));
    assert_eq!(f.status, KernelStatus::Unknown);
    // Unsupported guard shapes refuse instead of being ignored.
    let f = with(&|g| {
        g.push(Guard::KernargEq {
            offset: 30,
            size: 4,
            value: 1,
        })
    });
    assert_eq!(f.status, KernelStatus::Unknown);
    assert!(rules(&f).contains(&"guard-unsupported"));

    // Negative / oversized H, B, HD: the analysis follows its own constants,
    // so the facts carry the changed guards and a different expression; they
    // can never equal the HD256/B16 case.
    let base = analyze_guarded(&k, Arch::Gfx1201, &case());
    for (idx, g) in [
        (
            0,
            Guard::KernargEq {
                offset: 28,
                size: 4,
                value: 128,
            },
        ),
        (
            0,
            Guard::KernargEq {
                offset: 28,
                size: 4,
                value: 0xffff_ff00,
            },
        ),
        (
            1,
            Guard::KernargEq {
                offset: 32,
                size: 4,
                value: 17,
            },
        ),
        (
            1,
            Guard::KernargEq {
                offset: 32,
                size: 4,
                value: 0xffff_ffff,
            },
        ),
    ] {
        let f = with(&|c| c[idx] = g);
        assert_ne!(f.guards, base.guards, "{g:?}");
        if f.status == KernelStatus::Conditional {
            assert!(f.indexed.iter().all(|e| e.guards == f.guards));
        }
        if matches!(g, Guard::KernargEq { offset: 28, .. }) {
            assert_ne!(f.indexed, base.indexed, "HD enters the expression: {g:?}");
        }
    }
}

#[test]
fn the_single_position_writer_object_is_conditional_under_block_guards_only() {
    // kv_cache_write_fp8_e4m3 (same object family: positions at +16 read as
    // *P, no slot tables, no bid). The legacy lattice drops the loaded
    // position through its sign extension and multiplies, so the unguarded
    // analysis calls it Proven; the guarded analysis reports the same two
    // stores as indexed effects and needs the block shape to read the lane term.
    let k = kernel(SINGLE, "kv_cache_write_fp8_e4m3");
    let f = analyze(&k, Arch::Gfx1201);
    assert_eq!(f.status, KernelStatus::Proven);
    let guards = [
        Guard::BlockDim { axis: 1, value: 1 },
        Guard::BlockDim { axis: 2, value: 1 },
    ];
    let g = analyze_guarded(&k, Arch::Gfx1201, &guards);
    assert_eq!(g.status, KernelStatus::Conditional, "{:?}", g.unresolved);
    assert_eq!(g.indexed.len(), 2);
    assert_eq!(
        g.indexed.iter().map(|e| e.width_bytes).collect::<Vec<_>>(),
        [8, 2]
    );
    for e in &g.indexed {
        assert_eq!((e.base_arg_offset, e.mode), (0, AccessMode::Write));
        assert_eq!(e.index_loads[0].src_arg_offset, 16);
        assert_eq!(e.index_loads[0].byte_offset.to_string(), "c64:0");
    }
    let none = analyze_guarded(&k, Arch::Gfx1201, &[]);
    assert_eq!(
        none.status,
        KernelStatus::Unknown,
        "lane term needs the block guards"
    );
    assert_eq!(rules(&none), ["indexed-entry-guard", "indexed-entry-guard"]);
}
