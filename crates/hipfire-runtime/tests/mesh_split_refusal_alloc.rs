// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Admission refusals of `split_program` must be allocation-free.
//!
//! This binary installs a counting `#[global_allocator]` and runs every case
//! sequentially inside ONE `#[test]`. Inputs are built (allocating freely)
//! through `WeightEntry::layer` / `plan_manifest` and the `StepGraph` builder;
//! the counter is then armed immediately around the `split_program` call only.
//!
//! Counting is armed per thread (const-initialised `Cell<bool>`, no lazy
//! allocation) so unrelated libtest threads cannot perturb the count; the
//! tally itself is a process-global `AtomicUsize`.

use hipfire_dispatch::pipeline::gated_mlp::MlpLadder;
use hipfire_dispatch::pipeline::step_graph::{
    OpDecl, OpKind, ResourceClass, ResourceDecl, ResourceId, SplitRule, StepGraph, StepGraphError,
};
use hipfire_runtime::device_mesh::{DeviceMesh, DimKind};
use hipfire_runtime::mesh_placement::BandPlacement;
use hipfire_runtime::mesh_split::{split_program, SplitError, SplitPlan};
use hipfire_runtime::weight_manifest::{plan_manifest, ManifestPlan, ShardPolicy, WeightEntry};
use rdna_compute::DType;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static ARMED: Cell<bool> = const { Cell::new(false) };
}

struct CountingAlloc;

#[inline]
fn tally() {
    if ARMED.try_with(Cell::get).unwrap_or(false) {
        ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
    }
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        tally();
        System.alloc(layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        tally();
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        tally();
        System.realloc(ptr, layout, new_size)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

/// Run `f` with counting armed on this thread; return its result and the
/// number of allocator calls observed during `f`.
fn counted<T>(f: impl FnOnce() -> T) -> (T, usize) {
    ALLOCATIONS.store(0, Ordering::SeqCst);
    ARMED.with(|armed| armed.set(true));
    let out = f();
    ARMED.with(|armed| armed.set(false));
    (out, ALLOCATIONS.load(Ordering::SeqCst))
}

/// Call `split_program`, requiring `Err` and zero allocator calls.
fn refusal(case: &str, graph: &StepGraph, plan: &ManifestPlan, mesh: &DeviceMesh) -> SplitError {
    let (result, allocations) =
        counted(|| split_program(black_box(graph), black_box(plan), black_box(mesh)));
    assert_eq!(
        allocations, 0,
        "{case}: refusal path allocated {allocations} time(s)"
    );
    match result {
        Err(error) => error,
        Ok(_) => panic!("{case}: expected a refusal, got Ok(SplitPlan)"),
    }
}

/// One `Project` over a single manifest weight: `output = W · input`.
/// `shape` is the weight's logical shape `[out, in]`; `group` is the packed
/// block size recorded on the graph's weight resource.
fn project_graph(
    name: &str,
    layer: usize,
    shape: [usize; 2],
    group: usize,
) -> (StepGraph, ResourceId) {
    let mut graph = StepGraph::new();
    let input = graph
        .add_resource(ResourceDecl {
            class: ResourceClass::Input,
            shape: vec![shape[1]],
            alias: None,
        })
        .unwrap();
    let weight = graph
        .add_resource(ResourceDecl {
            class: ResourceClass::Weight {
                name: name.to_string(),
                layer: Some(layer),
                group,
            },
            shape: shape.to_vec(),
            alias: None,
        })
        .unwrap();
    let output = graph
        .add_resource(ResourceDecl {
            class: ResourceClass::Activation,
            shape: vec![shape[0]],
            alias: None,
        })
        .unwrap();
    graph
        .add_op(OpDecl {
            kind: OpKind::Project,
            layer: Some(layer),
            rule: SplitRule::Project {
                weight,
                input,
                output,
            },
            reads: vec![weight, input],
            writes: vec![output],
        })
        .unwrap();
    (graph, weight)
}

fn expect_plan(entries: &[WeightEntry], mesh: &DeviceMesh, n_layers: usize) -> ManifestPlan {
    let placement =
        BandPlacement::uniform(mesh, n_layers).expect("uniform placement on its own mesh");
    plan_manifest(entries, &[], mesh, n_layers, &placement).expect("manifest plans on its own mesh")
}

#[test]
fn split_program_refuses_without_allocating() {
    // Counter self-check: an allocation inside `counted` must be observed,
    // otherwise every zero below would be vacuous.
    let (_, observed) = counted(|| black_box(Box::new(black_box(7u64))));
    assert!(observed >= 1, "counting allocator did not observe a Box");

    // 1. Epoch mismatch: plan compiled on one rect, split on a second
    //    same-shaped rect (a different mesh generation).
    {
        let mesh_plan = DeviceMesh::rect(&[(DimKind::Tp, 2)]).unwrap();
        let mesh_other = DeviceMesh::rect(&[(DimKind::Tp, 2)]).unwrap();
        assert_ne!(
            mesh_plan.epoch(),
            mesh_other.epoch(),
            "same-shaped rects must still be distinct generations"
        );
        let entries = [WeightEntry::layer(
            "wq",
            0,
            vec![4096, 4096],
            DType::F16,
            ShardPolicy::ColumnShard { axis: 0 },
        )];
        let plan = expect_plan(&entries, &mesh_plan, 1);
        let (graph, _) = project_graph("wq", 0, [4096, 4096], 1);
        let error = refusal("epoch mismatch", &graph, &plan, &mesh_other);
        assert_eq!(
            error,
            SplitError::EpochMismatch {
                plan: plan.mesh_epoch,
                mesh: mesh_other.epoch(),
            }
        );
    }

    // 2. Ep axis larger than one is refused before any planning.
    {
        let mesh = DeviceMesh::rect(&[(DimKind::Ep, 2)]).unwrap();
        let entries = [WeightEntry::layer(
            "wq",
            0,
            vec![4096, 4096],
            DType::F16,
            ShardPolicy::Replicate,
        )];
        let plan = expect_plan(&entries, &mesh, 1);
        let (graph, _) = project_graph("wq", 0, [4096, 4096], 1);
        let error = refusal("ep axis", &graph, &plan, &mesh);
        assert_eq!(error, SplitError::UnsupportedAxis(DimKind::Ep));
    }

    // 3. RowShard geometry: parts * group must divide the split extent.
    //    Mandated geometry is dim 5120, group 256, Tp3 (5120 % 768 != 0).
    //    `plan_manifest` pre-rejects Tp3 on a 5120-wide RowShard axis
    //    (5120 % 3 != 0) and its behavior must not change, so the fixture
    //    plans a valid Tp3-divisible extent (6144 = 3 * 2048) through
    //    `WeightEntry::layer` + `plan_manifest`, then injects the refusal by
    //    overwriting only the planned placement's declared `logical_shape[axis]`
    //    with 5120 (the graph weight resource is declared with 5120 too). This
    //    mutates a derived plan, it is not a hand-written `ManifestPlan`.
    {
        const PLANNED_DIM: usize = 6144;
        const DIM: usize = 5120;
        const AXIS: usize = 1;
        assert_ne!(DIM % 3, 0);
        assert_ne!(DIM % (3 * 256), 0);
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, 3)]).unwrap();
        let entries = [WeightEntry::layer(
            "wo",
            0,
            vec![4096, PLANNED_DIM],
            DType::F16,
            ShardPolicy::RowShard { axis: AXIS },
        )];
        let mut plan = expect_plan(&entries, &mesh, 1);
        let placement = plan
            .weights
            .iter_mut()
            .find(|placement| placement.name == "wo" && placement.layer == Some(0))
            .expect("planned placement for wo");
        assert_eq!(placement.policy, ShardPolicy::RowShard { axis: AXIS });
        assert_eq!(placement.logical_shape[AXIS], PLANNED_DIM);
        placement.logical_shape[AXIS] = DIM;
        let (graph, weight) = project_graph("wo", 0, [4096, DIM], 256);
        let error = refusal("row shard geometry", &graph, &plan, &mesh);
        assert_eq!(
            error,
            SplitError::Geometry {
                resource: weight,
                extent: DIM,
                parts: 3,
                align: 256,
            }
        );
    }

    // 4. HeadLocal heads not divisible by Tp (kv_heads 3 % 2 != 0).
    {
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, 2)]).unwrap();
        let plan = expect_plan(&[], &mesh, 1);
        let mut graph = StepGraph::new();
        let q = graph
            .add_resource(ResourceDecl {
                class: ResourceClass::Activation,
                shape: vec![8 * 128],
                alias: None,
            })
            .unwrap();
        let attn = graph
            .add_resource(ResourceDecl {
                class: ResourceClass::Activation,
                shape: vec![8 * 128],
                alias: None,
            })
            .unwrap();
        graph
            .add_op(OpDecl {
                kind: OpKind::Attend,
                layer: Some(0),
                rule: SplitRule::HeadLocal {
                    heads: 8,
                    kv_heads: 3,
                },
                reads: vec![q],
                writes: vec![attn],
            })
            .unwrap();
        let error = refusal("head-local heads % parts", &graph, &plan, &mesh);
        assert_eq!(
            error,
            SplitError::Geometry {
                resource: q,
                extent: 3,
                parts: 2,
                align: 1,
            }
        );
    }

    // 5. Alias refusal. The graph builder rejects malformed aliases itself,
    //    so there is no way to hand `split_program` a forward / out-of-range
    //    alias; assert the builder refusals instead (these are not part of the
    //    zero-allocation guarantee).
    {
        let mut graph = StepGraph::new();
        let forward = graph.add_resource(ResourceDecl {
            class: ResourceClass::Activation,
            shape: vec![4],
            alias: Some(hipfire_dispatch::pipeline::step_graph::Alias {
                of: ResourceId(0),
                offset: 0,
            }),
        });
        assert_eq!(
            forward,
            Err(StepGraphError::AliasForward {
                resource: ResourceId(0),
                of: ResourceId(0),
            }),
            "self alias must be refused by add_resource"
        );
        assert!(
            graph.resources().is_empty(),
            "refused alias must not be recorded"
        );

        let base = graph
            .add_resource(ResourceDecl {
                class: ResourceClass::Activation,
                shape: vec![4],
                alias: None,
            })
            .unwrap();
        let past_end = graph.add_resource(ResourceDecl {
            class: ResourceClass::Activation,
            shape: vec![4],
            alias: Some(hipfire_dispatch::pipeline::step_graph::Alias {
                of: base,
                offset: 2,
            }),
        });
        assert!(
            matches!(past_end, Err(StepGraphError::AliasOutOfRange { .. })),
            "alias past the target extent must be refused, got {past_end:?}"
        );
        assert_eq!(
            graph.resources().len(),
            1,
            "refused alias must not be recorded"
        );
    }

    // 6. GatedMlp admission refusals: geometry (`intermediate % (parts *
    //    group)`, 384 % (2 * 256) != 0 while 384 % 2 == 0 plans), a down
    //    weight that does not split the intermediate with gate/up, and a
    //    ladder with no shard-safe form.
    {
        const HIDDEN: usize = 256;
        const INTER: usize = 384;
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, 2)]).unwrap();
        let split = [
            ShardPolicy::ColumnShard { axis: 0 },
            ShardPolicy::ColumnShard { axis: 0 },
            ShardPolicy::RowShard { axis: 1 },
        ];
        let mut mixed = split.clone();
        mixed[2] = ShardPolicy::Replicate;
        for (case, policies, ladder) in [
            ("gated mlp geometry", &split, MlpLadder::Trunk),
            ("gated mlp mixed policy", &mixed, MlpLadder::Trunk),
        ] {
            let shapes = [[INTER, HIDDEN], [INTER, HIDDEN], [HIDDEN, INTER]];
            let names = ["gate", "up", "down"];
            let entries: Vec<WeightEntry> = (0..3)
                .map(|i| {
                    WeightEntry::layer(
                        names[i],
                        0,
                        shapes[i].to_vec(),
                        DType::F16,
                        policies[i].clone(),
                    )
                })
                .collect();
            let plan = expect_plan(&entries, &mesh, 1);
            let mut graph = StepGraph::new();
            let mut add = |class, shape: &[usize]| {
                graph
                    .add_resource(ResourceDecl {
                        class,
                        shape: shape.to_vec(),
                        alias: None,
                    })
                    .unwrap()
            };
            let input = add(ResourceClass::Activation, &[HIDDEN]);
            let output = add(ResourceClass::Activation, &[HIDDEN]);
            let [gate, up, down] = [0, 1, 2].map(|i| {
                add(
                    ResourceClass::Weight {
                        name: names[i].to_string(),
                        layer: Some(0),
                        // Packed groups along the down weight's split axis.
                        group: if i == 2 { 256 } else { 1 },
                    },
                    &shapes[i],
                )
            });
            let op = graph
                .add_op(OpDecl {
                    kind: OpKind::GatedMlp,
                    layer: Some(0),
                    rule: SplitRule::GatedMlp {
                        input,
                        gate,
                        up,
                        down,
                        output,
                        ladder,
                    },
                    reads: vec![input, gate, up, down],
                    writes: vec![output],
                })
                .unwrap();
            let expected = match case {
                "gated mlp geometry" => SplitError::Geometry {
                    resource: down,
                    extent: INTER,
                    parts: 2,
                    align: 256,
                },
                _ => SplitError::UnsupportedPolicy { op, resource: down },
            };
            assert_eq!(refusal(case, &graph, &plan, &mesh), expected, "{case}");
        }
    }

    // 7. HeadMixer admission: an input projection the manifest does not plan
    //    (only `wo` is planned, row-split) is refused as `UnknownWeight`.
    {
        const HIDDEN: usize = 256;
        const HEADS: usize = 4;
        const HEAD_DIM: usize = 64;
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, 2)]).unwrap();
        let entries = [WeightEntry::layer(
            "wo",
            0,
            vec![HIDDEN, HEADS * HEAD_DIM],
            DType::F16,
            ShardPolicy::RowShard { axis: 1 },
        )];
        let plan = expect_plan(&entries, &mesh, 1);
        let mut graph = StepGraph::new();
        let mut add = |class, shape: &[usize]| {
            graph
                .add_resource(ResourceDecl {
                    class,
                    shape: shape.to_vec(),
                    alias: None,
                })
                .unwrap()
        };
        let input = add(ResourceClass::Activation, &[HIDDEN]);
        let output = add(ResourceClass::Activation, &[HIDDEN]);
        let weight = |name: &str| ResourceClass::Weight {
            name: name.to_string(),
            layer: Some(0),
            group: 1,
        };
        let qkv = add(weight("qkv"), &[3 * HEADS * HEAD_DIM, HIDDEN]);
        let wo = add(weight("wo"), &[HIDDEN, HEADS * HEAD_DIM]);
        let op = graph
            .add_op(OpDecl {
                kind: OpKind::GatedAttention,
                layer: Some(0),
                rule: SplitRule::HeadMixer {
                    input,
                    out_proj: wo,
                    output,
                    heads: HEADS,
                    kv_heads: HEADS,
                },
                reads: vec![input, qkv, wo],
                writes: vec![output],
            })
            .unwrap();
        assert_eq!(
            refusal("head mixer unknown weight", &graph, &plan, &mesh),
            SplitError::UnknownWeight { op, resource: qkv }
        );
    }

    // Keep the success type referenced so a future signature drift fails here.
    let _: Option<SplitPlan> = None;
}
