# peacemaker-author — map

> **Status:** `experimental` (PM kernel language milestone M0; build-time only).
> **Layer:** Kernel authoring (build time) — typed front of the `peacemaker-isa` builder; no runtime crate depends on it.

## Purpose

The typed core of the PM kernel language (design: `pm-kernel-lang-design.md` §4, §9, §12 M0). Kernel source is ordinary Rust run at build time; this crate puts types in front of a lowering `Backend` (`peacemaker-isa`'s `Builder` in production, `trace::Trace` for tests):

- `target`: `Gfx1100`, `Gfx1151`, `Gfx1201` as types with their wait-counter and barrier models; capabilities (`MmaIu4`, `SplitBarrier`) are trait bounds with `#[diagnostic::on_unimplemented]` errors. gfx942 (and its FNUZ fp8 type) is a documented future extension, not modelled.
- `lds`: `LdsRegion<R, Free | Writing | Published>`, the N-slot `Ring<R, Cur, Next, const D = 2>` (depth 2 to 4; `new_n`, `starting_at`, `depth`), `join`/`split`, and the barrier transitions `ready`, `retire`, `rotate`, `prime`, `retire_cur`, and the store-free `advance`.
- `wait`: `Pending` / `Drained` tokens; `ready`/`rotate` demand `Drained`.
- `scope`: `Workgroup` (barriers, `loop_carried`, `wg_skip_if`, `wg_if_else`, kernel exits `exit`/`exit_if`/`exit_unless`/`end`/`end_with`) dereferencing to `Wave` (LDS ops, waits, `skip_if`/`skip_unless`, `exec_if`, `forward` blocks of wave-uniform forward branches, raw ISA), affine `Uniform`/`WgUniform` SCC handles. `Forward::branch_if` and `branch_unless` preserve SCC1/SCC0 polarity. Every branch target (skip target, `Forward` target) joins the backend state of every path into it (`Backend::join`); branches only go forward, except a loop's back edge. `Wave::exit_unless` branches to an `End` with SCC0 and refuses later workgroup barriers. A handoff's reader receives `&End` as its third argument, so it can leave through that same exit without cloning ownership.
- `backend`: the lowering seam. Every state-changing entry point takes an `Auth` that only the core mints (one session per `Workgroup::new`); a backend keeps a `Seal` and refuses other sessions' tokens and anything after the kernel exit.
- `runtime`: the standalone RIP front end's checked `Driver`, with dynamic region/ring ids. It privately mints the same sealed `Auth`, mirrors the typed core's ownership and scope transitions, and issues the same backend calls. Wait ledgers, branch joins, loop fixed points and object-level safety analyses remain in the backend and existing Peacemaker IR passes.

The Halo VerifyAttn `lgkmcnt` race and the gfx11 FA2 mailbox race are compile errors (`tests/ui`), as are a lowering call without an `Auth` and a barrier in a wave scope, a `loop_until` body or a handoff's reader continuation.

## Gotchas

- Zero-cost is a gate, not a goal: every request maps onto the builder call a hand kernel makes, in the same order. The ported `peacemaker-isa` kernels (`iu4_v2c`, `iu4_v2b`, `iu4_gemm`) must re-emit byte-identical `.s` and `BuilderProof`s, except where a join or loop fixed point needs a guard the linear emission missed (or drops one it inherited from a path that never reaches the point).
- A barrier lowers its transitions as all retires, then all readies, each in tuple order (the slot lists every hand-written builder barrier passed).
- `Workgroup::scmp_wg_uniform` is the caller's claim that the compared SGPRs are workgroup-uniform; register values are pinned (hand `RegPlan`) in M0, so uniformity is not derived from dataflow yet.
- `Workgroup::new` seals the backend to one session: the `Builder`'s untyped `ds_store`/`ds_load`/`barrier*`/`loop_`/`control`/`lds_slot` refuse afterwards, and so does a second `Workgroup::new` over it (a wave scope escalating through `isa`). Raw `Builder::push` takes one plain statement: no branch, program end, trap, barrier, label, directive or line break. The `Builder`'s program, ledger and LDS slots are private (read-only `program()`/`ledger()`).
- A join keeps every wait and hazard guard either path needs, per ledger id: a counter whose pending operations are in-order loads of one family with equal shapes on both paths merges slot by slot (each slot carries both paths' ids, waits stay exact); on any other counter (stores, gfx11 LGKM with SMEM, gfx12 KM, or different shapes) an operation pending on one path only is `maybe` and its counter drains to zero. LDS slots merge `Published`+`Reading` to `Reading` and `Free`+`Publishing` to `Publishing`; other disagreements are refused. A join into a point nothing falls through to (after `s_branch`) takes the branch's state.
- A loop head is reached from the entry and the back edge: the builder re-emits the body from the entry's hazard trackers and LDS slots joined with the back edge's until the back edge adds nothing, and keeps that last emission (`loop_carried` carries the last run's state out). The back-edge ledger must still equal the entry's up to ids.
- The runtime driver's `Pending`/`Drained` distinction is a token kind, not “an event exists”: `begin_write` is Pending even before its first store. A raw counter wait retires backend events without changing that kind; the explicit driver `wait` still supplies publication evidence, matching the typed core.
- Runtime condition and exit handles are session-bound; split-barrier tokens name one signal instance. Joined write tokens retain one event representative per incoming path, so an explicit wait cannot lose a store outstanding only on a skipped path. Counter retirement itself is still evaluated only by the backend.
- Runtime `Phase` variants use `RtFree`, `RtWriting` and `RtPublished`, with the original human-readable debug labels. Distinct variant names keep rustc from qualifying the typed core's ownership-state names in existing compile-fail diagnostics.

## Crate map

<!-- crate-map:generated:begin -->

_Generated by `scripts/check-crate-maps.py` from the tree — do not edit inside the markers._

### Modules and exported symbols

- [`src/backend.rs`](src/backend.rs)
  - `Auth`
  - `Backend`
  - `check`
  - `end`
  - `EventId`
  - `is_sealed`
  - `Seal`
  - `seal`
  - `SlotTransition`
- [`src/lds.rs`](src/lds.rs)
  - `Advance`
  - `advance`
  - `AllFree`
  - `cur_index`
  - `depth`
  - `Free`
  - `FreeParts`
  - `into_steady`
  - `Join`
  - `join`
  - `JoinPart`
  - `LdsRegion`
  - `Lendable`
  - `Lowering`
  - `MAX_RING_DEPTH`
  - `new`
  - `new_n`
  - `next_index`
  - `Prime`
  - `prime`
  - `Published`
  - `Ready`
  - `ready`
  - `Retire`
  - `retire`
  - `retire_cur`
  - `RetireCur`
  - `Ring`
  - `Rotate`
  - `rotate`
  - `Slots`
  - `split`
  - `starting_at`
  - `State`
  - `Store`
  - `StoreTarget`
  - `Transition`
  - `Transitions`
  - `Writing`
- [`src/lib.rs`](src/lib.rs)
  - `backend`
  - `lds`
  - `runtime`
  - `scope`
  - `target`
  - `trace`
  - `wait`
- [`src/runtime.rs`](src/runtime.rs)
  - `Arrived`
  - `barrier`
  - `begin_write`
  - `branch_if`
  - `branch_unless`
  - `break_if`
  - `Breaks`
  - `Cond`
  - `consumed`
  - `cur_index`
  - `Driver`
  - `ds_load`
  - `ds_load_cur`
  - `ds_store`
  - `ds_store2`
  - `end`
  - `end_with`
  - `exec_if`
  - `Exit`
  - `exit`
  - `exit_if`
  - `Forward`
  - `forward`
  - `give`
  - `give_back`
  - `give_writing`
  - `goto`
  - `handoff`
  - `if_else`
  - `isa`
  - `join`
  - `label`
  - `lds`
  - `Lend`
  - `loop_carried`
  - `loop_until`
  - `new`
  - `next_index`
  - `Phase`
  - `Place`
  - `place`
  - `position`
  - `raw`
  - `region_phase`
  - `RegionId`
  - `relayout`
  - `ring`
  - `ring_depth`
  - `ring_n`
  - `ring_phases`
  - `ring_start_at`
  - `ring_steady`
  - `RingId`
  - `RingView`
  - `scmp`
  - `scmp_wg_uniform`
  - `signal`
  - `skip_if`
  - `skip_unless`
  - `split`
  - `take`
  - `take_writing`
  - `Transition`
  - `typed`
  - `view`
  - `wait`
  - `wait_all`
  - `wait_arrived`
  - `wave_exit_unless`
  - `wg_cond`
  - `wg_exit_unless`
  - `wg_if_else`
  - `wg_skip_if`
  - `WgCond`
- [`src/scope.rs`](src/scope.rs)
  - `Arrived`
  - `barrier`
  - `begin_write`
  - `branch_if`
  - `branch_unless`
  - `break_if`
  - `Carried`
  - `ds_load`
  - `ds_load_cur`
  - `ds_store`
  - `ds_store2`
  - `End`
  - `end`
  - `end_with`
  - `exec_if`
  - `exit`
  - `exit_if`
  - `exit_unless`
  - `Forward`
  - `forward`
  - `goto`
  - `handoff`
  - `if_else`
  - `isa`
  - `label`
  - `lds`
  - `loop_carried`
  - `loop_until`
  - `LoopExit`
  - `new`
  - `place`
  - `relayout`
  - `Scc`
  - `scmp`
  - `scmp_wg_uniform`
  - `Sealed`
  - `signal`
  - `skip_if`
  - `skip_unless`
  - `Token`
  - `Uniform`
  - `wait`
  - `wait_all`
  - `wait_arrived`
  - `Wave`
  - `wg_if_else`
  - `wg_skip_if`
  - `WgUniform`
  - `Workgroup`
- [`src/target.rs`](src/target.rs)
  - `BarrierModel`
  - `Full`
  - `Gfx1100`
  - `Gfx1151`
  - `Gfx11Waits`
  - `Gfx1201`
  - `Gfx12Waits`
  - `LdsCounter`
  - `MmaIu4`
  - `Sealed`
  - `Split`
  - `SplitBarrier`
  - `Target`
  - `WaitModel`
- [`src/trace.rs`](src/trace.rs)
  - `count`
  - `finish`
  - `new`
  - `raw`
  - `Trace`
  - `TraceFork`
- [`src/wait.rs`](src/wait.rs)
  - `Drained`
  - `Event`
  - `LdsWrite`
  - `Pending`
  - `Pendings`

### Dependencies (from `Cargo.toml`)

#### path

—

#### external

—

#### dev

- `trybuild`

#### build

—

### Reverse dependencies

Workspace crates with a path dependency on this crate:

- `peacemaker-isa`
- `peacemaker-rip`

<!-- crate-map:generated:end -->
