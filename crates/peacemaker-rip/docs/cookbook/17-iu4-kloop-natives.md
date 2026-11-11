# 17. Fine K-loop natives: place the waits, publishes and fetches of an `iu4_gemm` block yourself

[Chapter 11](11-production-blocks.md) composes `iu4_gemm` from whole blocks: `iu4::block` is one opaque native, so where a block's slab fetches, LDS publishes, waits and WMMA groups sit is not reachable from a script. This chapter splits the block. The K128 block, dense and 2:4-sparse, is a composition of finer typed natives that the generator itself is written over, and a script can spell the block out with them, in the generator's order (the committed twin does, byte for byte, for every variant) or in another one. The driver's typing is unchanged: a schedule that publishes into a buffer that is not `Free`, rotates before the wait, or reads or WMMAs a slot nobody published is a build error, not a wrong answer on the GPU.

The reference table of the natives is [reference §13](../REFERENCE.md); this chapter explains how they compose.

## One block, step by step

A K128 block on plane `h` is two 64-K slabs. Slab 0 sits in LDS slot 0 of the payload rings (A: activations, W: weights); slab 1 was fetched by the preceding block and is published into slot 1 while slab 0's WMMAs run. After the rendezvous the block reads slot 1, folds, and stages the next block's slab 0 into slot 0.

```text
block_begin                    label, register lifetimes
[step_descriptor]              block_stepping's descriptor advance
-- phase A (slab 0, slot 0) --
rip_fragments                   LDS -> registers, both 32-K sub-blocks
wmma sb0 (seed)    wmma sb1    the two WMMA groups
publish_slab                   slab 1 -> slot 1: Free -> Writing      } WAIT places it
fetch_next_meta                next token scale / row header
wait "slab"                    Writing pending -> drained
barrier rotate(A), rotate(W)   slot 0 retired, slot 1 published
-- phase B (slab 1, slot 1) --
fetch_next                     next block's slab 0 -> prefetch registers
rip_fragments  d_loads             slot 1; token scales
wmma sb0           scale_loads           (label _s0)
wmma sb1 (fold)    scale_tail            (label _s1)
publish_slab  publish_meta     next block -> slot 0 and the next metadata plane
wait "all"
signal rotate(A, W, DS, SZ)    split rendezvous
fetch_slab1 | fetch_slab1_next_group     next block's slab 1 -> prefetch registers
fold_scale                     the fold's stage 2
wait_arrived
```

The peeled last block of the K-loop (`fetch_next = false`) has no next block: no `fetch_next*`, no second publish, no rendezvous, and its phase B ends with `fold_scale`.

## The natives

| call | what it emits |
|---|---|
| `iu4::block_begin(wg, g, tag, h, end_label, entry_label, lds)?` | the block's label and register lifetimes |
| `iu4::fetch_slab1(wg, g, h)?`, `iu4::fetch_slab1_next_group(wg, g)?`, `iu4::fetch_next(wg, g, h)?`, `iu4::fetch_next_meta(wg, g, h)?` | the global loads into the prefetch registers (slab 1 of block `h`; the odd block's form for the next group; the next block's slab 0; its token scale and row header) |
| `iu4::publish_slab(wg, g, lds)?`, `iu4::publish_meta(wg, g, lds)?` | the LDS stores of the staged slab, or of the next metadata plane, into the ring's `Free` next buffer |
| `iu4::wait(wg, lds, set)?` | the counter wait for `"slab"`, `"meta"` or `"all"` rings; returns the `Drained` tokens |
| `iu4::fragments_body(wg, g, lds, loads)?` | the dense fragment loads as a script body: replays the script's `Vec<NativeLoad>` (DS loads only) on the A and W rings ([below](#the-fragment-loads-as-a-script-body)) |
| `iu4::d_loads(wg, g, lds)?`, `iu4::scale_loads(wg, g, tag, lds)?`, `iu4::scale_tail(wg, g, tag, lds)?` | LDS reads of the `Published` current buffers |
| `iu4::wmma(wg, g, sb, first, fold, lds)?` | the eight WMMAs of 32-K sub-block `sb`; `first` seeds the accumulators; `fold` interleaves the fold's stage 1 |
| `iu4::publish_slab_set(wg, g, set, lds)?`, `iu4::fetch_ahead(wg, g, h, slab)?`, `iu4::fetch_slab(wg, g, set, slab, desc, woff)?` | ring depth 3: publish prefetch set `set`; fetch a block one group on into set `slab`; fetch any block's slab into a set |
| `iu4::setup`, `iu4::split_trips`, `iu4::ring_cmp`, `iu4::group_end` | ring depth 3: the prologue without its ring part, the `q`/`r` split of the trip count, the entry branch's condition, a group's trip end |
| `iu4::fold_scale(wg, g)?` | the fold's stage 2 |
| `iu4::step_descriptor(wg, g, which)?` | the descriptor advance |
| `iu4::sparse_block_begin(wg, g, tag, h, end_label, entry_label, lds)?`, `iu4::sparse_fragments(wg, g, lds)?`, `iu4::sparse_wmma(wg, g, sb, first, fold, lds)?`, `iu4::sparse_scale_loads(wg, g, tag, lds)?`, `iu4::sparse_scale_tail(wg, g, tag, lds)?` | the sparse K-loop's steps that differ from the dense ones ([below](#the-sparse-k-loop)) |

Ring-touching natives take `lds` (the array of the four rings) and return it. The rendezvous is the generic driver's `wg.barrier((rotate(lds[0], da), ..))` or the split `wg.signal(..)` / `wg.wait_arrived(..)`; `iu4::wait` returns the tokens `rotate` takes (`rotate(ring)` alone also works: the driver checks the ring's own state, not the token).

## Typing rules

A ring crosses between a native and the driver in the state the Rust step is typed for (`Lend::take` for `Free`/`Published` buffers, `Lend::take_writing` for a `Writing` next buffer, whose `Pending` token travels with it). So:

| rule | refused as | test (`tests/iu4_gemm_fine_twin.rs`) |
|---|---|---|
| a publish into a buffer that is not `Free` | R0399 | `publish_into_a_non_free_buffer_is_refused` |
| a rotate before the wait | R0301 | `rotating_before_the_wait_is_refused` |
| a wait with nothing in flight, or an unknown ring set | R0399 | `waiting_without_a_publish_is_refused` |
| a fragment/scale read or WMMA of an unpublished slot | R0303 | `reading_or_wmma_of_an_unpublished_slot_is_refused` |
| a ring of the wrong kind, or one the script made | R0399 | `a_ring_of_the_wrong_kind_is_refused_by_the_fine_natives`, `a_ring_the_script_made_cannot_enter_the_fine_natives` |
| a fine step under wave-uniform control | R0304 | `a_fine_step_under_wave_uniform_control_is_refused` |
| a loop trip that ends with a publish in flight | R0399 | `a_trip_that_leaves_a_publish_in_flight_is_refused` |
| a dense step on a sparse spec, or a sparse step on a dense one | R0000 | `a_fine_step_of_the_other_mma_family_is_refused` |
| an `s24x64` second sub-block or `_s1` scale tail | R0000 | `an_x64_block_has_no_second_sub_block` |
| payload rings in different states, a split barrier, a re-laid-out ring, a bad slot | R0000 / R0399 | `the_script_fragments_body_keeps_the_typed_refusals` |
| a non-load in a body (R0399), a load whose defs/uses miss an operand (R0603), an unowned destination register (R0401) | R0399 / R0603 / R0401 | `a_fragments_body_outside_its_capability_is_refused` |

Each has a test in `tests/iu4_gemm_fine_twin.rs`, and the four below a whole failing script ([ERRORS.md](../ERRORS.md) points here from R0301, R0303 and R0399). The ring tags (`PayA`, `PayW`, `MetaD`, `MetaS`) are the typed lend's, so `iu4::publish_slab` with the W ring where the A ring belongs is refused. The loop rule is the driver's: `loop_carried` demands every ring leave the trip in the state it entered, which is why a block that publishes also waits and rotates inside the trip.

`fetch_*`, `fold_scale`, `wmma`, `sparse_wmma` and `step_descriptor` emit through the builder rather than `Driver::typed` (which the driver refuses while a split barrier is in flight), so they may sit between `signal` and `wait_arrived`; `wmma` and `sparse_wmma` are gated on the payload rings' phases instead (the current buffer must be `Published`).

## The fragment loads as a script body

The dense block's fragment loads are not a native call but a script function, `rip_fragments` in the twin: it builds the six `ds_load_2addr_stride64_b64` instructions (per 32-K sub-block one W load and two A loads) as `NativeLoad { role, insn }` values, with the slab offsets from `g.payload_base(role, slot)?`, and hands them to `iu4::fragments_body(wg, g, lds, loads)?`. The native lends the A and W rings in their state, replays each load through the typed `ds_load_cur` of the ring its role (`PayloadRole::A` or `W`) names, and gives the rings back. Its capability is a read of the payload rings: every instruction must be an LDS load whose declared defs and uses cover its operands, and its destinations must be registers the plan owns; anything else is refused before anything is emitted.

```text
fn rip_fragments<T: Target>(wg: &mut Wg<T>, g: &Iu4Gen, lds: Lds) -> Result<Lds, String> {
    // DS offset fields count 512-byte strides (`stride64` of b64).
    let slot = lds[0].cur_index();
    let a = g.payload_base(PayloadRole::A, slot)? / 512;
    let w = g.payload_base(PayloadRole::W, slot)? / 512;
    let mut loads = Vec::new();
    for sb in 0..2 {
        let f = g.reg(&format!("f{sb}")).ok_or("no fragment register")?;
        let fr_w = g.reg(&format!("fr_w{sb}")).ok_or("no W address register")?;
        let fr_a = g.reg(&format!("fr_a{sb}")).ok_or("no A address register")?;
        loads.push(NativeLoad { role: PayloadRole::W, insn: frag_load(f, fr_w, w) });
        for half in 0..2 {
            loads.push(NativeLoad { role: PayloadRole::A, insn: frag_load(f + 4 + 4 * half, fr_a, a + 2 * half) });
        }
    }
    iu4::fragments_body(wg, g, lds, loads)
}
```

`frag_load` and `ds_offsets` (in the full program below) format one load. The Rust `blocks::fragments` stays the generator's own step and the twin's independent reference: `the_script_fragments_body_emits_the_rust_loads_for_the_set_control` requires the script's loads to equal it, and `the_script_candidate_matches_the_cpu_fold_bitwise` runs the script's kernel on the CPU emulator against the exact CPU fold. Edit the offsets here and rebuild nothing; a legal but wrong offset is caught by the twin and the numerics (`a_legal_but_wrong_fragment_offset_is_caught_by_twin_and_numerics`).

## What the driver refuses

Four of the rules, as whole scripts (each the same two-function `iu4_gemm` kernel with one offending call; `tests/docs.rs` compiles each and requires its code). They are gfx1201-only, like every `iu4_gemm` script.

Publish into a buffer that is not `Free`: the second `publish_slab` finds the ring's next buffer `Writing`, its stores not yet waited and rotated.

```rip arch=gfx1201 fails=R0399
fn emit(spec: Spec) -> Result<Emitted, String> {
    let g = Iu4Gen::new(Iu4Spec {
        fold: Fold::K128, tile: Tile::T128x128x8, cacc: Cacc::One,
        epi: Epi::Set, act: ALayout::Token, arch: spec.arch, mma: Mma::Dense,
    })?;
    let mut b = Builder::new(g.kernel_spec(), g.plan()?);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    author(&mut wg, &g)?;
    b.finish()
}
fn author<T: Target>(wg: &mut Wg<T>, g: &Iu4Gen) -> Result<(), String> {
    let lds = iu4::declare_lds(wg, g)?;
    let end = wg.exit(iu4::END)?;
    let lds = iu4::prologue(wg, g, end, lds)?;
    let lds = iu4::publish_slab(wg, g, lds)?;
    // The slab above is still Writing: its stores were never waited and rotated.
    let lds = iu4::publish_slab(wg, g, lds)?;
    iu4::epilogue(wg, g, end)?;
    wg.end(end)
}
```

Fix: `iu4::wait(wg, lds, "slab")?` and a `wg.barrier((rotate(lds[0], da), rotate(lds[1], dw)))?` before the next publish, as `fine_block` does.

A wait with nothing in flight:

```rip arch=gfx1201 fails=R0399
fn emit(spec: Spec) -> Result<Emitted, String> {
    let g = Iu4Gen::new(Iu4Spec {
        fold: Fold::K128, tile: Tile::T128x128x8, cacc: Cacc::One,
        epi: Epi::Set, act: ALayout::Token, arch: spec.arch, mma: Mma::Dense,
    })?;
    let mut b = Builder::new(g.kernel_spec(), g.plan()?);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    author(&mut wg, &g)?;
    b.finish()
}
fn author<T: Target>(wg: &mut Wg<T>, g: &Iu4Gen) -> Result<(), String> {
    let lds = iu4::declare_lds(wg, g)?;
    let end = wg.exit(iu4::END)?;
    let lds = iu4::prologue(wg, g, end, lds)?;
    // Nothing was published into the next buffers: there is no store to wait for.
    let _drained = iu4::wait(wg, lds, "slab")?;
    iu4::epilogue(wg, g, end)?;
    wg.end(end)
}
```

Fix: wait only after the publish whose stores it drains.

A rotate before the wait: `s_barrier` does not wait for memory counters, so the driver refuses to publish the slab's undrained stores.

```rip arch=gfx1201 fails=R0301
fn emit(spec: Spec) -> Result<Emitted, String> {
    let g = Iu4Gen::new(Iu4Spec {
        fold: Fold::K128, tile: Tile::T128x128x8, cacc: Cacc::One,
        epi: Epi::Set, act: ALayout::Token, arch: spec.arch, mma: Mma::Dense,
    })?;
    let mut b = Builder::new(g.kernel_spec(), g.plan()?);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    author(&mut wg, &g)?;
    b.finish()
}
fn author<T: Target>(wg: &mut Wg<T>, g: &Iu4Gen) -> Result<(), String> {
    let lds = iu4::declare_lds(wg, g)?;
    let end = wg.exit(iu4::END)?;
    let lds = iu4::prologue(wg, g, end, lds)?;
    let lds = iu4::publish_slab(wg, g, lds)?;
    // The slab's stores are still pending.
    wg.barrier((rotate(lds[0]), rotate(lds[1])))?;
    iu4::epilogue(wg, g, end)?;
    wg.end(end)
}
```

Fix: drain first and hand the tokens to the rotate: `let (da, dw) = iu4::wait(wg, lds, "slab")?;` then `wg.barrier((rotate(lds[0], da), rotate(lds[1], dw)))?;`.

A WMMA group (or a fragment or scale read) of a slab nobody published: the rings are still `Free` as `declare_lds` made them.

```rip arch=gfx1201 fails=R0303
fn emit(spec: Spec) -> Result<Emitted, String> {
    let g = Iu4Gen::new(Iu4Spec {
        fold: Fold::K128, tile: Tile::T128x128x8, cacc: Cacc::One,
        epi: Epi::Set, act: ALayout::Token, arch: spec.arch, mma: Mma::Dense,
    })?;
    let mut b = Builder::new(g.kernel_spec(), g.plan()?);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    author(&mut wg, &g)?;
    b.finish()
}
fn author<T: Target>(wg: &mut Wg<T>, g: &Iu4Gen) -> Result<(), String> {
    let lds = iu4::declare_lds(wg, g)?;
    let end = wg.exit(iu4::END)?;
    // No block has been published: the first WMMA group has no slot to read.
    let lds = iu4::wmma(wg, g, 0, true, false, lds)?;
    let lds = iu4::prologue(wg, g, end, lds)?;
    iu4::epilogue(wg, g, end)?;
    wg.end(end)
}
```

Fix: run the prologue first (it stages and publishes block 0), then `rip_fragments(wg, g, lds)?` and the WMMA groups, as `fine_block` does.

## The twin composition

`iu4::block` is `kloop::block_with`, and `block_with` is the sequence above written in Rust over the same functions the natives call; `iu4::sparse_block` is `kloop::sparse_block_with`, the sequence of the next section. [`iu4_gemm_fine.rip`](../../../peacemaker-isa/src/kernels/iu4_gemm_fine.rip) is the committed twin of `iu4_gemm.rip` with `mma_block` calling a script function `fine_block` (the block above, only fine natives) for the dense variants and `fine_sparse_block` (only fine natives, no opaque `iu4::sparse_block*`) for the sparse ones. `tests/iu4_gemm_fine_twin.rs` requires it to emit, for every dense variant (`the_fine_composition_emits_the_shipped_bytes`) and for every sparse variant (`the_fine_sparse_composition_emits_the_shipped_bytes`: `k128-128x128x8-cacc1-{set,add}-{s24x64,s24x32}`), the bytes and proof of `iu4_gemm::emit`, so the opaque blocks and the spelled-out blocks are provably equal.

## The sparse K-loop

The 2:4-sparse blocks (`Mma::S24X64`, `Mma::S24X32`) are the same block with SWMMAC in place of WMMA: the same rings, publishes, next-block fetches, `d_loads` and fold, so those natives serve both. What differs is split into five steps, each the `iu4::sparse_*` twin of a dense one, and each refuses a dense spec; the dense `block_begin`, `iu4::fragments_body` (behind `rip_fragments`), `wmma`, `scale_loads` and `scale_tail` refuse a sparse spec in turn (`a_fine_step_of_the_other_mma_family_is_refused`), so a WMMA never lands in a SWMMAC kernel:

```text
                dense                   s24x64                       s24x32
head            block_begin             sparse_block_begin           sparse_block_begin
                                        (fragments end at _s0)       (W/index regs end at _s1)
fragments       rip_fragments           sparse_fragments             sparse_fragments
                (W + 2 A loads/sb)      (W + 8 ds_load_b64 B)        (= dense loads)
MMA groups      wmma sb0, wmma sb1      sparse_wmma sb0 (one K64     sparse_wmma sb0, sb1
                                        bundle; sb1 refused)         (index_key = sb)
scale loads     scale_loads (3, _s0)    sparse_scale_loads           sparse_scale_loads (2, _s0)
                scale_tail  (1, _s1)    (all 4, _s0; no _s1)         sparse_scale_tail  (2, _s1)
```

`first` seeds each accumulator with `magic8` (SWMMAC has no C operand; the dense WMMA passes `C = magic8`), and `fold` interleaves the fold's stage 1 between the SWMMACs, as dense. x64's single bundle per slab means `sparse_wmma` takes `sb = 0` only and its four scale loads all sit after the bundle; x32 keeps dense's two sub-blocks, so two scale loads sit between them and two after the second. A sparse spec admits the shipped wait placement (publishes after the slab's last bundle) only, so `fine_sparse_block` has no `WAIT` choice. `an_x64_block_has_no_second_sub_block` and `a_sparse_read_or_wmma_of_an_unpublished_slot_is_refused` pin the sparse-specific refusals.

## A wait-moved block

`const WAIT` in the twin picks where the next slab's publish sits between the WMMA groups: `"after"` is the shipped order; `"mid"` puts it between the two 32-K groups of each phase, `"before"` ahead of both. The stores then have more WMMA work between them and the wait, which is the knob the generator exposes as `Tune { wait: WaitPlace::MidMma | BeforeMma }`; the test requires `WAIT = "mid"`, `"before"` and `"window"` to emit exactly those generator bytes (`a_wait_moved_script_equals_the_generators_tune`). `"window"` (128x128x8 tile) is the schedule the shipped 27B gate/up `_b1s` export runs (`Tune::shipped_gate_up_slab`, `the_shipped_gate_up_export_is_the_window_twin`): slab 0's group, the slab-1 publish, then the first rendezvous split with sub-block 1's WMMA group (the twin's `raw_wmma_sb1`, pushed directly because a typed WMMA refuses a ring held by a split barrier) inside its window, and `iu4::d_loads` moved from phase B to right after phase A's fragments. The same block as a standalone, runnable `K == 256` kernel, with `WAIT = "mid"`:

```rip arch=gfx1201
const WAIT: &str = "mid";

fn emit(spec: Spec) -> Result<Emitted, String> {
    let g = Iu4Gen::new(Iu4Spec {
        fold: Fold::K128, tile: Tile::T128x128x8, cacc: Cacc::One,
        epi: Epi::Set, act: ALayout::Token, arch: spec.arch, mma: Mma::Dense,
    })?;
    let mut b = Builder::new(g.kernel_spec(), g.plan()?);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    author(&mut wg, &g)?;
    b.finish()
}

fn author<T: Target>(wg: &mut Wg<T>, g: &Iu4Gen) -> Result<(), String> {
    let lds = iu4::declare_lds(wg, g)?;
    let end = wg.exit(iu4::END)?;
    op(wg.isa(), "s_setprio 1", vec![], vec![])?;
    let lds = iu4::prologue(wg, g, end, lds)?;
    // K == 256 only: the peeled pair is the whole K-loop.
    wg.label(iu4::K_BEGIN)?;
    let lds = fine_block(wg, g, "t0", 0, true, iu4::label("t1"), true, None, lds)?;
    fine_block(wg, g, "t1", 1, false, iu4::EPI, true, None, lds)?;
    iu4::epilogue(wg, g, end)?;
    wg.end(end)
}

// One K128 block of the dense K-loop on plane `h`, written with the fine
// natives in the order the generator's `block` uses (`WAIT = "after"`; the other
// two placements are the generator's `Tune { wait: Mid | Before }`). `step`
// advances that descriptor right after the block's head, as `block_stepping`.
//
// Ring discipline, each line checked by the driver:
//   publish_*   next buffer Free -> Writing (a Pending token)
//   wait        Pending -> Drained (names the ring set)
//   barrier     rotate: cur retired, the Drained next one published
//   rip_fragments / scale_loads / wmma   read the Published current buffer
fn fine_block<T: Target>(wg: &mut Wg<T>, g: &Iu4Gen, tag: &str, h: usize, fetch_next: bool,
        end_label: &str, entry_label: bool, step: Option<usize>, lds: Lds) -> Result<Lds, String> {
    let lds = iu4::block_begin(wg, g, tag, h, end_label, entry_label, lds)?;
    if let Some(s) = step {
        iu4::step_descriptor(wg, g, s)?;
    }
    // Phase A: slab 0 from slot 0; slab 1 (fetched by the preceding block) is
    // published into slot 1.
    let lds = rip_fragments(wg, g, lds)?;
    let lds = if WAIT == "after" {
        let lds = iu4::wmma(wg, g, 0, true, false, lds)?;
        let lds = iu4::wmma(wg, g, 1, false, false, lds)?;
        iu4::publish_slab(wg, g, lds)?
    } else if WAIT == "mid" {
        let lds = iu4::wmma(wg, g, 0, true, false, lds)?;
        let lds = iu4::publish_slab(wg, g, lds)?;
        iu4::wmma(wg, g, 1, false, false, lds)?
    } else {
        let lds = iu4::publish_slab(wg, g, lds)?;
        let lds = iu4::wmma(wg, g, 0, true, false, lds)?;
        iu4::wmma(wg, g, 1, false, false, lds)?
    };
    if fetch_next {
        iu4::fetch_next_meta(wg, g, h)?;
    }
    let (da, dw) = iu4::wait(wg, lds, "slab")?;
    wg.barrier((rotate(lds[0], da), rotate(lds[1], dw)))?;
    // Phase B: slab 1 from slot 1, the fold, the next block into slot 0.
    if fetch_next {
        iu4::fetch_next(wg, g, h)?;
    }
    let lds = rip_fragments(wg, g, lds)?;
    let lds = iu4::d_loads(wg, g, lds)?;
    if !fetch_next {
        // The peeled group's last block: nothing to publish.
        let lds = iu4::wmma(wg, g, 0, false, false, lds)?;
        let lds = iu4::scale_loads(wg, g, tag, lds)?;
        let lds = iu4::wmma(wg, g, 1, false, true, lds)?;
        let lds = iu4::scale_tail(wg, g, tag, lds)?;
        iu4::fold_scale(wg, g)?;
        return Ok(lds);
    }
    let lds = if WAIT == "after" {
        let lds = iu4::wmma(wg, g, 0, false, false, lds)?;
        let lds = iu4::scale_loads(wg, g, tag, lds)?;
        let lds = iu4::wmma(wg, g, 1, false, true, lds)?;
        let lds = iu4::scale_tail(wg, g, tag, lds)?;
        let lds = iu4::publish_slab(wg, g, lds)?;
        iu4::publish_meta(wg, g, lds)?
    } else if WAIT == "mid" {
        let lds = iu4::wmma(wg, g, 0, false, false, lds)?;
        let lds = iu4::scale_loads(wg, g, tag, lds)?;
        let lds = iu4::publish_slab(wg, g, lds)?;
        let lds = iu4::publish_meta(wg, g, lds)?;
        let lds = iu4::wmma(wg, g, 1, false, true, lds)?;
        iu4::scale_tail(wg, g, tag, lds)?
    } else {
        let lds = iu4::publish_slab(wg, g, lds)?;
        let lds = iu4::publish_meta(wg, g, lds)?;
        let lds = iu4::wmma(wg, g, 0, false, false, lds)?;
        let lds = iu4::scale_loads(wg, g, tag, lds)?;
        let lds = iu4::wmma(wg, g, 1, false, true, lds)?;
        iu4::scale_tail(wg, g, tag, lds)?
    };
    let (da, dw, dd, ds) = iu4::wait(wg, lds, "all")?;
    let b2 = wg.signal((rotate(lds[0], da), rotate(lds[1], dw), rotate(lds[2], dd), rotate(lds[3], ds)))?;
    // The signal drained slab-0 stores: the retired payload ring can hold the
    // next K128's slab 1 while this block folds. The odd block runs before the
    // group offset advances, so it names the next group explicitly.
    if h == 0 {
        iu4::fetch_slab1(wg, g, 1)?;
    } else {
        iu4::fetch_slab1_next_group(wg, g)?;
    }
    iu4::fold_scale(wg, g)?;
    wg.wait_arrived(b2)?;
    Ok(lds)
}

// The fragment loads of a dense block, written here: per 32-K sub-block one
// W load (both row groups) and two A loads (column blocks 0-1, 2-3), from the
// payload rings' current slot. The script builds the six instructions;
// `iu4::fragments_body` lends the A and W rings in their state (Published
// current buffer, Free or Writing next, depth 2 or 3), replays each load on
// the ring its `PayloadRole` names through the typed core, and gives them
// back. Only LDS loads are accepted. Edit the body here: no rebuild.
fn rip_fragments<T: Target>(wg: &mut Wg<T>, g: &Iu4Gen, lds: Lds) -> Result<Lds, String> {
    // DS offset fields count 512-byte strides (`stride64` of b64).
    let slot = lds[0].cur_index();
    let a = g.payload_base(PayloadRole::A, slot)? / 512;
    let w = g.payload_base(PayloadRole::W, slot)? / 512;
    let mut loads = Vec::new();
    for sb in 0..2 {
        let f = g.reg(&format!("f{sb}")).ok_or("no fragment register")?;
        let fr_w = g.reg(&format!("fr_w{sb}")).ok_or("no W address register")?;
        let fr_a = g.reg(&format!("fr_a{sb}")).ok_or("no A address register")?;
        loads.push(NativeLoad { role: PayloadRole::W, insn: frag_load(f, fr_w, w) });
        for half in 0..2 {
            loads.push(NativeLoad { role: PayloadRole::A, insn: frag_load(f + 4 + 4 * half, fr_a, a + 2 * half) });
        }
    }
    iu4::fragments_body(wg, g, lds, loads)
}

// `ds_load_2addr_stride64_b64` of four VGPRs at `dst` from `v{addr}`, the two
// halves `o` and `o + 1` strides apart.
fn frag_load(dst: u8, addr: u8, o: u32) -> Instruction {
    let text = format!("ds_load_2addr_stride64_b64 v[{}:{}], v{}{}", dst, dst + 3, addr, ds_offsets(o, o + 1));
    Instruction::new(text, vec![vr(dst, 4)], vec![v(addr)]).memory(MemoryClass::DsLoad)
}

// DS offset fields as objdump prints them: a zero field is omitted.
fn ds_offsets(o0: u32, o1: u32) -> String {
    let mut t = String::new();
    if o0 != 0 {
        t.push_str(&format!(" offset0:{o0}"));
    }
    if o1 != 0 {
        t.push_str(&format!(" offset1:{o1}"));
    }
    t
}
```

Move any step the same way: put `iu4::fetch_next_meta` before the first WMMA group, split the publish across the groups, delay `iu4::fold_scale` past the `wait_arrived`. A move that breaks the ring discipline fails at the offending native; a move that is legal builds, is certified like any script, and its result bits are the shipped ones (a reorder of independent steps changes only the schedule; the loads read the same published slots).

## Timing a variation

`pm_loop --kernel iu4_gemm ... --rip FILE` takes the full twin, not the standalone kernel above: `crates/peacemaker-isa/src/kernels/iu4_gemm_fine.rip` is the file to copy. For the 27B gate_up shape (M = 17408, N = 8192, K = 5120):

```sh
pm_loop --kernel iu4_gemm --arch gfx1201 --shape M=17408,N=8192,K=5120 \
        --fold k128 --tile 128x128x8 --cacc 1 --epi silu --alayout slab \
        --rip crates/peacemaker-isa/src/kernels/iu4_gemm_fine.rip
```

(`pm_loop` refuses, exit 3 stage `rip`, a file whose symbol, kernarg layout, workgroup size or LDS differs from the generator's; the fine twin keeps all of them.) Change `WAIT`, the body of `fine_block` or `rip_fragments` and rerun: no rebuild.

## Ring depth 3

`const FOLD = "chain"` in the twin builds the generator's `Tune { fold_sched: FoldSched::Chain }` (the shipped 27B gate/up export's fold, with `WAIT = "window"`): the spec's `fold_sched` makes `iu4::wmma` (fold) and `iu4::fold_scale` emit the chained fold and the priority drop to 1; the script adds `s_setprio 2` after the block's descriptor step. `tests/iu4_gemm_fine_twin.rs` (`a_chain_fold_script_equals_the_generators_tune`) pins it.

`const RING = 3` in the twin builds the generator's `Tune { ring: RingDepth::Three }` for the dense K128 variants (every epilogue but the fused GDN projection, whose epilogue re-lays out the K-loop's LDS): the A and W payload rings are three LDS slabs deep (+8192 bytes on the 128x128x8 tile: the launch's dynamic LDS stays the shipped 20480 and the object declares the extra slab as static LDS, `group_segment_fixed_size` 8192, so a launcher that passes 20480 is right for it: 28672 per workgroup), two slabs are published ahead, and the metadata rings (DS, SZ) stay two deep. `tests/iu4_gemm_fine_twin.rs` (`a_ring_3_script_equals_the_generators_tune`) requires the script to emit exactly the generator's depth-3 bytes for all three `WAIT` placements and all twelve variants that have a generator; the result bits equal depth 2's (the schedule moves, not the math: `tests/iu4_gemm_ring3.rs` on the CPU emulator at K = 256, 512, 768, 1024, 5120).

**Schedule.** Two prefetch register sets hold the next block's slabs, set 0 its slab 0 and set 1 its slab 1 (the second set takes the eight VGPRs `magic8` held: each block's first WMMA group seeds its accumulators from `cacc[3][1]`, which no WMMA writes until the group's last, instead of from a constant register; the 128x128x8 kernel stays at 187 VGPRs). At block `b`'s entry slab `2b` is the current buffer, slab `2b + 1` is published ahead and the sets hold block `b + 1`:

```text
phase A   fragments(cur) ; WMMA groups ; publish set 0 (b+1's slab 0) into next
          fetch_next_meta ; wait ; barrier rotate(A, W) ; fetch b+2's slab 0 -> set 0
phase B   fragments(cur) ; d_loads ; WMMA groups ; publish set 1 (b+1's slab 1) and metadata
          wait ; signal rotate(A, W, DS, SZ) ; fetch b+2's slab 1 -> set 1 ; fold ; wait_arrived
```

Block `b + 2` is the same plane one group on, so each plane's activation descriptor steps at the head of its block (`iu4::step_descriptor(wg, g, h)`) and the group's trip end only advances the group offset (`iu4::group_end`). The peeled group's even block publishes the odd block's slabs and fetches nothing (`"t0"`); its odd block publishes nothing, so phase A's barrier is `advance(lds[0]), advance(lds[1])`: a store-free rotation (retire the current buffer, the ahead buffer becomes current).

**Addressing.** Every LDS address stays an immediate. With `n = K/256 - 1` full groups, `r = n mod 3` and `q = n / 3`, slab `s` lives in slot `(s - r) mod 3`: after `r` groups the next group starts on slot 0, and so does the peeled group after the loop's `q` trips of three groups. The script (`author3`) is the entry branch (`iu4::split_trips`, then `wg.wg_if_else(iu4::ring_cmp(wg, g, 0)?, ..)` nested once for `r` = 0, 1, 2), arm `r` = `lds[0].starting_at((3 - r) % 3)` and `lds[1].starting_at(..)`, the prologue's ring part (`ring_part3`: `publish_slab`, `publish_meta`, `wait`, `signal(prime(..))`, `fetch_slab`, `wait_arrived`, `publish_slab_set`, `wait`, `barrier(rotate(..))` over the vacant current buffer, `fetch_slab`), and `r` full groups; then `wg_skip_if(q == 0)` around a `loop_carried` body of three groups counted by `q`; then the peeled group. Three arms, three body groups and the tail: about 2.3 times the depth-2 code (the 27B gate/up kernel: 14084 -> 32192 code bytes).

**Legality.** `Spec::validate` refuses what depth 3 does not generate, with the numbers: the sparse blocks (their fine steps exist, but no depth-3 composition of them is written), the fused GDN projection, a tile whose second prefetch set is wider than the eight freed VGPRs, a VGPR count over the occupancy ceiling (192) and an LDS size (static + dynamic) that does not fit per workgroup (64 KiB) or per WGP at the occupancy the ceiling allows. The 27B gate/up node (`--fold k128 --tile 128x128x8 --cacc 1 --epi silu --alayout slab`) and the 16-wave tile are legal.

```sh
pm_loop --kernel iu4_gemm --arch gfx1201 --shape M=17408,N=8192,K=5120 \
        --fold k128 --tile 128x128x8 --cacc 1 --epi silu --alayout slab --ring 3 --offline
```

`pm_loop --rip` takes the twin with `const RING = 3` (the candidate's depth is read from that line); `--ring` is the generator's own knob and is refused with `--rip`.
