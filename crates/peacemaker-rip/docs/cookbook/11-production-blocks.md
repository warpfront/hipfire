# 11. Production blocks: vary a shipped generator

The shipped `iu4_gemm` kernel is a Rust generator. Its phase blocks are natives (`iu4::*`, [reference §13](../REFERENCE.md)), so a `.rip` script composes the same kernel and can change the composition without a rebuild: how the K-loop is structured, which blocks run, what sits between them, how the epilogue is written. Block internals stay Rust and typed: a block still cannot read a `Free` LDS buffer, and a ring of the wrong kind (the weight ring where the activation ring is expected) is refused by the driver's tag check.

The committed twin [`iu4_gemm.rip`](../../../peacemaker-isa/src/kernels/iu4_gemm.rip) is the starting point. For every valid `Spec` on gfx1201 it emits the bytes of `iu4_gemm::emit` (`tests/iu4_gemm_twin.rs`), so any difference you see later is yours. Copy it, rename it, edit it.

## The composition

```text
declare_lds -> wg.exit(END) -> prologue -> kloop -> epilogue [-> gdn_epilogue] -> wg.end
```

`kloop` is itself a script function in the twin: a `wg_skip_if` over the full trips (taken when `K == 256`), a `loop_carried` trip of two `block`s and the trip-end bookkeeping, then the peeled last group.

```text
fn kloop(wg, g, lds) {
    wg.label(iu4::K_BEGIN)?;
    let no_trip = iu4::trips_cmp(wg, g, "s_cmp_eq_u32")?;
    let lds = wg.wg_skip_if(no_trip, iu4::tail_label(0), lds, |wg, lds| {
        let lds = wg.loop_carried(iu4::K_LOOP, lds, |wg, lds| {
            let lds = iu4::block(wg, g, "m0", 0, true, iu4::label("m1"), true, lds)?;
            let lds = iu4::block_stepping(wg, g, "m1", 1, true, iu4::K_LOOP_END, true, 0, lds)?;
            iu4::trip_end(wg, g)?;
            Ok((lds, iu4::trips_cmp(wg, g, "s_cmp_lg_u32")?))
        })?;
        wg.label(iu4::K_LOOP_END)?;
        Ok(lds)
    })?;
    let lds = iu4::block(wg, g, "t0", 0, true, iu4::label("t1"), false, lds)?;
    iu4::block(wg, g, "t1", 1, false, iu4::EPI, true, lds)
}
```

`lds` is the array of the four LDS rings. Each `iu4::block(..)` takes it and returns it; underneath, it is one checked `Driver::typed` call that lends the rings to the typed block and takes them back. That is why the driver's loop rule covers the `loop_carried` above: the body must leave every ring in the state it found it, and `block` returns the rings in the state `block` found them.

`block_stepping` is `block` plus one descriptor step: the production odd block advances the even-block descriptor (`step_descriptor` 0) right after its slab-1 fetch, which is inside one block, so it cannot be written as a call between two `block`s. Moving it is not a free choice: block 1's own next-block fetch reads that descriptor next.

## Variation 1: a `K == 256` specialization

A kernel that only ever runs `K == 256` has no full trip. Drop the trip compare, the skip and the rolled loop and run only the peeled pair. The first block keeps its own label (`entry_label = true`): nothing skips to it any more.

```rip arch=gfx1201
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
    let lds = iu4::block(wg, g, "t0", 0, true, iu4::label("t1"), true, lds)?;
    iu4::block(wg, g, "t1", 1, false, iu4::EPI, true, lds)?;
    iu4::epilogue(wg, g, end)?;
    wg.end(end)
}
```

The kernel is only correct for `K == 256`; it is a specialization, not a replacement. The loop-free program has no `.Liu4_k_loop` and is shorter than the shipped one.

## Variation 2: a timing experiment

Every shipped symbol raises the wave priority for its K-loop (`s_setprio 1`) and lowers it in the epilogue, so co-resident K-loops win issue arbitration over a finishing tile. To ask what that is worth, delete the line in `author`:

```text
    op(wg.isa(), "s_setprio 1", vec![], vec![])?;
```

Results are unchanged; only scheduling moves.

## Variation 3: a hand-written epilogue

`iu4::epilogue(wg, g, end)` places `iu4::EPI` and writes the store path of the spec's epilogue family. A script may place `iu4::EPI` itself and write its own stores with the generic helpers (`mem`, `bstore_b128`, `gather_offset`, ...) and the registers the generator assigned (`g.reg("acc")` is the base of the 64 accumulator VGPRs, `g.reg("arg_y")` the output pointer's SGPR pair). A hand-written epilogue is raw instructions on the builder (`wg.isa()`), so it gets the builder's ledger and the instruction validator, and the final object is certified like any `.rip`; it does not get a typed ring check, because it touches no LDS ring.

## Variation 4: the 2:4-sparse K-loop

The twin's variants carry an optional `-s24x64` / `-s24x32` suffix: `k128-128x128x8-cacc1-set-s24x64` is the same composition with `iu4::sparse_block` / `iu4::sparse_block_stepping` in place of the dense blocks (`mma_block` in the twin picks by `g.spec().mma`). The weights must be packed with `iu4_gemm::sparse24::pack_block`; the staging, trips, descriptor steps and epilogue are the dense script's. To try a different block order for the sparse kernel, edit `kloop` in the copy of the twin exactly as above; a sparse block refuses a dense generator and a dense block refuses a sparse one, so a mixed-up composition fails at build, not on the GPU.

## Timing a variation

`pm_loop --kernel iu4_gemm --rip FILE.rip [knobs...]` is the loop (build once with `cargo build --release -p rdna-compute --features lab --example pm_loop`, run on a gfx1201 card through `fabric-run.sh --class timing`). The knobs pick the `Spec` (tile, fold, epilogue, layout: the launch geometry and kernarg layout); `FILE` is compiled for that `Spec`'s variant name, refused (exit 3, stage `rip`) unless its symbol, kernarg layout, workgroup size and LDS equal those of `iu4_gemm::emit`, then linked with the native writer, certified (M7), checked against the CPU reference and the shipped object, and ABBA-timed. The record carries `candidate.rip = {path, sha256, variant, identical_to_generator}`. Iterating on generator logic is: edit the `.rip`, rerun; no Rust rebuild. Only `iu4_gemm` blocks are exposed to `.rip` so far (`--kernel qwen4_moe_sym --rip` is a usage error). The variant names are the shipped ones (`k128-128x128x8-cacc1-slab-silu-bf16`; the grammar is in the twin's header). `peacemaker-rip check FILE --variant V --arch gfx1201` already runs the front-end checks on its own.

## Why

- **One implementation.** `iu4_gemm::emit` and its K-loop are written over `iu4_gemm::blocks`, the functions the natives call. The twin is a re-expression of the composition, not a second generator; the byte-identity test keeps them equal.
- **Checked, not trusted.** Everything a block emits reaches the builder through the typed core's authorized calls, so the wait ledger, the LDS slot machine and the loop fixpoint see it. The bridge (`Driver::typed`) lends only rings the typed core built, with their phase, tag and owner checked; it mints no ring, region or authorization.
- **gfx1201 only.** The blocks are typed for gfx1201; on another target a typed call is an error.
- **A block can be split further.** The dense block is itself a composition of finer typed natives (`iu4::fragments_body` under the script body `rip_fragments`, `iu4::wmma`, `iu4::publish_slab`, `iu4::wait`, `iu4::fetch_*`, ...), so where its publishes, waits and fetches sit is a script choice: [chapter 17](17-iu4-kloop-natives.md).
