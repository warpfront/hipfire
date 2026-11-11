# 12. Fine blocks: move a wait, a barrier, a decode, a fold in `mq6_i8_gemm`

[Cookbook 11](11-production-blocks.md) composes `iu4_gemm` from coarse blocks: each block owns a whole K128 block including its LDS ring transitions, so a script can reorder blocks but cannot move a wait or a barrier. `mq6_i8_gemm` (the MQ6G256V2 x `block_i8_128` exact-INT8 GEMM, gfx1201) is split finer. Its natives are instruction builders; the LDS ring operations are script statements, checked by the driver ([reference §14](../REFERENCE.md)). That is what lets a script move a wait, the position of a split-barrier signal, where the decode sits among the WMMAs, and the order of the fold.

The committed twin [`mq6_i8_gemm.rip`](../../../peacemaker-isa/src/kernels/mq6_i8_gemm.rip) is the starting point. For every valid `Spec` on gfx1201 (`128x128x8-set`, `128x128x8-set_bf16`, each also `-linear`, and the same four on the `256x128x16` and `128x256x16` tiles: twelve) it emits the bytes **and the proof** of `mq6_i8_gemm::emit` (`tests/mq6_i8_gemm_twin.rs`), and `emit` itself is pinned to the digests recorded before the blocks existed (`peacemaker-isa/tests/mq6_i8_gemm_digests.rs`), so any difference you see later is yours. Copy it, rename it, edit it. Every edit below is applied to the committed twin by that test file; the claims about what builds, what is refused and what keeps the output are the assertions there.

## What the script owns

```text
author:   declare_lds -> wg.exit(END) -> s_setprio 1 -> prologue -> kloop -> label EPI -> epilogue -> wg.end
prologue: front, exits, scalars, lanes, fetch0, zero_acc, block-0 staging, signal(prime ..), seed, wait_arrived
kloop:    wg_skip_if(no full trip) { loop_carried { block m0; block m1; trip_end } }, peeled blocks t0 t1
block:    phase A (K16 steps 0..4 + queued work) -> B1 split barrier -> phase B -> fold -> B2 split barrier
```

The script writes all of it, down to the rings:

```text
let a0 = slot(wg, g, "A0")?;  ...  [Ring::new(a0, a1), Ring::new(w0, w1), Ring::new(d0, d1), Ring::new(s0, s1)]
let _ = wg.begin_write(st.a);                          // stage the next slab into the other buffer
wg.ds_store(st.w, insn)?;                              // insn from mq6::store_w_insns(g, slot, r)
wg.ds_load_cur(st.a, loads[1])?;                       // loads from mq6::step_load_insns(..): the Published buffer
let (da, dw) = wg.wait_all((st.a, st.w))?;             // drain the stores
let b1 = wg.signal((rotate(st.a, da), rotate(st.w, dw)))?;   // B1, first half
...                                                    // work that overlaps the rendezvous
wg.wait_arrived(b1)?;                                  // B1, second half
```

The natives return `Instruction`s for everything that touches LDS and emit straight into `wg.isa()` for everything else. A native owns no ring, so there is nothing to lend: the driver sees every store, load, wait and barrier and checks it as it does any script.

Between the WMMAs the block runs queued work units (`enum Work { Insn, StoreW, StoreA, FetchA, FetchMeta, Meta }`). `phase(wg, g, st, slot, first, start, per, io_at, io_n, hold_at, hold, ..)` issues four K16 steps and, after each WMMA from index `start` on, runs `per` units; `io_n` more units run after WMMA `io_at`, and the `hold` units are appended and drained after WMMA `hold_at` (`99` = never; the trailing fold arguments are likewise off). The order of the `st.work.push(..)` lines, `start`, `per`, `io_at` / `io_n` and `hold_at` are the placement.

## Move a wait

The staging registers of the next slab's raw activations are still sources of the previous half's LDS stores until those stores have read them. The script waits before it refills them (`Work::FetchA`):

```text
Work::FetchA(slab, sb, srd) => {
    // The raw staging registers are still sources of the previous half's stores.
    mq6::wait_all(wg)?;
    mq6::fetch_a(wg, g, slab, sb, srd)?;
}
```

Delete `mq6::wait_all(wg)?;` and the build is refused: `fetch_a(slab 1, sb 1): clause requires 2..=32 independent instructions of one memory class without waits`. The wait is not decoration; the builder needs it there. A second wait right after the refill (the wait kept and another added) builds and is a different program: that is the experiment for what an extra wait costs.

The ring waits are the same kind of statement. `wg.wait_all((st.a, st.w))` before the B1 signal drains the stores the rotate publishes. Drop it (`rotate(st.a)` with no drained token) and the driver refuses: `barrier would publish ring@0 with an undrained LDS store: wait for it first`.

## Move the split-barrier signal

B1 is split: `wg.signal` after the stores drain, the next block's raw slab-0 fetch (VMEM) overlaps the rendezvous, `wg.wait_arrived` completes it. Production issues the next block's token word and row header (`mq6::fetch_meta`) as the queued unit `Work::FetchMeta`, right behind the half-1 refetch after WMMA 7, so it is long landed by the signal. To make that fetch overlap the rendezvous instead, drop the `st.work.push(Work::FetchMeta(..))` and issue it after the signal:

```text
    let (da, dw) = wg.wait_all((st.a, st.w))?;
    let b1 = wg.signal((rotate(st.a, da), rotate(st.w, dw)))?;
    if fetch_next {                                    // was the queued `Work::FetchMeta`
        mq6::fetch_meta(wg, g, h, if h == 0 { 4 } else { group })?;
    }
    if fetch_next {
        mq6::fetch_a(wg, g, 0, 0, srd_a[h ^ 1])?;
        mq6::fetch_w(wg, g, if h == 0 { 8 + 96 } else { group + 8 })?;
    }
    wg.wait_arrived(b1)?;
```

The program has the same instructions in another order, builds, and produces the same bits on the CPU interpreter (`moved_decode_split_barrier_and_fold_order_keep_the_output_bit_identical`).

B2 is the same kind of edit. It is signalled after the fold's LDS reads and waited at the block end; the next block's slab-1 fetch sits between. Move `frag(wg, g, 2, 0, ..)` above the `let arrived = ..` block and the signal comes one fold fragment later: the first packets of that fragment no longer overlap the rendezvous. It builds and keeps the output (`signalling_b2_one_fragment_later_keeps_the_output`).

What the driver refuses, wherever you put the signal: a read of a ring the barrier in flight holds (`wg.ds_load_cur(st.s, ..)` between `signal` and `wait_arrived`: `split barrier in flight`), and a `signal` with no `wait_arrived` (`ring ring@0 is held by a split barrier in flight until wait_arrived`).

## Decode placement relative to the MMAs

The weight decode (`mq6::decode_insns(g, r)`, 31 VALU instructions per round) is spread between the WMMAs so the tensor and vector pipes overlap. In `phase(wg, g, &mut st, 0, true, 8, 3, ..)` the first K16 step's bundle is the seeded one: the seed octet *is* the decode scratch, so decode may only start after the last seeded WMMA (index 8). From there three work units run after every WMMA. Change it:

```text
    phase(wg, g, &mut st, 0, true, 12, 3, ..)?;   // decode starts after the 12th WMMA
    phase(wg, g, &mut st, 1, false, 6, 2, ..)?;   // two units per WMMA in phase B
```

Both build, keep every instruction and leave the output bit-identical (interpreter). A start below 8 is the experiment the seed forbids: the decode overwrites the `0x4B400000` octet the first WMMAs still read. `phase(.., 0, true, 4, 3, ..)` builds, because the builder has no model of that arithmetic, and computes another result. The interpreter is what catches it (`decoding_before_the_seed_is_consumed_builds_but_changes_the_output`), so run a variation there before the card.

The queue order is a placement knob as well: `Work::StoreW(0)` after the first decode round is the earliest the decoded operand can be stored; `Work::FetchA(.., sb = 1, ..)` is queued right after the first `StoreA` so the raw staging registers refill as soon as their stores have drained, and `StoreA(1)`, the last consumer of that refetch, is held to after WMMA 24 (`hold_at`) so the refetch has the whole span to land.

## VOPD fold pairing

The fold of a block is 160 VOPD packets: per accumulator fragment `k = nb * 2 + rg` four stages, `mq6::fold_stage(wg, g, k, stage, sc, zp)` for stage `0..4` (8 `mul`/`add` packets, 4 `fmac` pairs, 4 `mul` pairs, 4 `fmac` pairs). Which two operations form a packet, and which register banks and parities make the pair legal, is inside the stage (Rust: `vopd::validate_pair` refuses a bad assignment at build time). The script decides which stage runs when, for which fragment, with which `sc` / `z'` octets:

```text
fn frag(wg, g, nb, rg, sc, zp) {
    mq6::emit_msn(wg, g, nb)?;
    for stage in 0..4usize { mq6::fold_stage(wg, g, nb * 2 + rg, stage, sc, zp)?; }
    Ok(())
}
```

Row group 1's four fragments are independent (each reads its own `cacc` octet and accumulators), so their order is free:

```text
    for nb in [3usize, 2, 1, 0] { frag(wg, g, nb, 1, cacc, scratch)?; }
```

It builds, keeps all 160 packets and the output bits. The stages inside one fragment are not free: stage 2 overwrites the product octet (`fold_t`) that stage 1 still reads. `for stage in [0usize, 2, 1, 3]` builds, because every packet is individually legal, and computes another result: `a_wrong_fold_stage_order_builds_but_changes_the_output` runs it on the interpreter and sees the difference. The builder checks hazards, waits, lifetimes and VOPD legality; it does not check arithmetic. The interpreter does.

## Timing a variation

```text
cargo build --release -p rdna-compute --features lab --example pm_loop
pm_loop --kernel mq6_i8_gemm --shape fn:m10240k2560-n4096 --epi set --rip my_mq6.rip
```

`my_mq6.rip` is compiled for the `Spec`'s variant name (`128x128x8-set`; `--tile 256x128` or `--tile 128x256` selects the wide tiles' variants), refused unless its symbol, kernarg layout, workgroup size and LDS equal those of `mq6_i8_gemm::emit`, then linked with the native writer, certified (M7), checked bit for bit against the CPU reference and the shipped object, and ABBA-timed against the shipped kernel. The record carries `candidate.rip = {path, sha256, variant, identical_to_generator}`: the unmodified twin reports `identical_to_generator: true`. Run it on a gfx1201 card through the fabric (`fabric-run.sh --class timing`). Iterating on generator logic is: edit the `.rip`, test it on the interpreter, rerun `pm_loop`. No Rust rebuild.

## Why

- **One implementation.** `mq6_i8_gemm::emit`, its K-loop and its prologue are written over `mq6_i8_gemm::blocks`, the functions the natives call. The twin re-expresses the composition; the byte-identity test and the pinned digests keep the three equal.
- **The seal stays.** No native owns a ring, and no `Auth` and no `Lend` reach the script. Every ring transition, wait and barrier is a driver call (`begin_write`, `ds_store`, `ds_load_cur`, `wait_all`, `signal`, `wait_arrived`, `loop_carried`), checked as in any script.
- **What stays Rust.** Tile geometry (rows x tokens x waves: 128x128x8, 256x128x16, 128x256x16), the LDS layout, the register map and the DS/SZ ring depth (two buffers). They are spec axes of the generator: a change needs a rebuild. The A/W ring depth is the script's own `const RING` (2 or 3, see below). The script reads the tile only through `g.w_rounds()` (a tile with one weight round queues one decode / store).
- **gfx1201 only.** `Mq6Gen::new` refuses another target with the Rust message `mq6_i8_gemm targets gfx1201 only`.
- **`set_nr` is generator-only.** The Flash-Next QSA q / k epilogue (`Epi::SetNr`, `256x128` tile, entry `gemm_mq6g256v2_mmq_i8_pm_set_nr_gfx1201_t256x128`) re-lays the K loop's LDS out as one region (`nr_epilogue.rs`: staged token rows, inverse RMS, RoPE partner exchange) with typed barriers, then runs the unchanged F32 store loop. Its K loop is the `256x128` `set` loop (the same blocks, byte for byte: `set_nr_keeps_the_hot_loop_and_adds_the_documented_epilogue`); the twin has no `set_nr` variant, so `pm_loop` and `--rip` time the plain epilogues and `crates/rdna-compute/tests/qsa_norm_epi_oracle.rs` checks and times `set_nr`.

## Ring depth 3 (`const RING = 3`)

`const RING: u32 = 2;` at the top of `mq6_i8_gemm.rip` is the A / W payload ring depth. With `3` the script composes the generator's depth-3 K-loop (`Tune { ring: Three }`, `kernels/mq6_i8_gemm/kloop3.rs`) from the same natives, byte for byte (`the_ring3_twin_matches_the_generators_depth_3`): slabs are published two ahead through `Ring::new(a0, a1, a2)`, the raw fetch runs two ahead through two prefetch register sets (`g.pf_set(0)` / `g.pf_set(1)`; the half-1 refetch of a set takes no ledger-wide wait, `fetch_a` settles the stores' hazard itself), and the K-loop runs over a three-group period: `mq6::ring3_split` computes `r = (K/256 - 1) % 3` and `q = (K/256 - 1) / 3`, `wg.wg_if_else` picks the arm that stages slabs 0 and 1 with the rings `starting_at((3 - r) % 3)` and runs `r` groups, then `q` trips of three groups (`mq6::group_end`, `mq6::trips_dec`), then the peeled group, whose last block's B1 is the store-free `advance(ring)`. `Mq6Spec { .., ring: 3 }` (optional field, default 2) builds the matching `Mq6Gen`.

Depth 3 needs a third A+W slab of LDS and a second prefetch set of VGPRs, so `Spec::validate` refuses what does not fit, with the numbers: the shipped `128x128x8` tile fits (53248 B of LDS, 202 VGPRs against the 256 that two workgroups per WGP allow), the `256x128x16` and `128x256x16` tiles do not (79872 B against the 65536 B workgroup limit; 8 / 7 more VGPRs over the 192 budget at two workgroups per WGP). The depth-3 object keeps the depth-2 dynamic launch size and declares the third slab as `group_segment_fixed_size`, so fixed + dynamic covers the layout.
