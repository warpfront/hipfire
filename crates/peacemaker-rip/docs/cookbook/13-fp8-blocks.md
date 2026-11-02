# 13. The raw Builder: move a barrier, a fold, a delay in `fp8_gemm`

[Cookbook 11](11-production-blocks.md) and [12](12-mq6-blocks.md) compose typed-core generators (`Workgroup`, sealed `Ring`s). `fp8_gemm` (MQ4V2 weights x E4M3 activations, `v_wmma_f32_16x16x16_fp8_fp8`, gfx1201) is not one: the shipped generator is an untyped-`Builder` kernel, so its twin is written on the raw `Builder` surface ([reference §15](../REFERENCE.md)). The natives are instruction builders; every LDS store and load, every barrier, every branch and the K-loop are script statements. Nothing is sealed, and nothing is unchecked: the Builder's slot machine, wait ledger, hazard guards and loop fixpoint refuse a wrong edit.

The committed twin [`fp8_gemm.rip`](../../../peacemaker-isa/src/kernels/fp8_gemm.rip) is the starting point. For each of the twelve valid `Spec`s (`ratio-256x128x8-row.set` ... `k128.qkvza`; `silu-bf16` and `qkvzagdn` are row-scaled only) it emits the bytes and the proof of `fp8_gemm::emit` (`tests/fp8_gemm_twin.rs`), and `emit` is pinned to the digests recorded before the blocks existed (`peacemaker-isa/tests/fp8_gemm_digests.rs`). Copy it, rename it, edit it. The claims below are the assertions in that test file.

## What the script owns

```text
emit:    Builder::new -> enable_delay_alu -> declare_lds -> prologue -> kloop -> fp8::epilogue -> label END -> s_endpgm
prologue: front, branch END, lanes, fetch A slab 0, publish A0, barrier [ready A0]
kloop:    block(first) ; trips ; loop_ { block ; block ; trip_end ; branch } ; peeled block
block:    fetch A1 -> K16 steps 0,1 on A0 -> publish A1 -> B1 barrier -> K16 steps 2,3 on A1 -> publish next A0 -> fold -> B2 split barrier
```

The barrier lists are plain statements:

```text
let mut b1 = vec![retire(0), ready(1)];
if next { b1.push(ready(2 + p)); if k128 { b1.push(ready(4 + p)); } }
b.barrier_with_scope(b1, "lds_only")?;
...
let mut b2 = vec![retire(1), retire(2 + p), ready(0)];
b.barrier_signal(b2)?;
b.barrier_wait()?;
```

## What the Builder refuses

- **A slot read before it is published.** Reading the first slab's fragments from slot 1 (`step(b, g, u, st, first, 1)` in the first half) before the B1 barrier publishes it is refused, naming `A1`.
- **A barrier over a slot that was never stored.** Adding `ready(4)` (the K128 D plane) to the row kernel's B1 list is refused: `invalid LDS transition Ready(4)`.
- **A dropped retire.** Removing `retire(1)` from the B2 list leaves A1 Published; the next block's publish into it is refused: `A1 cannot be stored while Reading`.
- **A K128-only block on a row spec.** `d_prepare`, `d_store_insn`, `d_restore` and the `fold_k128_*` natives refuse: `only the K128-scaled kernel has a D ratio`.
- **An out-of-range index.** `fp8::mma(b, g, unit, st, first, i + 16)` is refused by name.

## A variation that builds

`b.enable_delay_alu();` asks the builder to insert `s_delay_alu` hints. Delete the line and the script builds a different, shorter program with no `s_delay_alu` (`a_variation_without_delay_alu_builds_and_differs`). Moving a wait or a signal works the same way as in cookbook 12: edit, rebuild with `peacemaker_rip::compile`, and time it.

## Time it

`pm_loop` runs the twin on a card against the shipped bundle, with the same launch contract as the runtime (the shipped repacker once, then the GEMM), at a production shape:

```text
pm_loop --kernel fp8_gemm --rip fp8_gemm.rip --epi silu --shape M=17408,N=512,K=5120
```

`correctness.pass` is bitwise identity with the shipped symbol, and the record's `reference` is `none` (there is no CPU reference for this kernel; the record says so). `--emu` reports the emulator's own refusal: dense FP8 WMMA is unqualified there, and `pm_loop` does not model it.
