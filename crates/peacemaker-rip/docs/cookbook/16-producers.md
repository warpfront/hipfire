# 16. The PM producers: move the norm reduction, the prefetch, a wait in `iu4_c8_producer_pm`

The producers of the A3B prefill route (`iu4_c8_producer_pm`: rotate, norm, sigmoid-gated rotate and norm-out; `moe_rotate_i4_pm` and its grouped-position twin; `gdn_chunk_prep_pm`) are untyped-`Builder` kernels, so their twins are written on the raw `Builder` surface like [cookbook 13](13-fp8-blocks.md) ([reference §18](../REFERENCE.md)). The natives are instruction builders over a generator handle (`C8Gen`, `RotGen`, `GdnGen`); the norm's LDS exchange, the loops of `gdn_chunk_prep` and the order of the phases are script statements, and the Builder's slot machine, wait ledger and hazard guards refuse a wrong edit.

Each twin ([`iu4_c8_producer.rip`](../../../peacemaker-isa/src/kernels/iu4_c8_producer.rip), [`moe_rotate_i4_pm.rip`](../../../peacemaker-isa/src/kernels/moe_rotate_i4_pm.rip), [`gdn_chunk_prep.rip`](../../../peacemaker-isa/src/kernels/gdn_chunk_prep.rip)) emits the bytes and the proof of the Rust generator for every export (`tests/producers_twin.rs`), and the Rust emit is pinned to the digests recorded before the blocks existed (`peacemaker-isa/tests/producers_pinned.rs`). Copy one, rename it, edit it. The claim below is an assertion in the twin test.

## What the c8 script owns

```text
emit:       Builder::new -> enable_delay_alu -> lds_slot (norms) -> c8::prologue -> front -> fwht_scale_store -> quantize -> store_record
front:      rotate_front | sigmoid_rotate_front | norm_front
norm_front: norm_row_loads -> norm_prefetch -> norm_sumsq -> norm_reduce_pre
            -> ds_store [norm_store_insn] -> barrier_with_scope [ready 0] -> norm_reduce_addr
            -> ds_load x2 [norm_load_insns] -> norm_reduce_post -> norm_rms -> norm_apply
```

`norm` and `norm_out` share the sequence; they differ inside `norm_reduce_pre` / `norm_reduce_post` (the fused RMSNorm reduces the wave first and then the eight wave sums, `rmsnorm_f32` order reduces across waves first and then the wave) and in what `norm_apply` stores.

## A fusion-relevant edit: issue the next phase's loads after the reduction's first input

`norm_prefetch` issues the next phase's loads (this lane's x, weight, AWQ and the two sign tables) right after the row loads, so they are in flight during the sum of squares and the reduction. Moving the sum of squares ahead of them is one swap:

```text
    c8::norm_row_loads(b)?;
    c8::norm_sumsq(b)?;
    c8::norm_prefetch(b)?;
```

Nothing numeric changes (the same instructions, every result bit identical): only when the loads are issued relative to the per-thread fma chain, which is how you test whether the loads or the reduction set the producer's latency before fusing the next consumer into it. The Builder accepts the new order and emits a different program (`moving_the_prefetch_behind_the_sum_of_squares_changes_the_program`); `pm_loop --kernel iu4_c8_producer_pm --export norm --rip FILE.rip` then runs M7, checks the outputs against the shipped producer and CPU reference, and times it. The LDS exchange is checked: reading the partials with the `ds_load`s before the barrier that publishes them is refused by the builder's slot machine (`reading_the_norm_partials_before_the_barrier_is_refused`).

## What stays Rust

The rotate (`FWHT128` / `FWHT256` butterflies), the C8 quantizer, the record store, the prologue of each producer, `gdn_chunk_prep`'s gate and row bodies (one trip of each loop, back branch included) and the conv_state advance are single blocks: their interiors are ordered instruction streams with no LDS statement to move. The script owns the order of the phases, the loops (`b.loop_`) and the unit-dispatch branches of `gdn_chunk_prep`, which export a `moe_rotate_i4_pm` / `down_gp` flavour is built, and where the drains (`prod::wait_all`) sit.
