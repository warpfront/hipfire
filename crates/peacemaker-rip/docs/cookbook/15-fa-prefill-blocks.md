# 15. The raw Builder, twice: move a barrier or a subtile in `fa_prefill`

[Cookbook 13](13-fp8-blocks.md) composes `fp8_gemm` on the raw `Builder`; `fa_prefill` (the gfx1201 GQA-fused FA2 prefill attention for H16 / KV2 / D256, [reference §17](../REFERENCE.md)) is the same kind of kernel with two schedules in one generator. Q8_0 K/V keep two 32-KiB planes resident (`k_plane`, `v_plane`, one barrier pair per tile); native fp8 E4M3 K/V reuse **one** 32-KiB stage sequentially, so a tile is fill K, publish, QK, retire, fill V, publish, softmax/PV, retire (four barriers). The twins [`fa_prefill.rip`](../../../peacemaker-isa/src/kernels/fa_prefill.rip) (variants `direct`, `preconvert`) and [`fa_prefill_fp8.rip`](../../../peacemaker-isa/src/kernels/fa_prefill_fp8.rip) (`direct`) write both schedules statement by statement. Each re-emits the bytes and the proof of `fa_prefill::emit` / `emit_preconvert` (`tests/fa_prefill_twin.rs`), and `emit` is pinned to the digests recorded before the blocks existed (`peacemaker-isa/tests/fa_prefill_digests.rs`). Copy one, rename it, edit it. The claims below are the assertions in that test file.

## What the script owns

```text
slots:    g.lds_slots() -> b.lds_slot     (k_plane, v_plane | stage)
prologue: fa::prologue
tile:     loop_(fa::LOOP) { tile_setup ; fill | fill_k ; labels ; barriers ; qk ; softmax_sub 0..4 ; barriers ; next_tile }
epilogue: fa::epilogue   (label EPI .. END, s_endpgm)
```

The Q8_0 tile, as the script spells it:

```text
fa::tile_setup(b, g)?;
fa::fill(b, g)?;
fa::label(b, fa::FILL_DONE)?;
b.barrier_with_scope([ready(0), ready(1)], "lds_only")?;
fa::label(b, fa::QK)?;
fa::skip_if_helper(b, fa::TURN)?;      // the fifth (helper) wave takes part in every barrier, nothing else
fa::qk(b, g)?;
for sub in 0..4 { fa::softmax_sub(b, g, sub)?; }
fa::label(b, fa::TURN)?;
b.barrier_with_scope([retire(0), retire(1)], "lds_only")?;
fa::next_tile(b)
```

## What the Builder refuses

- **A slot the kernel does not have.** `ready(2)` in the Q8_0 barrier, or `ready(1)` in the fp8 one (a single stage), is refused by the slot machine.
- **A dropped retire.** Removing `retire(1)` from the Q8_0 `TURN` barrier, or the `retire(0)` between QK and the V fill of the fp8 tile, leaves the slot Published/Reading, and the next store into it (the loop's next fill, or the V fill) is refused.
- **The other codec's fill.** `fa::fill` in the fp8 twin: `only the Q8_0 kernel fills K and V jointly`; `fa::fill_k` in the Q8 twin: `only the fp8 kernel fills K and V separately`.
- **An out-of-range subtile.** `fa::softmax_sub(b, g, 4)`: `softmax_sub: subtile 4 is outside 0..4`.
- **A spec outside the Rust gate.** `FaSpec { heads: 8, .. }` or any arch but gfx1201 is refused with the Rust message (`only heads 16, kv_heads 2, head_dim 256 are supported`, `built for gfx1201 only`).

## Variations that build

Two edits are different programs that break no rule, so they build and differ from the production bytes (`reordered_subtiles_and_an_extra_wait_build_and_differ`):

```text
for k in 0..4 { let sub = 3 - k; fa::softmax_sub(b, g, sub)?; }   // subtiles in reverse order
fa::qk(b, g)?; b.wait(Counter::Ds, 0)?;                           // an extra DS wait after QK
```

Whether the reversed order is still *attention* is not the builder's question (the online softmax is order dependent); timing it and checking the output are.

## Time it

`pm_loop` runs a twin on the card against the shipped bundle with the runtime's launch (the shipped preconvert once in setup, then the direct kernel), at an A3B chunk shape:

```text
pm_loop --kernel fa_prefill --codec q8_0 --rip fa_prefill.rip     --shape a3b:n4096-s4096
pm_loop --kernel fa_prefill --codec fp8  --rip fa_prefill_fp8.rip --shape a3b:n4096-s4096
pm_loop --kernel fa_prefill --codec q8_0 --export preconvert --rip fa_prefill.rip --shape a3b:n4096-s0
```

`correctness.pass` is bitwise identity with the shipped symbol. The direct kernel's CPU reference is **approximate** (an f64 causal attention over the dequantized K/V, tolerance 5e-3; no exact reference exists at production size) and the record says so; the preconvert's is exact (f16 RNE). `--emu` runs the candidate export on the emulator at N=16 and reports its own refusal where an instruction has no semantics.
