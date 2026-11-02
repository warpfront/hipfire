# 14. Per-tile branches, waits and the gate/up hand-off in `qwen4_moe_sym`

[Cookbook 12](12-mq6-blocks.md) moves a wait in a kernel whose blocks return `Instruction`s. `qwen4_moe_sym` (the MQ4 MoE expert GEMMs: `gate_up` with SwiGLU and `down`, NT1 / NT4 / NT8 tiles, plus the grouped-position `down_gp`) is written the same way. The committed twin [`qwen4_moe_sym.rip`](../../../peacemaker-isa/src/kernels/qwen4_moe_sym.rip) re-emits `qwen4_moe_sym::emit` / `emit_down_gp` byte for byte (`s_text` and proof) for all eight entries: `gate_up`, `gate_up_nt4`, `gate_up_nt8`, `down`, `down_nt4`, `down_nt8`, `down_gp_nt4`, `down_gp_nt8` (the pm_loop `--epi` names, gfx1201). Edit it and re-run; nothing is rebuilt.

## What the script owns

```text
author:   [gate_up: wg.lds::<Gate>("swiglu_gate", 0, g.lds_bytes())] -> wg.exit -> prologue -> kloop -> epilogue
prologue: front, exit_if(outside), expert, exit_if(sentinel), run, loop_until(run walk), leader,
          exit_if(follower), cnt, slot loads, descriptors, per-tile {offset, epoch-0 loads}, init
kloop:    wg_skip_if(no full trip) { loop_carried { load_w 1, fold 0, advance, load_w 0, fold 1, trips } },
          the tail: tail_pair, or if_else(odd) { tail_pair } { fold 0 }
fold:     compute_head, then per tile { wait_tile, skip_unless(cnt > j) { fold_tile }, load_x(next epoch) }
epilogue: down: store_tiles ; gate_up: bf16_roundtrip, handoff(up waves) { publish } { gate loads, silu, store_tiles }
```

Every native is a straight run of instructions emitted into `wg.isa()` (or an `Instruction` for a compare). The same Rust functions are what the generator calls, so there is no second copy of any instruction to drift.

## Move a wait

The fold of tile `j > 0` reads that tile's `d` and activation registers, which the previous fold issued loads for. The script waits for exactly those, then branches:

```text
if j > 0 {
    moent::wait_tile(wg, g, j)?;
    let live = wg.scmp(moent::tile_live_insn(g, j)?)?;
    wg.skip_unless(live, g.label(&format!("{}_dead{}", site, j)), (), |w, st| {
        moent::fold_tile(w, g, p, j)
    })?;
}
```

Delete `moent::wait_tile(wg, g, j)?;` and the build is refused (`clause requires 2..=32 independent instructions of one memory class without waits`: the next `load_x` clause is issued under pending loads). Put it after the `scmp` instead and the build is refused (`SCC was redefined between its compare and the branch that consumes it`: the wait's own SCC-setting counter instruction lands between the compare and the branch). The wait is where the script puts it, and the builder checks it.

## Move the descriptor step or a load in the K-loop

```text
moent::load_w(wg, g, 1)?;
compute(wg, g, 0, "l0", Some(1))?;      // fold epoch 0, re-issue each tile's epoch-1 activations
moent::advance(wg)?;                      // weight and activation descriptors one trip on
moent::load_w(wg, g, 0)?;
compute(wg, g, 1, "l1", Some(0))?;
```

Swap `advance` and the second `load_w` and the script builds (every instruction is legal) but emits a different program (`identical_to_generator: false`), and the second load now reads the stepped descriptors. A schedule that is wrong is caught on the card: `pm_loop` compares the candidate's output bitwise with the shipped entry's and then against the CPU reference before it times anything.

## The gate/up hand-off

On NT4 / NT8 `gate_up`, waves 0-1 hold the gate rows and waves 2-3 the up rows of the same tiles. After the BF16 round trip the gate waves publish their rounded rows through LDS and leave; the up waves read them back, take the SwiGLU and store. The script writes both sides:

```text
let up = wg.scmp(moent::up_insn())?;
wg.handoff(up, g.label("up"), end, r,
    |w, r| {                                      // the gate waves: write, publish, leave
        let _ = w.begin_write(r);
        for j in 0..g.nt() { for insn in moent::publish_insns(g, j)? { w.ds_store(r, insn)?; } }
        Ok(())
    },
    |w, r, end| store_tiles(w, g, end, Some(r))   // the up waves: loads, silu, stores
)
```

Drop the stores and the driver refuses the publication (`invalid LDS transition Ready(0)`: the region was never written). Change the reader condition (`up_insn` to a tile-liveness compare) and the program differs. Tile geometry and the LDS size are the generator's (`g.lds_bytes()`); the hand-off's statements are yours.

## Run it

```text
cargo build --release -p rdna-compute --features lab --example pm_loop
fabric-run.sh --class timing --card C --mem-gb 6 --max-secs 1500 -- \
  pm_loop --kernel qwen4_moe_sym --shape fn:gate_up_nt4-t4096 \
          --rip crates/peacemaker-isa/src/kernels/qwen4_moe_sym.rip
```

The record's `candidate.rip` carries `{path, sha256, variant, identical_to_generator}`. Unedited, the twin is `identical_to_generator: true` and the paired timing reports `identical_to_baseline` against the shipped entry. An edit that keeps the 64-byte ABI is checked on the card (bitwise against the shipped baseline and the CPU reference) and ABBA-timed.
