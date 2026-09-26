# Retained PM4 replay for the Qwen4 declarative program

## Status and purpose

**Status: implementation complete and REDLINE §7 gates 1–7 collected on
`feat/qwen38-flash-next`; gate 8 partly collected; no promotion claim.** G1–G3
plus G4's A2 (scope), C2 (census) and B1 (admission contract) landed: the
specialized sealed-MoE route is admitted, and a Qwen4 `.mq4r` decode body
captures, prepares and *replays* through retained PM4 on gfx1151 with
byte-identical output.

The §7 evidence is recorded in the certification section below: multi-position
state parity against ordinary HIP and the exact-kernarg HIP oracle (126,623,888
state bytes per position), the `tools.redline` route-proof ledger on both arms,
serve coherence, long-context behaviour to position 1547, request reset, and the
induced replay-failure contract. What is **not** collected: gate 8's model-swap
row (blocked by a pre-existing loader/VMM teardown guard that reproduces with the
route disabled) and the §8 author declarations, so nothing here is an admission,
a promotion, or a performance claim. AQL transport and prefill/speculative
bodies are out of scope and uncertified.

This record is the tracking document for enabling Redline retained PM4 replay for
Qwen3.8 Flash-Next (`hipfire-arch-qwen4`) **at the shared engine/dispatch level**,
without adding PM4-specific code to the architecture crate. `hipfire-arch-qwen4`
still contains zero Redline references after G1–G3.

Follow [the design-record lifecycle](README.md): this file is intent plus a
progress ledger, not a claim of implementation, admission, or speed. Runtime
validation authority remains [`../VALIDATION.md`](../VALIDATION.md); the retained
replay contract is [`../REDLINE.md`](../REDLINE.md); performance protocol is
[`../methodology/perf-benchmarking.md`](../methodology/perf-benchmarking.md).
G3/G4 decisions get their own linked record; do not rewrite this intent once
decided.

## Objective

Make one Qwen4 single-token decode program (the `Step` list built in
`crates/hipfire-arch-qwen4/src/gpu_forward.rs` and executed by
`hipfire-dispatch::pipeline::steps`) a **capture-complete, lowerable retained
tape** whose dynamic values are declared by the program and patched by the replay
layer.

Non-objectives, explicitly:

- No new Redline/PM4 code in `crates/hipfire-arch-qwen4/src/`. The family keeps
  binding typed descriptors; the engine owns the replay contract.
- No prefill, speculative, MTP, or multi-token retained body. REDLINE §3 scope is
  ordinary sequential single-token AR continuation.
- No promotion claim. Certification is the REDLINE §7 ladder, and §8 governs any
  number quoted.

## Baseline (measured, this box)

Fixture: `~/.hipfire/models/qwen3.8-flash-next.mq4r` (`.mq4r` + gfx1151 + single
GPU ⇒ `retained_redline_default` is true, `crates/hipfire-runtime/src/config.rs`).

```bash
# Reproduced 2026-09-20. Isolated HOME is required: the daemon takes an flock on
# $HOME/.hipfire/daemon.pid, and config.json must carry max_seq=2048 for qwen4.
mkdir -p /tmp/qwen4probe/.hipfire
jq -c '.max_seq=2048 | .port=11499' ~/.hipfire/config.json > /tmp/qwen4probe/.hipfire/config.json
HOME=/tmp/qwen4probe HIPFIRE_LOCAL=1 \
  target/release/hipfire run ~/.hipfire/models/qwen3.8-flash-next.mq4r "say hi" -n 8
```

Observed (default config): the engine's own retained default arms, then the
declarative program refuses at preflight — **before any decode work**:

```text
[redline] enabling fail-closed retained default on gfx1151 (model_arch=qwen4, drafter=off, transport=pm4)
[qwen4] 2560d 48L 248320 vocab
... qwen4_ar forward_chunk prefill failed: ... preflight Qwen4 typed program:
    Hip("sealed_moe: specialized sealed MoE has no retained-replay pointer contract; refusing before launch")
```

Observed with `HIPFIRE_REPLAY_BACKEND=hip` (same command family): the model loads
and generates on the ordinary HIP path — this is REDLINE gate 1 (healthy
baseline) and the A/B arm for later work. The artifact requires `max_seq=2048`.
The only failure seen with a small token budget is a caller-side framing check
(`open think span at end of generation (validation)`) for a reasoning model that
cannot close `<think>` inside the budget; it is not a GPU/dispatch error.

Two facts follow, and they order the plan:

1. The refusal is the sealed-MoE pointer-contract guard
   (`crates/hipfire-dispatch/src/pipeline/sealed_moe.rs`, `specialized_route &&
   gpu.replay.is_enabled()`), not a missing hook. `is_enabled()` is true for every
   non-`Hip` backend *including* manual shadow, so **no complete Qwen4 tape can be
   captured today**, for any model size.
2. The retained path is already the automatic default for this artifact on this
   arch, so the end-to-end census (G1's acceptance) is only observable once G4's
   guard is replaced. G1 and G2 are therefore verified by their own unit and
   structural evidence now, and by the end-to-end census as soon as G4 lands.

## Why this is an engine-level problem

Evidence established from this branch (symbols are anchors, re-read before
editing):

| Fact | Evidence |
|---|---|
| One recorder entry for the whole engine | `Gpu::launch_maybe_blob_bound` → `ReplayController::record_hip_launch_typed_bound` (`crates/rdna-compute/src/dispatch.rs`, `crates/rdna-compute/src/replay.rs`) |
| PM4 lowering is engine-owned and arch-selected, not arch-authored | `Pm4Architecture::from_name` (`replay.rs`); gfx11 family admitted, gfx1151 among them |
| Step lowering is dispatch-owned | `crates/hipfire-dispatch/src/pipeline/layer_ops.rs` |
| Positions are program data, not architecture logic | `GatedDeltaNetOp.start_position`, `IndexedAttentionState.position` (`layer_ops.rs`) |
| The AR loop is shared engine code | `generate_ar_with_forward` (`crates/hipfire-generate/src/ar.rs`), fed by two closures in `crates/hipfire-generate/src/qwen.rs` |
| The dispatch layer already gates on replay state | `gpu.replay.is_enabled()` guard (`sealed_moe.rs`) |

The one thing the architecture crate cannot supply from outside is *semantics of
its own scalars* (which kernarg word is a function of position, under which
formula). That is exactly what G2 makes declarable, and it is declared at the
*lowering* owner where the value is computed — still not the arch crate.

## Gap inventory

| Gap | What | Owner | Status |
|---|---|---|---|
| G1 | Half the decode program never reaches the recorder: `tensor_ops.rs` was 31 wrappers, 31 raw `Gpu::launch_kernel_blob`, 0 funnel launches | `crates/rdna-compute` | **done** (branch-implemented) |
| G2 | Binding vocabulary could not express a quotient/modulo of position, and declared bindings could not be attached to a launch | `crates/rdna-compute`, `hipfire-dispatch` | **done** (branch-implemented) |
| G3 | Position-switched kernel symbol and dynamic shared memory in QSA (`complete > 0` conditional launch, grid growing with position) | engine (lowering) + a kernel contract decision | **decided + done** (branch-implemented; route arming deferred to the G4 hook) |
| G4 | Sealed MoE "no retained-replay pointer contract" refusal: `route_policy` is `Some(Qt44Qt53Grouped)` for qwen4 decode | engine (dispatch) | **A2 + C2 + B1 landed**: scoped refusal, engine-side boundary, census, admitted with per-launch proofs — the route replays on gfx1151 |

## G1 — recorder funnel coverage (done)

**Problem.** `crates/rdna-compute/src/tensor_ops.rs` owns every Qwen4 trunk state
op (GDN step/conv/params/gate, HC hyper read/write/norm/activation, all QSA ops,
bf16 roundtrip, argmax). Every one of the 31 wrappers launched through the raw
`Gpu::launch_kernel_blob`, which returns before the recorder is consulted. A
capture that "succeeds" therefore contains a truncated tape; REDLINE §7 gate 2
rejects it, and the `capture_blobs.len() == recorded_launches().len()` parity
invariant does not catch it for qwen4 because qwen4 has no HipGraph path.

**Landed.** `Gpu::launch_blob_recorded` is the blob-shaped twin of
`Gpu::launch_maybe_blob`: same recorder entry, same exact-byte capture, same
HipGraph capture-blob accounting, same `last_kernel` attribution, and the same
`Result` shape. `Gpu::launch_kernel_blob` remains the raw entry used by the
funnel itself and by the recorded-HIP oracle, which must not re-record. All 31
`tensor_ops.rs` sites migrated; the duplicated artifact-alias table in the funnel
and in `scratch.rs` collapsed into one `recorded_launch_artifact` resolver, which
the new entry uses too.

**Evidence.**

- `tensor_ops.rs` contains zero `gpu.launch_kernel_blob(` calls.
- GPU-backed test `tensor_ops::tests::recorded_blob_launch_enters_the_tape_with_the_bytes_it_launched`
  (gfx1151): with no recording window the launch leaves no tape entry; inside one
  it records exactly one entry whose kernel, grid, resolved artifact and kernarg
  bytes (`ptr`, `elements`, `scale`) match what was launched, while the kernel
  still executes (readback equals the expected product). The test fails against
  the pre-G1 raw entry.
- `cargo test -p rdna-compute --lib` (260) and `cargo test -p hipfire-dispatch`
  (285 + 1 ignored) pass; the QSA/HC GPU tests in the module exercise migrated
  sites.
- Ordinary-HIP end-to-end probe (below) still runs the model.

**Not yet evidenced.** The end-to-end census (recorded launches == compute
launches for one decode forward) needs G4, because the sealed-MoE guard refuses
before any decode forward completes.

## G2 — declared dynamic bindings (done)

**Problem.** The vocabulary was grid `PositionCeilDiv` (rounds up, narrow-only)
and kernarg `PositionPlusU32` / `GdnFrameU32`. Qwen4 already computes, in dispatch
lowering, scalars the vocabulary cannot name:

- GDN conv `cursor = (start_position + row) % history_rows` (`layer_ops.rs`);
- QSA `block_count = final_position / compress`, `budget_blocks = budget / compress`
  (`layer_ops.rs`).

Today such a difference is discovered only by differencing two recordings
(`synthesize_position_bindings`), which fails closed on anything non-affine.
REDLINE §4 requires dynamic fields to be *named* with an owner; the fix is a
declared binding, not a smarter heuristic.

**Landed.**

1. `ReplayKernargBinding::PositionDivU32 { offset, addend, divisor }` and
   `PositionModU32 { offset, addend, modulus }`, applied through the single shared
   helper `apply_kernarg_bindings_for_dispatch`, so PM4 and the recorded-HIP
   oracle cannot diverge. Zero divisor/modulus, out-of-range offset, and u32
   overflow are explicit errors; all four kinds write through one bounds-checked
   `write_kernarg_u32`.
2. A declared set travels with the launch into `RecordedHipLaunch` and is merged
   at prepare by `merge_declared_kernarg_bindings`, which fails closed on a second
   owner for one `(dispatch, offset)` and rejects a non-position-derived
   declaration (the GDN frame counter is owned by the replay layer, which derives
   it from the recorded launch).
3. `synthesize_position_bindings` skips declared offsets: a named field is not an
   unexplained difference.
4. Declared bindings are part of tape identity (`replay_sequence_hash`), and the
   three duplicated offset `match` arms collapsed into `ReplayKernargBinding::offset()`.
5. First consumers, both at the lowering that computes the value:
   `gated_delta_conv` (ring cursor, `addend = row_index`) and
   `gated_delta_conv_batched` (chunk `start_cursor`, `addend = 0`). The new
   `GatedDeltaConv::row_index` datum is supplied by `layer_ops`, so the
   declaration is exact for a single-token decode and for a multi-row chunk
   alike. Kernel width 1 declares nothing (no ring to index).
6. QSA quotient declarations are deliberately **not** made; they belong to G3.

**Evidence.** Six new unit tests in `crates/rdna-compute/src/replay.rs`
(`position_div_binding_rederives_the_host_quotient`,
`position_mod_binding_rederives_the_ring_cursor`,
`declared_binding_is_not_an_unexplained_kernarg_difference`,
`declared_binding_collision_fails_closed`,
`a_declared_frame_counter_is_not_a_position_binding`,
`declared_bindings_are_part_of_tape_identity`); 87 `replay::` tests pass.

**Not yet evidenced.** Nothing here has been through PM4 preparation, because a
Qwen4 tape cannot yet be captured or prepared (G3/G4).

## G3 — QSA geometry, shared memory, and position fields

**Status: decided and implemented (branch-implemented, uncertified).** Decision:
capacity-fixed pool grid, capacity-pinned LDS with a constant symbol, and every
position-derived field declared at the launch that computes it. Evidence below;
the alternatives and their weights are kept in this record.

### The problem, precisely

The Qwen4 QSA step lowers ~13 launches per full-attention layer. Retained replay
requires a fixed launch sequence (no conditional presence), a fixed symbol, a
fixed block and shared-memory size, a grid that is fixed or only narrows from a
recorded maximum, and every position-derived field either absent or declared.
Four properties of the QSA step violate that, and one of them fails *silently*:

**P1 — the pool launch appears and grows.** `indexed_attention_pool_rope_f32` is
launched only when `block_count > 0` (`layer_ops.rs`), and its `grid.x` is that
position-derived count. The kernel itself is mask-safe (`if (block >= block_count
|| compress <= 0) return;` precedes every read, `kernels/src/tensor_ops.hip`), so
an oversized grid is legal; the wrapper rejects `block_count == 0`
(`crates/rdna-compute/src/tensor_ops.rs`). The count is
`complete = (position + rows) / compress`, monotone non-decreasing, so the
condition flips exactly once, at `complete == 1`.

**P2 — dynamic shared memory varies with position.** Both remaining QSA shapes
size their LDS from position-derived lengths: `indexed_attention_select_*` uses
`block_count * 4` bytes, `indexed_attention_attention_*` uses
`max_selected * 8` where `max_selected = min(end_position, capacity, budget *
compress + compress - 1)`. Each wrapper picks between a batched (dynamic LDS) and
a `_serial` (shared memory 0) symbol when the request exceeds a 64 KiB limit.
REDLINE §4 makes symbol + grid + block + shared memory one identity contract and
requires a "replay-stable fixed/tiled design" for a changing shared-memory shape,
so the *value* must become constant even though the symbol happens not to flip.

**P3 — one position-derived device pointer.** `raw_batch = view(raw_index_keys,
initial_position * index_kv_width, …)` (`layer_ops.rs`) bakes a position-shifted
address into the `copy_rows_strided_f32` destination pointer. Pointers are not
scalars; no binding kind covers an address that moves with position.

**P4 — every position-derived field must be *declared*, because the automatic
route has no calibration pass.** `synthesize_position_bindings` (which
differences two recordings and classifies what changed) is called only from the
manual/speculative path; the automatic MQ4R route goes `Captured → Ready` on a
single recording (`docs/REDLINE.md` §3). There is therefore no mechanism that
notices a stale position-derived kernarg field. A tape whose QSA launches carry
undeclared `position_start` / `block_count` / shifted pointers would replay the
capture-position values: **wrong output, no error, no fallback**. G2's declared
bindings are the mechanism that makes this class explicit, and they are mandatory
here rather than an optimization.

### Admitted geometry (source-derived)

| Quantity | Value | Source |
|---|---|---|
| `max_seq` | exactly 2048 (admission requires it) | `crates/hipfire-loader/src/admission.rs` |
| `compress` / `budget` | 4 / 2048 | `crates/hipfire-arch-qwen4/src/config.rs` |
| indexer heads / kv heads / index_dim | 4 / 1 / 128 | same |
| main heads / kv heads / head_dim | 24 / 2 / 256 | same |
| `qsa_selected_capacity` | `budget + compress - 1 = 2051` | same |
| `pooled_capacity` | `ceil(max_seq / compress) = 512` | `crates/hipfire-arch-qwen4/src/state.rs` |
| pool `grid.x` | `complete`, 1…512 | wrapper + config |
| select LDS | `4 * complete` ≤ 2048 B | wrapper |
| attention LDS | `8 * max_selected` ≤ 16 408 B | wrapper |
| attention grid | `[24, 1, 1]` (24 workgroups) | `n_heads=24`, `head_dim=256`, block 256 |

Consequences that shape the decision: **neither 64 KiB switch flips anywhere in
the admitted range** (attention needs `max_selected > 8192`); for `complete ≥ 1`
(positions ≥ 3) the symbol set is already constant (batched everywhere); and both
dynamic-LDS requests are small enough that a capacity-sized reservation fits the
64 KiB device limit with room to spare. The attention grid is 24 workgroups on a
40-CU device, so LDS reservation cannot become an occupancy limiter there.

### Decision and measurement (2026-09-20)

**Decided: A1 + B1 + C1.** Capacity-fixed pool grid, capacity-pinned LDS with a
constant symbol, position-derived fields declared at the launch that computes
them. A2 (dynamic grid narrowing) was dropped once A1 measured free: it would add
an environment gate, a single-queue restriction, and a PM4-only grid binding that
the recorded-HIP oracle does not share, for no measurable gain. B2 was dropped
because the select `_serial` variant launches one thread per row. C4 stays the
fallback if the declared-field count ever becomes unwieldy.

What the measurement showed (fixture: `qwen3.8-flash-next.mq4r`, md5
`fda74d3760dc803e778e9b30a2fe0ebd`; binary md5s recorded in `/tmp/qsa-shape-cost.log`;
prompt `benchmarks/prompts/qwen4_ar_primes.txt`, md5 `0508eec29a44323f62e70fa77d92b834`;
greedy `-n 512 -t 0`, HIP backend, isolated `HOME` with `max_seq=2048`):

| Arm | runs (tok/s) | stream |
|---|---|---|
| position-derived shapes (before) | 10.9, 11.5, 11.4, 8.1, 11.5 → median 11.4 | `2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37`, 116 tokens, `finish=stop` |
| capacity-pinned shapes | 11.4, 11.4, 11.4, 11.4, 11.4 → median 11.4 | byte-identical to the arm above in all 5 pairs |

Interleaved fresh-process pairs. The pinned arm showed no spread at all in five
samples while the derived arm showed one 8.1 outlier; that variance difference is
recorded as an observation, not a claim (5 samples, unknown cause). The
conservative direction is the measurement itself: both knobs are at their worst at
a short context (the attention LDS reservation is pinned at its maximum while the
live length is small, and the pool grid masks the largest share of workgroups), so
this is the arrangement most likely to expose a cost.

Unit evidence, all on gfx1151:

- `tensor_ops::tests::pinned_qsa_shapes_are_bit_identical_to_derived_shapes` —
  an oversized masked pool grid, a larger select LDS reservation, a larger
  attention LDS reservation, and the batched select symbol at an active count of
  zero (against the serial symbol it replaces) are each bit-identical.
- `tensor_ops::tests::qsa_position_fields_are_declared_to_the_recorder` — with a
  recording window open, each QSA launch reports the declared binding the lowering
  intends, and the recorded bytes *at that offset* equal the value the launch
  used, so a drifted offset fails in the test rather than at replay.
- `test_copy_rows_strided_f32_parity` (7 cases, now including
  `row-absolute/dcol=7*len`) — the relaxed `copy_rows_strided_f32` contract is
  bit-exact against the per-row `copy_d2d` reference.
- `cargo test -p rdna-compute --lib` 263 pass; `cargo test -p hipfire-dispatch`
  285 pass.

**Not evidenced.** No tape has been captured, prepared, or replayed: the shapes
are validated on the ordinary HIP path only, and G4 still refuses any Qwen4
forward with a replay backend enabled. The pool launch also remains absent for the
first `compress-1` positions (`layer_ops` keeps `if complete > 0`), so the route
that eventually arms a tape must arm once `complete > 0` — a monotone, one-time
condition — or the pool kernel must accept a zero count. That arming decision
belongs with the G4 hook and is deliberately not pre-empted here.

**Bound-of-record.** `pooled_capacity = ceil(max_seq / compress)` and
`qsa_selected_capacity = budget + compress - 1` are both derived from the admitted
geometry, so the pinned shapes cover every position the admitted configuration can
reach. A different `max_seq` is a different tape identity, which is correct: it is
a reload with different capacities.

### Options

**P1, pool launch presence and grid**

- **A1 — capacity-fixed grid (recommended).** `grid.x = pooled_capacity` (512),
  `block_count` passed as a declared `PositionDivU32 { addend: 1, divisor:
  compress }`. The kernel masks, so active work is unchanged; the cost is one
  compare-and-return per inactive workgroup (≤512 per layer per token, only until
  `complete` saturates). Pros: no new mechanism, no env dependency, no kernel
  change, one tape for every position. Cons: bounded wasted work (~0.5–1 % of a
  token by workgroup-count arithmetic, unmeasured), and the grid no longer
  encodes the active count (readability: the binding and the scalar must stay
  consistent).
- **A2 — dynamic grid narrowing.** Keep `grid.x = complete` at capture and
  declare `ReplayGridBinding::PositionCeilDiv { axis: 0, addend: 1, divisor:
  compress }` (`ceil((p+1)/4) == complete` for `rows == 1`, verified identity),
  prepared at a declared maximum position (`set_prepared_max_position`, with the
  existing `position > prepared_max_position` refusal). Pros: exact grid, zero
  wasted work. Cons: needs `HIPFIRE_REPLAY_PM4_DYNAMIC_GRID` (off by default) to
  even record the binding, forces single-queue PM4, and the grid binding is
  PM4-only, so the recorded-HIP oracle launches a different grid than the PM4
  route (numerically identical because the kernel masks, but it is one more
  difference to explain in the parity ledger).
- **A3 — kernel change: capacity tiling.** Rejected: same result as A1, but pays
  a kernel/ABI re-certification for no functional gain.
- **A4 — keep the conditional by arming later (orthogonal, recommended).** Require
  the retained route to arm only once `complete > 0` (position ≥ 3). The
  condition is monotone, so the tape captured after that point stays valid
  forever, and the first ≤3 decode steps run on HIP. Removes the presence
  problem *and* the zero-count select edge case in one move, with no wrapper or
  kernel change. Alternative D2 below if a uniform-from-position-0 tape is
  preferred.

**P2, dynamic shared memory**

- **B1 — capacity-pinned LDS (recommended).** Size the reservation from the
  declaration instead of the active value: select `= pooled_capacity * 4` (2048 B),
  attention `= qsa_selected_capacity * 8` (16 408 B), with the symbol chosen from
  the pinned shape and the active lengths left as declared scalars. Both kernels
  index their dynamic LDS by the active counts, so a larger reservation is not
  read. Pros: constant symbol/block/shared-memory (the whole P2 contract becomes
  position-free), arithmetic untouched (LDS size never changes which values are
  computed, only where they are staged), free at these grid sizes. Cons: it is an
  *assumption about the kernels* — "reservation ≥ active need is safe" must be
  pinned by a test per variant, and the capacity arithmetic must stay under 64 KiB
  if `max_seq` ever grows.
- **B2 — force the `_serial` variants.** Pros: shared memory 0 everywhere, the
  simplest possible contract. Cons: disqualifying for select — `_serial` launches
  with block `[1,1,1]`, i.e. one thread per row scanning every candidate block;
  for attention `_serial` recomputes the dot per pass (bit-identical, but more
  work). Only viable for attention, and only if B1's assumption fails.
- **B3 — patch `shared_mem` at replay.** Rejected: REDLINE §4 forbids a changing
  shared-memory assumption outright; the PM4 packet field is patchable but the
  semantics are not admissible.
- **B4 — kernel change: explicit `lds_capacity` argument.** Fallback if B1's
  premise is falsified. Pays a kernel re-certification; no benefit over B1 while
  the premise holds.

**P3/P4, position-derived fields**

- **C1 — declare scalars; replace the shifted pointer with the existing offset
  parameter (recommended).** `copy_rows_strided_f32` already takes `dst_col_offset`
  as an `i32` kernarg (offset 32 in its blob). Pass the *base* `raw_index_keys`,
  `dst_row_stride = index_kv_width`, and `dst_col_offset = position *
  index_kv_width` — the row mapping `dst[r * index_kv_width + position *
  index_kv_width + c]` is identical for decode and for a `rows > 1` chunk — and
  declare that slot with a new sibling `PositionMulU32 { offset, factor }`.
  Declare the remaining scalars with the vocabulary G2 landed: `position_start`
  as `PositionPlusU32 { addend: 0 }` on the norm/RoPE, cache-append, select and
  attention launches, `block_count` as `PositionDivU32 { addend: 1, divisor:
  compress }` on pool and select. Pros: no kernel change, no new pointer class,
  uniform for decode and chunked prefill, and it removes the silent-staleness
  hazard for exactly the fields that carry it. Cons: ~9 declared bindings per QSA
  layer (host-side 4-byte patches between replays — cheap, but it is per-layer
  bookkeeping to keep honest), plus `PositionMulU32` is a third arithmetic form in
  the vocabulary, and the params-shaped funnel needs a bindings-aware entry
  (or the copy converts to the blob entry).
- **C2 — keep the shifted view, add a pointer binding.** Rejected: an 8-byte
  address patch needs the capture position and the base-address relationship
  inside the tape contract; strictly more machinery than C1 for the same result.
- **C3 — write raw index keys with the existing cache-append kernel** (pass the
  same tensor as key and value): no new binding kind, but it writes the same 128
  values twice and misuses an append contract for a keys-only cache. Viable,
  less honest than C1.
- **C4 — move position into a device buffer** (kernels read position from memory;
  the pattern the other MQ4R models use for the position-buffer H2D). Pros:
  removes position scalars from the tape entirely. Cons: 4–5 kernel signature
  changes plus re-certification, and it does not address P1 or P2 at all. Keep as
  a fallback if the declared-scalar count becomes unwieldy, not as the first move.
- **C5 — rely on recording differencing (do nothing).** Rejected outright: the
  automatic route never calls `synthesize_position_bindings` (P4), so this is a
  silent-wrongness option, not a cheap one.

### Weighing

| | Mechanism cost | Device cost | Kernel/ABI risk | Failure mode if wrong |
|---|---|---|---|---|
| A1 + B1 + C1 + A4 | lowering only | ≤512 masked workgroups/layer/token; LDS reservations free at 24-worker grids | none | loud: binding/owner mistakes fail at prepare |
| A2 + B1 + C1 + A4 | lowering + env flag + prepared max | none | none | clamp at `prepared_max_position` refuses (fail closed) |
| A1 + B2(attention) + C1 + A4 | lowering only | attention dot recomputed twice | none | loud |
| A1 + B4 + C1 + A4 | lowering + kernel | unknown | kernel re-certification | kernel change invalidates the base tape |

The recommended package is **A1 + B1 + C1 + A4**: it is the only column with no
kernel change, no environment dependency, and no silent failure mode. Its total
device cost is bounded by ~512 masked workgroups per QSA layer per token plus two
constant LDS reservations, and both are unmeasured — which is what the next step
must fix.

### Landing

- `IndexedAttentionPoolRope` takes a declared `grid_bound` (the lowering passes
  `pooled_capacity`) and, when the caller declares a position source, the active
  count as `PositionDivU32 { addend: rows, divisor: compress }`; the wrapper
  verifies the caller's count *equals* the declared formula, so the declaration
  cannot drift from the launch.
- `IndexedAttentionSelectBatch` takes a declared `shape_blocks` (LDS + symbol come
  from the bound, not the active count) and declares both its active count and
  `position_start`. `IndexedAttentionAttentionBatch` takes a declared
  `shape_selected` and declares `position_start`.
- `indexed_attention_norm_rope_batch` and `indexed_attention_cache_append_batch`
  declare `position_start`.
- The index-key write no longer bakes `position * index_kv_width` into a device
  pointer: `copy_rows_strided_f32` receives the base tensor plus a
  `dst_col_offset` scalar (its row-absolute contract is now bounded by the
  destination extent rather than by one row pitch) and declares it with the new
  `ReplayKernargBinding::PositionMulU32`. Kernarg offsets are captured where the
  scalar is written (`args.len() - 4`), never hand-counted.
- The `HIPFIRE_QSA_STABLE_SHAPES` measurement gate is gone: the pinned shape is the
  only production shape, so the HIP and replay paths cannot diverge.

**Remaining for the route (with G4):** arm the tape once `complete > 0`, or teach
the pool wrapper to accept a zero count. Also outstanding: the launch census
(REDLINE §7 gate 3) that asserts every position-derived kernarg in a *recorded*
Qwen4 forward is declared — it needs a capturable forward, i.e. G4.

## G4 — the sealed MoE pointer contract and the refusal's blast radius

**Status: analyzed, no decision taken.** No code changed. Evidence below is
source-derived (two read-only surveys plus direct reading); option weights are
judgements and are marked as such.

### The problem, precisely

`crates/hipfire-dispatch/src/pipeline/sealed_moe.rs:1829-1837`:

```rust
let specialized_route = match &self.params {
    SealedParams::Decode(params) => params.route_policy.is_some(),
    SealedParams::Prefill(params) => params.route_policy.is_some(),
};
if specialized_route && gpu.replay.is_enabled() {
    return Err(invalid(
        "specialized sealed MoE has no retained-replay pointer contract; refusing before launch",
    ));
}
```

Qwen4 binds that policy unconditionally (`crates/hipfire-arch-qwen4/src/program.rs`
decode and prefill, plus the legacy `execute_moe`), `is_enabled()` is
`request != Hip && state != Fallback`, and the guard sits in `validate_for_gpu`,
which runs at **preflight** — before any launch, for **every** forward, including
prefill.

Two distinct defects are tangled in that one predicate:

**D1 — the refusal's blast radius exceeds the retained contract.** The tape never
exists for prefill (REDLINE §3 requires prefill to stay outside it), and
`Fallback`/`Hip` forwards would run HIP anyway. Refusing them means a `.mq4r`
Qwen4 artifact on gfx1151 (where `retained_redline_default` arms automatically)
**cannot serve at all**: no prefill, no generation, no fallback. The reproduced
probe in the Baseline section is exactly this. The engine's own fallback semantics
(REDLINE §3: a poisoned route falls back to HIP; an ineligible forward never
records or replays) describe the correct scope, and the current guard does not
implement it. There is no Qwen4 arming/poison hook at all
(`crates/hipfire-arch-qwen4` contains no replay reference), so even a scoped guard
would leave the capture window erroring per token instead of degrading to HIP.

**D2 — the route has no *stated* contract, so the engine refuses instead of
validating.** The survey says the route is already close to conformant:

| Fact | Evidence |
|---|---|
| Expert pointer tables are built **once at load** from load-time device addresses and uploaded once; nothing rewrites them (no Qwen4 weight pager) | `gpu_forward.rs:629-658`, freed only in `free_gpu:694-706` |
| Every kernarg pointer is a model/`Gpu`-lifetime tensor, a pure address-arithmetic slice view, or geometry — no per-forward host-built table, no per-forward H2D inside the sealed call | `mod.rs:393-405`, `dispatch.rs:317-325`, `gpu_forward.rs:643/654` are the only `memcpy_htod_auto` on the path |
| **No position-derived scalar exists anywhere in this route** — every non-pointer kernarg is m/k/batch/top-k/n_exp | `gemv.rs:11884-11892`, `moe.rs:1797-1810` |
| Pointer identity is already re-proved **every seal** against identities frozen at a one-shot bind | `validate_live_binding` (`sealed_moe.rs:3957-4015`), `bind_live` one-shot (`:1148-1172`), `build_live_binding` (`:3476-3545`) |
| A per-expert pointer **mapping fingerprint** already exists | `mapping_fingerprint`, `sealed_moe.rs:3523-3545` |
| Single-rank, single-device only; EP/root-routed/gather paths are never entered | `:1759-1764`, `:1822`, `:1777-1791` |
| The generic path is **not** an alternative: `MQ4G128V2` down requires the policy, and the k=10 generic path is CPU-host-routed and separately refused under capture | `sealed_moe.rs:2511-2516`, `families/moe.rs:493`, `moe_program.rs:1131-1134`, `sealed_moe.rs:1746-1753` |

What is genuinely missing is therefore small and specific:

1. **Table *contents* are not proved on the Single path.** `validate_pointer_table`
   checks dtype and byte capacity only; the *compact* path additionally proves each
   owned entry points at its own local tensor (`sealed_moe.rs:3660-3680`), the
   Single path does not. A stale entry would be dereferenced by the kernel while
   the live-expert check looks satisfied.
2. **The mapping fingerprint is not part of any plan identity.** It is
   dispatch-private, so a retained plan cannot pin "the pointer mapping this tape
   was captured against" and re-prove it at prepare/replay.
3. **Lifetime hazards outside the route's own checks are unnamed**: `gpu.scratch`
   FWHT sign tables are lazily allocated on first use (`gemv.rs:3437-3443`,
   `3560-3562`, `scratch.rs:366-420`) — if first touched *inside* a recorded body
   the tape holds an address nobody promised to keep, and a scratch rebuild while
   a plan lives would dangle it; `HIPFIRE_DUMP_HIDDEN` performs a D2H inside the
   route; `invalidate_for_kv_mode_switch` and model reset/swap must invalidate the
   plan (the switch path already poisons, `dispatch.rs:4253-4255`).
4. **No census exists.** Launch count and geometry stability across positions are
   unverified (REDLINE gate 2), and the census needs a capturable forward — which
   the refusal prevents, while the census is one of the things that would justify
   admission.

### Options

**D1 scope — how wide should the refusal be?**

- **A1 — keep the blanket guard.** Rejected: it makes the model unservable, blocks
  the prefill evidence as well, contradicts REDLINE's fallback semantics, and is
  the one behavior in this area with a product-visible regression. Cost if kept:
  every `.mq4r` Qwen4 forward fails on gfx1151.
- **A2 — scope to the eligible retained body (recommended, independently).**
  Refuse only when the controller would record or route
  (`is_recording() || should_route_pm4() || should_route_aql()`); prefill, `Hip`,
  and `Fallback` forwards launch normally, and a failure inside the capture window
  poisons the controller (sticky fallback) instead of erroring each request. Needs
  the Qwen4 arming/poison/fallback hook — which is also where G3's "arm once
  `complete > 0`" condition lives, so the two remaining items share one hook.
  Pros: restores serving and the prefill evidence; keeps the tape fail-closed
  (capture still refused until D2 is closed); small, testable change (a `.mq4r`
  serve with Redline armed must reach HIP and log a fallback reason). Cons: no
  tape yet; adds a hook whose correctness (poison-on-capture-failure) matters and
  must be tested.
- **A3 — A2 plus admission (D2).** The actual goal.

**D2 contract — how to obtain an admissible route?**

- **B1 — validate-and-admit by extending the existing machinery (recommended).**
  (i) Prove table contents on the Single path (mirror the compact path's entry
  check); (ii) expose the mapping fingerprint as plan identity and require
  capture/prepare/replay to observe the same fingerprint; (iii) name the lifetime
  rules: sign tables resident before the capture window, no scratch rebuild while a
  plan lives, `HIPFIRE_DUMP_HIDDEN` refused while recording, existing invalidation
  paths wired to the plan; (iv) replace the blanket guard with these checks plus
  *named* refusals for the sub-cases that stay out (paged residency, EP>1,
  host-routed fallback); (v) census as gate-2 evidence, with a negative test per
  refused shape. Pros: the route already satisfies the substance per the inventory;
  the checks are cheap and two of them already exist in the compact path; no kernel
  or lowering change; the refusal becomes narrow and self-explaining. Cons: the
  proof burden lands here; the census needs a capturable forward, which needs
  either this admission to be complete or the diagnostic route below.
- **B2 — make stability structural.** Absolute (base-relative) expert addressing or
  a device-side indirection so that residency/placement changes never touch a
  recorded kernarg, plus load-time sign tables. Pros: the contract becomes trivial
  and survives a future weight pager. Cons: kernel/dispatch work for hazards
  nothing currently exercises (no pager, single rank, table written once); the
  cheap half (sign tables resident before capture) belongs in B1 regardless.
  Defer.
- **B3 — run the retained body over the generic route.** Rejected on evidence:
  `MQ4G128V2` down is refused without an architecture-declared policy, and the k=10
  generic path is host-routed (CPU top-K with readback) and separately
  capture-refused. There is no second route to fall back to.
- **B4 — do not admit; keep the route HIP-only.** Honest and cheap. Two spellings:
  (a) add a Qwen4 carve-out to `retained_redline_default` — the Muse Glimmer
  precedent, whose comment says automatic admission is withheld "until that
  lowering lands", so a carve-out would be policy-consistent; and/or (b) rely on
  A2's scoping. Pros: no risk, removes the brick, keeps the product honest about
  an unadmitted route. Cons: G1–G3 stay latent, the census stays unreachable, and a
  carve-out would be a policy decision taken *without* the measurement that would
  justify or refute it. Fallback position, not a first move.

**C — evidence sequencing.**

- **C1 — contract first, then census.** The REDLINE-correct order, but the census
  is the evidence that tells us whether the contract's assumptions hold (launch-set
  and geometry stability across positions), so it is partly circular.
- **C2 — diagnostic capture first, then contract (recommended with B1).** Add an
  explicit, non-default diagnostic that lets a capture proceed with the specialized
  route for *measurement only*: it must not install a plan, must not serve, and
  must never count as route proof — the same posture `HIPFIRE_REPLAY_MANUAL_CAPTURE`
  already has. This yields the launch census and the shape-stability answer (G3's
  last outstanding item) before the admission rule is frozen. Pros: breaks the
  circularity with an existing precedent; cheap. Cons: it is a bypass of a
  fail-closed guard, so its scope, naming, and evidence class must be explicit in
  the code and in this record.

### Weighing

| Package | Cost | Risk of silent wrongness | Evidence produced | Verdict |
|---|---|---|---|---|
| A1 | none | none (refuses) | none | reject — model unservable |
| A2 | low (scope + poison + Qwen4 hook) | low (fallback path must be tested) | serving restored; prefill evidence | **do first**, independently of D2 |
| A2 + B1 + C2 | medium (contents proof, fingerprint identity, census, hook, negative tests) | low — every new check fails closed | census, mapping identity, named refusals, then the REDLINE ladder | **recommended path to admission** |
| A2 + B2 | high (kernel/dispatch) | low | trivial contract | defer; fold the sign-table half into B1 |
| A2 + B3 | — | — | — | reject (no such route) |
| A2/B4 | low | none | none | fallback if the census refutes B1's assumptions |

Recommendation: **A2 now** (it is a defect in its own right: a contract question
must not make a model unservable), then **B1 with C2** for admission, with B4 held
as the honest fallback if the census shows the launch set or geometry moving with
position.

### Landing (A2)

**Problem D1 fixed.** The refusal is now scoped to the retained body instead of
"any replay backend enabled": `ReplayController::retained_body_active()` is
`is_recording() || should_route_aql() || should_route_pm4()`
(`crates/rdna-compute/src/replay.rs`), and the guard refuses only then, through the
single admission point
`hipfire_dispatch::pipeline::sealed_moe::specialized_sealed_moe_retained_admission()`
— so the launch guard and the arming hook read one rule and cannot drift.

**Qwen4 gained the retained-body discipline** at the engine level
(`crates/hipfire-generate`: `ar::retained_body_action` + the Qwen4 AR producer's
closures), keeping `hipfire-arch-qwen4` free of replay code:

- prefill marks itself ineligible, so it can neither record nor consume a tape;
- plain single-token decode is the eligible forward; with admission refused it
  **poisons before running the body** and logs the reason, so this and every later
  forward runs HIP instead of arming a capture the route would refuse per token;
- a capture window that fails inside the body poisons rather than failing every
  following forward;
- a routed state (`Ready`) fails closed while no plan can be prepared for this
  family — it never silently runs HIP behind a plan the engine believes is in use.

Deliberate deviation from the earlier sketch: the "arm once `complete > 0`" gate is
**not** implemented here. Arming is refused wholesale today, so the gate would be
dead code; it belongs with B1, where capture-completeness (the pool launch being
present) is knowable. This is recorded rather than half-built.

**Evidence.**

- `replay::tests::retained_body_scope_is_the_eligible_forward_not_the_backend_choice`
  — armed-but-idle, captured-but-unprepared, ineligible-inside-a-window, and
  poisoned states are all *not* the retained body.
- `ar::route_scope_tests::retained_body_action_keeps_hip_for_every_non_eligible_or_refused_forward`
  — the action matrix across all seven replay states.
- End-to-end, two fresh processes, `.mq4r` artifact with the Redline default
  **armed** (the configuration that previously failed at prefill preflight):

  ```text
  [redline] enabling fail-closed retained default on gfx1151 (model_arch=qwen4, drafter=off, transport=pm4)
  [redline] qwen4 retained body unavailable: specialized sealed MoE has no retained-replay pointer contract; refusing the retained body (the model runs on HIP)
  {"content":"2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37","tokens":116,"tok_s":11.4,"finish_reason":"stop"}
  ```

  The decoded stream and tok/s are identical to the explicit
  `HIPFIRE_REPLAY_BACKEND=hip` baseline, i.e. the scope fix changes nothing about
  the HIP path and restores serving. Capture stays fail-closed: `is_recording()`
  remains part of the refused scope, pinned by the unit test above; no Qwen4 code
  path arms a window while admission is refused, so the guard is a backstop rather
  than a reachable refusal today.

### Diagnostic capture and census (C2)

`HIPFIRE_REPLAY_DIAGNOSTIC_SPECIALIZED_MOE_CAPTURE=1` lets the retained body be
*recorded* while admission is refused, for measurement only: the hook closes the
window, reports the census, and poisons, so no plan is ever installed and no
forward routes. Any number it produces is discovery evidence, never route proof.

Two fresh processes, `.mq4r` artifact, greedy, two different prompt lengths:

```text
[redline] qwen4 capture census: launched=2787 recorded=2787 unique_kernels=34 sequence_hash=a0c943b0e4046077 launch_count=2787
[redline] qwen4 capture census: launch-complete
[redline] qwen4 capture census: effects inside window — htod=2 dtod=13 dtoh=0 memset=48
[redline] qwen4 capture census: effect-incomplete — 13 device copy(ies) and 48 memset(s) inside the window are state the tape cannot replay (kernelise them, or move them into the external boundary)
```

`launched` is an independent count from `hip_bridge::launch_counters` (every
launch, funnel or raw), so the comparison is not the recorder grading itself.

**What the census found.** Three things, only one of which was expected:

1. **A recorder gap G1 missed.** The first run reconciled at 2786/2787: one kernel
   per decode forward escaped the tape. The diagnostic named it by
   loaded-kernel delta: `Gpu::add_f32` (`norm.rs`) was a raw launch while a
   funnel-based twin (`add_f32_graph_safe`) existed for the same kernel. `add_f32`
   is now the funnel path and the twin is gone (callers migrated), which moved the
   census to 2787/2787 and changed the tape identity (34 unique kernels, new
   sequence hash) — a silent one-launch-per-forward truncation that no test
   covered.
2. **Launch stability across positions.** Both runs, at different prompt lengths,
   produced the same count, the same unique-kernel set and the same sequence hash:
   the decode body's launch set and geometry are position-independent, which is the
   discovery form of REDLINE §5 stage 5 and closes G3's last outstanding question.
3. **The tape is launch-complete but effect-incomplete.** Per decode forward, the
   window also contains 48 `memset`s (one `Step::Clear` per layer) and 13
   device-to-device copies (one `selected_indices` copy per full-attention layer) —
   device work that is not a kernel launch, so the recorder cannot replay it, plus
   2 host→device uploads (token ids, PLE staging). `dtoh=0`, i.e. nothing reads
   back inside the window. REDLINE §4 makes this decisive: *"Incomplete recorder
   coverage is not a smaller valid tape unless every omitted launch belongs to a
   named external adapter boundary and launch accounting reconciles exactly."*

Those three items are closed (see "Completion" below), which is what turned the
census from "launch-complete but effect-incomplete" into the admission evidence.

### Completion (B1)

1. **In-window effects are recorded dispatches.** The layer `Clear` now launches
   `zero_f32` (a recorded kernel; multiply-by-zero would preserve NaN) and the two
   device copies — the QSA `selected_indices` row and the PLE depthwise
   `streams→query` copy — go through the recorded `copy_f32_buffer`. The census
   over the same forward reports `dtod=0 memset=0 dtoh=0`.
2. **The external input boundary is inside the Qwen4 forward.** Everything that
   materialises inputs — token-id upload, embedding, HC stream init, PLE
   prefetch/stage/upload/gather — runs before the boundary on every forward, HIP or
   replayed; the tape covers only the body. The forward declares eligibility
   (`n == 1 && wide_hidden_capture.is_none()`, so the speculative shape can never
   enter the tape), arms the capture and submits a prepared plan at that boundary.
   REDLINE §3 assigns this boundary to the model adapter, so this is a boundary
   marker rather than PM4 machinery — the engine still owns the policy, the tape
   machinery and the route-proof instrumentation.
3. **A replayed forward derives its host bookkeeping.** The lowering advances
   per-layer QSA lengths and positions while executing, which a replay cannot read
   back; both paths now derive them from the pre-body plans, and the HIP path
   cross-checks the derivation against the readback, so drift is a loud error
   instead of a silent state divergence.
4. **The admission rests on per-launch proofs, not on an argument.**
   `specialized_sealed_moe_retained_admission()` now answers `Admitted` (with
   `DiagnosticCapture` still available as a measurement-only override), and a
   retained body additionally requires, at launch time: the host expert pointer
   entries that the device tables were uploaded from must be present *and* must
   match the live expert tensors (the contents proof the compact path already had),
   and each table's pointer mapping must be the one this tape latched
   (`ReplayController::note_route_identity`, keyed per table because one model has
   one mapping per layer). A mismatch refuses the launch, and the forward's capture
   failure poisons the route to HIP.

**End-to-end evidence (gfx1151, `.mq4r`, greedy, `benchmarks/prompts/qwen4_ar_primes.txt`):**

```text
[redline] qwen4 capture census: computed=2848 recorded=2845 external=3 unique_kernels=33 sequence_hash=e4a03ba84c837714
[redline] qwen4 capture census: effects inside window — htod=2 dtod=0 dtoh=0 memset=0
[redline] qwen4 retained body prepared: transport=pm4 dispatches=2845
HIPFIRE_REPLAY_ROUTE_PROOF transport=pm4 position=65 request_id=run replays=115
{"content":"2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37","tokens":116,"tok_s":12.0,"finish_reason":"stop"}
```

`external=3` is the named input boundary (embedding, HC stream init, PLE gather);
the 2845 dispatches are the body, replayed 115 times — i.e. every decode step after
the capture warmup — producing the byte-identical stream the HIP path produces.

**Cross-turn evidence (serve harness `--mode chain`, same daemon, capped think
span):** one capture and prepare per daemon process — `dispatches=2845` with
`sequence_hash=e4a03ba84c837714`, the same tape identity the CLI run produced — then
every later turn's decode ran on retained replays: `replays=104` at `position=168`
and `replays=132` at `position=284`, i.e. one replay per decoded token across the
per-turn state reset, with coherent answers. The harness's empty turns (3/5 with
the route, 4/5 on the HIP arm under the same budget) are the artifact's think-span
framing under a capped thinking budget; they appear on both arms and are not a
route effect.

**A/B (3 interleaved fresh processes per arm, greedy, 116 tokens):** retained
11.9 / 8.6 / 12.2 tok/s (median 11.9) vs HIP 11.4 / 11.4 / 11.5 (median 11.4),
byte-identical output in all six runs and in the earlier 5-pair shape A/B. **No
performance claim** follows: three samples with one outlier, within this box's
within-session spread, and no timed-arm route proof.

**Not claimed:** REDLINE §7 gates 4–8 (multi-position HIP/PM4/recorded-blob shadow
parity, the `tools.redline` timed-arm route-proof ledger, serve health, long
context, reset and model swap) and any performance promotion. The census is
discovery evidence; the route is implemented, not certified.

### Acceptance evidence per option

- A2: a `.mq4r` Qwen4 serve with a replay backend armed loads, prefills, and
  generates on HIP, logging a fallback reason; a capture-window failure poisons
  rather than erroring the request; a unit test pins the guard's new predicate
  (`is_recording`/`should_route_*`, not `is_enabled`).
- B1: table-content proof with a negative test (a mutated entry must refuse);
  mapping fingerprint recorded with the tape and re-proved at prepare with a
  negative test; sign tables proven resident before the capture window;
  `HIPFIRE_DUMP_HIDDEN` refused while recording; census (REDLINE §7 gate 2) with a
  reconciled count; then gates 3–8 of the REDLINE ladder.
- C2: the diagnostic override is non-default, cannot install a plan, and any number
  it produces is labelled discovery-only in this record.

Until G4 lands, an explicit `HIPFIRE_REPLAY_BACKEND=hip` run is the healthy
baseline (REDLINE gate 1), and it is also the A/B arm for the later census.

## Certification status: REDLINE §7 gates (evidence collected 2026-09-20)

Instrument: `tools/redline bench` (the route-proof product harness) against the
Qwen4 `.mq4r` fixture on gfx1151, plus the CLI, the serve harness and the
diagnostic census. Raw reports are archived locally under
`.codeinsight+research/qwen4-retained/` (gitignored):
`qwen4-route-proof-coherent.json`, `qwen4-route-proof-settled.json`,
`qwen4-route-proof-longctx.json`, `qwen4-swap-serve.log`.

| Gate | Required evidence | Collected | Where |
|---:|---|---|---|
| 1. Baseline correctness | Ordinary HIP stable and coherent on the exact fixture | **yes** | `HIPFIRE_REPLAY_BACKEND=hip` CLI runs (116-token greedy stream, `finish=stop`); bench HIP arm `measurement_validation.valid=true`, median 13.584 tok/s; `coherence: HIP passed` |
| 2. Capture completeness | compute/external/retained counts, unique kernels, ordered sequence hash, fresh-process stability | **yes** | census `computed=2845 recorded=2845 external=0 unique_kernels=33 sequence_hash=e4a03ba84c837714`, non-launch effects `htod=0 dtod=0 dtoh=0 memset=0`; the same hash from four independent fresh processes (CLI, serve daemon, two bench runs) |
| 3. ABI/artifact validation | every symbol resolves to the exact artifact plus loader metadata; padded blobs, geometry, effects, dynamic bindings validate | **yes** | prepare-time: `certified_artifacts=16/16`, `certified_launches=2845 fallback_launches=0 unknown_launches=0`, PM4 wait audit `covered=2844`, prepare reached `ready` with `dispatches=2845 packets=1 queue_id=2 dwords=67508 queues=1 phases=1`; any failure would have refused before `Ready` |
| 4. Multi-position shadow parity | HIP, retained PM4 and the exact HIP-kernarg-blob oracle agree for logits, KV, recurrent state, guards, blobs | **yes** | `qwen4-gate4-shadow.json`: the retained transport is bit-exact against ordinary HIP at 5 positions (129–133) over **126,623,888 state bytes per position** — QSA full/raw/pooled keys, the circular partial block, the indexer's selected indices, GDN recurrent + convolution, PLE convolution and hyper feedback, the position/length bookkeeping, and logits — and the recorded-HIP oracle is bit-exact at its capture position; harness failures `[]` |
| 5. Route proof | request/transport, preparation, `Ready`, observed replay at multiple positions, dispatch/packet/queue/dword identity, sequence hash, no fault, fallback reason per arm | **yes** | auto arm: `route_proof.valid=true`, `retained_rows=5`, `observed_positions=[1500,1547]`, `prepared_identities=[[2845,1,2,67508,1,1]]`, `sequences=[[2845,33,"e4a03ba84c837714"]]`, `errors=[]`, `lifecycle_route_proof.valid=true` with `retained_rows=36` at positions `[1500,1501,1511,1547]`; HIP arm `route_proof.valid=true`, `retained_rows=0`, `state=hip`, `fallback_reason=null` |
| 6. Production serve | user-facing generation with healthy output, finish state, attractors, framing | **yes** (bench coherence both arms; chain coherent answers) | `coherence.mode=custom`, `hip: valid=true`, `auto: valid=true` (CLI/serve prompt `benchmarks/prompts/qwen4_ar_primes.txt`, expected `2, 3, 5, 7`); serve-harness chain produced coherent answers under the route (`replays=104/132` at positions 168/284) with the empty-turn artifact appearing on *both* arms (3/5 routed vs 4/5 HIP) |
| 7. Stationary matched performance | identical binary/model/prompt/settings/clocks; tok/s and ms/token | **yes** (one fully valid run) | long-context fixture: `hip` 8.347 vs `auto` 8.507 tok/s, `speedup=1.019`, `valid=true`, both arms stationary and route-proven, coherence both arms. The short-context (128) fixture produced arm-symmetric harness rejections across three attempts — all with valid route proofs, `retained_rows=5`, coherent output on both arms, and stable medians (`hip` 13.570-13.584, `auto` 14.164-14.248) — failing only the measured 5-row window's policy limits (`spread<=1.0%`, `slope<=0.05%/row`, `drift<=0.5%`): `auto` drift 0.141%/0.215% in two attempts, and in the long-settle attempt `hip` itself at spread 1.027%/slope -0.101%/row from a single first-row downclock outlier. Not route-attributable; the valid gate-7 numbers are the long-context run |
| 8. Long-context and lifecycle | dynamic position, geometry, KV growth, recurrent state, request reset, failure behaviour, model swap | **partly** | long context ✓ (positions 1500–1547, see gate 7); recurrent/convolution + KV growth ✓ (gate 4, bit-exact at 5 positions); request reset ✓ (multi-turn `replays=104/132` across turns); capture-window failure ✓ (poisons to HIP, no plan installed); **replay failure ✓** (induced: the plan refuses a boundary position past its prepared window — the long-context-growth hazard — the forward errors, the route poisons with the named reason, and the next forward is on HIP and bit-exact against clean HIP); model swap **blocked by a pre-existing loader/VMM teardown guard** (see below) |

Claims carried by this table: gates 1, 2, 3, 5, 6 and one gate-7 run. **No
promotion, admission, or performance claim** follows — gate 4 is uncollected and
gate 8 is partial.

### Gate 4 instrumentation

Gate 4 compares state, not decoded text, so it needs three arms over the same
positions. What was added:

1. **`redline_shadow_qwen4`** in `crates/hipfire-generate/src/redline.rs`, next
   to `redline_shadow_gemma4`/`redline_shadow_deepseek4`, dispatched from
   `handle_redline_shadow`'s downcast chain. It resets + primes each arm, drives
   single-token forwards, and snapshots every position: logits and their argmax,
   the per-layer QSA active marks, the GDN recurrent/convolution tensors, PLE
   convolution and hyper feedback, and every capacity-pinned arena slice the
   model can still read (arenas are read to their active mark; the bounded
   partial block and selected-index list are read whole).
2. **`ShadowBodyRoute`** on the controller (`rdna-compute`): a manual shadow
   controller states, for the next eligible forward, whether the body submits
   the prepared plan, re-executes the recorded HIP prefix (the exact-kernarg
   oracle), or runs ordinary HIP. Production adoption never sets it — it reads
   the executor from the prepared plan — and the field is cleared by every model
   reset, so no shadow state can leak into a serving route. The boundary selects
   the replay entry from the plan that is actually installed
   (`prepared_pm4_plan_ready`/`prepared_aql_plan_ready`), because a shadow arm
   prepares the transport it chooses rather than the one the controller is
   configured with.
3. **`--qwen4` mode in `scripts/redline_daemon_harness.py`**, gating
   `QWEN4_EXACT_FIELDS` per position plus the capture-position oracle row.

Two semantics worth stating, both learned by running it:

- **The recorded-HIP oracle is exact only at its capture position.** It
  substitutes position through the controller's *synthesized-binding
  calibration*; the Qwen4 route instead **declares** its position bindings in
  the program (G1/G2), which the oracle path does not read. So the oracle is
  compared once, at the capture geometry, while the retained transport carries
  the multi-position claim — which is the stronger half anyway. Observed
  directly: with the tape captured at position 128, the oracle matched HIP at
  the capture window and diverged from 129 onward, while the retained arm stayed
  bit-exact at 129–133.
- **`prepared_route_identity` now follows the installed plan**, not the
  configured transport. It previously answered `None` for a PM4 plan on a
  controller whose configured transport was AQL — i.e. it could report an
  installed, replaying route as an absent one. That is a route-reporting bug
  independent of this work; the shadow harness would have certified an
  unidentified plan.

### Gate 8 blockers found

- **Model swap: pre-existing loader/VMM guard, route-independent.** Reloading the
  Qwen4 artifact after serving a different model in the same daemon fails with
  `refusing load: prior VMM teardown still pending (… 16 live VMM tensor owner(s)
  remain …)`. The identical failure reproduces with
  `HIPFIRE_REPLAY_BACKEND=hip` (control run), so it is not a retained-route
  defect: the swap *away* from Qwen4 succeeds and the retained plan is reset by
  `configure_model_default` on the intervening load, but the loader refuses the
  swap *back*. This needs a loader/VMM lifecycle fix (or a documented
  single-model-per-process requirement) before gate 8's model-swap row can pass.
- **Replay-execution failure: collected** through the plan's own position bound
  rather than an injected hook. The probe re-prepares the plan with a
  `prepared_max_position` one below the boundary position this forward replays
  at, which is exactly what long-context growth past the prepared window
  produces. Observed: the plan refuses before the body
  (`Qwen4 retained replay failed: position 128 exceeds prepared max_position
  127`), the route poisons with that reason, and the next forward runs on HIP
  and is bit-exact against two clean HIP runs at the same geometry. The hazard
  this rules out is the same-forward fallback claim: the failing forward does
  **not** let HIP finish the transition the retained body started.

### Record schema (REDLINE §8)

The bench report carries host, model path/bytes/SHA-256, daemon and CLI digests,
`git_commit`, UTC start, transport, every `HIPFIRE_REPLAY_PM4_*` knob, KV mode,
context/iterations/runs/warmups, device visibility, automatic clocks, stationarity
per arm, coherence (prompt file, digest, expected substrings, thinking, max
tokens, sampling) and the per-arm route proofs. Environment identity for the
runs recorded here:

| Field | Value |
|---|---|
| Host | `halo` |
| GPU | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S (Strix Halo, gfx1151) |
| PCI | `1002:1586`, subsystem `2014:801D`, driver `amdgpu` |
| Kernel | `7.0.9-cachyos-lto` |
| Runtime | `libamdhip64.so.7.2.53211` (ROCm 7.2.x, via the merged root `/home/bjoern/.hipfire/rocm-merged`) |
| Timed prompt digest | `benchmarks/prompts/qwen4_ar_primes.txt` — md5 `0508eec29a44323f62e70fa77d92b834`, sha256 `16c282e391cce0c1ad4b702cdacc1b25c126c474de842f2e9c6d32b7ebd9a752` |
| Clock policy | automatic (`automatic_clocks: true`), DPM settle + stationarity gate per arm |

Still missing for a promotion record, and all of it report-level rather than route
work: the timed arm's prompt/token-stream digest where the harness synthesises
tokens, the timed arm's sampler seed, and the author's predeclared promotion rule
plus disposition. No promotion rule exists yet, so nothing here may be cited as
an admission.

## Verification ladder

| Stage | Route | State |
|---|---|---|
| Unit: G1 funnel entry | GPU test `tensor_ops::tests::recorded_blob_launch_enters_the_tape_with_the_bytes_it_launched` (gfx1151; skips elsewhere) | **passing** |
| Unit: G2 binding vocabulary | 7 tests in `crates/rdna-compute/src/replay.rs` | **passing** |
| Unit: G3 pinned shapes + declarations | `tensor_ops::tests::{pinned_qsa_shapes_are_bit_identical_to_derived_shapes, qsa_position_fields_are_declared_to_the_recorder}` (gfx1151) | **passing** |
| Contract: row-absolute column offset | `test_copy_rows_strided_f32_parity` 7/7 cases bit-exact (needs `--features lab`) | **passing** |
| Real-model shape pin A/B | 5 interleaved fresh-process pairs, greedy 116-token stream bit-identical, tok/s medians equal | **passing** (HIP path only) |
| Structural: no raw launch left in the program's op owner | `tensor_ops.rs` launch-discipline review (0 raw sites) | **passing** |
| Ordinary HIP baseline | `HIPFIRE_REPLAY_BACKEND=hip` daemon probe: loads, generates (`PARIS`), commits | **passing** |
| G4 A2 scope | armed-default `.mq4r` probe: serves on HIP, logs the refusal reason, stream identical to the HIP baseline | **passing** |
| G4 A2 policy | `replay::tests::retained_body_scope_*`, `ar::route_scope_tests::retained_body_action_*` | **passing** |
| G4 C2 census | `HIPFIRE_REPLAY_DIAGNOSTIC_SPECIALIZED_MOE_CAPTURE=1` + `.mq4r` probe: launched==recorded (2787/2787), identical hash across two prompt lengths, effect audit printed | **passing** (discovery evidence, not route proof) |
| End-to-end census: recorded launches == compute launches | needs G4 | blocked by design |
| Multi-position HIP vs recorded-blob vs PM4 parity | needs G3, G4 | blocked by design |
| Serve health, stationary matched performance | REDLINE §7 gates 6–7 | blocked by design |

Harness gap to be resolved with G3/G4 (not now): no dispatch-level or engine-level
capture hook exists, and the arch-coupled bench harness
(`crates/hipfire-generate/src/redline.rs`) downcasts to existing model types. The
qwen4 probe today is the daemon auto-default path (`hipfire run` with an isolated
`HOME` holding `max_seq=2048` and a distinct port, because the daemon takes an
flock on `$HOME/.hipfire/daemon.pid`), which is a diagnostic, not route proof.

## Progress ledger

Append-only. One line per landed change with the commit hash once it exists.

- 2026-09-20 — plan written. Baseline refusal reproduced (see above). G1/G2 in
  progress.
- 2026-09-20 — **G1 landed** (branch-implemented) in `b12185138`:
  `Gpu::launch_blob_recorded` + 31 `tensor_ops.rs` migrations + one shared
  `recorded_launch_artifact` resolver (funnel, scratch, new entry).
  Evidence: GPU tape test, `cargo test -p rdna-compute --lib` 260 pass,
  `cargo test -p hipfire-dispatch` 285 pass, HIP end-to-end `PARIS`.
- 2026-09-20 — **G2 landed** (branch-implemented) in `b12185138`:
  `PositionDivU32`/`PositionModU32` bindings, declared-binding carriage on
  `RecordedHipLaunch`, prepare-time merge with one-owner-per-slot +
  position-derived-only rules, synthesis skip for declared offsets, binding
  identity in the sequence hash, `offset()` accessor replacing three duplicated
  match arms, and the first two declarations (GDN conv ring cursor, batched chunk
  start cursor) with `GatedDeltaConv::row_index` supplied by `layer_ops`.
  Evidence: 6 new unit tests (87 `replay::` tests pass).
- 2026-09-20 — Default-path probe re-run after G1/G2: still the sealed-MoE
  preflight refusal, i.e. G4 remains the gate for any capture. Unchanged behavior
  is the expected result here, not a regression.
- 2026-09-20 — status paragraph brought in line with the collected §7 evidence
  (gates 1–7, gate 8 partly); no promotion claim.
- 2026-09-20 — **Gate 8 replay-failure row collected**: the plan's position bound
  is the induction (no test-only hook), and the observed behaviour is error →
  named poison → correct HIP recovery. Model swap stays blocked on the
  pre-existing VMM teardown guard.
- 2026-09-20 — **Gate 4 collected**: Qwen4 multi-position state-parity shadow
  arm (`ShadowBodyRoute`, `redline_shadow_qwen4`, `handle_redline_shadow`
  dispatch, `--qwen4` harness gate). Retained PM4 is bit-exact against ordinary
  HIP over 126,623,888 state bytes per position at positions 129–133, and the
  recorded-HIP oracle is bit-exact at its capture position.
- 2026-09-20 — lifecycle + bench carrier + certification record in
  `561b0f166` (retained-body lifecycle owned by the forward; `RetainedBodyAction`
  moved to `hipfire-dispatch`), `51473026a` (Qwen4 bench carrier arm),
  `10ed7bfe0` (this section).
- 2026-09-20 — **Certification evidence collected** (no promotion): gates 1, 2, 3,
  5, 6 and one fully valid gate-7 run are collected through the route-proof
  product harness; gate 4 is blocked on a Qwen4 shadow arm (plus manual-capture
  arming at the boundary) and gate 8 is partial (long context and request reset
  pass; model swap is blocked by a route-independent loader/VMM teardown guard;
  replay-failure behaviour needs the `serve-fault-inject` driver). Details and raw
  report paths in the certification-status section.
- 2026-09-20 — **G4 complete (B1 + admission), route replays end to end**
  (branch-implemented, uncertified) in `67704ed34`: the in-window memsets and device copies are
  recorded launches, the eligible-forward boundary moved into the Qwen4 forward so
  the tape covers only the body, a replayed forward derives its per-layer QSA
  bookkeeping (cross-checked against the HIP readback), and admission is granted
  with per-launch proofs (host pointer-table contents vs live experts; per-table
  pointer-mapping identity latched into the tape). Evidence: census 2845/2848 with
  `external=3` and zero in-window effects, then `dispatches=2845` prepared and
  `replays=115` observed with the byte-identical decoded stream. Certification
  (REDLINE §7 gates 4–8) not claimed.
- 2026-09-20 — **G4 C2 landed** (branch-implemented, uncertified) in
  `1182b046f`: the
  diagnostic capture + launch census + effect audit. It found a recorder gap G1
  missed (`Gpu::add_f32` raw while a funnel twin existed for the same kernel; now
  one funnel path, twin deleted, callers migrated), proved launch stability across
  two prompt lengths, and quantified the effect gap that blocks admission (48
  memsets + 13 D2D copies + 2 H2D per decode forward inside the window). Admission
  remains refused; the three remaining work items are listed in the C2 section.
- 2026-09-20 — **G4 A2 landed** (branch-implemented, uncertified) in
  `0c758739b`: the refusal is
  scoped to the retained body (`ReplayController::retained_body_active()` plus one
  admission point in dispatch), and Qwen4 gained the engine-level retained-body
  discipline in `hipfire-generate` (prefill ineligible; decode poisons when the
  route cannot be retained; body failure inside a window poisons; a routed state
  fails closed). Verified by two policy unit tests and two fresh-process armed
  probes that now serve on HIP with the refusal reason logged and a stream
  identical to the HIP baseline. The `complete > 0` arming gate is deferred to B1
  (arming is refused wholesale today, so it would be dead code).
- 2026-09-20 — **G4 analyzed** (no decision): split into D1 (the refusal's blast
  radius — it disables prefill, fallback and every forward, making the `.mq4r`
  artifact unservable) and D2 (the unstated pointer contract). The route is already
  close to conformant: load-time pointer tables that are never rewritten, no
  per-forward host table or H2D, no position-derived scalar anywhere in the route,
  and per-seal pointer-identity re-proof with an existing mapping fingerprint. The
  real gaps are table-*content* proof on the Single path, a plan-pinned mapping
  identity, names for the lifetime hazards (lazy FWHT sign tables, dump-hidden D2H,
  invalidation paths), and the census. Options A1–A3, B1–B4, C1–C2 recorded with
  weights; recommended A2 now and B1+C2 for admission, with B4 as the honest
  fallback.
- 2026-09-20 — **G3 decided and landed** (branch-implemented, uncertified) in
  `360a44ab1`:
  capacity-fixed pool grid + capacity-pinned LDS with a constant symbol + declared
  position fields (including the new `PositionMulU32` for the index-key row offset,
  which replaced a position-shifted destination pointer) + `copy_rows_strided_f32`
  bounded by the destination extent instead of one row pitch. Evidence: 2 new GPU
  tests (shape pin bit-exactness incl. the zero-count symbol swap; declarations
  point at the recorded slots), 7/7 copy-parity cases, 5 interleaved fresh-process
  A/B pairs with a bit-identical 116-token greedy stream and equal tok/s medians,
  263 + 285 regression tests. The `HIPFIRE_QSA_STABLE_SHAPES` measurement gate was
  consumed and deleted. Route arming (`complete > 0`) still belongs with G4.
- 2026-09-20 — **G3 analyzed, no decision taken** (this section): problem split
  into P1 pool presence/grid, P2 dynamic shared memory, P3 a position-shifted
  destination pointer in the index-key write, and P4 the finding that the
  automatic route has no calibration pass, so an undeclared position-derived field
  is a *silent* wrong-output hazard rather than a loud failure. Admitted geometry
  computed (compress 4, budget 2048, pooled_capacity 512, qsa_selected_capacity
  2051, attention grid 24 workgroups, LDS ≤ 16 408 B); neither 64 KiB switch flips
  in the 2048-token range. Options A1/A2, B1–B4 and C1–C5 recorded with
  pro/contra and a measurement plan; recommendation A1 + B1 + C1 + A4.

### Next

1. G3 measurement plan (bit-exactness of pinning, cost of the masked pool grid and
   the pinned LDS reservations, binding census) — none of it needs G4, but the
   end-to-end form does.
2. G4: until the specialized-route pointer contract exists (or the refusal is
   narrowed to the exactly-unstable sub-case), no complete Qwen4 tape can be
   captured, so no census, parity, or route proof is reachable.
3. Then the REDLINE §7 ladder, notably the state oracle (QSA full/raw/pooled
   keys, selected indices, GDN recurrent/conv state, PLE history).
