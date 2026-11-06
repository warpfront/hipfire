# PeaceMaker EXPAND_LIVE (P0 template 1, gfx1201)

Status: lab, no product route. `crates/peacemaker-isa/src/kernels/p0_expand_gfx1201.rs`
(wrapper) and `p0_expand_gfx1201/body.rs` (`expand_body`, the native task-body region).
No runtime crate depends on `peacemaker-isa`; this change touches no runtime
`Cargo.toml`.

## Scope

EXPAND_LIVE is a producer that turns the static many-tile grid of the few-row (R2)
gate/up kernel into a compact list of live workgroups. It covers **only** the
`gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4` launch (tid 16 in the M0 layer pool). The
down-projection `dni` kernel (tid 18) stays a static grid with **no producer** (see
[DNI](#dni-tid-18-stays-static)). In the P0 pool the producer is template id 1 in the
b128 (block 128) pool; PmPrereq confirmed this.

## Kernarg ABI (exactly 64 bytes)

| Offset | Field | Type / access |
|---|---|---|
| 0 | `expert_tile_ids` | `u64` pointer, read-only |
| 8 | `sorted_slot_index` | `u64` pointer, read-only |
| 16 | `live_list` | `u64` pointer, write-only |
| 24 | `count_cell` | `u64` pointer, **read-write** (`actual_access` read_write: atomic RMW); host-zeroed before launch |
| 32 | `M` | `u32` |
| 36 | `grouped_rows` | `u32` |
| 40 | `n_experts` | `u32` |
| 44 | `gx` | `u32`, `ceil(M / 16)` |
| 48 | `gy` | `u32`, `ceil(grouped_rows / 2)` |
| 52 | `capacity` | `u32`, `<= gx * gy` and at least the actual live count |
| 56, 60 | reserved | `u32` |

Grid z is 1. Constants: `TEMPLATE_ID = 1`, `SYMBOL = "pm_expand_live_gfx1201"`,
`KERNARG_BYTES = 64`, `SGPRS = 40`, `VGPRS = 8`, `LDS_BYTES = 0`.
The region reads `s[0:1]` (kernarg pointer) and `v0` (flat thread id, read-only), owns
`s2..s39` / `v1..v7`, retires all memory, restores full EXEC and contains no `s_endpgm`.

## Valid domain (fixed ABI contract, no runtime validation)

Real M0 is well inside these bounds.

- `M` and `n_experts` are `<= i32::MAX` (hipcc `int`s).
- `grouped_rows <= u32::MAX / 4` (byte indexing).
- `gx * gy <= u32::MAX / 16` (32-bit LiveList byte offsets).
- `expert_tile_ids` covers `ceil(grouped_rows / 16)` ids; `sorted_slot_index` covers
  `grouped_rows` slots.
- The pointers/resources do not overlap.
- Undersized `capacity`: no out-of-bounds write; the full count is still published
  (consumer poison). The list is only valid when the actual count is `<= capacity`.

## LiveList format

`live_list` is an array of 16-byte entries `u32 {vx, vy, vz = 0, pad = 0}`. The count
is a `u32` in `count_cell`, which the host zeroes before each launch (reset precondition); the
producer publishes it with a device-scope release `global_atomic_add_u32` (an EventCells
signal), not a plain store, so the j8292 bind refusal of a plain-store count is fixed. The host header carries an implicit `vcount` of 1 per entry.
Order is **y ascending, then x ascending** (y-major). Entries are released before the
final count publication, and the count atomic add is a release; it is retired before exit. The multi-wave body takes a barrier
after every wave's record release and before flat-tid-0 publishes the count.

The input pointers come from the o4 consumer's kernarg offsets **+8** (tile ids)
and **+16** (sorted slots), not its expert-weight pointer at +0. The route scatter
initializes both arrays to -1
(`kernels/src/qwen4_moe_scatter_fused_k8.hip:68-83`), installs valid flat slot
indices at `:169-172`, and fills each expert's padded 16-slot tile ids at
`:194-202`. Expert offsets/counts determine that grouping (`:119-127`) but are not
themselves read by EXPAND_LIVE; liveness is taken from the same two arrays the
consumer reads.

## Predicate

A pair `y` (the R2 subtile `blockIdx.y`, `slot_start = 2*y`) is **live** iff all hold:

1. `2*y < grouped_rows`;
2. `t = expert_tile_ids[(2*y) >> 4]` satisfies `0 <= t < n_experts` (signed);
3. at least one of `sorted_slot_index[2*y]` or, when `2*y + 1 < grouped_rows`,
   `sorted_slot_index[2*y + 1]` is signed `>= 0`.

An x is live iff `16*x < M` (every block row of 16 contains a valid output row).
A mixed live/pad pair (one slot `>= 0`, one `< 0`) is **live and emitted**.

## Dead-write proof (source: `kernels/src/gemm_mq4g256v2p_moe_grouped_top10_o4_r8_x4.gfx1201.hip`)

The R2 entry is `gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4` (`:311-328`), which
instantiates `grouped_top10_o4_body<2>` with `xs[2*64]` of LDS. Grid comment `:311`:
`[ceil(M / 16), ceil(grouped_rows / 2), 1]`, block 128. In the body
(`:85-291`):

- `:106` `row0 = (blockIdx.x * 4 + wave) * 4` (x covers rows `16*x .. 16*x+15`),
  `:107` `slot_start = INVERSE ? mapped_start : blockIdx.y * R` (R = 2 here).

Three block-uniform early exits, all **before the first LDS stage and every global
store** (the first LDS write is `:223`, and the only global stores are `:277-288`):

1. `:109` `if (slot_start >= grouped_rows) return;`
2. `:115-121` `flat_raw[slot] = sorted_slot_index[...]` (clamped read),
   `expert_id = expert_tile_ids[slot_start >> 4]`, then
   `if (expert_id < 0 || expert_id >= n_experts) return;`
3. `:133-152` per-slot `live[slot] = flat >= 0` (`:145-146`, with `flat = -1` for
   `grouped_slot >= grouped_rows`) and `any_live`;
   `:152` `if (R < 8 && !any_live) return;`

So a workgroup whose pair fails predicate 1, 2 or 3 stores nothing: its `Y_grouped`
words are never written, and R2 has no zero-fill store on those exit paths. The
three exits are exactly the negation of the pair predicate, so omitting those
workgroups removes executions that return before any LDS or global write.

For a pair that passes, stores happen at `:277-288` by `tid == 0` only:
`:281` `if (grouped_slot >= grouped_rows) continue;` then (`:282-284`) under `HF_NODZ`
`if (!live[slot]) continue;`, then `:286-288` per `r` with `row_valid[r]`
(`row_valid[r] = row0 + r < M`, `:133`) `Y_grouped[grouped_slot * M + row0 + r] = sum[r][slot]`.
`HF_NODZ` is **not set** for the R2 entry: in `crates/rdna-compute/src/gemm.rs` the
`#define HF_NODZ 1` source (`:50922`) is built only under
`if !few && !self.moe_dead_stores && self.arch_caps.is_gfx1201()` (`:50919`); the
comment at `:50914-50915` states the few-row R2 arm keeps its entry. The R2 entry name
is selected at `:50900-50911` (`few` -> `..._o4_r2_x4`, 2 slots) and the launch grid is
`[m.div_ceil(16), grouped_rows.div_ceil(slots), 1]`, block 128 (`:50965-50972`). Hence
in a live pair with one pad slot (`live[slot] == false`, `flat = -1`, `x_row = 0`,
`:147`) the pad slot still stores its computed zeros for `row_valid` rows, so a mixed
pair must be **emitted** (not dropped) to keep those zero stores. Row-valid x: only
x with `16*x < M` have any valid row, and within x, rows `>= M` are never stored
(`:125`, `:133-134`, `:287`).

The skipped pad slot's accumulators start at `+0.0` (`:175-181`) and are never
updated (`:229`), proving the mixed pair's pad stores are zeros rather than
unspecified arithmetic results. No assumption about downstream readers is needed.

## CPU byte oracle

`crates/peacemaker-isa/tests/p0_expand_live.rs` assembles and lifts the emitted
kernel, executes it on the PeaceMaker interpreter, and compares every LiveList
and count byte with an independent host enumeration. It also checks input
immutability, unused output suffixes and canaries. The committed j8284 route
prefixes and their original dump/manifest SHA-256 values are in
`tests/fixtures/p0_expand_live/SOURCE.json` under the crate. That route has 45 live
R2 pairs (30 mixed live/pad), hence 3,600 entries at gx=80; its 38 live 16-slot
expert tiles are not the R2-pair count.

## DNI (tid 18) stays static

`kernels/src/gemv_mq4g128v2_moe_down_top10_dni.gfx1201.hip` (`dni_down_k640`,
`:88-183`): `:98` returns only on block-level range checks; `:104-105`
`float res = 0.0f; if (expert >= 0 && expert < n_exp) {...}` and `:182`
`if ((tid & 3) == 0) obase[tid >> 2] = res;` store `res` (zero for an invalid expert)
**unconditionally** for every in-range block. An invalid-expert block therefore
zero-fills its output and is observable; dropping it would leave stale words. Main
decided to keep the dni grid static, with no producer.

## Standalone image

`cargo run -p peacemaker-isa --features lab --example emit_expand_live -- 128 /existing/dir/expand_live_b128.co`
writes the native-assembled gfx1201 code object (BLOCK 32/64/128/256; output parent must exist).

## Notes

- The standalone wrapper `emit(block)` (32/64/128/256) establishes flat `v0` as the
  lifted pool does (`readfirstlane` of the wave's first tid + `v_mbcnt_lo`), releases
  at device scope and ends; CFG wait verification is strict over the body.
- The recipe/launcher source consulted for the no-`HF_NODZ` claim is `gemm.rs` above;
  no separate recipe file exists in this worktree.

## Exercised CPU checks

- `cargo test -p peacemaker-isa --test p0_expand_live`: 14 passed, zero ignored.
  Includes real j8284 bytes on blocks 32/64/128/256; empty routes; the full
  38,400-entry all-live cap; expert-boundary and negative sentinels; mixed/odd
  pairs; fixed-seed random routes; capacity/canaries; widths above one block.
- `cargo test -p peacemaker-isa --lib`: 169 passed.
- `cargo test -p railgun-cert --lib`: 13 passed.
- `cargo test --release -p peacemaker-ir --lib`: 495 passed, 13 ignored, including
  the runtime-dependency test excluding PeaceMaker from runtime crates.

All builds used the host build-slot wrapper, a private target directory and
`CARGO_INCREMENTAL=0`. No card job or hardware-equivalence claim is part of this
producer receipt; P0-vs-eager hardware equality belongs to the integrating harness.
