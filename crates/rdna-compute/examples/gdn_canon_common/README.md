# gdn_canon_* — S0 kill oracles (CHUNK-CANONICAL-DESIGN §6 S0)

Four independently runnable examples, one shared support module (`#[path = "gdn_canon_common/mod.rs"] mod canon;`).
All need `--features lab,deltanet`; kills 1/2 and the device legs of 3/4 need a gfx1201 card
(run through `/home/kaden/qcal/locks/fabric/fabric-run.sh`). The CPU legs of 3/4 run with `--no-gpu`.

| example | design item | subject |
|---|---|---|
| `gdn_canon_producer_kill` | S0.1 | QKVZA prepared-IU4 route vs short-call route, per field, n = 1/25/63/64/537 (+ extras) |
| `gdn_canon_scan_prefix_kill` | S0.2 | C64 scan prefix stability for every r = 1..63 (heads 0/64/512, nonzero Q8+EF+conv, zero/NaN/huge padding) |
| `gdn_canon_reference_kill` | S0.3 | retained-512 reference vs bounded-tile replay (CPU) and current production vs single call (device) |
| `gdn_canon_snapshot_kill` | S0.4 | snapshot Open at p = 1/25/63/64/65/511/512/513/521, clobber, restore, continue, seal |

Common flags: `--json <receipt.json>`. Receipts live in
`reliability/chunkfix/runs-canon/<slice>/`. Envelope:
`{oracle, receipt_schema:1, design, git_head, arch, verdict: pass|fail|oracle_invalid, result:{…}}`.

Exit codes: `0` pass, `1` the code under test fails the kill (fail-before for current production), `2` the
oracle's own controls are invalid (correct model failed, a fault mutant escaped, determinism/sensitivity
control failed), `3` environment (no gfx1201, init failure, unsupported flag).

## Plug-in hook for S2/S3/S4

`stream::GdnStreamUnderTest` (typed trait object) is the only integration surface: `reset`, `meta`, `append`
(raw **pre-conv** QKV, raw alpha/beta; outputs are `rows*hv*128` f32), `seal`, `peek_seal` (non-mutating),
`snapshot` / `restore` (opaque deep copy), `clobber_live` (destroys every live persistent buffer).
Implement it for a real stream and call

```rust
kills::evaluate_candidate(&mut yours, &mut reference, &init, &data, &scheds, RefMode::SameSchedule, &RunCfg{recipe_id})
// or, when only one device stream can exist:
kills::evaluate_self_single(&mut yours, &init, &data, &scheds, &cfg)
```

Schedules: `Schedule::{single, fixed_chunk, random_partition, snapshot_at}` (steps `Append(n)`, `Seal`, `Peek`,
`SnapshotClobberRestore`; implicit final `seal`). `matrix::compare_runs` is **bitwise**: every published row,
every `Seal`/`Peek` state, the final Q8 codes/scales/EF/conv, and (when `reports_stream_meta()`) the metadata
against `model::expected_meta` (absolute 512 seams, absolute C-grid, first tile/frame clipped from an off-grid
`phase_start`, `pending_rows = position - tile_base`).
`gpu::GpuProductionStream` is the current production device route as such a stream (fail-before subject).

## Pinned fixtures

Seeds: `util::SEED_BASE | {0x11 model, 0x22 data, 0x33 init, 0x40.. random partitions, 0x101..0x105 producer}`
(GPU legs: `gpu::gen_gpu_initial/gen_gpu_rows`, lcg seeds 1..12 identical to `gdn_chunk_tail_kill`).
CPU geometry: hk=1, hv=2, head_dim=128 (`q8ef::HD`), C=64, commit 512, conv history 3, Q8Ef and plain Q8.

## CPU tile arithmetic is a stand-in

`model::tile_scan` is deterministic CPU math that preserves two structural properties of the C64 scan: a
tile-local gate cumulation (rounding depends on the absolute tile origin, so a partial tile's end state is not
a valid entry for a shifted grid) and the causal prefix property. It does **not** model WMMA numerics; the
device legs and kill 2 cover the real kernel. Quantization is `gdn_q8ef_reference` unchanged.

## Fault mutants (`model::Fault`)

One per design §6 S0.4 item, each must be detected (otherwise verdict `oracle_invalid`):
`call_end_quantization` (= current production), `f32_entry_without_raw_carry`, `missing_pending_raw_carry`,
`missing_tile_entry_state`, `missing_ef_read`, `snapshot_drops_ef`, `stale_q8_read`, `duplicate_conv`,
`call_index_rounding` (plain Q8 only), `pending_capacity_read_poison`.
