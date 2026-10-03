# Peacemaker schedule lint

Run offline, without a GPU:

```sh
cargo run -p hipfire-isa --features toolchain --bin peacemaker -- lint path/to/module.co
cargo run -p hipfire-isa --features toolchain --bin peacemaker -- lint path/to/bundle.hxaco --routes route.json
```

The input must be an ELF code object or an uncompressed HIP bundle supported by the Peacemaker lifter. The command emits JSON to stdout, one record per kernel and per CFG natural loop containing WMMA/SWMMAC. A rejected or incompletely decoded module is an error, never an empty/clean report. `--routes` takes a JSON object keyed by **exact symbol**, for example:

```json
{
  "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1151_nt4": {
    "rows": [17, 31, 0, 64], "tile_rows": 16, "k128_epochs": 2
  }
}
```

`rows` is the captured number of real routed rows for **each expert**; `tile_rows` is the kernel's actually executed row tile, declared by its launch contract. Padding is `(sum(ceil(rows/tile_rows)*tile_rows)-sum(rows))/sum(ceil(rows/tile_rows)*tile_rows)`; zero-row experts execute no rows. `k128_epochs` is the kernel's declared number of K128 epochs in one hot-loop trip (not inferred from the WMMA instruction form). The optional `epoch_starts` map contains CFG loop-header block IDs as string keys and the exact instruction-layout positions of that loop's K128 epoch starts (first position must be the loop's first instruction). For example, `{"epoch_starts":{"2":[120,196]}}` for a two-epoch loop with header block ID 2. These boundaries are producer-supplied evidence, **not** guessed from waits. Omit declarations when unknown: corresponding JSON metrics are `null`. Nonempty routes with zero tile width, all-zero routed rows, or inconsistent epoch markers are errors.

`wmma_per_trip` counts emitted matrix instructions once per natural-loop trip; `wmma_per_k128` divides by the declared K128 epoch count. `independent_chains` counts distinct vector accumulator destinations, not folded FP32 sums. `valu_per_wmma` counts emitted VALU **packets** (a VOPD pair is one packet), and VMEM packets and operand width are divided by WMMAs in the same loop. `wait_depth_before_wmma` is the minimum VMEM pending units immediately before any WMMA according to the CFG-converged wait replay; zero means a complete VMEM drain was observed. With declared epoch boundaries, `prefetch_distance_epochs` is the greatest loop VMEM-load distance to its first retiring wait, including waits on the next loop trip; without those boundaries or if a load has no known retiring wait, it is `null`, not a fabricated zero. Loop overlaps are reported separately, not added as dynamic execution counts. This is a *schedule census*, not a predicted TOPS or source of byte-identity proofs. The `single_accumulator_chain` and `prefetch_drained_before_wmma` findings are advisory and do **not** block CI or custom builds in v0.4.1.

For a before/after pair, `cost_lint::drain_pressure` ranks lost overlap as `(before_depth-after_depth)_+ × after_vmem_per_wmma / after_wmma_per_trip`. This is a *unitless ranking only*, not a latency/speedup estimate. The pinned calibration fixture `tests/fixtures/cost_lint/wait_delete.json` contains the 26 measured process-ratio triplets from `pm-wait-delete/three-process-timings.json` and static census from the before/after code objects. The CPU test recomputes 6/6 regressions detected, 0/20 false positives, and Spearman ρ≥0.8 on pairs whose median ratio difference exceeds 3σ of the 20 neutral controls (149 decisive pairs; observed ρ=0.835285). All-pair ρ=0.7303 is **not** claimed as passing: differences among controls within the noise floor cannot be reliably ordered by static ISA. The optional mounted-object unit test additionally re-lifts the actual historical code objects; CI without that external corpus skips only this replay, not the checked-in numerical fixture.

`crates/hipfire-isa/tombstone.toml` records killed variants and the reports containing their measured numbers. It is evidence for avoiding already-killed searches, not a list of currently disabled production kernels.
