# iu4frag/attrib2 — kernel-only ablation marginals (N=512 gate_set)

Current kernel (`fd0969955`), ordinal 1, gfx1201, same session 2026-09-18
~08:27–08:45 UTC. One runner binary JITs 8 TU copies from disk
(`gen_attrib_tus.py` + `tmp_iu4_attrib.rs`); module `tmp_iu4_attrib_<tag>`
keeps per-variant cache entries (`.hip/.hsaco/.radiowave.json` in ab1).
3 brackets each (fresh process; each bracket = 10 warmup + 5×20 median).

## Marginals table (median-of-3 brackets, µs/call)

| # | variant | b1 | b2 | b3 | med | FULL−med (marginal) |
|---|---|---|---|---|---|---|
| 1 | full | 500.6 | 498.9 | 503.8 | 500.6 | — |
| 2 | no-stage (staging loads/stores after prologue → `if(false)`; barriers + re-points stay) | 456.5 | 461.1 | 457.8 | 457.8 | **42.8** |
| 3 | no-fold (fold nest → `if(false)` + full-cacc XOR checksum sink in-trip) | 447.6 | 448.1 | 446.6 | 447.6 | **53.0** |
| 4 | no-compute (4 bundles → `if(false)`; cacc = volatile-loaded opaque init, zero WMMA) | 381.1 | 378.7 | 376.0 | 378.7 | **121.9** (incl. its 24 frag LDS reads) |
| 5 | no-store (store nest → `if(false)` + full-acc fadd sink) | 523.7 | 524.6 | 526.6 | 524.6 | **−24.0** (negative: tail-drain/scheduling artifact) |
| 6 | no-barrier (all 7 static `__syncthreads()` → no-op) | 449.9 | 446.6 | 447.0 | 447.0 | **53.6** (combined with rescheduling) |
| 7 | sync-stage (prefetch deleted; slab1 sunk below compute-0, next-A0 next to next-W0; same op counts, zero overlap) | 499.0 | 499.8 | 499.3 | 499.3 | **1.3** (≈0: prefetch overlap is worth nothing) |
| 8 | single-buffer (slab1 publishes to W0; 1-line change) | 504.3 | 511.1 | 509.6 | 509.6 | double-buffer saves **9.0** |

Bracket spread ≤1.5% throughout. Do NOT sum marginals (overlapping cover;
Sol's no-invalid-%sum rule): each row is FULL minus one combined
intervention.

## Opcode/resource deltas (static, full_set symbol; whole-fatbin WMMA)

| variant | WMMA (fatbin/sym) | fold cvt | gld_b64 | barriers | VGPR (JIT) | spill |
|---|---|---|---|---|---|---|
| full | 128 / 32 | 68 | 12 | 42 | 183 | 0 |
| no-stage | 128 / 32 | 68 | 4 | 42 | 163 | 0 |
| no-fold | 128 / 32 | 0 | 12 | 42 | 105 | 0 |
| no-compute | 0 / 0 | 12* | 12 | 42 | 117 | 0 |
| no-store | 128 / 32 | 68 | 12 | 10 | 166 | 0 |
| no-barrier | 128 / 32 | 68 | 12 | 0 | 179 | 0 |
| sync-stage | 128 / 32 | 68 | 12 | 42 | 188 | 0 |
| single-buffer | 128 / 32 | 68 | 12 | 42 | 183 | 0 |

* nocompute fold partially restructures over opaque init (12 cvt vs 68) but
executes the fold DAG on defined values — valid, no undef/poison (Sol review
incorporated). v3 sink overhead censused: +1 static `v_xor` (rolled
64-iter checksum) + 1 volatile store per trip ≈ sub-µs. fmac/fadd static
counts wiggle across variants (compiler CSE noise from guard structure);
cvt counts confirm fold presence/absence cleanly.

## Counters: full vs no-stage (rocprofv3 `--pmc SQ_BUSY_CYCLES GRBM_GUI_ACTIVE`, 110 dispatches)

Units: SQ_BUSY = cycles/instance/dispatch with ≥1 wave present (32
instances/dispatch; medians over instances×dispatches); GRBM_GUI_ACTIVE =
engine-active cycles/dispatch (1 instance); dispatch wall in ns from the
same rows.

| variant | wall med | SQ_BUSY med-sum/disp | per-inst busy | GUI_ACTIVE med | GUI/wall @2.93GHz |
|---|---|---|---|---|---|
| full | 490.2µs | 39.92M | 1.247M (87% of 1.436M elapsed) | 1.265M | 88% |
| no-stage | 445.8µs | 36.57M | 1.143M (87% of 1.307M elapsed) | 1.163M | 89% |

Ratios nostage/full: SQ 0.916, GUI 0.919, wall 0.910 — proportional. Removing
staging shortens wall ~9% with identical occupancy character (87% busy both).

## Clocks

Box DPM table max 2350 MHz (`pp_dpm_sclk`); GRBM-implied effective during
execution ≈ 2.6–2.9 GHz (1.265M GUI cycles over 490µs wall ⇒ 2.58 GHz at
100% engine-active, 2.93 at 88%). Use 2.93 for comparability with Sol's
issue model; the DPM table does not capture the boost level. R9700/Navi48:
32 WGP / 64 CU / 128 SIMD32; our OCC "3 WG/CU" = 3 WG per WGP (HIP reports
WGP count; `hip_mp_count_to_cu_count` ×2). rocminfo: CU 64, SIMDs/CU 2.

## Limiter (no fixes this pass)

- K-loop core (WMMA fetch+issue): 122µs. Fold VALU: 53µs. Staging work: 43µs.
- Barrier/sync epochs: 54µs (overlapping; removal also reschedules).
- Prefetch overlap: ~0µs (sync ≈ full) — the machine never exploited it;
  matches Sol's serial-wait ISA finding. Double-buffer: 9µs.
- Store epilogue is NOT a limiter: removing it costs 24µs (tail cover).
- Main's "~200µs outside the K-loop": staging 43 + barrier epochs ~54 +
  Sol's prologue/tail/drain + unassigned waits (~111) ≈ 208 as an
  upper-bound envelope (NOT a sum — see Cautions).
- Biggest single removable item is the compute+fold issue chain itself
  (~175µs); no ≥100µs non-compute overhead stands alone.

## Cautions — Sol review (read before quoting marginals)

- v3 sink cost is NOT sub-µs. Verified in `tu-3-nofold.dis.txt` (full_set
  CBC4..CD38): the checksum is UNROLLED — 1× `v_xor_b32` + 31× `v_xor3_b32`
  per trip (my static census grepped only `v_xor_b32` and missed the xor3s),
  ≈ 14.85µs issue-only @2.93GHz across 40 trips plus dependent-chain delays,
  with a `flat_store_b32 SCOPE_SYS` + `s_wait_storecnt` every trip (added
  serialization), and all-WG aliasing on `Y[tid]`.
- Corrected reading: v3's 53µs is a contaminated fold-removal net
  (fold − ~15µs+ sink), not comparable to modeled fold arithmetic. Fold true
  cost is higher by the sink. v5's once-per-WG sink (≈0.5 MB) is negligible
  — v5's +24 stands as an observed combined-intervention effect, not causal
  proof of tail cover. The clean v3 fix (not run — Sol: no rerun needed):
  accumulate the XOR across trips in a register, store once per WG to a
  WG-exclusive slot.
- The "≈208 reconciled" line in the Limiter section is REJECTED as a sum:
  43/54/111 share effects (staging waits cover fold issue and vice versa).
  Upper-bound envelope only, not an addition. All rows are combined
  interventions; none is an additive phase time.

## Files

- `gen_attrib_tus.py` (generator) + `tu-<tag>.hip` (8 TU copies, worktree
  only — too big to commit; regenerate with the script)
- `brackets/<tag>-{a,b,c}.log` (24 bracket logs)
- `counters/full_both_results.db`, `nostage_both_results.db` (worktree only)
- `counters/*.stdout/*.stderr`, `build-attrib.log`
- Runner: `crates/hipfire-runtime/examples/tmp_iu4_attrib.rs` (kept, tmp_*
  precedent; reads TU path at runtime, no rebuild per variant)
- JIT cache entries `tmp_iu4_attrib_<tag>.<hash>.*` in ab1 (per-variant
  `.hip/.hsaco/.radiowave.json` for Sol's ISA work)
