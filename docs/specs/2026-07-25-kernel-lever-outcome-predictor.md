<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev> -->

# Kernel lever-outcome predictor

| Field | Value |
|---|---|
| State | **planned** |
| Date | 2026-07-25 |
| Target | Predict end-to-end throughput delta from an isolated kernel measurement |
| Homes | [`crates/hipfire-atlas`](../../crates/hipfire-atlas), [`crates/radiowave`](../../crates/radiowave) |
| Data owners | `autoresearch/corpus/`, `redline/corpus/` |
| Related | [`2026-07-19-cross-arch-utilization-campaign.md`](2026-07-19-cross-arch-utilization-campaign.md), [`2026-07-23-deepseek4-mq2r-e8-recipe.md`](2026-07-23-deepseek4-mq2r-e8-recipe.md) |

## 1. Decision

Build a **lever-outcome predictor**: given a proposed kernel change and its
isolated microbenchmark result, predict the end-to-end throughput delta before
spending an acceptance run.

Do **not** build a cycle simulator, a general GPU performance model, or a neural
network. See §8.

The predictor ships in two tiers. **Tier 0 is pure arithmetic and requires no
training data.** Tier 1 is a small fitted correction over the existing corpora,
gated on label volume that does not yet exist.

## 2. Problem: prediction, not inspection

The engine already inspects kernels well. `radiowave` extracts VGPR/SGPR/LDS/
scratch/occupancy from the code object; the redline PM4 wait audit computes
launch-stream resource dependencies; `rocprof` traces segment per decode token.
None of that is the bottleneck.

The bottleneck is that **isolated microbenchmark deltas do not translate to
end-to-end deltas, and the error runs in both directions.** Measured on the
DeepSeek V4 MQ2R gfx1151 campaign:

| lever | isolated | projected | measured e2e | verdict |
|---|---|---|---|---|
| weight cache policy (B2) | — | ~1.5 ms | 1.10 ms, **+2.94%** | ACCEPTED |
| `topk_kv_gather` LDS transpose | 72.23 → 19.57 µs (**−72.9%**) | ~0.6–0.9 ms | **+3.3%** | accepted |
| top-k E2 bounded bitonic | 564 → 52.8 µs | 3.2 ms | 0.89 ms | accepted |
| small-M dense k-split4 | 1.27–1.40× | 1.2 ms | 0.12 ms (**0.31%**) | REJECTED |
| attention cross-lane lowering | 176.392 → 175.274 µs (**+0.64%**) | 1.5 ms | **0.046 ms** | REJECTED |
| attention large-context serial | 176.268 → 116.259 µs (**−34%**) | 2.46 ms | **−0.19 ms (−0.52%)** | REJECTED |

Two of these consumed a full six-sample fresh-process ABBA at roughly 14 minutes
each, plus build and screening time, to establish a result that was arithmetically
predictable. One (`attention large-context serial`) recorded a 34% isolated win
that measured **negative** end to end with exact output parity.

The projections in column three are unaided human/agent estimates. They are wrong
by up to 30× and in both directions. That is the defect this spec addresses.

## 3. Tier 0 — arithmetic, no training data

```
Δ_e2e  ≈  isolated_Δ  ×  share_of_token  ×  affected_call_fraction

  isolated_Δ             from the microbenchmark
  share_of_token         kernel_ms / token_ms, at the ACCEPTANCE depth
  affected_call_fraction calls entering the changed code path / total calls
```

The third term is the one that has never been computed, and it is where the
misses come from.

### 3.1 Worked example — attention large-context serial

At 2,048 KV depth the segmented decode token is **2,328 launches, 38.77 ms busy**,
of which `deepseek4_attn_swa_topk_scoregrid_f32_buf` is **41 calls, 3.38 ms, 8.7%**.

```
isolated_Δ            = -34%
share_of_token        = 3.38 / 38.77 = 8.72%
naive projection      = -2.96%          (assumes every call site is affected)
measured              = -0.52%
implied affected      = 0.52 / 2.96 ≈ 18%
```

The mechanism is known and was documented during the campaign: scoregrid's fast
path applies only at `n_total <= 64`. At the benchmark depth most of the 41 call
sites never enter the modified body, so the isolated bench — taken at a single
representative shape — misrepresented the kernel by roughly 5×.

Tier 0 with a correct `affected_call_fraction` predicts −0.52%, which is below
the §5 threshold and would have rejected the lever without a build.

### 3.2 Worked example — topk_kv_gather

```
isolated_Δ            = -72.9%   (72.23 -> 19.57 µs)
share_of_token        = 1.33 / 38.77 = 3.43%
naive projection      = -2.50%
measured              = +3.3%
```

This one **over**-delivered, which is itself signal: the kernel's scattered
512-float-stride stores were degrading the memory system for neighbouring
kernels, so the fix paid twice. Tier 0 under-predicts here, and that residual is
exactly what Tier 1 exists to learn.

## 4. Tier 1 — small fitted correction

A ridge regression or shallow gradient-boosted tree over the residual between
Tier 0's projection and the measured delta.

Features, all already collected:

| feature | source |
|---|---|
| `isolated_delta`, measurement shape | microbench |
| `share_of_token` | segmented rocprof trace |
| `affected_call_fraction` | `redline/corpus/launch_shapes.jsonl` |
| `vgpr`, `sgpr`, `lds`, `scratch`, `waves_per_simd` | `radiowave::ResourceContract` |
| `mem_busy`, `occ`, `l2_hit_pct` | `attempts.jsonl` `roofline{}` |
| `lever_family` | **not currently recorded — see §6** |
| `arch`, `kv_depth`, `model` | run metadata |

**Do not exceed this.** The labelled set is ~932 clean rows with ~115 on gfx1151
(§5). A deeper model overfits, and worse, becomes uninterpretable — which this
campaign would correctly ignore, since its operating discipline is measured
evidence over inference. Tier 1 must emit a point estimate **and** a confidence
band, and must be able to say "insufficient data for this lever family."

Gate Tier 1 on **≥ 200 labelled gfx1151 rows.** Below that, ship Tier 0 alone.

## 5. Data inventory

Measured 2026-07-25. Row counts are total; label counts are what is actually
usable as a supervised target.

### 5.1 autoresearch — the clean labels

| file | rows | contents |
|---|---|---|
| `attempts.jsonl` | 932 (115 gfx1151) | `lever`, `kernel`, `arch`, `delta_pct`, `verdict`, `rounds`, `mwu_dominance`, `roofline{mem_busy, occ, vgpr, l2_hit_pct}`, `profile` auto-diagnosis, `extra{base_runs, var_runs}` |
| `kernels.jsonl` | 157 | static: `vgpr`, `sgpr`, `lds`, `scratch`, `isa_fingerprint`, `bound_class`, `shape_bucket`, `repro_cmd` |
| `bod.jsonl` | 25 | per-arch decision summaries |

This is the only corpus where a lever is bound to a measured outcome **and** a
taxonomy. It is the training set.

### 5.2 redline — rich features, thin labels

| file | rows | usable |
|---|---|---|
| `bench.jsonl` | 2,354 | `speedup` present on only **213**; `classification` null on 1,525 (65%); `family` null on 2,349 (**99.8%**) |
| `captures.jsonl` | 1,718 | `sequence_hash`, `launches`, `unique_kernels`, `tok_s`, `us_per_token`, `context_tokens`, `kv_mode` |
| `launch_shapes.jsonl` | 515 | `kernel`, `grid`, `block`, `shared_mem`, `kernarg_bytes`, **`occurrences`**, **`positions`**, `shape_key`, `distinct_sequences` |
| `aql_contracts.jsonl` | 391 | AQL packet contracts |

Arch coverage in `bench.jsonl`: gfx1100 658 · **gfx1151 541** · gfx1201 434 ·
gfx1030 259 · gfx1010 240.

**`launch_shapes.jsonl` is the key asset.** `kernel × shape_key × occurrences`
is the `affected_call_fraction` denominator, already harvested — behind it sit
the ~1.1M individual launch records collected by `scripts/harvest_redline.py`.
Tier 0 does not need to recompute it from a trace each time.

### 5.3 What is missing

Not volume. **Shape.** `bench.jsonl` records outcomes without the lever taxonomy
that makes `attempts.jsonl` useful. See §6.

## 6. Required schema additions

Every candidate evaluation must emit a training row. Six fields; `radiowave`'s
`CandidateRecord{target, gpu_round, attempt, resource_assessment,
correctness_artifact, timing_artifact, verdict, recorded_unix_seconds}` already
carries four.

```
lever_family            REQUIRED  — null in 99.8% of bench.jsonl today
isolated_delta          REQUIRED  — with the shape it was measured at
share_of_token          REQUIRED  — at the acceptance depth, not a proxy depth
affected_call_fraction  REQUIRED  — from launch_shapes or trace bucketing
resource_contract       have      — radiowave::ResourceContract
measured_e2e_delta + CI have      — timing_artifact
```

**The `affected_call_fraction` requirement is most of the value of this spec,
independent of whether any predictor is ever fitted.** Forcing an answer to
"which call sites does this change actually reach?" before a lever earns an
acceptance run would have prevented both attention ABBAs. Treat it as a gate,
not a field.

Backfill from the existing per-run evidence directories where recoverable;
`.codeinsight+research/ds4-gfx1151-campaign/` retains traces and per-arm
summaries for most of the current campaign.

## 7. Placement

Both halves exist and are unconnected.

**`crates/hipfire-atlas`** — `AtlasRow` (`schema.rs`), `suggestions_for_row`
(`suggest.rs`, 96 lines of single-row heuristics), `eval_task` (`eval.rs`). The
suggestion path has the right shape and no link to measured outcomes.

**`crates/radiowave`** — `CandidateRecord` is the only record in the tree that
binds static resource assessment, correctness and timing in one row. It is the
natural carrier.

Proposed split:

- `radiowave::campaign` — owns the record and the six required fields.
- `hipfire-atlas` — owns the Tier 0 projection and, later, the Tier 1 fit;
  extends `suggestions_for_row` to consume outcome history rather than one row.
- Corpus join (`attempts.jsonl` × `launch_shapes.jsonl` × `bench.jsonl`) lives in
  `scripts/`, alongside the existing `harvest_ledgers.py` / `harvest_redline.py`.

## 8. Non-goals

- **Not a cycle-accurate simulator.** ROCm's `rocjitsu` exists and decodes
  RDNA3.5, but it solves running kernels without a GPU. The fleet has gfx1010,
  gfx1030, gfx1100, gfx1151 and gfx1201 on real hardware. Emulating a 284B MoE
  forward is not tractable and would not answer the prediction question anyway —
  the misses in §2 are about shape distribution and share-of-token, not
  instruction semantics.
- **Not a general GPU performance model.** AMD's own PerfXpert `gpu_specs.yaml`
  stops at gfx1100 and is CDNA-first. This predictor is scoped to *this engine on
  this fleet*, which is precisely why it can work at n≈200 where a general model
  could not.
- **Not a neural network.** See §4.
- **Not a replacement for acceptance measurement.** The predictor decides whether
  a lever earns an ABBA. It never substitutes for one, and it never overrides the
  byte-identical output gate.

## 9. Acceptance

Tier 0 ships when:

1. It reproduces all six §2 rows within 2× of measured, given the correct
   `affected_call_fraction`.
2. It is wired into the candidate flow such that a lever cannot reach acceptance
   without an `affected_call_fraction` recorded.
3. The corpus join runs against `attempts.jsonl` and `launch_shapes.jsonl`
   without a GPU.

Tier 1 ships when:

4. ≥ 200 labelled gfx1151 rows exist with the §6 fields populated.
5. Held-out residual error beats Tier 0 alone on a temporal split — train on
   rows before a cut date, test after. Random k-fold is not acceptable here;
   levers within a campaign are correlated.
6. It emits calibrated confidence bands and abstains on unseen lever families.

## 10. Notes

This corpus is unusual and worth saying plainly: it binds diagnosis to measured
outcome across 932 attempts on an architecture family that AMD's own performance
tooling does not cover. The diagnosis strings in `attempts.jsonl` — e.g.
`"L2-hit 14-17% CACHE-NOT-FIXED(DRAM-thrash; SLC/stream the write-once weights,
keep x/KV resident)"` and `"VGPR 40->48; occ 0.5->0.3% DROPPED-WAVES(you
overloaded = the row-reuse regression mode)"` — are already better grounded than
a single-trace advisor, because each is attached to a `delta_pct` and a
`verdict`. The predictor is the interface that corpus has been missing.
