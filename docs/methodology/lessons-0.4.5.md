# Engineering lessons from the 0.4.5 cycle

**Status: authoritative.** These rules come from the 0.4.5 Flash-Next serving
campaign (2×R9700 EP2, November 2026). Where [`AGENTS.md`](../../AGENTS.md),
[`CLAUDE.md`](../../CLAUDE.md) or an older campaign note says otherwise, this file
wins. Timing protocol (identity, warmup, sampling policy, dispositions) stays in
[`perf-benchmarking.md`](perf-benchmarking.md); validation routes stay in
[`VALIDATION.md`](../VALIDATION.md). This file adds the rules those owners did not
have and corrects the ones that turned out wrong.

Each lesson has a one-line **rule**, the **why**, and the in-repo **evidence**:
commits on `beta`, fabric job ids (`jNNNN`; landed changes quote theirs in the
`CHANGELOG.md` entry of the commit, killed experiments have only the id) and docs. The numbers quoted are
examples, bound to the fixture that produced them. §11 lists the ones that
must not be reused on other hardware.

Terms: a **cell** is one (workload, concurrency C) point of a served
benchmark (for example code C8 = eight code streams arriving together). A
**load** is one fresh `hipfire serve` process. **τ** is committed tokens per
verify step per lane. **Bitwise** means the change is byte-identical to the
incumbent on every gated output.

## 1. Decompose before choosing levers

**Rule.** Before ranking levers for a served cell, split the cell's aggregate
tok/s into TTFT, decode step time and τ with a counterfactual model, and spend
effort on the factor that owns the gap.

**Why.** Aggregate tok/s includes TTFT (sum of tokens ÷ (last end − first
start)). Two cells with the same aggregate gap can need opposite work. Without
the split, step-time work gets built for cells whose gap is prefill.

**How.** Per cell and load, keep each stream's decode duration
`end − (start + ttft)`. Recompute the aggregate with TTFT replaced (for
example by the comparison engine's TTFT) and decode durations scaled by a
candidate step speedup. Report medians over loads. The model must reproduce
the measured medians exactly before you use it. A model of the form
`agg = N / (TTFT + steps × step + tail)` matched code C1 within 1% and
underestimated code C4 (tail and overlap effects), so quote measured paired
differences, not the model, when they disagree.

**Evidence.**
- On the cp15 baseline (`7d893583a7`), story cells were 93–97% step and code
  C4/C8 were 69–72% TTFT. That sent code work to prefill: lane-WMMA prefill
  (`4aeeacf834`, code TTFT C1/C4/C8 0.139→0.116 / 0.296→0.207 /
  0.511→0.335 s, j8721/j8728), the TTFT host levers (`0ae7e4e8c5`, j8640) and
  the sequence-parallel sub-512 waves (`566f26b40a`, `ea94a8e415`). Story work
  went to the step (`0cbabb3d57`, `bd14d9c48b`).
- The same decomposition showed that a planned "+7.4% τ restore" credit for
  code C1 did not exist: the τ drop was a reducer artifact (§4.2). Code C1 had
  to come from TTFT and step.

## 2. Kill experiments before building, ceiling first

**Rule.** Before building a lever, compute its ceiling (Amdahl: the share of
the step or TTFT it can remove at best) and run the cheapest experiment that
can kill it. Declare the kill line before measuring.

**Why.** Most candidate levers died at this stage, for the cost of one job or
a CPU-only replay. Building first would have cost days and twins each.

**Evidence.**
- **symlab** (exact-rewrite search over the decode layer): the opaque
  routed-expert launch is 85–95% of the modelled step and the 28 launches per
  rank cost 56 µs of gap. Removing every other overhead and activation byte
  leaves at most 113 / 187 µs (story C1 / code C8) against a 200 µs threshold,
  so no exact rewrite can pay. The search was stopped on that ceiling, not on
  a failed run. Its cost model had also charged zero-copy views as memory
  writes, a phantom saving of more than 200 µs that a candidate could have
  "won": check the cost model before trusting a search hit.
- **Draft-round host turnaround:** a profiled attribution of about 3.5 ms
  turned out to be mostly the step head. Unprofiled turnaround was 8–14 µs per
  boundary (j8428), a ceiling of 0.08 / 0.24 ms/step. Killed before any build.
- **AR skew gather-in-AR:** ceiling ≤ 0.42 / 0.55 ms (story / code C8) against
  a multi-day cross-rank design plus two twins, and it needed a weight source
  that does not exist under the trunk shard. Parked.
- **Story acceptance:** block verification, draft temperature, sibling trees
  and a draft distribution over 16 exact candidates were all killed offline from one acceptance
  trace (j8735, 23.7k verified drafts) and one own-head probe (j8725). No
  kernel or twin was built for any of them.
- **Aging loan bank:** killed by a CPU replay of the logged expert traffic
  (26 loads; −25.6 MiB/step [−28.7, −22.5] against a −50 MiB/step bar).
- **Explicit `s_prefetch_inst`:** killed in one job (j8440). The kernel
  descriptor already prefetches about 32.6 KB of instructions, which also
  explained why a fully unrolled 60 KB capture kernel lost (j8417–j8419).
- **Profiled ceilings overstate.** The profiler inflates idle and launch gaps
  (in-loop dispatch gap 3.1 µs profiled, about 0.7 µs unprofiled on this
  fixture). Take ceilings from unprofiled timers or scale the profiled figure.

## 3. An isolated kernel win is not a served win

**Rule.** A kernel that is faster in an isolated bench is a candidate, not a
result. Land on a served A/B, unless the change is bitwise and its projected
step effect is below what a served A/B can resolve. In that case land on an
in-module re-time plus the bitwise gates (§7), and say so.

**Why.** In the model, the kernel sees other shapes, a warm or cold L2,
cross-rank waits and host time that the bench does not have.

**Evidence.**
- **NT1 down selection:** 5–6% faster in the isolated bench at 16–36 rows.
  Served, pooled over 10 loads per arm: story −0.32 / +0.25 / +0.11 ms, code
  +0.19 / +0.27 / −0.26 ms (C1/C4/C8). One 4-load repetition even showed story
  C4 +0.92 ms [+0.28, +1.57] slower (j8839). Parked, not landed.
- **P3 BF16-WMMA prefill tile:** 1.3–3.5× per GEMM in isolation and about 4%
  per wave forward in the lane oracle. Served code TTFT was flat over 8+8 loads
  (C8 ×0.982, C4 ×1.003, C1 ×0.999; j8850 + j8854): about 29 ms of the
  family's time is in shapes P3 does not tile. Parked.
- **Sub-512 sequence-parallel singleton prefill:** the oracle showed −9 ms,
  but served code C1 TTFT moved only 0.116 → 0.113 s (`566f26b40a`).
- **Bitwise, below served resolution:** the HC8 rip-search winners
  (`74378cbd28`, about −0.23 ms/step code C8) and the X8 rot load order
  (`bd14d9c48b`, about −0.18 ms story C8) landed on in-module re-times on two
  cards plus lane-oracle identity, because a served A/B cannot resolve
  0.2 ms on this fixture.

## 4. Measurement hygiene

### 4.1 Use nonces or distinct prompts in C>1 greedy comparisons

**Rule.** Every C>1 comparison that includes a greedy (T=0) arm sends each
stream a distinct prompt (a nonce line or a distinct prompt). Check that the
lanes' decoded texts differ.

**Why.** Identical greedy lanes produce identical text. The routed MoE
deduplicates experts across a step's rows, so eight identical lanes read
about one lane's experts, and the arm looks faster than any real workload.

**Evidence.** A "sampled verify is 18% slower than greedy" finding at story C8
(31.5 vs 25.7 ms) was this artifact: the T=0 arm ran without nonces (j8615),
MoE gate_up and down took +2.91 / +1.54 ms more per step in the diverse arm
(j8616). With nonces the gap was +1.8 ms (j8617); a clean top_k=1 control on the
full sampled path was +0.03 ms (j8618). `hipfire bench --concurrency` already
tags every request uniquely ([CLI.md](../CLI.md#bench-concurrency---concurrency)).

### 4.2 Check the reducer before believing a delta

**Rule.** The reducer that turns serve logs into step ms and τ must classify
bursts by the harness's known layout, drop lead and warm-up bursts, and report
EOS-on cells and fixed-length rows separately. Re-run a changed reducer on the
older logs before comparing across checkpoints.

**Why.** Pooling bursts of different kinds moves τ and step ms by more than
the effects being measured.

**Evidence.** A code C1 τ "regression" from 5.78 to 5.38 between two
checkpoints was mostly 1-token lead requests that finished early, formed
separate bursts and were counted as code C1 decode steps. Faster prefill
(`4aeeacf834`) made the lead finish before the main burst, which caused the
split. Dropping bursts of ≤ 5 log lines gave 5.615; the remaining difference
was −0.18 [−0.56, +0.23] τ, inside load noise. The fixed-256 rows have a
different τ from the EOS-on cells (code C1 5.1–5.6 vs 6.1–6.4) because the
tail past the natural end is free-running text. A controlled four-arm
Latin-square run (j8772, j8773, j8775, j8777) showed that neither landed
lever moved τ.

### 4.3 Check host load yourself, even when the scheduler says clean

**Rule.** For every timing job, record host runnable load and foreign CPU use
over the run, look for per-load dips, and re-run on a quiet host instead of
quoting a dirty job.

**Why.** The scheduler's confounded flag measures one thing (unpartitioned CPU
demand against a threshold). Other host load still bends serve timings,
because the decode loop has a host turn per step.

**Evidence.**
- j8775 ran with runnable mean 21.0 against a threshold of 16 and was flagged
  clean. Its default arm aggregated 208.2 tok/s vs 224.6–226.6 on the other
  loads.
- A checkpoint pair (j8851, j8853) ran beside other timing jobs on the host:
  runnable mean 17.4 / 16.7 and per-load dips of 15–35% in four cells, both
  flagged clean. It was discarded and re-run (j8859, j8869).
- One load of a sampled-verify control jittered 28–72 ms per step with
  unchanged τ while a foreign job started. It was excluded; the median over
  the other loads stood.

### 4.4 Time a throwaway arm first

**Rule.** In an interleaved timing run, the first arm of each cell is a
throwaway (clock ramp and per-cell warm-up). Put sacrificial objects or a
discarded arm first and keep the reference arm out of the first slot.

**Why.** The first arm of a cell runs on a GPU that is still ramping. If the
reference always goes first, it reads slow and candidates look like wins.

**Evidence.** In the rip-search HC8 re-time (j8744, j8745) the first reference
arm of every cell, and the first candidate after it, read 1–6% slow. The
confirmation runs (j8748, j8749) put two sacrificial objects first and skipped
the first three reference arms. Only then were the winners called
(`74378cbd28`).

### 4.5 Interleave at least four fresh loads per arm

**Rule.** A served A/B uses at least 4 fresh loads per arm (6–12 when the
expected effect is under about 1 ms/step), interleaved ABBA or Latin-square
across slots, on the same binary where possible (kill switch as the A arm).

**Why.** Load-to-load spread on this serving fixture is larger than most
single levers: code C1 τ per load ranged 5.0–6.4 within one binary, and one
4-load repetition can show a CI that excludes zero in either direction.

**Evidence.** NT1's first repetition (4 loads) showed a significant story C4
regression; the pooled 10 loads did not (§3). The elastic reserve was judged
on 12 loads per arm in two ABBA jobs (j8669/j8670, `240207c6e3`). The peer
reserve used 12 loads per arm (`7d893583a7`). The τ-drop study used a
Latin-square order so every arm took every slot once.

### 4.6 Report bootstrap CIs and pooled medians with load counts

**Rule.** Report every served delta with a 95% bootstrap CI (paired over
bursts or over loads, say which). A bar claim ("faster than X") quotes the
pooled median, the CI of the ratio and how many loads cleared the bar.

**Why.** Medians of 4 loads move by several percent between jobs; a median
above a bar with a CI that includes 1.0 is not a win.

**Evidence.** cp18 (`bd14d9c48b`): code C8 median 890.9 tok/s was 1.012× the
external bar. Pooled over 8 loads it was 1.026×, but the bootstrap CI of the
ratio was [0.985, 1.052] and 6 of 8 loads cleared the bar, so the claim is
"above on the median, not separated". The elastic cap raise (`74d78df1ff`)
quoted a step gain whose baseline had one outlier load; its CHANGELOG entry
carries that caveat.

### 4.7 Measure the build that lands, and prove the lever ran

**Rule.** Quote the gain measured on the land build (after twins and rebase),
not the lab build. In every measured run, show the log line or counter that
proves the lever was active.

**Why.** Twin re-emission, rebases and default changes move the number, and a
lever that silently did not engage measures as noise.

**Evidence.** The X8 glue fold measured −3.09 ms on its lab build and −1.60 ms
on the twin build that landed (`85351b1e66`). Checkpoint receipts quote the
hot-unit loan and elastic lend lines per rank (cp18: loan active on both
ranks in 8 of 8 rank-lines), and a run that did not show the elastic lend was
investigated rather than averaged in.

### 4.8 Run an A/A control on any oracle you rely on

**Rule.** Before a comparison oracle decides anything, run it off-vs-off.

**Evidence.** The D2 retire-decision lines differed between two identical
"off" runs, so that oracle could not judge the retirement-overlap change; the
decision rested on wall time instead.

## 5. Per-architecture cost tables are measured, never borrowed

**Rule.** Any cost table or prior that drives a runtime choice (depth,
route, tile) is measured on the architecture it runs on. A table from another
arch is a bug, even when it is the only table.

**Why.** Relative costs differ by arch (memory bandwidth, launch cost,
row-scaling). A borrowed table picks the wrong option silently, and the
outputs stay correct, so no correctness gate catches it.

**Evidence.** EP lanes on gfx1201 priced draft depth with the gfx1151
`MTP_WINDOW_COST` table: about 10% of story windows drafted nothing. Pricing
the AR window with the measured gfx1201 step cost (`44c3c7838a`,
`HIPFIRE_QWEN4_LANE_MEASURED_DEPTH`) gave story tok/ms ×1.041 / ×1.049 /
×1.035 (CIs above 1, 6 loads per arm), AR windows 8% → 0, code flat. A first
version that priced every depth with a linear fit over-drafted (×1.032 / 1.020
/ 1.007, CIs straddling 1), because the fit under-priced deep windows. Nonzero
depths still use the gfx1151 table on gfx1201: that is open debt, not a
design. A depth with no measured C>1 cost must never be chosen: K=7 at code
C8 costs 169 ms per step.

## 6. Residency first for MoE weights that do not fit VRAM

**Rule.** When routed experts do not all fit in VRAM, make the hot experts
resident before tuning the kernels that read host-mapped weights.

**Why.** A host-mapped expert read is PCIe-bound. Kernel work on it cannot
beat moving the bytes into VRAM.

**Evidence.**
- Hot-unit loan (`b829080894`): code step −1.05 / −3.53 / −2.38 ms (C1/C4/C8),
  bitwise.
- Peer reserve for the expert tail (`7d893583a7`): code C4 −3.43 ms
  [−6.62, −0.25], C8 −1.78 [−2.77, −0.79], 12 loads per arm.
- Elastic expert reserve (`89b32a0f23`, receipts `240207c6e3`): code step
  −2.70 / −4.40 / −5.79 ms, tok/s 178.8→194.3 / 431.7→481.8 /
  636.1→695.3, no TTFT regression. Cap raised to 2048 MiB/rank in
  `74d78df1ff`.
- A down-kernel rewrite was killed: the down GEMM is memory-bound and the
  remaining time per layer was host-mapped weight tiles, i.e. residency.
- **`hipMemGetInfo` lags after a free.** The driver releases a freed buffer
  object asynchronously, so free memory read right after a free is stale.
  The first elastic build refused two 16k prompts with HTTP 503 under stress
  (j8650). The reclaim now waits (at most 250 ms) until free memory shows the
  released bytes (`ec6d74bb5f`). Any admission decision taken right after a
  free must do the same.

## 7. Bitwise-gate discipline

**Rule.** A change that claims to be bitwise passes all of these; a change
that is not bitwise says so and passes the KLD route instead.

1. **Twins.** A new kernel, entry or mode gets a PeaceMaker twin (Route A for
   hipcc kernels, [`leviathan-pm-twin.md`](../leviathan-pm-twin.md)). Host-only
   reuse of existing entries needs none (`74b0fce95a`).
2. **Lane oracle**: token ids equal (`IDS_EQUAL`) plus every verify-logit dump
   byte-equal, candidate vs incumbent on the same binary (kill switch).
3. **Served C1 byte-equal**: greedy replies at C1 identical between arms.
4. **Numeric changes**: KLD per prompt-length band against the BF16 teacher,
   paired on-vs-off with CIs, plus a greedy diff list. `4aeeacf834` passed
   (every band within its paired CI). The int8 route for sub-512 bursts was
   killed on KLD before any build (paired KLD +0.0055…+0.0141, all CIs above
   0, absolute 0.081 over the 0.08 bar).

**Row-exact off means lane ≠ singleton is expected.** With
`HIPFIRE_QWEN4_LANE_ROW_EXACT` off (the default, `20c7b6de0e`), lane verify
steps of 9..=64 rows use routes whose per-row arithmetic depends on the
co-batched rows. Concurrent MTP output may then depend on batch composition;
C1 and AR stay exact. Compare a candidate against the incumbent under the same
setting, never a lane against the singleton. A lane-vs-singleton failure
that appears in both arms is this known divergence, not the candidate (the
lane-trace exit 1 seen while gating `3639de7b26` was exactly that). The gates do catch real defects: the
default-on exact EP prefill wave was found not byte-identical for ≤ 64-row
lanes (first-token KL max 0.114, one top-1 flip in 17 lanes, j8439) and was
fixed by raising its row floor (`2919b59bbc`). The later small-lane wave
(`HIPFIRE_EP_LANE_PREFILL_WAVE_SMALL_LANES`, default on) re-admitted those
lanes on a burst-oracle pass, but served C1-vs-C8 identity (0.5.0 D1 check)
showed they are still not byte-identical to a solo prefill: by default they are
distribution-equivalent, and only `serve.batch_exact` (which keeps them out of
the wave) is byte-identical.

## 8. Gate gaps found this cycle

- **railgun-cert scans the whole repository and must always run.** Its
  recording-inventory test (`every_recording_predicate_in_the_tree_has_an_inventory_row`)
  reads every crate, so a land gate that skips it "because the change is not in
  railgun" lets a red inventory land. `240207c6e3` landed that way; the SITES
  row came in `8aab98f1e0` and the land gate now always runs railgun-cert.
  New recording/capture predicates need their row in the same change
  (`ea94a8e415`, `3639de7b26`).
- **Run `scripts/pre-commit-ci.sh` and clippy on the final commit**, after the
  last rebase, squash or conflict fix, not on an earlier one. Clippy's
  deny-by-default lints fail it even without `-D warnings`: a test's
  redundant comparison failed the land gate (`566f26b40a`).
- **Additive merge conflicts are routine** in
  `crates/rdna-compute/src/feature_flags.rs`, `docs/env-vars.md`,
  `CHANGELOG.md`, `Cargo.toml` and kernel registries when several default-on
  levers land in a row (async prepare `94442ad0db`, S1 + X8 `85351b1e66`).
  Keep both sides, then re-run the whole gate on the merged tree:
  `check-env-docs.py`, `check-lifecycle.py` and crate maps catch a half-kept
  side.

## 9. Search pays where the targets were never swept

**Rule.** Point automated schedule search at kernels nobody has swept, ranked
by served µs per step. A kernel already hand-swept is a calibration, not a
target.

**Evidence.**
- Phase 1 on the MQ2 xbatch GEMV (already hand-swept): 720 certified objects,
  720/720 hardware byte-identical (j8590), no winner beyond noise. It
  validated the pipeline and nothing else.
- Phase 2 on the HC8 family (never schedule-swept): 438 certified, 342 unique
  objects, all byte-identical on hardware; down_n3 −16.1%, down_n2
  −7.8% / −4.5%, up_n1 −1.9% per call (`74378cbd28`). In the generator's own
  unroll space the shipped value was already optimal; the wins came from a new
  axis (issue the activation loads before the weight loads).
- For hipcc kernels with no Rust generator, the route is knobbed HIP source:
  `#ifndef` constants whose default build is sha256-identical to the committed
  anchor, each candidate re-sealed through `hip_twin` (Route A). The X8 rot
  activations-first change (`bd14d9c48b`, −3.66% per call, 6/6 passes beyond
  2× A/A) came this way.

## 10. Clean up after merge

**Rule.** When a branch lands or is killed, delete its worktree build caches
(`target/`) and bulky dumps (verify-logit dumps, rocprof traces, binaries over
about 50 MB). Keep receipts: logs, small JSON, result notes and the commit.

**Why.** Each lever leaves tens of GB (a full verify-dump set is about 1.2 GB
per arm). Disk pressure stalled timing jobs this cycle; one sweep of landed
and killed work freed about 99 GB, and later per-branch cleanups 21 and 42 GB.

## 11. Host-specific: do not generalize

These numbers were measured on one fixture: 2× R9700 (gfx1201) EP2,
Flash-Next mq4-xts with the hot-unit and HC8 sidecars, `hipfire serve --spec
mtp --max-seq 16384`, November 2026. Use them as examples of scale only.
Re-measure before using any of them on other cards, models, host CPUs or
builds.

| Quantity | Value on this fixture | Source |
|---|---|---|
| In-loop dispatch gap, profiled / unprofiled | 3.1–3.2 µs / ≈ 0.7 µs per launch | rocprof census, story C8 |
| Kernel-descriptor instruction prefetch | ≈ 32.6 KB (bodies above it pay icache misses) | j8440, j8442 |
| Safe hot-unit loan | 768 MiB per rank | `b829080894` |
| Elastic reserve lendable | ≈ 1620 MiB per rank; reclaim p95 4.79 ms (gate 5 ms) | `240207c6e3`, `74d78df1ff` |
| `hipMemGetInfo` settle after free | 2.84 ms max observed; wait capped at 250 ms | `ec6d74bb5f`, `240207c6e3` |
| Singleton prefill, 511 vs 512 rows (before `4aeeacf834`) | 247 ms vs 146 ms | route change at 512 rows |
| Gap owner at cp15 | story 93–97% step; code C4/C8 69–72% TTFT | §1 |
| Load-to-load τ spread, code C1, one binary | 5.0–6.4 per load | §4.2 |
| Host-load threshold that still let confounded runs through | runnable mean 16 | §4.3 |
| First-arm warm-up penalty | 1–6% per cell | §4.4 |
| Lane depth | K=5 kept (code C8 1.18 ms per lane-token vs 1.37 at K=3) | j8484–j8486 |
| cp18 vs the external bars | code C8 1.026× pooled (CI [0.985, 1.052]), code C4 0.938×, code C1 0.786×; story 0.875–0.924× of the reference engine | j8859 + j8869 |
