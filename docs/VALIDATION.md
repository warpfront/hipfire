# Validation routes

Maps claim class → validation route. **Merge authority is direct maintainer
review backed by evidence selected for the changed behavior**, plus the
required no-GPU CI checks named below. **`beta` advances only through the
[promotion lane](#promotion-lane)**: every required hardware row is `current`
on the installed release archive of the exact candidate SHA. Executable
behavior lives in the scripts and workflows named in each route. Methodology
numbers and Redline certification prose live in their owners
([`INDEX.md`](INDEX.md)).

[`hw-gate`](../.github/workflows/hw-gate.yml) is PR-time evidence automation:
useful for a reviewer, not how `beta` moves, and not a substitute for the
lane, the proof the claim needs, or an approving human review. Authors may
request hardware routes with the PR template's `<!-- hw-gate-request -->`
block; no local planning tool is merge evidence.

| Field | Value |
|---|---|
| Original route inventory date | 2026-07-22 (historical provenance) |
| Original inventory source ref | `202282de8759dfa6963ea5184ad2bf2b9259cef6` (historical provenance) |
| Original comparison base | `origin/beta` @ `202282de8759dfa6963ea5184ad2bf2b9259cef6` (historical provenance) |
| Documentation audit | 2026-10-13, `origin/beta` @ `c7a732ad5`: source and documentation review only. No hardware certification was rerun and the route inventory was not remeasured. |
| Integrated commit / tree hashes | External Git/CI only. |

## Rules

1. Pick the **narrowest** route that covers the changed surface.
2. Every route below names an **existing** path, or is marked **blocked**.
3. **Fail closed** on unknown claim classes: do not improvise a gate, and
   do not treat a green unrelated route as coverage.
4. There is **no universal replacement gate**.
5. Harness success without the proof the claim needs is insufficient
   (especially Redline route proof — see [`REDLINE.md`](REDLINE.md)).
6. **Admissions** are recorded only in [`admissions.yml`](admissions.yml).
   A passing route does not create an admission row.
7. **Static review ≠ hardware proof.** Required no-GPU CI and a clean static
   read never substitute for GPU/model evidence the claim map requires.
   Hardware harness output never substitutes for required CI or an approving
   review.

## Merge bar (required)

Branch protection on `master` requires these CI job names from
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml):

- `build (workspace, no GPU)`
- `unit tests (lib, no GPU)`
- `gates (ratchets, layering, registers)`

plus **one approving review**.

Those three jobs stay required. Do not disable them. `build` and
`unit + integration tests (no GPU)` are aggregate checks over the matrix jobs
`build shard (...)` / `test shard (...)`: they pass only if every shard passes,
so the check names above are stable while the shards are rebalanced. Test
shards run under the `ci` Cargo profile (`cargo test --profile ci`: optimised
where the exhaustive CPU oracle tests live, debug assertions and overflow
checks on). Local equivalent of one shard: copy its command from the matrix in
[`ci.yml`](../.github/workflows/ci.yml). These checks gate a merge into the
`beta-candidate` branch; `beta` moves only on a green
[promotion record](#promotion-lane). `hw-gate` is not a substitute for
either.

Every change still owes the evidence named by the [claim → route map](#claim--route-map)
for the surfaces it touches. The author or reviewer may gather that evidence
with the manual harnesses below and/or optional automation; the maintainer
who approves judges that the evidence matches the claim.

The `Windows MSVC (no GPU)` job in the same workflow is required (it was
advisory until its first green run on `beta`, 2026-10-26). It builds the native
CLI/daemon, runs config/registry/client
library tests, checks that `scripts/*.ps1` parse and that `install.ps1`'s
parameters bind (guard for #979), smoke-checks version and parseable JSON
diagnostics without HIP, lists the DLLs the packaged `daemon.exe` and
`hipfire.exe` import against the allowlist in
[`scripts/check-pe-imports.ps1`](../scripts/check-pe-imports.ps1), and
packages/uploads an unsigned developer ZIP plus SHA-256. It does not run
inference or certify an AMD GPU, driver, runtime tuple, kernel pack, signed
installer, or update path. Until a GitHub run exercises it, the Windows job and
PowerShell packaging are unverified; see
[the artifact description](GETTING_STARTED.md#ci-developer-artifact). GPU
evidence for Windows (native or WSL2) comes only from community test reports
filed through the Windows test report issue form (native installs run
[`scripts/windows-check.ps1`](../scripts/windows-check.ps1)); see
[Help test Windows](GETTING_STARTED.md#help-test-windows).

## Promotion lane

Kaden's decision (2026-11-14, reversing the earlier "hw-gate is optional"
rule): candidates land on `beta-candidate`; `beta` fast-forwards to a
`beta-candidate` tip only when every required hardware row of every blocking
card class is `current` on the **installed release archive** of that exact
SHA. Two regressions reached beta through the GPU-free gate in one week (the
DFlash forced-advance crash and the stale `no-caller` pack hole that cut A3B
one-shot speed 42%); the lane exists to stop that class of escape.

### What runs

1. The candidate builds its own archive with
   [`scripts/build-binary-archive.sh`](../scripts/build-binary-archive.sh) at its SHA,
   with the release rustc pinned in `scripts/release-toolchain.txt` (the
   compiler the release binaries use; recorded in the archive's `manifest.json`).
2. [`scripts/promotion-lane/lane.py`](../scripts/promotion-lane/lane.py) `run`
   installs it into a fresh directory and runs
   [`scripts/device_mesh_matrix.py`](../scripts/device_mesh_matrix.py)
   `--rows <required> --host-class <class>` against it with an isolated
   `HOME`, an empty kernel cache and **no device compiler**: compiler
   directories are filtered out of `PATH` and `HIPFIRE_NO_DEVICE_COMPILER=1`
   is set (resolved-root discovery finds `hipcc` regardless of `PATH`). A
   kernel the pack lacks therefore fails the request instead of compiling
   quietly, and any `hipcc compiled (` / `but hipcc is unavailable` line in a
   row's log fails that row.
3. The wrapper, not the runner's exit code, is the verdict: a required row
   that is missing, `hardware-blocked` (absent or mismatched fixture or
   draft), `rerun-required` or `failed` is red. **Skip is not pass.** The
   runner's own exit status is also a verdict now (0 only when every selected
   row is `current`).
4. One `promotion-record-v1` JSON per class (candidate SHA, archive file and
   sha256, gate SHA and policy/matrix hashes, verified GPU arch, every row
   with `current|failed|skipped|waived`, waivers applied, wall time against
   the class budget); `lane.py merge` folds the classes into a
   `promotion-candidate-v1` record whose verdict decides the fast-forward.

Rows run product defaults (no `HIPFIRE_DETERMINISTIC`): developer modes route
to kernels the pack legitimately omits (deterministic mode forces
`gemm_hfq4g256_residual_wmma_k2`, packed for gfx1151 only), so a lane row
under them would fail on a hole no user hits.

### Required rows

Selection lives in [`scripts/promotion-lane/policy.json`](../scripts/promotion-lane/policy.json);
the rows are in [`scripts/device-mesh/matrix.json`](../scripts/device-mesh/matrix.json)
(milestone `lane`).

| Row | What it proves | Classes |
|---|---|---|
| `dense-27b-ar` | 27B mq4-xts AR, reasoning on and off, 8-token think cap (forced advance), an appended turn served from the held conversation (`cached_tokens > 0`), an attempt aborted at `commit_ready` (rollback), clean warm/cold replays reproducing every answer | gfx1100, gfx1201, gfx942 |
| `dflash-think-close` | the same session with the pinned DFlash draft (`dflash_mode on`); pins the 2026-11-13 forced-advance escape | gfx1100, gfx1201, gfx942 |
| `cold-archive-9b` | 9B one-shot from a cold install: pins the 2026-11-13 pack hole | gfx1100, gfx1201, gfx942 |
| `cold-archive-27b`, `cold-archive-a3b` | the same for 27B mq4-xts and A3B `.mq4r` | gfx1100, gfx1201, gfx942 |
| `cold-archive-flashnext` | Flash-Next single-card default route (host-mapped experts) | gfx1201 |

Card classes: gfx1100 and gfx1201 block every candidate and every release.
gfx942 (MI300X) is a **secondary** class (Kaden 2026-11-15: secondary until AMD
interest; the MI300X rental is released): it never blocks a promotion or a
release tag. When a host for it is configured in the landing inventory the
lane runs it and the candidate record and `check-release` report it
(`secondary: gfx942 green|red|no record`); a red secondary record is a
warning. The policy gate values are `candidate` (blocks every candidate),
`rented` (blocks candidates while a host of the class is configured, and every
release), `release` (every release only) and `secondary`. Budget: 15 minutes
per card for gfx1100 and gfx1201 including the performance rows (60 for
gfx942). A `.kldseq` row is not in the set: the evaluator (`eval_hipfire`) is not
in the release archive, and building it would put a compiler back in the lane.

### Performance rows

Correctness rows alone let a 4.4% DFlash per-cycle loss and a 42% cold-start
loss through, so each blocking class also carries performance rows
(`policy.json` `perf`, [`scripts/promotion-lane/perf.py`](../scripts/promotion-lane/perf.py)).
They run after the correctness rows, in the same fabric job (class `timing`:
exclusive card, build slots paused), on the same installed archive.

| Row | Measures | Classes |
|---|---|---|
| `perf-27b-ar-decode` | 27B mq4-xts AR decode tok/s, committed 134-token code prompt, 128 tokens | gfx1100, gfx1201 |
| `perf-27b-dflash-decode` | the same with the pinned DFlash draft, reasoning off | gfx1100, gfx1201 |
| `perf-27b-ttft-134`, `perf-27b-ttft-5900` | 27B prefill TTFT at 134 and 5,909 tokens | gfx1100, gfx1201 |
| `perf-27b-cold` | first request of a fresh process with an empty kernel cache (wall tok/s) | gfx1100, gfx1201 |
| `perf-a3b-mq4rs-decode`, `perf-a3b-mq4rs-cold` | Qwen3.6-35B-A3B `.mq4rs` AR decode (the Redline default) and its cold first request | gfx1201 |
| `perf-flashnext-decode` | Flash-Next GPTQ3 monolith, single-card default route (native MTP + n-gram), decode tok/s | gfx1201 |
| `perf-27b-dflash-warm-ttft` | relation row: the DFlash process's median warm TTFT over the AR process's (same candidate, same job, prompt `code-134`) may exceed 1 by at most `max_excess` (0.1 on gfx1100 and gfx1201); catches a speculative request that leaves work queued on the device for the next request's prefill (the pre-0.5.0 terminal seed forward: ratio 1.50 on gfx1201 and 1.17 on gfx1100, against 1.04 with the deferral). Cold TTFT is not used: the spec-floor load calibration runs DFlash windows before `loaded`, so the DFlash process has no cold first request | gfx1100, gfx1201 |

- **Measurement.** Every number comes from a fresh `hipfire bench` process
  (own `HOME`/`HIPFIRE_HOME`, empty kernel cache, no device compiler,
  `HIPFIRE_*`/`HIP_*`/`GPU_*` knobs scrubbed except the fabric's card
  selectors) with `--warmups 0` and `--min-tokens` = `--max-tokens`: sample 0
  is the cold row, the first `discard` samples are dropped for the warm rows.
  Processes run grouped by model (arms interleaved inside the group), so one
  model's load does not evict the next one's page cache mid-row.
  Prompts are committed under `scripts/promotion-lane/perf-prompts/` and the
  model files are pinned by size and sha256 in the policy.
- **Host pressure.** Fabric's timing class locks the card, not the host CPU
  or memory, and cold rows are CPU-bound. Each process records its load
  average and the PSI `some` stall fraction over its lifetime
  (`/proc/pressure` cpu, memory, io). A process above `max_pressure` (cpu
  10%, memory 0.5%; Flash-Next memory 3%, its host-mapped experts reclaim
  page cache by themselves) is rejected, kept in the record, and re-run up to twice
  after the host's 10 s averages settle; a row still without clean samples
  is `skipped` (red), never measured.
- **Statistic.** Each process contributes the median of its measured runs;
  an arm is the median of its 3 process medians; the row is the ratio of the
  arms against the gate's own beta on the same card (new/old for tok/s,
  old/new for TTFT, so below 1 is always a loss), with a 95% bootstrap CI over
  processes. The process is the unit because runs inside one process are not
  independent (on gfx1100 every process's third 5.9k-token prefill is ~19%
  slower). A row **regresses** when the CI upper bound is below
  `1 - max(2%, 3 x A/A half-width)`; the half-width per class and row comes
  from an A/A calibration (`lane.py calibrate`) recorded in the policy.
- **Relation rows.** A row with a `reference` proc judges the candidate
  against itself rather than against beta: the median over its processes of
  the row's metric may exceed the reference proc's median, measured in the
  same job, by at most the class's `max_excess`. It needs no baseline and no
  A/A calibration, and adds no processes when both procs already have rows.
- **Baseline.** The previous promotion record of the gate's beta for the
  same class supplies the baseline numbers when host, card (fabric GPU
  UUID), HIP runtime, PCI slot and measurement spec match and the numbers are
  at most 7 days old; only the candidate runs. Otherwise the beta's archive
  runs interleaved with the candidate (`pair`). A regression seen against
  reused numbers is confirmed by a `pair` run before it counts.
- **Verdict.** A `regressed`, `failed` (candidate process failed) or
  `skipped` (no usable baseline, fixture or prompt mismatch) perf row is red
  and blocks promotion like a correctness row; waivers name perf rows the same
  way. The record's `perf` section carries each row's ratio, CI, threshold,
  mode and every process's raw samples.
- **Cost.** A full `pair` run of every perf row takes about 10 minutes on
  gfx1201 (Flash-Next is about 3.3 of them) and 5 on gfx1100; a `reuse` run
  takes about half that.

### Three CPU rules in CI ([`scripts/ci-gates.sh`](../scripts/ci-gates.sh))

Each has a planted-negative self-test that fails CI if the rule stops firing.

- **(a) Skip is not pass.** hw-gate grades the declared mode set: a skipped
  mode fails its fixture (`scripts/hw-gate/run.py --self-test`); the matrix
  and the lane wrapper count a non-executed required row as red
  (`device_mesh_matrix.py --self-test`, `lane.py --self-test`).
- **(b) A production-reachable module cannot be allowlisted `no-caller`.**
  `kernel_registry_coverage` fails a `no-caller` row once production code
  calls the function that loads the module; the planted negative replays the
  `gated_delta_net_q8_fast_stream` sequence (`ci.yml`, rdna-compute-abi).
- **(c) A required row exercises forced advance, append and rollback.**
  `lane.py check` requires every transition in `transitions_required` to be
  structurally present in a required row of a blocking class (a capped
  `max_think_tokens`, an assistant turn quoting a captured answer, an
  `abort_at_commit_ready` or reset, judged by `wire_transitions`).

### Gate policy, waivers, attribution

- **Who judges whom.** The lane scripts, matrix and policy come from the
  previously promoted `beta`; the candidate contributes only its archive and
  its waiver file, so a candidate cannot soften the gate that judges it.
- **Gate-policy changes** (paths listed in `policy.json` `gate_policy_paths`:
  the lane, matrix, runner, hw-gate, the kernel allowlist and coverage test,
  layering, ratchets, CI workflows) land only in a candidate that contains
  nothing else and whose commits carry `Gate-Policy-Approved-By: Kaden` or
  `Gate-Policy-Approved-By: fivetide`. The landing script refuses mixed or
  unapproved policy candidates.
- **Waivers** live in [`scripts/promotion-lane/waivers.json`](../scripts/promotion-lane/waivers.json):
  SHA, class, rows, reason, `granted_by` (Kaden or fivetide, never an agent),
  expiry at most 7 days after the grant, rollback plan, `closed_by`. No release
  is tagged with an open waiver (`lane.py check-release`).
- **Attribution.** Agent-authored commits end with `Agent: <model>/<seat>`,
  human ones landing through the lane with `Human-Authored: <name>`
  ([CONTRIBUTING.md](../CONTRIBUTING.md#commit-attribution));
  [`scripts/check-attribution.py`](../scripts/check-attribution.py) runs in CI
  (well-formedness) and in the landing script (`--require`).
- **Escape hatch.** Reverts always proceed. A card class whose host is down is
  recorded `down` and owed a rerun; promoting past it needs an explicit
  `--allow-down <class>` and at least one class that ran green.

### Commands

```sh
python3 scripts/promotion-lane/lane.py plan --class gfx1100
python3 scripts/promotion-lane/lane.py run --archive hipfire-<tag>-linux-rocm-gfx1100-x64.tar.gz \
    --class gfx1100 --candidate <sha> --out <dir> --models-dir <dir> --repo <checkout> \
    [--perf-baseline <beta's record.json>] [--baseline-archive <beta's archive>]
python3 scripts/promotion-lane/lane.py merge --candidate <sha> --out candidate.json <record.json>...
python3 scripts/promotion-lane/lane.py check-release --sha <sha> --records <dir>
# perf rows alone (A/A calibration, kill experiments), then the A/A half-widths
python3 scripts/promotion-lane/lane.py perf --archive <A> --baseline-archive <A> --class gfx1100 --out <dir> --models-dir <dir>
python3 scripts/promotion-lane/lane.py calibrate --class gfx1100 <dir1>/perf.json <dir2>/perf.json
```

`lane.py run` must execute inside a fabric job on a card of its class. The
landing flow (build, per-class runs over fabric, merge, fast-forward) is the
maintainers' `land-pick-lane.sh`, outside the tree.

## hw-gate (PR evidence automation)

[`hw-gate`](../.github/workflows/hw-gate.yml) collects load/serve/kernel
evidence and posts seat commentary on a PR. It does not approve PRs, does not
move `beta` (the [promotion lane](#promotion-lane) does) and does not own
`master`. When it runs, two automated reviewer seats (seat 1: diff review and
route selection; seat 2: investigation) may assist under their own automation
identities; a human still owns the approving review. Diffs that touch no
hardware-relevant surface need no hardware route.

### Bucket selection (`scripts/hw-gate/bucket_select.py`)

Changed paths → buckets `load` / `serve` / `kernel` (first match wins per path;
`serve` and `kernel` also imply `load`):

| Bucket | Path rules (summary) |
|---|---|
| **kernel** | `kernels/**`, `crates/rdna-compute/**`, `crates/hipfire-dispatch/**`, `crates/hip-bridge/**`, `crates/saddle-core/**` |
| **serve** | `crates/hipfire-engine/**`, `crates/hipfire-generate/**`, the HTTP gateway `crates/hipfire-cli/src/serve/**` and `crates/hipfire-client/**`, daemon `slots.rs` / `serve*.rs`, runtime emit/eos/dflash/spec/reset surfaces, arch `serve`/`generate`/`spec` sources |
| **load** | loader/daemon/runtime load+config surfaces, arch `load*`/`weights*`/`carrier.rs`, `crates/hipfire-config/**`, `crates/hipfire-registry/**`, `registry/**`, workspace/`crates/*/Cargo.toml` and lockfile |
| **none** | everything else (docs, benchmarks, most scripts, tests, markdown, …) |

Touching **policy** paths (`.github/workflows/**`, `.github/CODEOWNERS`,
`scripts/hw-gate/**`, leanup/ratchet scripts, `registry/**`) always needs a
human decision; automation must not merge those. Exec-sensitive paths (build
scripts, manifests, toolchain, CI, shell/python) are useful context for
execution-risk judgment; they are not a gate by themselves.

### The author's request (`<!-- hw-gate-request -->`)

A PR body may carry a fenced JSON block after the marker:
`{"routes":[{"mode":"battery"|"chain","tag":"registry:tag"}],"claim":"..."}`.
Author and seat-selected routes are limited to `battery` and `chain`
(`scripts/hw-gate/review.py` drops other modes); the `-dflash` variants below
come only from the bucket manifest. When hw-gate runs, seat 1 treats the claim
as a claim, runs the requested routes when the tag exists on the runner,
reports unknown or absent tags as unavailable (not failed), and states in its
verdict whether the claim was proven, disproven, or not exercised. The PR
template ships the skeleton. Authors may instead attach equivalent manual
harness output; either path is evidence for direct review, not an acceptance
pass by itself.

### Seat 1 — diff review and route selection (when automation runs)

Seat 1 (read-only tools in a PR checkout) reads the
diff before anything runs and decides `run_hardware`: the hardware job builds
the PR and runs its daemon as the maintainer's user on their workstation, so
seat 1 refuses diffs that reach outside the process (network, filesystem beyond
model/cache/temp, env or credential reads, process spawning, unexplained build
or dependency changes, obfuscated blobs, unexplained `unsafe`). A maintainer's
**`hw-run`** label only ever forces a run and is removed after each run and on
every push. Seat 1 also composes the route list: the mandatory fixtures for
the touched buckets, the author's requested routes, and its own additions.

After hardware, seat 1 reads every decoded turn and returns `greenlight` /
`needs-human` / `block` with regressions cited by `file:line` and fixture. Seat
1 never merges and never approves; its review is a comment that may inform the
human reviewer.

### Fixtures and harnesses

Every fixture runs through [`scripts/serve_harness.py`](../scripts/serve_harness.py)
— never a single `hipfire run`, which is one request against a fresh daemon
and proves nothing about turn-to-turn state — with reasoning off. Mandatory
fixtures are registry tags pinned by sha256 in
[`scripts/hw-gate/fixtures.json`](../scripts/hw-gate/fixtures.json); a
missing or mismatched pinned fixture **fails that automation run**. Requested
extra tags resolve through `registry/v1.json`; one absent from the runner is
reported as unavailable.

`serve_harness.py --mode` accepts `battery`, `chain`, `session`, `images`
(tiny diffusion-pipeline fixture, not vision input) and `prefix-fanout`; hw-gate
drives only `battery` and `chain`, plus `--dflash on` variants of both
(`battery-dflash`, `chain-dflash`, from `HARNESS_MODES` in
`scripts/hw-gate/run.py`). A `-dflash` mode needs a fixture-declared draft whose
sha256 matches the manifest. A lane without that draft records the mode as
`skip` and the fixture **fails** (skip is not pass); it never falls back to
plain decode.

- **`load` bucket** — `battery` and `battery-dflash` on every fixture (varied
  prompts, expect substrings, attractor / runaway / empty detection).
- **`serve` bucket** — adds `chain` and `chain-dflash` (related turns through
  the prefix cache: reset, cache, and terminal semantics).
- **`kernel` bucket** — `battery` plus
  [`scripts/redline_daemon_harness.py`](../scripts/redline_daemon_harness.py)
  capture + HIP/PM4 parity on the dense trunk (`--pm4` with the manifest's
  capture/measure/decode iteration counts).

Decoded assistant text is posted **verbatim** in the evidence comment. Reading
it is part of review. The per-turn prefill/decode rates in that table are
harness-side timings (HTTP streaming, sampling) and run well under
`hipfire bench`; they are context, never a performance claim. Perf claims go
through [`docs/methodology/perf-benchmarking.md`](methodology/perf-benchmarking.md).

The same harnesses are the manual evidence path when automation is not used:
run them locally, attach the artifacts, and let the maintainer review.

### KV backend / K–V axis changes (load + serve)

When a change touches KV backend selection, Qwen K/V axes, or the Qwen
implicit default, authors filing PRs use the disclosure checkbox in the
PR template — do **not** copy that checklist here:

- [`.github/PULL_REQUEST_TEMPLATE.md`](../.github/PULL_REQUEST_TEMPLATE.md)
  test-plan item: *I inspected loaded/bench/harness KV backend fields
  (`kv_backend`, `kv_backend_legacy`, `kv_backend_reason`); if any run
  used legacy, I disclose the `HIPFIRE_KV_BACKEND=legacy` token…*

Manual load/serve evidence for those changes (in addition to the
`battery` / `chain` fixtures above):

1. **Qwen path and tag.** Load the same Qwen artifact once by filesystem
   path and once by registry tag. Confirm both routes report the same
   effective backend and the same effective K/V axes (loaded ACK /
   harness `--out` / bench `--json` fields and stderr markers agree).
2. **Default axes.** On exact **gfx1201**, unpinned eligible Qwen must
   keep native **fp8/fp8** (`KV cache: Fp8 vmm (` or equivalent admission
   marker). On a **non-gfx1201** arch, unpinned Qwen HFQ must
   default **q8/q8** (not the former FWHT3/Q8).
3. **Card-B baseline guard.** Run
   [`scripts/guard_gfx1201_baseline.py`](../scripts/guard_gfx1201_baseline.py)
   on the pinned card-B fixture. The guard sets `HIPFIRE_GRAPH=1`; treat
   a green run as graph-capture **and** replay coverage for that
   baseline, not a portable perf claim. Attach its `fixture.json` /
   `summary.json` when the change can affect gfx1201 FP8 VMM selection
   or allocation.

Numerical/state parity for KV still needs a path-specific oracle (claim
map row below); this procedure only covers user-facing load/serve
semantics and the disclosure path for agents filing PRs.

### Seat 2 — investigation (when automation runs)

Seat 2 reads the diff, the evidence, and seat 1's verdict — and then, when
that evidence does not prove the change, may gather more evidence itself. It
runs with a real shell in a sandboxed checkout of the PR head on the hardware
host: every GPU on the host is reserved for the session (a host-level lock
serializes seat-2 sessions and excludes the lane runs), the PR and the base
branch are both built for A/B, every registry artifact on the host is
available read-only, and everything seat 2 writes to `$HW_GATE_EVIDENCE` is
uploaded with its decision. There is no fixed route vocabulary: seat 2 chooses
what proves the change — a multi-GPU load for new topology code, a
refused-load-then-generate sequence for a moved refusal, Redline parity for a
kernel, a base-vs-PR A/B for a "no behavior change" refactor. The sandbox
bounds reach, not judgment: no network, no writes outside its
home/evidence/build tree, no other GPUs, no credentials, no `gh` (the script
posts), a wall-clock budget (`HW_GATE_MAX_MINUTES`, default 45), and only
registry artifacts.

Seat 2 returns a decision comment with an `investigation` table
(question → route run → evidence file → result), an `unproven` list for what
this host could not exercise, and an announcement written for the author. It
may disagree with seat 1 and must say why. **Seat 2 does not replace the
required approving review and does not promote `master`.** Seat commentary is
input to the maintainer, not an acceptance pass or auto-merge contract.

### The floor (`scripts/hw-gate/review.py`)

When automation runs, the workflow's own rule is split in two:

| Tier | Fires on | Who can override |
|---|---|---|
| **hard** | a failed fixture or harness, an attractor, a policy-file change, a `RATCHET-RAISE:` commit without the `ratchet-raise` label | nobody within the automation — evidence stays failed / policy stays held |
| **soft** | coverage gaps, confidence < 0.8, seat 1's `needs-human`, an unparseable verdict | seat 2, with a stated reason |

Hard-floor failures are strong signals for the human reviewer; they are still
not a separate required GitHub check beyond the [merge bar](#merge-bar-required).

## Automatic checks vs manual evidence

| Class | When it runs | Authority |
|---|---|---|
| **Automatic (no GPU CI, required)** | PR / push via [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) | **Required** merge bar: `build (workspace, no GPU)`, `unit tests (lib, no GPU)`, `gates (ratchets, layering, registers)`. **Not** model coherence, serve semantics, or perf admission. |
| **Automatic (Windows MSVC, advisory)** | PR / push via [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) on `windows-latest` | Native build, selected CPU library tests, no-HIP version/JSON smoke, unsigned ZIP packaging. Initially advisory; promote after the first green `beta` run. **Not** GPU/runtime certification. |
| **Promotion lane (required for `beta`)** | Every `beta-candidate` tip `land-pick-lane.sh` promotes; inside fabric jobs on each blocking card class | **Required**: `beta` fast-forwards only on a green promotion record. See [§ Promotion lane](#promotion-lane). |
| **Automatic (hw-gate, PR evidence)** | PR via [`.github/workflows/hw-gate.yml`](../.github/workflows/hw-gate.yml) when that automation runs; hardware run after `hw-run` when buckets need it | Evidence for the reviewer. Not a required status check and not promotion. |
| **Automatic (path-gated hooks)** | Local `pre-commit` on matching staged runtime paths | Runs the hotspot guards selected by the staged path set. Documentation-only staged sets do not trigger a separate docs hook. **Not** a full product matrix. |
| **Manual local no-GPU equivalent** | Human/agent invokes `scripts/no-gpu-ci.sh` outside CI | Same checks as the workflow script body; still **manual invocation**, not automatic CI. |
| **Manual (GPU / model)** | Human or agent on hardware with an explicit model path | Required for claim classes that name GPU/model routes (parity oracles, serve semantics, perf protocol, Redline promotion ladder, admissions) when those surfaces change — whether or not hw-gate ran. |
| **Direct maintainer review** | Human approval on the PR | **Required** (one approving review). Judges claim-matched evidence; automation comments are inputs, not substitutes. |

No-GPU CI green never substitutes for GPU/model evidence the claim map
requires. A green promotion record or hw-gate run does not create an
admission, skip claim-specific oracles named below, or replace the approving
review.

### Automatic entrypoints

| Route | Path | Role |
|---|---|---|
| No-GPU CI (required jobs) | [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) (+ [`scripts/no-gpu-ci.sh`](../scripts/no-gpu-ci.sh) body) | **Required** automatic CI: workspace build, lib unit tests, ratchets/layering/registers gates. |
| Windows MSVC (required) | [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) + [`scripts/package-windows.ps1`](../scripts/package-windows.ps1) | Native Windows CPU build/tests/smoke and unsigned developer artifact; no GPU certification. |
| **Promotion lane** | [`scripts/promotion-lane/`](../scripts/promotion-lane/) + [`scripts/device_mesh_matrix.py`](../scripts/device_mesh_matrix.py) | **Required** for `beta`: required rows on the installed archive, no compiler, one promotion record per candidate. See [§ Promotion lane](#promotion-lane). |
| **hw-gate (PR evidence)** | [`.github/workflows/hw-gate.yml`](../.github/workflows/hw-gate.yml) + [`scripts/hw-gate/`](../scripts/hw-gate/) | Evidence automation: path → buckets, pinned fixtures, hardware run, seat commentary. See [§ hw-gate](#hw-gate-pr-evidence-automation). |
| Pre-commit hooks | [`.githooks/pre-commit`](../.githooks/pre-commit) | **Automatic** when hooks are installed (`scripts/install-hooks.sh`). Selects HOTSPOT / SERVE_HOTSPOT / PP_HOTSPOT runtime guards from staged paths; documentation-only staged sets exit without a separate docs gate. |
| Dispatch `bind_thread` invariant | [`scripts/verify-bind-thread.sh`](../scripts/verify-bind-thread.sh) (via pre-commit on matching paths) | **Automatic** when hooked: every public `dispatch.rs` entry must bind the HIP thread. Not a kernel numeric test. |
| Env/docs drift check | [`scripts/check-env-docs.py`](../scripts/check-env-docs.py) | **Automatic** through `scripts/no-gpu-ci.sh`; checks that referenced `HIPFIRE_*` names are documented and production reads are config-owned. |

### GPU lib tests are not concurrency-safe (run them with `--test-threads=1`)

`cargo test -p rdna-compute --lib` flake on **4 tests** (`gdn_q8_verify_matches_
single_row`, `gdn_q8_state_tracks_f32_on_every_route`, `gdn_q8_state_tracks_f32_
on_every_route`'s siblings in the same module, and
`batched_select_ranking_matches_the_serial_selection_sort`) is a **harness
property, not a regression**, and the mechanism is worth knowing before
chasing it:

`gdn_q8_verify_matches_single_row` captures a hipGraph on the device's active
(default) stream — `gpu.graphs.begin_graph_capture(&gpu.hip, gpu.device_id,
&stream)` after `gpu.active_stream.take()`. Capture state is per-stream and
global to it, so while that capture is open, **another test's kernel launch on
the same stream is absorbed into the capture** and fails with
`hipModuleLaunchKernel(extra blob): operation failed due to a previous error
during capture`. Only one test in the module captures; the others fail as
collateral, and which ones fail varies run to run.

Evidence it is the harness: the same suite is **542 passed / 0 failed** with
`--test-threads=1`, and each failing test passes in isolation.

Two real fixes, neither taken yet — capture on a **dedicated stream** so
concurrent default-stream launches are not absorbed (the correct one, and it
also hardens the product path if two slots ever capture concurrently), or
serialize every GPU test in the module on a `static Mutex<()>`, the idiom
`hipfire-arch-qwen35`'s `EP_TEST_LOCK` already uses. Until one lands, run GPU
lib suites serial:

```bash
cargo test -p rdna-compute --lib -- --test-threads=1
```

### Manual local equivalents (not automatic)

| Route | Path | Role |
|---|---|---|
| No-GPU script | [`scripts/no-gpu-ci.sh`](../scripts/no-gpu-ci.sh) | Manual local run of the same body CI uses: `cargo check --workspace --examples`; selected no-GPU Rust and native control-plane tests; CPU pytest; env/docs coverage. |
| Env/docs drift check | [`scripts/check-env-docs.py`](../scripts/check-env-docs.py) | Fast standalone env-name coverage and ownership check. No GPU or model download. |

### Documentation checks

The shipped automated documentation-specific checks are
[`scripts/check-env-docs.py`](../scripts/check-env-docs.py), which validates
environment-variable reference coverage and config ownership, and
[`scripts/check-pins.py`](../scripts/check-pins.py), which validates
[`fixtures/pins.toml`](../fixtures/pins.toml) (schema, provenance of current
KLD pins, prompt md5s against the committed files, mirrored digests) and that
every `pin:<kind>/<id>` cited in the root rule files, `docs/`, `.agents/` and
`scripts/` exists. Neither is a general Markdown link, lifecycle-label, or
semantic-drift checker. Beta does
not currently ship a standalone `check-docs-reliability.py` command or a
documentation-only pre-commit gate. For documentation-only changes, run the
env/docs and pins checks plus `git diff --check`, and inspect changed links and commands
against the current tree.

### Maintained manual harness roles

Narrow roles. Do not widen a harness into a universal gate.

| Harness | Path | Role | Not this harness |
|---|---|---|---|
| **gates.sh** | [`scripts/gates.sh`](../scripts/gates.sh) | Maintained **manual** wrapper: optional Redline capture, generic serve battery, optional fresh-process perf compare (`probe_commits.sh`). Requires `--model`. | Not CI-default. Not universal. Does not call retired coherence-gate scripts. |
| **serve_harness.py** | [`scripts/serve_harness.py`](../scripts/serve_harness.py) | **Model-agnostic** user-facing serve behavior (`--mode` `battery` / `chain` / `session` for text turns; `prefix-fanout` for prefix-cache concurrency; `images` for the tiny diffusion-pipeline fixture): finish reasons, runaway/empty, prefix cache, prefill/decode timing, recall hooks. | Not LFM thinking-frame specifics. Not Redline route proof. Not vision-input (VL) coverage. |
| **serve_harness.py (LFM tag)** | [`scripts/serve_harness.py`](../scripts/serve_harness.py) | LFM2.5 serve smoke with the exact registry tag; use registry sampling or `recipe:nothink` for non-thinking framing. | Not a substitute for numerical parity oracles. |
| **redline_daemon_harness.py** | [`scripts/redline_daemon_harness.py`](../scripts/redline_daemon_harness.py) | Resident-daemon **Redline** capture, phase fingerprint, shadow/parity, and timing evidence under manual-capture env. | Discovery/correctness evidence ≠ product timed-arm route proof by itself. Does not enable AQL routing. |
| **dispatch_profile** | [`tools/redline/dispatch_profile.py`](../tools/redline/dispatch_profile.py) | Manual **attribution-only** steady-state retained-PM4 per-dispatch span diagnostic (instrumented GFX12 tape). | **Not** route proof, certification/admission, or absolute/pure kernel timing: timestamp commands add observer overhead and spans include preceding PM4 boundary packets. |
| **tools.redline golden** | [`tools/redline/golden.py`](../tools/redline/golden.py) | Reproduce the exact checked-in MQ4R TG128 model/benchmark/route fixtures through the route-proof-capable product harness. | Not a universal GPU gate, a new-route certification shortcut, or an admission. |

### Supporting manual tools (existing; claim-scoped)

Use only when the claim class below names them. They are not universal.

| Tool | Path | Typical claim |
|---|---|---|
| Kernel channel tests | `target/release/examples/test_kernels` (build: `cargo build --release --features deltanet --example test_kernels -p hipfire-runtime`) | Kernel numeric vs CPU reference on the detected arch. **Not** dispatch bind coverage. |
| Dispatch bind check (manual) | [`scripts/verify-bind-thread.sh`](../scripts/verify-bind-thread.sh) | Same bind invariant as the hook; run explicitly if hooks are not installed. |
| Speed regression floor | [`scripts/speed-gate.sh`](../scripts/speed-gate.sh) | Prefill/decode vs committed `tests/speed-baselines/<arch>.txt` when that path’s policy applies. |
| Fresh-process commit probe | [`scripts/probe_commits.sh`](../scripts/probe_commits.sh) | Optional A/B invoked from `gates.sh --perf`. |
| Perf protocol | [`docs/methodology/perf-benchmarking.md`](methodology/perf-benchmarking.md) | How to measure; not an executable gate. |
| Redline certification ladder | [`docs/REDLINE.md`](REDLINE.md) | What evidence is required before Redline-attributed promotion. |
| Path-specific parity / state oracle | Arch-owned example or test named by the change (when one exists) | Hidden-state, logit, KV/conv, or graph parity. If none exists for the surface → **blocked**. |
| RADV/ACO MQ4 pilot | [`pm_aco_oracle`](../crates/rdna-compute/examples/pm_aco_oracle.rs), with the pinned offline producer under `tools/aco-export/` | Frozen artifact/config/entry-ABI identity; original Vulkan vs PM/HIP vs hipcc whole-buffer and guard equality, negative controls, eager and hipGraph replay. Native-writer relift and raw typed round-trip checks precede GPU execution. Separate matched-data timing is diagnostic only and requires a passing parity report; no product-route admission is implied. |

### S1 slow-link layer-major prefill (claim-scoped)

Scope: Flash-Next MQ4 packed on the measured gfx1100 evidence host behind a measured narrow
(<16 GB/s) discrete host link, `HIPFIRE_QWEN4_SLOWLINK_PREFILL`
(`developer.qwen4_slowlink_prefill`): AUTO is the default selector (unset selects the candidate automatically from 1,537 rows: the first genuinely multi-chunk request, one row past the 1,536-row tile bound), `1` forces it as a shipping-path candidate (a capacity shortfall fails the load with its exact byte deficit instead of falling back), `0` is the kill switch. The selector is the sealed capability check, not a reintroduced exact-arch gate: an unqualified load is declined before any reserve or mutation. Nothing here is a claim
about any other arch, codec or link: fast links, unified memory and an
unmeasured link never select the route under either selector, and a fast-link GPU result is never
inferred from the XTX. H0's one-layer synthetic transfer evidence (copy-bound
at ~99.9% of 7.09 GB/s) is a ceiling, and its 1,400+ tok/s pp16384 figure is a
historical projection; the measured base on the XTX at pp16384 is 1,738.9-1,750.2 tok/s (accepted release, ~1.87x beta), and only the fresh full-path receipts below support a delivered
number. The 1,400 tok/s pp16384 figure was the original historical S1 gate
floor and is not release acceptance; 2,000 tok/s is not a release block, and no future MQ2 performance goal is promised here.
Route-relative caps are long-context prefill teacher-forced ROUTE-vs-ROUTE caps, not a same-route decode claim: mean KL(beta4096||candidate) 16K 0.03206548313867323 against the benign-beta upper 0.03948458079446915, and 4K 0.024549946338726408 against 0.027284316982763658. Same-route repeats are the determinism criterion only (7/7 native bit-identical plus coherent decoded text; not beta-id equivalence, not a full-corpus teacher claim). The required followup stack (ordered placement + IU4 + QSA, thin 2054) is not promotion and not a base gate: QSA currently fails 4K (0.029107700567698017, excess +0.001823383584934359) and no waiver exists. No f32 GDN carry ships in base: the f32 carry is DROPPED entirely and is never resurrected; the inert Q8 WMMA trunk lever is removed. Future D2 pair and stack planning does not change the
current S1 hold.

Harness (CPU-testable parts run in `no-gpu` CI; GPU parts only inside a hipx
fabric hold):

| Piece | What it proves |
|---|---|
| [`qwen4_slowlink_gate`](../crates/hipfire-arch-qwen4/examples/qwen4_slowlink_gate.rs) `--model ABS --fixture ABS.json --output ABS.json --mode primitives\|state\|product\|perf --route incumbent\|layer-major --manifest ABS.json` | `primitives`: production panel transport/dispatch/ordered combine on real host-expert sources (payload and table identity, repeated epochs at the same addresses, delayed readers, within-expert panels with canaries, exact canonical combine). `state`: per-prompt route receipts (the purpose profile's tile: 768 rows product, 512 rows primitive self-test; never the 8,192-row incumbent chunk and never a retained 8,192-row/full-prompt scratch; resident plus host layers close on the 48-layer trunk with a per-host-layer byte bound; ring+lookahead reserve charged, one submitted payload sweep per window, exactly one window at 16,384 rows), same-route repeat digests of every fixed state and the final logits, launch-cut and resume chronology, session restore and divergent continuation at the fixture's boundaries, the four named failure injections (restore or contain, then a clean re-run), bounded head-row scoring (never N x vocab), and eager vs retained-body replay decode after the prefill. `product`: greedy ids plus decoded text per prompt (a human reads the text; the attractor heuristic is not a quality gate). `perf`: untimed warmup, pp512/pp8192/pp16384 timing and decode, with spans, the barrier cost and the cross-layer overlap evidence. `--route` is a force inside the one production selector and safety predicate, not another implementation. A required check that did not execute makes the run `incomplete`; only `pass` exits 0. The manifest pins source base/tip, binary SHA-256, model identity, PLE source/format/sidecars, kernel code-object hashes, host/device/link and the memory ledger; the binary hash and model size are re-observed and a mismatch fails before GPU work. |
| [`qwen4_greedy_ids`](../crates/hipfire-arch-qwen4/examples/qwen4_greedy_ids.rs) `--receipt` | **Historical/diagnostic, NOT S1 acceptance.** Exact full-`token_ids`/finish_reason/prompt-count equality between the beta autoregressive arm and the candidate over byte-identical id fixtures (512, 1536, 1537, 8192, 16384 rows plus a `resume_at` irregular resumed prefix). The ids file is unchanged by the receipt flag; the receipts name the route each record took. |
| [`qwen4_kld`](../crates/hipfire-arch-qwen4/examples/qwen4_kld.rs) `s11ref --decode` then `pair` | **Historical/diagnostic, NOT S1 acceptance.** The full 24-sequence, 6,120-row code24 decode corpus (a teacher-paired tool row: shared-expert quantization error against the BF16 source teacher) against the existing fp8 `code24` reference (`code24.fp8.s11ref`, md5 `56d93a4931631e306244e95ea64aecfc`; `code24.corpus.json`, md5 `6145e25c0858e31e08527c1e5278c074`; never regenerated). `pair` aggregates KL per corpus sequence, then the paired candidate-minus-beta 95% t interval; pass when `lo <= 0`, else `hi <= +0.010`. A partition, a partial corpus or a different reference/corpus/config is refused. The decode scorer prefills the unscored prefix and records whether each prefix took S1 (`slowlink_prefix_routes`); it does not by itself prove 16K prompt scoring, and the cold-prefill all-row scorer (N x vocab logits) never exercises S1. |
| [`qwen4_parity --mode slowlink-state`](../crates/hipfire-arch-qwen4/examples/qwen4_parity.rs) | Raw-byte SHA-256 digests of GDN/QSA/PLE state and a final-logit summary over repeated whole-prompt prefills; same-route determinism, one route per setting. |
| [`slowlink_prefill_hw`](../crates/hipfire-arch-qwen4/tests/slowlink_prefill_hw.rs), the S1 test in [`session_cache_hw`](../crates/hipfire-arch-qwen4/tests/session_cache_hw.rs) | Focused hardware tests (primitives on the admitted profile, repeated epochs, injected failures, decode-after-prefill replay, session restore through the layer-major route). The un-ignored receipt/hold-guard tests are GPU-free. |

Because S1's mixer tile (768 rows for the product profile, 512 for the primitive
profile; never the 8,192-row incumbent chunk) differs from the incumbent's 8,192-row
manifest, a layer-major result is never labelled bitwise-identical to the
incumbent: the historical cross-route gate sentence below is historical/diagnostic, NOT S1 acceptance (exact greedy ids and a paired KLD verdict were that gate's tools). The current S1 rule is the route-relative caps plus same-route determinism plus coherent text: long-context prefill teacher-forced ROUTE-vs-ROUTE mean KL(beta4096||candidate) within the benign-beta uppers, 7/7 same-route determinism, coherent decoded text. Identical digests are required only between repeats of one route.
Selection truth: AUTO is the default selector (unset selects the candidate automatically from 1,537 rows; automatic and forced share that floor); `0` is the kill switch and forces the incumbent without probing the link; `1` forces the shipping-path candidate and a capacity shortfall fails the load with its exact byte deficit instead of falling back. Fast links (>=16 GB/s), unified memory and unmeasured links refuse under both selectors. S1 changes prefill routing only: decode and <=4-row verify keep their truth (the per-load measured D2 default or an explicit `memory.offload_exec`, unchanged by S1), and the CPU expert splice path is declined before any S1 reserve. Request and load shapes the selector refuses before any mutation (short context, no host layers, deficits, geometry) are a named incumbent bypass, never a silent fallback. The caps above are route-relative same-route caps, never teacher-paired; when the selector declines, the incumbent runs byte for byte as before with zero S1 charge.
A receipt without a cross-layer event intersection (copy of layer L+1
overlapping compute of layer L) cannot support an overlap claim. Decode after
the S1 prefill is proved on Qwen4's retained-body replay (the only decode body
replay Qwen4 production has); an XTX `qwen4_g2_replay_gate` SKIP is not that
proof.

CPU commands (sequential, `ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1`):
`cargo test -p hipfire-arch-qwen4 --features lab,reference-parity --example qwen4_slowlink_gate -- slowlink_receipt --test-threads=1`,
`cargo test -p hipfire-arch-qwen4 --features lab --example qwen4_greedy_ids -- --test-threads=1`,
`cargo test -p hipfire-arch-qwen4 --features lab --example qwen4_kld -- --test-threads=1` and
`python3 scripts/check-env-docs.py`. At most three XTX holds are budgeted
(schedule correctness, matched full-path A/B with the baseline code24 row,
candidate code24 and regression closure); any missing cell is `incomplete`,
not a pass.

Speed-evidence protocol (base, no 2000 block): the accepted base 1,738.9-1,750.2 tok/s at pp16384 (~1.87x beta) is an observed historical exception only (two fresh processes, six samples per arm); it claims no final base CI/serve PASS. A qualified new-stack claim needs ABBAAB with 3 fresh processes per arm after quality passes, flat all-nine candidate medians >= 2000 tok/s. Serve-kit disclosure: thinking-off battery requests derive the off sentinel cap 1 (None), never an explicit 0 that the daemon reads as uncapped on effort-carrying requests; corrected from `max_think_tokens=0` to `None`.

Historical development receipts (NOT current acceptance; no final base CI/serve PASS is claimed). The last correctness hold
(j1759) is **incomplete**: its primitive 80 s cap expired during the 82 s model
load, so only `identity.manifest` passed and every other check was skipped.
That hold supports no correctness, determinism, quality, memory or speed claim.
Its cap covered the 80-110 s model load as well as the primitives; that was a
harness budgeting error, not a finding about the route. The replacement hold is
one correctness hold over all cases at the same 11 resident layers, route
with a 1,500 s maximum; no case, quality criterion or flag is
narrowed or relaxed. [Historical plan text; the selector state above is current.]

[Everything from "(j1759) is **incomplete**" through the `j1855` probe paragraph is one historical development-receipt block, NOT current acceptance.]
The replacement hold (j1761) **failed**: identity and load passed (11 resident
layers; 76 host-mapped entries as the load receipt records them), then the
first payload primitive hit a real `hipMalloc` out-of-memory after about 110 s.
The aggregate primitive report then carried that one initialization error on all
seven cases (they are not seven arithmetic mismatches), every later check was
skipped, and no product
state cell ran, so the product headroom criterion below was not exercised. The
primitive cell's external minimum free VRAM (21,327,872 B) and the in-process
`hipMemGetInfo` end figure (75,497,472 B) are informational; the residual is
non-gating. This is a finding about the primitive self-test's allocation, not a
correctness, quality or speed result. The loader therefore takes an explicit
admission purpose: the primitive self-test (`PrimitiveSelftest`) and the product
route (`Product`) are admitted separately, with the product load keeping the
512 MiB headroom rule. The aggregate report is also corrected: each primitive
runs and reports independently, a case that cannot run is reported as
`status:"error"` with `mismatches:null` and its own `detail.error` (never a
fabricated mismatch count), the batch continues past it, and the gate fails
only that case with its own reason. A global start refusal is one
`primitive.init` failure that leaves the seven cases unproved (missing), not
seven cloned failures. Another hold is approved only after the admission proof
passes on the CPU. [Historical hold-gating text.]

Memory acceptance for that hold (product state cells): the minimum observed
free VRAM over the whole run MUST be at least 512 MiB (536,870,912 B), taken
from an independent 20 ms sysfs used/total sampler together with the existing
in-process `hipMemGetInfo` `pre_load`, `post_load` and `end` points. Primitive
identity, 7-case, load and teardown checks must pass; the memory census, the
residual and the peak are informational only and are not driver attribution.
The census is source-owned and records the actual admitted placement separately
from its declared reference projection; neither is an actual driver figure.
GPU-pool parked bytes are counted as used. This is
a headroom product question and does not select a driver explanation. A
missing cell is `incomplete`, not a pass.

Purpose profiles (Main's choice from the joint admission frontier). The load
admits one tile per purpose and a request never falls to the other's: the
**product** profile (every state, headroom, greedy and perf cell, and any later
product claim) is a **768-row** tile with the largest resident-layer count that
fits the actual post-initialization memory quote. Ten layers is preferred, not
a floor; zero is legal when the base budget fits and a host layer remains. The
**primitive_selftest** profile (`--mode primitives`) is a **512-row** tile that
funds the self-test diagnostic inside the reserve. The primitive cell therefore
proves kernel correctness at 512 rows only; it is not a product-shape
qualification, and its report carries `evidence.s1_profile` saying so
(`expected_purpose`/`expected_tile_rows` from the binary at startup;
`admitted_purpose`/`admitted_tile_rows` are `null` until a load succeeds and are
then read back from the loaded bundle's own profile, so a failed or declined load
never reports an expected value as admitted). One H1
manifest holds both sealed profiles as full source ledgers,
`memory_ledger.profiles.product` and `memory_ledger.profiles.primitive_selftest`
(each: `purpose`, `tile_rows`, `resident_expert_layers`, `window_rows`,
`page_count`, `device_weights_bytes`, `s1_tile_census_bytes`/`_terms`,
`s1_extras_bytes`/`_terms`, `device_core_bytes`, `sequential_diag_bytes`,
`device_owned_bytes`, `runtime_budget_bytes`, `safety_margin_bytes`), with no
flat fallback. The root keeps the shared card/fabric/host figures and
`device_owned_bytes` as the MAX of the two per-purpose owned figures (they load
sequentially; the peak is not product core plus the primitive diagnostic). The
state, product and perf cells select `product` (tile 768, no diagnostic bytes);
the primitive cell selects `primitive_selftest` (tile 512, with diagnostic
bytes), and its census diagnostic reads that profile alone. `identity.manifest`
fails before GPU work if a profile is incomplete, its arithmetic does not close
(core = weights + tile census + extras; owned = core + diagnostic), the tiles are
equal or above 1,536, the root owned bytes are not the MAX, or either tile
differs from the tile this binary's loader admits for that purpose. Every
layer-major receipt of a cell must name exactly its own profile's tile (and
retain no more scratch rows than it), and `load.placement` fails unless the
bundle's actually admitted purpose, tile, window rows and resident layers equal
the selected sealed profile. A product receipt at 1,536, 1,024 or 512 rows is
therefore a failure even though it is under the historical bound. A manifest
without both profiles describes an older binary and is refused.
Each profile's `s1_tile_census_terms` and `s1_extras_terms` are arrays of
`{name, bytes}` with unique names whose checked integer sums equal
`s1_tile_census_bytes` and `s1_extras_bytes`; the cold BF16-to-F16 shadow bound is
exactly one `f16_weight_shadow_bytes` term in the tile census terms and none in
the extras, and a stated `f16_shadow_cold_bound_bytes` must equal that one term,
so it is never charged twice. The hardware tests (`slowlink_prefill_hw`) load the
`Product` purpose for state, determinism, injection and replay and judge receipts
at the bundle's admitted product tile; only the primitives test loads
`PrimitiveSelftest`. The generic `Limits::default()` remains for GPU-free
shape-agnostic units only.
The generic 1,536-row
limit (`Limits::default`) remains only for historical validators and as the
bound of the approved named incumbent bypass for requests of at most 1,536 rows,
which is unchanged. The `tile <=1,536 rows` in the gate table above is that
generic bound, not the expected tile. No cut, repeat or case was removed.

Hold 1 attempt 4 (`j1778`, historical) corrected the proof, not the gates. Its primitive cell passed, but the state cell failed S1
initialization on the route-region allocation and then ran the incumbent until
the 720 s state cap expired, losing every in-memory timing and memory/cache
observation; seconds cannot be apportioned from that run. The source fixes
recorded since, each proved on the CPU only: the placement census and the
runtime now read one shared capability predicate for the model-lifetime
BF16-to-F16 weight shadows (the census had charged them on exact gfx1201 only,
so gfx1100 charged 0 of the real artifact's 1,298,923,520 B; the existing
512-row threshold and cold-owner bound are unchanged); S1 admission evaluates
(tile, resident layers) jointly; that repair retained the then-current >=10
policy, a tile of at most 1,536 rows (the admitted purpose profile above), the full 16,384-row window
and the lookahead charge; every admitted
S1 owner (page ring, route region, HC arena, window scratch, boundary stages)
is bound once at the selected load and requests only consume it, an allocation
failure being a named load failure that reports whether cleanup parked or
quarantined the owners; and the packaged `tensor_ops` index now lists the
supported Q8 exports, each indexed symbol being checked against the compiled
object's exports. No new env or config knob, no cache reservation or eviction
path and no lowered floor, window or lookahead came with them; the packaged
index legitimately differs from the legacy runtime cache hash. No
performance claim follows from any of this.

R5 supersedes that resident-count policy after `j1811`: the primitive load
refused N9 before any kernel or state case ran. The outer `pre_load` free point
was 25,713,180,672 B, but admission after link/copy initialization observed
25,560,088,576 B. That measured 146 MiB interval loss is not a universal tax.
The model-free pinned-only `j1855` probe isolated it at the first nonblocking
HIP stream creation, before any copy or event; synchronous pinned copying
initialized the same state implicitly. Copy size and `HSA_ENABLE_SDMA=1` did
not avoid it in the tested HIP paths.
Projection, frontier and admission now consume one pure post-initialization
quote; its witness proves the selected count fits and N+1 does not. The
512 MiB safety, remaining runtime budget, window and lookahead are unchanged.
The manifest's resident count is reference data, never an expected value
presented as observed. Product-state headroom still gates at 512 MiB in both
counter domains; no primitive, identity, state, greedy or KLD gate is relaxed.


Durable gate progress. Every `qwen4_slowlink_gate` run appends bounded JSONL
records to `<output>.progress.jsonl` (no new flag), each written and
`sync_data`-flushed as it is observed, so a SIGTERM, timeout or crash keeps
what ran: `case_start`/`case_end` and `repeat_start`/`repeat_end` per state
prompt (a repeat carries its raw receipts, the compact per-call window list,
wall time, state and final-logit digests, the determinism comparison with the
first repeat and the session-cache occupancy and eviction counter), the
primitive cases one at a time (each raw report, independent), named
`stage_*`/`phase_*` records that keep apart the prompt repeats, the
marker-boundary restores, the strict 16K restores, the failure injections, the
bounded head scoring and the natural-prompt replay/eager decode, every check
row, the error text, and the existing `pre_load`/`post_load`/`end` point
samples (plus a `load_failed` or `run_failed` point when the GPU is still owned
after a failure; it is not the HIP free at the failing allocation, which the
gate does not observe). Each record has UTC and a monotonic offset that the
stderr progress echo repeats, for aligning the untimestamped stderr and the
sampler. The sidecar is evidence only and is never promoted. `--output` is
written once, atomically (temporary file, fsync, rename), and only a complete
run reaches `pass`: an aborted or panicking run writes `incomplete` with the
error and a summary of the sidecar in which the last completed repeat, the one
in flight and every unexecuted required check are named, never filled in. A
sidecar write error fails the required check `progress.durability`. The legacy
`perf` cell keeps its informational timing path unchanged and persists only its
envelope. The CPU filter `slowlink_receipt` covers the progress, torn-tail,
corruption, write-error and atomic-publish cases.

### Flash-Next product-default receipts (perf claims)

A Flash-Next (`qwen3.8:flash-next`) perf claim MUST state the startup
configuration it was measured under and whether that configuration equals the
default pull for the claimed arch (`gfx1151` or `gfx1201`). This applies to
every published claim, not only gate receipts: a performance number in the
README, `BENCHMARKS`, release notes, or any other public text MUST name its
config and say whether it equals the default pull (the product default).
Generate the
receipt block with `python3 scripts/perf_config_receipt.py --log <daemon.log>
--arch <gfx1151|gfx1201> --receipt <receipt.json> [--model <path>]`. It writes
`startup_config` (raw startup lines plus normalized PLE row format, expert
hot-set attachment, trunk overlay, NPU trunk state, route plans, model
sha/path when given) and `product_default_check` (`match`, `mismatch`, or
`unknown` with reasons) into the receipt, preserving every other field.
`unknown` means the log lacked the evidence; it is not a pass. The CPU tests
are `python3 -m unittest tests/test_perf_config_receipt.py`.

The fresh mq4 pull is the split layout: the base
`qwen3.8-flash-next-base.mq4-xts` carries no PLE rows, and `hipfire pull` adds
the shared sidecar chosen by `memory.ple_format=auto`, which is a **q4** sidecar
(rank order) on gfx1151 and a **q8** sidecar (row order) on gfx1201. A log that
shows those rows matches. Embedded q8 rows (`qwen4 ple rows: q8 from embedded
model file`) come from a grandfathered monolithic install, which is not a
fresh-pull layout, so the receipt reports a mismatch for them. So is q8 on
gfx1151 or q4 on gfx1201: that is an explicit override, not the default (a
historical rank-q4 result on gfx1201 MUST NOT be relabelled as the default).
The receipt also labels which layout produced the log (split base
`eafaff94…` or grandfathered monolith `8b15b6fe…`, from the hot-set identity,
the bound base sizes, or `--model`). Flash-Next gates on `mq4-xts` pin
`HIPFIRE_PLE_FORMAT=q4` explicitly, so on gfx1201 their receipts are expected to
show the override mismatch; quote the `qwen4 ple rows:` line in the receipt.
Product defaults by arch: trunk overlay applies on gfx1151 only; expert hot
set applies on gfx1201 only (absent hot set with whole-layer placement on
gfx1201 is a mismatch); NPU trunk is `auto` on gfx1151 and counts only with an
eligible calibration/request plan (plans routed `gpu` are not NPU-executed
evidence); `memory.ple_hot_store_bytes` defaults to 0. Absence of overlay/NPU
lines on gfx1201 is expected, not a mismatch. Do not combine different
binaries, K depths, prompt sets, or q4/q8 PLE arms into one band.

Pinned reference bands (source: `product-config-audit.md`, 2026-10-16; each
is a *historical* band, not a matched product-default baseline):

| Arch / config | Band | Status |
|---|---|---|
| R9700 (gfx1201), explicit rank-q4 PLE + expert hot set, greedy, K3, 256 tokens, 16384 seq, eleven prompts (`fn-floor-hotset-r9700/GATE.txt:13-19`, `runs/dec-b-tier-fix-1`) | 50.47–50.77 tok/s (beta 47.14–49.16; hot set disabled 41.29) | Not the product default (the split-layout default on gfx1201 is a q8 PLE sidecar + pinned hot set; q4 is an explicit override); no matched default-pull reference exists. |
| Halo (gfx1151), nine-prompt MTP, sampled (.7/top-p .8/top-k 20/presence 1.5, seeds 123, 256 tokens, 16384 seq), forced K4 (`pm-wave/mtpdepth/runs/hd-smp-p0/kf4`) | 60.23–60.57 tok/s (sampled beta K4 54.30–58.63 is a separate binary/depth band) | Rank-q4 + overlay, the gfx1151 default PLE format and overlay, but measured before the split layout and forced K4; not a matched split-layout default-pull receipt. |
| Halo (gfx1151), AR, eight-prompt (`records_parser_1k` absent), hipx `pm-wave/fnds/runs/sar/base/r{1,2,3}/decode.json`, all rank-q4 + overlay | 34.33–34.69 tok/s | Not a matched nine-prompt AR baseline; no matched nine-prompt AR receipt exists. Do not compare against the nine-prompt MTP set. |

## Claim → route map

| Claim / change class | Minimum route(s) | Evidence kind |
|---|---|---|
| Docs, env-var tables, no-GPU Rust/Python/CLI only | Automatic: `.github/workflows/no-gpu-ci.yml` invokes `scripts/no-gpu-ci.sh`; for a fast docs-only local check run `python3 scripts/check-env-docs.py` plus `git diff --check` | Automatic CI and/or manual local equivalent — **not** GPU/model evidence |
| New/changed `.hip` kernel (numeric) | `test_kernels`; then model-level manual route for the arch under test | Manual + channel |
| gfx1151 Qwen4 F16 WMMA grouped MoE X-LDS entries (always on for the F16 WMMA route) | CPU: `cargo test -p rdna-compute --lib qwen4_moe_x_lds` (the route predicate). GPU: [`qwen4_moe_f16_lds_oracle`](../crates/rdna-compute/examples/qwen4_moe_f16_lds_oracle.rs) (`cargo build --release -p rdna-compute --features lab --example qwen4_moe_f16_lds_oracle`; `--mode check`; the production launchers take the LDS entries); then `serve_harness.py --sampling greedy` on long prompts against a build from before the X-LDS entries with `--compare-transcript`. Exercise the F16 route (asymmetric routed headers, or `HIPFIRE_QWEN4_MOE_SYM_IU4=0`); the default symmetric IU4 route is not a substitute. | Manual kernel oracle: all bytes, poison/guards, asymmetric experts, routing/run-length coverage, negative controls; manual serve for semantics; speed claims need matched fresh-process prefill-matrix runs. `--mode time` is synthetic kernel timing only. |
| Dispatch `bind_thread` / public `dispatch.rs` bind surface | `.githooks/pre-commit` → `scripts/verify-bind-thread.sh` (or run that script manually) | Automatic hook or manual bind check — **not** `test_kernels` |
| Forward / fusion / KV **numerical or state parity** | Path-specific parity/state oracle for that arch/surface; **blocked** if no oracle exists | Manual oracle — **not** `serve_harness.py` |
| Speculative **whole-prompt exact retry** (#1198: a cold exact retry reuses its whole prompt, zero prefill rows, via `Speculator::prompt_end_checkpoint`; Qwen3.5-family MTP and DFlash routes) | CPU: `cargo test -p hipfire-generate --test generation_route_matrix_tests prompt_end` (descriptor is the only whole-prompt authorisation; wrong length/fingerprint/conversation/cache policy keep the strict-`<len` plan), `cargo test -p hipfire-generate --test generation_route_matrix_tests local_candidate_never_covers_the_whole_prompt`, `cargo test -p hipfire-generate --test qwen_cancel_keep_committed_tests`, `cargo test -p hipfire-runtime --lib mtp_adapter_exposes_drafter_checkpoint_positions`. GPU, MTP (dense Qwen3.5/3.6, fixtures `HIPFIRE_MTP_PROMPT_END_MODEL`, optional `HIPFIRE_MTP_PROMPT_END_HEAD`, `HIPFIRE_MTP_PROMPT_END_TOKENS`): `cargo test --release -p hipfire-arch-qwen35 --test mtp_prompt_end_state_equality -- --ignored --nocapture` — fresh prefill → decode → exact retry, byte-equal last-row logits, DeltaNet bytes + stream meta, target and head KV prefixes, `prev_hidden` (+ position), first token and continuation; greedy and fixed-seed sampled, eager and deferred-tail prefill; negative controls (superseded prompt, reset) fail closed; skips without the model variable, fails on a GPU init failure when it is set. Then claim-scoped `scripts/serve_harness.py` (retry / append / rollback semantics) on eager **and** graph capture/replay (`HIPFIRE_GRAPH=1`), fresh process per arm. | Manual oracle + serve semantics; not perf evidence (perf needs the protocol row below: ≥3 fresh processes per arm, prompt and binary MD5s, decoded output inspected) |
| Speculative **whole-prompt exact retry, DFlash route** (#1198 step 2b: `DflashSpeculator` captures a prompt-end payload after the cold prefill's final seal and before the first token — sealed DeltaNet snapshot, last-row logits, the draft's `target_hidden` cursors, and on the windowed draft the `4 x min(len, W) x E x D`-byte `target_hidden` window the decode ring overwrites (200 MiB for the 27B DFlash2 draft: E=5, D=5120, W=2048); only from memory the admitted context does not need; a payload that cannot be held keeps the checkpoint reuse) | CPU: `cargo test -p hipfire-arch-qwen35 --lib dflash_spec::prompt_end` (descriptor, write-below invalidation, key, exact-request predicate). GPU (27B dense target + paired DFlash2 draft, fixtures `HIPFIRE_TEST_TARGET` / `HIPFIRE_TEST_DRAFT`): `cargo test --release -p hipfire-arch-qwen35 --features deltanet --test dflash_prompt_end_state_equality -- --ignored --nocapture --test-threads=1` — fresh cold prefill → decode past the draft window (the digest then differs from fresh, so the ring wrap is proven) → exact retry (`prefill_start == prompt.len()`, empty fill): byte-equal last-row logits, DeltaNet bytes + stream meta, target-KV prefix, `target_hidden` window rows, `target_hidden` cursors/watermarks, request RNG, first token; same continuation and end-of-decode digest; a second independent fresh prefill digests equal to the first; greedy and fixed-seed sampled, short and longer-than-window prompts; negative controls (changed token, shorter prompt, superseded prompt, reset, rewind below the prompt end, suffix prefill below it) fail closed or drop the payload; prints payload bytes, capture/restore microseconds and cold-prefill vs exact-retry milliseconds; skips without the model variables, fails on a GPU init failure when they are set. Then the claim-scoped `scripts/serve_harness.py` retry / append / rollback run on eager **and** graph capture/replay (`HIPFIRE_GRAPH=1`), fresh process per arm. | Manual oracle + serve semantics; not perf evidence |
| Qwen4 validation-tool split-base PLE loading | CPU: `cargo test -p hipfire-arch-qwen4 --features reference-parity --lib state_loader_split_base_refuses_missing_q4_without_derivation`. All four real-model state/profile loaders attach existing rows with the shared non-deriving validation policy before admission and pass the admitted row handle to bundle assembly; missing explicit q4 refuses instead of deriving. Hardware: rerun the failed `qwen4_parity --mode state` case with the same format selection, model, tokens and criteria, preserving other passed gate receipts. | CPU attachment regression + manual state parity; not new model admission |
| Forward / fusion / sampling / KV **user-facing serve semantics** | `scripts/serve_harness.py` with the exact model (after parity route if the change can break numbers/state); add `scripts/gates.sh` when the Redline+serve+optional perf wrapper is desired | Manual serve (semantics only) |
| KV backend rename / Qwen K–V axes / Qwen implicit default (load+serve semantics) | Load/serve harness route above **plus** [KV backend / K–V axis changes](#kv-backend--kv-axis-changes-load--serve): Qwen path **and** tag with matching backend + K/V axes; exact gfx1201 native fp8 and non-gfx1201 q8 defaults; card-B [`scripts/guard_gfx1201_baseline.py`](../scripts/guard_gfx1201_baseline.py) including graph replay (`HIPFIRE_GRAPH=1`). PR authors check the legacy-disclosure item in [`.github/PULL_REQUEST_TEMPLATE.md`](../.github/PULL_REQUEST_TEMPLATE.md) — do not duplicate that checklist here. | Manual load/serve + guard; disclosure via PR template |
| LFM2.5 chat framing / thinking output | `scripts/serve_harness.py` with an `lfm2.5:*` registry tag | Manual LFM |
| Flash-Next image pathway (arch 16) | CPU: `cargo test -p hipfire-arch-qwen4 --lib vision`, `cargo test -p hipfire-arch-qwen4 --lib image_context`, `cargo test -p hipfire-loader --lib vision`, and `cargo test -p hipfire-generate --lib qwen4_vision`. QSA rotary/pooling: `cargo run --release -p rdna-compute --example qsa_mrope_parity --features lab` against an independent CPU reference, including rectangular image coordinates and decode delta. Vision tower: exact-checkpoint HF vision-only dump and comparison, with pinned input/weight hashes. Load contract: schema-default `auto` and explicit `resident` without a sidecar stay text-only; `auto`/`resident` with an explicit bad sidecar and `on` without a valid sidecar fail closed; `off` skips even an explicit tower. Flash Next maps `auto`/`on`/`resident` loads with a tower to resident storage; other models retain their beta vision routes. Record the pre-placement resident-weight plus maximum-image-scratch reservation, including FC2 padding and projected rows, through the shared budget (`vision_reserve=`). Serve: `scripts/check_qwen4_vision.py` against the actual loaded trunk and sidecar (`--expect-vision on` for loaded image capability, including resident; `off` for no tower), covering stream/non-stream, image replacement and cancellation followed by a new turn; inspect every answer. Text-after-image, mode-off text regression and default-mode text identity with/without a sidecar must pass too. See [scope and reproduction](qwen4-vision.md); its 2026-10-23 receipts are historical, not validation of later merges. | Manual oracle + image-bearing serve; no registry admission or performance claim implied |
| VL vision-tower forward numerical parity (arch-5 / arch-11 carriers) | Dump-and-diff vs an HF `transformers` reference for the exact checkpoint, pixel inputs pinned by hash (`benchmarks/vision/dump_hf_reference.py` precedent; the archived VL family specs are historical design provenance, not current validation authority, so the route is owned by that script and the source it exercises); **blocked** for a checkpoint with no reference dump. Expected floor on 4:2:0 JPEG inputs: `patches` rel-L1 ≈ 5e-3 (zune-jpeg IDCT/chroma vs PIL's libjpeg-turbo; `jpeg-decoder` only narrows it to ≈ 3.5e-3; since d147fccb3 hipfire decodes JPEG with `libjpeg-turbo-rs`, byte-identical to PIL/libjpeg-turbo on all five committed fixtures, and the `patches` floor is the CatmullRom-vs-PIL-bicubic kernel alone: ≈ 2.6e-3 on doge, max one u8 step; 2026-09-07); 4:4:4 inputs reach ≈ 1e-4. Judge tower parity on the shape of the per-block curve and the serve battery, not on beating that floor | Manual oracle — not `serve_harness.py`; a green VL serve battery is *not* parity evidence |
| VL image-bearing serve semantics (`generate_vl` over `/v1/chat/completions`) | Manual OpenAI-compatible battery through `hipfire serve` with the exact VL artifact: committed fixtures under [`../benchmarks/vision/images/`](../benchmarks/vision/images/) + the fixed desc/ocr prompts of [`comparison-2026-05-23.md`](../benchmarks/vision/comparison-2026-05-23.md), greedy temp 0; stream **and** non-stream typed-emission check (reasoning vs `content` deltas; no literal `<think>`/`<|im_end|>` chunks in content); client-disconnect probe mid-stream followed by an immediate follow-up turn (no slot wedge); eyeball every decoded output; record artifact sha256, fixture hashes, binary md5s. No scripted harness exists (**blocked** until one lands); `scripts/serve_harness.py` is text-only today and does not exercise this surface | Manual serve (semantics only) |
| Retained replay / PM4 / AQL graft | `scripts/redline_daemon_harness.py` **and** the certification steps in `docs/REDLINE.md` | Manual Redline; promotion still policy-gated |
| Perf improvement claim | Protocol in `methodology/perf-benchmarking.md` + stationary matched runs; `speed-gate.sh` or `gates.sh` perf arm when applicable. Flash-Next claims also carry a [startup-config / product-default receipt](#flash-next-product-default-receipts-perf-claims) (`scripts/perf_config_receipt.py`) | Measured; not admission |
| Existing sealed MQ4R Redline fixture reproduction | `python3 -m tools.redline golden` for its exact gfx1100/gfx1151/gfx1201 fixture only | Measured reproduction; exact identity + route proof + stationary floor; not admission. Does **not** imply that every default-eligible MQ4R-SKU model file is one of the sealed fixtures. |
| Arch port | `methodology/arch-port-validation.md` (channel + speed; no retired coherence battery as acceptance) | Manual |
| Model/route **admission** (registry evidence) | Row in [`admissions.yml`](admissions.yml) | Schema v2; exactly one evidence-bound record (LFM2.5-350M MQ4 gfx1201 retained-PM4). No inferred/wildcard rows. Registry admission/evidence is distinct from runtime wiring: the sealed LFM row does **not** select a runtime default and current automatic selection does not use it. |
| MQ4R **runtime** automatic Redline default | Source predicate `mq4r_redline_default` in `crates/hipfire-runtime/src/config.rs`; policy in [`REDLINE.md`](REDLINE.md) | Cross-model automatic runtime predicate (the gfx1201 Qwen3.5 dense and Qwen4 rows below are the other product defaults). Runtime-only: exact GPU arch `gfx1100`, `gfx1151`, or `gfx1201`; PP=1; TP=1; the MQ4R SKU (HFQ `hipfire_route_sku`, else the registry card `quant`; never the extension) → retained PM4/Auto unless disabled with the config wizard's built-in `hip` profile, another explicit backend selection, or `HIPFIRE_REPLAY_BACKEND=hip`. Model-family agnostic apart from the exclusions applied by `retained_redline_default` in the same file: Muse Glimmer (arch 14) is withheld until its PM4 lowering lands, Gemma 4 (arch 13) is never admitted (unvalidated retained tape), expert-parallel loads (`ep` > 1) are never admitted, and a Qwen3.5 process using CPU-executed partial offload gets no retained default. Selection by file extension was removed in 0.5.0 (the SKU comes from HFQ metadata or the registry card). `gfx1200` and all other arches remain opt-in. Existing LFM `.mq4` registry evidence is not auto-selected because it is not `.mq4r`, not because LFM is categorically exempt; any usable non-default retained route must still prove route support and fail closed when unsupported. **Not** registry admission, **not** Section 7 certification, and **not** a sealed-fixture claim for every default-eligible `.mq4r` model. |
| gfx1201 Qwen3.5 dense **runtime** automatic Redline default | Source predicate `retained_redline_default` in `crates/hipfire-runtime/src/config.rs`; policy in [`REDLINE.md`](REDLINE.md) §3 | Runtime-only: exact GPU arch `gfx1201`; `model_arch` `qwen3_5` (dense, any weight format); PP=1; TP=1; no drafter → retained PM4/Auto for plain-AR decode unless disabled with `replay.backend = "hip"`, the built-in `hip` profile, another explicit backend selection, or `HIPFIRE_REPLAY_BACKEND=hip`. Every new gfx1201 decode kernel must pass the graph-vs-PM4 parity gate (logits, hidden, DeltaNet state, KV at ctx 512/8,192/32,768, 256 greedy tokens). **Not** registry admission and **not** Section 7 certification. |
| gfx1201 Qwen4 (Flash-Next) **runtime** automatic Redline default | Source predicate `retained_redline_default` in `crates/hipfire-runtime/src/config.rs`; policy in [`REDLINE.md`](REDLINE.md) §3 | Runtime-only: exact GPU arch `gfx1201`; `model_arch` `qwen4` (any weight format); PP=1; TP=1; no drafter (a native MTP head is a drafter, so only MTP-off loads replay) → retained PM4/Auto for plain-AR decode, same opt-outs as the Qwen3.5 row. A change to the Qwen4 decode tape must pass `scripts/redline_daemon_harness.py --qwen4 --pm4` (HIP/PM4/blob bit-exact over the shadow positions) and greedy text equality against `HIPFIRE_REPLAY_BACKEND=hip`. **Not** registry admission and **not** Section 7 certification. |
| gfx1100 Qwen3.5 dense **runtime** automatic Redline default (qualified decode weight formats) | Source predicate `retained_redline_default` + `gfx1100_dense_weights_qualified` in `crates/hipfire-runtime/src/config.rs`, identity `DenseWeights` in `crates/hipfire-runtime/src/route_sku.rs`; policy in [`REDLINE.md`](REDLINE.md) §3 | Runtime-only: exact GPU arch `gfx1100`; `model_arch` `qwen3_5`; the trunk's decode weight format (tensor index + symmetric marker, never the file name) MQ3G256V2 (either zero mode) or symmetric MQ4G256V2; PP=1; TP=1; EP=1; no drafter → retained PM4/Auto (with the GFX10/GFX11 scratch rings for MQ3G256V2) for plain-AR decode, same opt-outs as the gfx1201 Qwen3.5 row. Adding a format, or changing a qualified format's tape, must keep served `generate` greedy identity vs `HIPFIRE_REPLAY_BACKEND=hip` with a per-request `HIPFIRE_REPLAY_ROUTE_PROOF replays>0` marker on a file of that format (`.mq3-xts` / `.mq4-xts` run the AWQ rotation kernels): `python3 tools/redline/serve_identity_ab.py ident --daemon <daemon> --model <file> --arms dflt,hip --out <dir>` (acks `commit_ready`; an un-acked driver prints no marker) and `... ab --arms on,hip` for served decode tok/s; `cargo test -p hipfire-runtime --lib route_sku::` proves a rename does not change the answer. **Not** registry admission and **not** Section 7 certification. |
| railgun typed-program authoring / validation (G0 recording invariance, G4 probe, G7 shadow diff, `HIPFIRE_RAILGUN_CHECK`) | `cargo test -p railgun-cert` (recording-inventory source scan); `scripts/redline_daemon_harness.py --g0 --inventory <json>` (add `--g0-program`; `--dflash-cycle --draft` for the DFlash window); shadow/check procedures in [`REDLINE.md`](REDLINE.md) §5 | Manual diagnostic evidence for the railgun program model only. railgun authors and validates typed programs (kernel nodes, resources, effects, hazard edges, lowering); Redline is the retained-replay transport. They are not the same system and railgun is not a full Redline replacement: a green railgun check is not Redline route proof, certification, or admission, and the default route stays plain retained PM4. |
| VMM continuous-batching serve route (`serve.vmm_batch`, default off) | State oracle `cb_vmm_state_oracle` (k=1..8, unequal prompts, contexts 512/8,192/32,768, 256 greedy steps, negative controls; under `--spec dflash` the pre-window `singleton.checkpoints.positions` compare is plan-aware: DeltaNet checkpoint positions follow the prefill chunk plan, whose ceiling is memory-admitted (8192 down to 512) and so lower for lane walks with other lanes resident, hence each walk records its seed ceiling, equal ceilings compare raw positions, and differing ceilings must each equal the engine's simulated `take_dn_checkpoint` result (`dflash_seed_checkpoint_positions`) for their own ceiling; every other surface stays an exact compare and no `HIPFIRE_PREFILL_CHUNK_ROWS` pin is needed; extended with the 63/64/65/96/128-row aggregate cases and wide-verify on/off, HIP and PeaceMaker arms for `HIPFIRE_CB_VERIFY_CHUNK128`) for exactness; kernel byte oracle `cb_verify_gemm_oracle` for the 64..=128-row exact verify GEMMs (DFlash only, gfx1201 dense MQ4G256V2); flag-off load ack/route identical to base; `scripts/serve_harness.py --concurrent-clients 1,2,3,4,5,6,7,8` (battery/chain only; greedy runs assert per-client isolation equality) with eyeballed outputs and exactly-once terminals for serve semantics; loaded ack must report `continuous_batch_route=vmm`, `vmm_batch_kv_backend=vmm` without any `kv_backend`/`max_seq` override. Default promotion needs every gate in the CB plan, not a throughput win. | Manual oracle + serve; perf per `methodology/perf-benchmarking.md` |
| VMM-backed **shared paged** KV (`serve.multi_slot_paged` + `serve.multi_slot_paged_backend=vmm`) | **Experimental; not an admission route.** Evidence so far: `cargo run -p hip-bridge --features lab --example vmm_page_arena_smoke`, `cargo run -p rdna-compute --features lab --example vmm_paged_kernel_parity` (two distinct block tables), `vmm_page_mapper_smoke`, `vmm_paged_pool_smoke` (share/COW), and `vmm_paged_swap_smoke` (page-offset capture/restore); `scripts/serve_harness.py --no-spawn --concurrent-clients 2` and `--concurrent-clients 4` on gfx1100/Qwen3.5-0.8B q8 passed (4-slot: 20 turns, `empty=0`, `attractor=0`). Prefix-cache hit (`cached_tokens=384/430`) is observed. A 2-page pool + 430-token prompt returns retryable 503 `page demand exceeds pool`; `/health` stays ok and the next request succeeds. The gfx1100 matrix now includes 0.8B, 4B, 9B, and 27B two-slot concurrent requests. `python3 scripts/check_slots_vmm_swap.py --paged-vmm` passes swap-out→swap-in with continuation reuse (31/61). COW divergence is checked at the pool/mapper layer; a 27B DFlash2 attach smoke loaded the paired draft and served without a fault. MTP is exercised end-to-end on the 4B `.mtp` fixture (trace-confirmed draft acceptance). Batched VL passes the image/follow-up/long-context/stop/json/concurrency harness on the paged VMM backend. 4 × 40,000-token contexts were admitted and generated on the 27B fixture (all four HTTP 200, 39,999 prompt + 1 output). MoE prompt chunks run in one `_vmm_paged` forward apart from decode; lockstep paged decode can take retained PM4 when the slots Gpu is armed (2026-10-26 gfx1100 `nex-n2.5-mini.mq4r` q8 paged-VMM receipts: 1 slot / 320 greedy tokens with a 256→512 bucket rearm is byte-identical to `HIPFIRE_REPLAY_BACKEND=hip` (125.5 vs 94.4 tok/s), 2 slots in lockstep are byte-identical on both lanes (182.5 vs 157.1 agg), and 2 slots at deliberately different prompt lengths are byte-identical on both lanes with the 1,417-launch `live_slots=2` tape while 94/94 two-live steps were ragged positions (185.4 vs 151.1 agg); a prefill row stays eager and the tape rearms on a context-bucket or live-slot change. Parity needed three fixes — one upload path and one `max_ctx_len` bound, the few-row MQ4V2 GEMV instead of zero+GEMM, and `ldsstage` admitted under recording — see the plan's Phase 6 log. **Not** a perf claim and not an admission row). **Not yet accepted:** gfx1151/gfx1201, graph capture, hardware PM4/MoE serve, and the exhaustive leak/error sub-matrix. **Known open defect (gfx1100):** mixed-length concurrent prefill does not reliably reproduce the solo completion — `scripts/check_vmm_paged_regressions.py` (chunk pinned) reports 24/144 mismatches at 127/128/129/255/256/257, a lane can return another length's completion, and four identical prompts in one batch can disagree; also seen once on the A3B `.mq4r` route (360/25-token pair); not the merge, the batched lm_head, the arena backend or the VMM mapper (both builds tested), mechanism open. Treat concurrent == solo as **unproven** for this route; log in [`investigations/2026-10-30-vmm-paged-kv-fixlog-and-revalidation.md`](investigations/2026-10-30-vmm-paged-kv-fixlog-and-revalidation.md). Until then the arena-paged and per-request VMM routes remain the supported routes. |
| Multi-slot KV-storage route (`serve.multi_slot`, fixed slab vs the shared paged pool with `serve.multi_slot_paged` + `serve.multi_slot_paged_backend=vmm`; `kv_backend=vmm` alone is refused) | `cargo run --release -p hipfire-runtime --features lab --example test_slots_vmm_oracle -- <model>` (slab vs paged-VMM in one process: composition determinism on both routes, paged first-token isolation, optional `HIPFIRE_SLOTS_DECODE_GRAPH=1` arm asserting slab capture/replay (the shared paged pool decodes eagerly by design); `HIPFIRE_SLOTS_ORACLE_DUMP` writes ids for a graph-on/off diff, which must be identical; `HIPFIRE_SLOTS_ORACLE_DRAFT=<DFlash2 sidecar>` runs the same gates with DFlash armed on both storage routes, and `HIPFIRE_SLOT_TRACE=1` must show `DFlash draft`/`DFlash accept` lines); `cargo test -p hipfire-arch-qwen35 --lib serve_vmm` (tier gate, reservation bound, snapshot-tag separation); `hipfire bench --concurrency 1,2,4 --backend slots` with `HIPFIRE_SERVE_MULTI_SLOT_PAGED=1 HIPFIRE_SERVE_MULTI_SLOT_PAGED_BACKEND=vmm` (`--kv-backend vmm` alone is refused) for load + concurrent service; `python3 scripts/check_slots_vmm_swap.py --model <trunk>` (3 conversations on 2 slots: `swap-out`, `swap-in`, and a `continuation HIT` with reused > 0 after restore — a VMM restore that does not publish `snap.seq_len` silently re-prefills and fails this); `python3 scripts/check_slots_vmm_vision.py --model <trunk> --vision <sidecar>` for image turn → **text follow-up** (continuation HIT, sequential M-RoPE), a prefill-chunk-boundary fact recall, stop sequences (`finish_reason: "stop"`, no stop bytes, next request served), malformed stop 400, `response_format: json_object` (returns a JSON object), `logprobs`/`top_logprobs` (finite chosen log-prob, candidate list, malformed contract 400), and concurrent lanes; vision with a committed `benchmarks/vision/images/` fixture; MTP (`--spec mtp`) with `HIPFIRE_MTP_TRACE=1` windows plus an `--spec off` byte-equality control on the same prompt (both eyeballed). Container recipe: `docker build -f Containerfile --target full -t hipfire:dev .` then `docker compose -f docker/compose.vmm-9001.yaml up` (the compose `build:` stanza runs the same command), then `/health` + text + image requests on port 9001. Fail-closed gates: rotated tiers and legacy (non-DFlash2) drafts with a named reason before allocation; the VMM prefix cache uses host snapshots, not the arena page pool. Output equality with the sequential route is **not** a gate (row-count-dependent kernels: same as the arena route). Not a perf claim. |
| Shared-weight host import (`HIPFIRE_WEIGHT_HOST=1`, developer, default off; dense Qwen3.5 arch 5, exact gfx1201, PP=TP=1, no drafter/MTP/vision, `HIPFIRE_REPLAY_BACKEND=hip`) — [`CONFIG.md`](CONFIG.md#shared-weight-host-developer-linux-default-off) | Pure CPU: `cargo test -p hipfire-runtime weight_host`, `cargo test -p hipfire-arch-qwen35 shared_weights`, `cargo test -p hipfire-daemon --bin daemon gpu_lock` (identity/canonical-hash/negative-manifest/shared-claim fixtures). G1: `shared_weight_state_oracle` (build with `--features lab`) `weights` / `compare-weights` modes — every one of the 1,347 owners byte-equal private vs import, second importer adds no W-sized physical allocation (sysfs `mem_info_vram_used`). G2: two daemon builds with different executed code objects (`HIPFIRE_MODULE_LOAD_AUDIT=1`) import one epoch under correct-class fabric admission; mismatched recipe/source/admission and malformed clients refused before any FD; tenant SIGTERM, live-tenant drain and host SIGTERM; VRAM returns to baseline. G3 (acceptance): `shared_weight_state_oracle` own-solo vs concurrent **byte** equality of f32 logits, hidden/DeltaNet/KV state, token IDs and UTF-8 for each of two builds, HIP eager and `HIPFIRE_GRAPH=1`, ≥3 fresh processes per arm, contexts 512/8,192/32,768 where two tenants fit, matched private-vs-import load parity and the ≤20% warm weight-ready gate. Default-off CI (`no-gpu-ci.sh`) must stay green; adjacent architectures keep their legacy path. | Manual oracle + lifecycle evidence; timing only through a separate exclusive (or memory-isolated) run. **Not** a performance claim, not Halo/UMA, Flash-Next or host-mapped support. |
| Embedding encoder (arch 17) forward / attention kernel / weight-format change | `scripts/oracle_pplx_embed.py` — a torch-free NumPy reference of the whole text encoder (embedding gather, zero-centered RMSNorm, fused q+gate split, per-head Q/K norm, partial interleaved mRoPE, bidirectional GQA attention, the GatedDeltaNet causal recurrence, SwiGLU, Dense head) — diffed against the artifact under test; **plus** `scripts/check_embed_quant.py` when the quantizer or a weight format is involved, which loads every artifact (dir or `.hfq`, any format) and asserts each inside its bit-width band (`q8` 3e-2, 6-bit 8e-2, 4-bit 2.5e-1 — an order under the O(1) a broken dequant arm produces). Both are manual; the crate's own unit tests cover the GPU-free parts (config parse, arch mapping, skiplist mask, L2 normalize, MaxSim, image splice, dequant dispatch). | Manual oracle + quantization matrix — **not** `serve_harness.py`: that harness drives text generation and does not exercise the non-chat `embed` verb |
| Embedding (arch 17) **multi-slot / batched forward** change (`serve.multi_slot`, `encode_embeddings_batch`, the slot engine, the admit/batch path) | `scripts/check_embed_multislot.py --model <dir> [--slots N] [--oracle]` — encodes the same inputs through the ordinary daemon and through `experimental_multi_slot` and requires them **bit-identical** (the batched forward is deterministic: the GDN state is reset per segment and one weight copy serves the batch), then optionally diffs the slot route against `scripts/oracle_pplx_embed.py` for an independent leg. Exact, not statistical, on purpose: a segment that reads another segment's rows or delta-rule state still returns plausible vectors, so a tolerance would pass the very bug the route can have. Also `cargo run --release -p hipfire-arch-qwen35-embed --example batch_parity --features lab -- <model>` for the forward alone, and the HTTP path (`/v1/embeddings` through a multi-slot `hipfire serve`) when the change touches admission or the client's reply routing. | Adjacent to `serve_harness.py` (same serve route) but that harness drives generation; the batched forward is only reachable through the `embed` verb |
| Embedding (arch 17) perf claim | `scripts/bench_embed.py` against committed `benchmarks/prompts/embed_*.txt` (JSONL or HTTP transport), fresh process, warmups then medians, prompt md5s recorded in the JSON. The encoder has **no `tg`** — no LM head exists (`tie_word_embeddings: true`, zero `lm_head` tensors), so there is no decode loop; only the single bidirectional pass (`pp`) is meaningful. | Measured per `methodology/perf-benchmarking.md`; not admission |
| EmbeddingGemma 2 (arch 18) forward / attention / weight-format change | `scripts/check_embeddinggemma2_parity.py --model <dir> --ref <ref.npz>` — drives the daemon `embed` verb end-to-end and diffs served 768-d embeddings and their Matryoshka truncations against a float32 CPU `transformers` reference (generated by the committed `scripts/make_embeddinggemma2_reference.py`; band 2e-2 rel-L2, expected ~1e-3), plus the image and audio legs (six cases each: sizes across the ≤280 range, non-multiple pooling grids, tiny, cap-hitting, and clips from 0.1 s to 15 s) and the video legs (five clips — 3, 4, 5, 32 and 77 frames, so the ≤32-frame window and the uniform `max_frames` sampler are both exercised, with a non-square aspect and a mid-text marker) from the same generator, plus the edge-case contract (marker-count mismatch refused per modality, a tower-less image/audio/video input refused by modality, a video frame that is not a PNG/JPEG refused with the extraction to run, empty text refused, oversized `dims` refused, >8192-token truncation lands on exactly 8192). The script loads the daemon with `vision_mode=auto` and emits the shared `.vl` tower pack itself via the quantizer, so it needs no prebuilt artifact; `vision_mode=off` (the schema default) is text-only and refuses media. The tokenizer leg is implicit — `n_input_tokens` must equal the reference's HF tokenization length per case (the crate's SP-BPE merge path is what makes that hold; the greedy default diverges). | Manual oracle + edge-case contract — not `serve_harness.py` (drives generation) |
| EmbeddingGemma 2 (arch 18) multi-slot / batched forward change | Same route as arch 17: sequential-vs-slot bit-identity plus the parity script against the slot daemon. | Manual oracle |
| EmbeddingGemma 2 (arch 18) perf claim | Same route as arch 17: `scripts/bench_embed.py --daemon target/release/daemon --model <tier>.hfq` over the committed `benchmarks/prompts/embed_*.txt`, one fresh process per arm, warmups then medians, `prompt_md5` plus the binary and model digests recorded in the JSON. It drives the `embed` verb directly, so the encoder's single bidirectional pass (`pp`) is the whole measurement — there is no `tg`. **Text only:** the tool loads with an empty params object, so `vision_mode` stays `off` and no media leg is timed; a media perf claim needs `vision_mode=auto` and its own per-modality fixture, and neither the multi-slot collector nor an HTTP client is exercised here. | Measured per [`methodology/perf-benchmarking.md`](methodology/perf-benchmarking.md); not admission |
| SKU application-capacity emulation / estimation (`HIPFIRE_EMU`, `hipfire bench --emu`, `hipfire emu list\|validate\|estimate`, the `emu_validate` daemon verb) — [`EMU.md`](EMU.md) | CPU, no GPU (set `ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1`): `cargo test -p hipfire-config emu` (registry truth tables for all ten ids, host-compatibility checks, UMA scenario math, budget ledger incl. concurrent reservations and faults, derating model), `cargo test -p hip-bridge emu_budget`, `cargo test -p redline-rocr emu_kernarg_budget_tests`, `cargo test -p rdna-compute --lib emu`, `cargo test -p hipfire-cli emu` (parsing, `--emu` over `HIPFIRE_EMU` precedence, unknown id refused before spawn, estimate on a sample input), plus `python3 scripts/check-env-docs.py`. Hardware (host-specific, through the fabric scheduler only; never a CU mask, persistent kernel or host-flag poll): `hipfire emu validate <id> --json` on each compatible host (R9700 `rx9070xt`/`rx9070`/`rx9060xt*`, XTX `rx7900xt`/`rx7800xt`/`rx7700xt`/`rx7600`, Halo `strix-point`/`krackan`); the receipt's capacity, host-topology and graph sections must pass (nonzero exit otherwise). Capacity-capped model runs add their own placement/parity oracle for the touched route, and the no-profile (`HIPFIRE_EMU` unset) regression stays byte-identical. | CPU unit + manual hardware receipt. Capacity enforcement and the receipt only: **not** target tok/s, not a CU/clock/cache reproduction, not model or route admission. An estimate is model output (`TopologyScenario` unless measured on target hardware with held-out validation), never a measurement. |
| Raon-1-mini-preview (arch 19) forward / weight-format / serve-route change | `raon_oracle` module gates vs the BF16 reference (mhc, norm, head, mla, rda, moe), state/graph parity (`raon_oracle state --mode graph`), KLD vs a BF16 teacher on the pinned WT2 and code24 windows, `raon_oracle bench` pp/tg, thinking on/off serve smoke. **Receipt (2026-10-30, `raon-1-mini-preview.mq4-xts`, sha256 3828f980…; 3 fresh processes × 3 reps, medians, tok/s):** KLD vs BF16 teacher WT2 0.081, code24 0.0138 (gfx1151 code24 0.0139); mla module oracle 24/24 on gfx1100, gfx1201, gfx1151. pp512/pp2048/pp8192/pp16384/tg128: gfx1100 RX 7900 XTX (shader clock capped at 2,400 MHz; gfx1100 PeaceMaker twins + capacity-based Q8 MLA mirror) 3,327/3,303/3,101/2,652/64.1; gfx1201 R9700 2,927/2,895/2,742/2,545/71.4 (decode varies across processes on this card, ~39 in about 1 of 3; the same card-level dip appears in eager mode and in other engines, so it is not route selection); gfx1151 Strix Halo 1,348/1,322/1,240/1,149/39.2. Serve harness (OpenAI/Anthropic, thinking on/off, tools, multi-turn, greedy needle 2K/8K/27K) all pass on gfx1100 and gfx1201; chat-template render parity 20/20 cases. RX 7900 XT (20 GB): not measured on the card; estimated for RX 7900 XT from capacity-emulated measurements (`HIPFIRE_EMU=rx7900xt` on the XTX, W4A8 routed experts: 3,531/3,490/3,262/2,770/63.6, scaled per cell by the XT/XTX ratio of another engine measured on both cards) at about 3,154/2,809/2,417/2,055/54.4. Under the 20 GiB emulation, 195,782 tokens of context are proven with exact needle recall; 262,144 does not fit (latent-cache growth exhausts memory near 237K rows). | Manual + channel |
| Raon-1-mini-preview (arch 19) prompt / session cache change (`SessionState` impl, snapshot grid, `cache_capable`) | `raon_oracle session --model M --tokens F` (build with `--features lab`; cold vs resident follow-up vs pool restore after an unrelated prompt: final logits, RDA/conv slab, greedy decode ids and logits bit-exact; pool bytes within the budget), at the admitted chunk and with `HIPFIRE_RAON_PREFILL_CHUNK_ROWS=512`; `scripts/serve_harness.py --sampling greedy` cache on vs `HIPFIRE_SESSION_CACHE_BYTES=0`, fresh serve per arm alternating, rows diffed (content, reasoning, tool calls, finish, gen, request md5): `--mode chain --tool-chain benchmarks/sessions/raon_agent_chain.json` thinking on and off, `--mode session`, `--mode prefix-fanout --fanout-shared-tokens 8192`, plus a small-budget arm and the battery against the base binary. **Receipt (2026-10-31, v2 `mq4-xts-a8attn-s.shared-q8f16`; every cache-on/off pair greedy-identical; total TTFT over the chain, s):** R9700 thinking off (16 requests, 9.9K -> 34K tokens) 99.0 off -> 31.6 on (median of 3; total request wall 133.5 -> 66.0), thinking on (6 requests) 36.4 -> 22.9; 7900 XTX thinking off (18 requests) 176.8 -> 48.4 (wall 247.2 -> 77.1), thinking on 68.6 -> 29.5. Per turn, R9700 thinking on: 2.9/4.3/4.6/5.9/8.0 -> 1.4/1.7/0.9/2.3/2.9 s. 8K shared system prompt, server TTFT median: R9700 1.60 -> 0.86 s (1, 4, 8 clients); XTX 2.58 -> 0.23 s (1, 4 clients), 0.75 s (8 clients). Decode tok/s unchanged; no speculative verify on this route. Hosts were under build load during part of the run (alternated arms; timing receipts carry the job ids). | Manual oracle + measured A/B |
| Raon-1-mini-preview (arch 19) railgun AR decode owner (`HIPFIRE_RAON_RAILGUN` tri-state `auto`(default)|`1`|`0`, with the corpus in `HIPFIRE_RAON_RAILGUN_CORPUS=<dir>` or `<root>/kernels/railgun-corpus` beside the installed pack, developer; exact gfx1201, singleton plain-AR, graph decode mode) | CPU, no GPU: `cargo test -p railgun-jitcorpus` (program-sequence parser, loaded-digest check, conditional/unknown reporting), `cargo test -p railgun-cert recording` (the `raon_ar_gfx1201` inventory program), `cargo test -p rdna-compute --lib --features deltanet raon_` (the registry inventory of the decode body's JIT modules and the embedded `gated_delta_vector_pm` image). Corpus (CPU, hipcc): `railgun-jit-corpus build --only-program raon-ar-decode-gfx1201 --program-sequence SEQ.jsonl --out DIR`, then `railgun-jit-corpus check` with the same flags and `--corpus DIR/v2/<host key>`. `SEQ.jsonl` is the real decode body's launch sequence, one `{"kind":"kernel","symbol","module","sha256",..}` per launch (`"fill"` lines carry no object); **Install route** (how the runtime loads exactly the corpus bytes): the runtime probes `<exe dir>/kernels/compiled/<arch>`, then `./kernels/compiled/<arch>` under its CWD, then `$HIPFIRE_HOME/bin/kernels/compiled/<arch>` for the pack, and loads a module whose `<module>.index.json` matches its packaging key verbatim before any hot-cache or JIT lookup; so copy `DIR/v2/<host key>/pack/<arch>` to `<private root>/kernels/compiled/<arch>`, run the processes with that root as CWD and `HIPFIRE_KERNEL_CACHE` set to a private hot cache (nothing under `~/.hipfire*`). `railgun-jit-corpus verify-install --corpus DIR/v2/<host key> --install-root <private root> --only-program raon-ar-decode-gfx1201 --program-sequence SEQ.jsonl --arch gfx1201` is the CPU-only dry run of that loader: it exits 1 naming every module whose resolved object is not a file of the pack with the corpus record's SHA-256 (embedded images: the `pm_native` record) and checks the corpus toolchain pin against the local compiler. `check` exits 1 and names every node with no verified record, **including a node whose recorded loaded object digest differs from the corpus record of that module** (the program would not run the certified bytes). Conditional nodes (valid only under the runtime's generation-bound witness) and Unknown nodes are printed as `NOTE` lines, not errors: the owner supplies the witness or declines. Product oracle (GPU, one fresh process per arm, caller-assigned card): `raon_oracle railgun-state --arm graph` and `--arm railgun --corpus DIR` from the same model, prompt and `--ctx` (also 127/128/129 via `--boundaries`), N greedy steps on argmax-fed ids, then `raon_oracle railgun-state --compare G.json R.json` (and a `--flip-byte` negative control): logits bytes every step, the persistent RDA/conv/seq slab, live latent and mirror K/V rows and the mirror watermark, mHC streams, host and device positions, weights digest before/after; reports requested versus executing backend, Declined reasons and owner stats. A requested-railgun arm that executed no step, a quarantined owner or any byte mismatch fails. | Manual CPU contracts + manual GPU state oracle for the Raon railgun route only (default `auto`: no owner without exact gfx1201 and a discovered corpus; `0` is the kill; `railgun-state` runs with `1` or an explicit corpus). **Not** registry admission, **not** Redline route proof, **not** a perf claim (perf needs the three-fresh-process `raon_oracle bench` protocol of `methodology/perf-benchmarking.md`), and **not** coverage of Raon DFlash verify or gfx1100. The corpus check certifies facts about the exact objects, never that the program ran; only the oracle compare does. Local corpus (CPU, no hipcc/GPU/network): `railgun-jit-corpus from-cache --arch gfx1201 --program raon-ar-decode-gfx1201 --cache <hot cache> [--pack <pack arch dir>] --out <dir>` certifies the user's own pack/hot objects under the runtime's current key (`cargo test -p railgun-jitcorpus --test from_cache`); exit 2 with a one-line JSON refusal publishes nothing. |
| Unknown surface | **Blocked** until an owner adds a row here | Fail closed |

## Retired gates (historical only)

The following are **retired as current acceptance evidence**. They must not
be required for merge, promotion, or benchmark claims.

| Pattern | Status |
|---|---|
| `scripts/coherence-gate-*.sh` (formerly e.g. `coherence-gate-qwen35-dspark.sh`, `coherence-gate-qwen3-dspark.sh`, `coherence-gate-minimax.sh`, `coherence-gate-cohere2moe.sh`, `coherence-gate-deepseek4-*.sh`, …), `_coherence_runner.py`, `awq_coherence_check.sh` | **Removed in 0.5.0** (git history only). Never promotion or acceptance. There is **no universal successor**: use the claim-specific route in the [claim → route map](#claim--route-map) (`scripts/serve_harness.py` battery/chain for serve-level coherence), or treat the surface as blocked. |
| `tools/change_gate/`, `.github/agentic-review/`, and the pre-hw-gate agentic static-review route | **Retired.** Superseded as a review path; historical references only. Not merge evidence. |
| Other gate scripts **not named anywhere in this selector** | Do not treat as canonical acceptance unless a future INDEX/VALIDATION revision names them. Supporting tools already listed above stay in force. |

Campaign-specific guidance (for example an LFM effort that omits coherence
gates) is **not** a universal rule for every arch or model. Universality is
blocked on purpose.

## Explicit non-routes

| Anti-pattern | Disposition |
|---|---|
| One script that “replaces all gates” | **Rejected** — no universal gate. |
| Green no-GPU CI as proof of GPU correctness | **Rejected** |
| `serve_harness` as numerical/state parity or universal forward proof | **Rejected** — semantics only; oracle or **blocked** |
| `serve_harness` success as Redline route proof | **Rejected** |
| `redline_daemon_harness` fingerprint as installed product PM4/AQL route | **Rejected** without `REDLINE.md` ladder |
| railgun G0/G7/check result as Redline route proof or as a replacement for the Redline ladder | **Rejected** — railgun validates typed programs; it does not certify a retained route |
| Coherence-gate pass as current acceptance | **Rejected** |
| `change_gate` / agentic-review route as current acceptance | **Rejected** — retired; historical only |
| hw-gate success as auto-approval or auto-promotion | **Rejected** — PR evidence only; `beta` moves on the promotion lane, merges on direct review + required no-GPU checks |
| A skipped, missing or `hardware-blocked` required lane row as a pass | **Rejected** — skip is not pass; only a recorded waiver (Kaden or fivetide, at most 7 days) lets a red row through, and no release is tagged with one open |
| Bench number without protocol + identity hashes | **Rejected** as promotion evidence |
| Inferred or “signed” `admissions.yml` row without earned fixture evidence | **Rejected** — schema v2 forbids inferred/wildcard rows; only the exact admitted record applies |

## Related owners

- Navigation / lifecycle: [`INDEX.md`](INDEX.md)
- Admission registry: [`admissions.yml`](admissions.yml)
- Redline policy: [`REDLINE.md`](REDLINE.md)
- Perf protocol: [`methodology/perf-benchmarking.md`](methodology/perf-benchmarking.md)
- Testing playbook (setup, smoke tests, reporting bars): [`testing.md`](testing.md)
- Fixture and quality pins: [`fixtures.md`](fixtures.md), values in [`../fixtures/pins.toml`](../fixtures/pins.toml) (`scripts/check-pins.py`)
- Engineering lessons (bitwise-gate discipline, gate gaps, measurement rules): [`methodology/lessons-0.4.5.md`](methodology/lessons-0.4.5.md)
