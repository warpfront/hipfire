# Post-land GPU smoke

`scripts/post-land-smoke.sh` runs a fixed set of greedy 32-token generations on the multi-card and
multi-architecture routes after every land of `beta`, pins the committed token ids, and checks the
invariants between cells that the multi-GPU contracts promise. A land is judged against the ids pinned
by the previous clean land, so a route that silently breaks, changes arithmetic, or stops loading is
reported with the exact cell, the first differing token position, and the lane that ran it.

The runner is `scripts/post_land_smoke.py` (Python standard library only; its module docstring is the
reference for the plan schema and the daemon protocol), the plan is `scripts/post-land-smoke/plan.json`,
and the orchestrator is `scripts/post-land-smoke.sh` (bash). CPU tests:
`scripts/test_post_land_smoke.py` (runner, verdict, pins, hold, watcher) and
`scripts/test_post_land_smoke_orchestrator.py` (orchestrator end to end with a fake daemon, fake
`fabric-run.sh`, fake ssh/rsync). Both run in `scripts/no-gpu-ci.sh`.

## Purpose

Catch multi-card and per-architecture regressions within an hour of a land, instead of when someone next
runs that route by hand.

The incident that motivates it (2026-10-18): the D2 view override `7f6f5af32` broke Flash-Next EP2.
Every EP2 load died in its first prefill with `sealed_moe: live gate/up live buffer identity differs`,
followed by `qwen4 EP mesh is poisoned by an earlier failure (reload the model)`. Single-card runs, the
unit tests and the no-GPU CI all passed, and the regression sat on `beta` unnoticed for hours. Both EP2
cells of this smoke fail on that commit, on the first generate, with those strings in the daemon's error
event.

The smoke is a correctness check, not a benchmark: the timings it records are wall clocks of shared-fabric
holds, never exclusive measurements.

## Cells

Every cell is one fresh daemon process: load, one generate (`temperature 0`, 32 tokens, thinking off via
`assistant_prefix = closed_think`, prompt "Explain why the sky is blue in three short sentences."), unload.
Loads are named `<cell>-load` in the plan (the runner rejects a load and a cell with the same name); pins
and invariants use the cell names. Times are the single observed validation runs from the recipes
(load / generate / process wall, seconds); the generate time includes cold first-use kernel compilation
because every load gets a fresh HOME.

| Cell | Host / cards | Artifact | Route proof (required in addition to 32 ids, `finish_reason=length`) | Contract / invariant | Measured load / gen / wall |
|---|---|---|---|---|---|
| `fn1-ar` | hiptrx, A (lane A,E) | Flash-Next mq4 split base `qwen3.8-flash-next-base.mq4-xts` (70,888,270,360 B) + `.ple-q4` | stderr `kernels: Ar route plan (N modules)` (the checked-in plan applied), `kernels: Ar route ready, open (never sealed): …/… predicted … modules … compiled before ready … 0 unplanned kernel requests during load`, `drafter=off`; loaded `arch=qwen4, pp=1, mtp_mode_effective=off, max_seq=2048` | one-card AR reference; `fn1-ar == fn1-mtp == ep2-ar` | 88.8 / 0.8 / 91.7 |
| `fn1-mtp` | hiptrx, A | same | stderr `kernels: NativeMtp route plan (N modules)`, `kernels: NativeMtp route ready, open (never sealed): …/… predicted … modules … compiled before ready … 0 unplanned kernel requests during load`, `qwen4 native MTP speculator enabled`, `drafter=mtp`; loaded `mtp_mode_effective=on` | greedy native MTP equals AR | 96.6 / 0.7 / 99.5 |
| `ep2-ar` | hiptrx, A+E | same | stderr `qwen4 EP plan: 2 ranks`, `native MTP off`; loaded `ep=2, pp=1, mtp_mode_effective=off` | EP2 equals one card; `fn1-ar == ep2-ar`, `ep2-ar == ep2-mtp` | 127.4 / 31.3 / 160.1 (fixed build; see Known state) |
| `ep2-mtp` | hiptrx, A+E | same | stderr `qwen4 EP plan: 2 ranks`, `qwen4 native MTP speculator enabled`, `drafter=mtp`; loaded `ep=2, mtp_mode_effective=on` | `fn1-mtp == ep2-mtp` | 124.3 / 51.8 / 177.6 (fixed build) |
| `pp2-ar` | hiptrx, B+C (lane B,C) | Flash-Next **mq2-xts** export (47,038,936,600 B) + `.ple-q4` | stderr `qwen4 PP plan: 2 bands`; loaded `pp=2, mtp_mode_effective=off` | PP2 is byte-identical to one card on the same artifact; `pp2-ar == pp2-mtp` | 52.2 / 32.7 / 85.3 |
| `pp2-mtp` | hiptrx, B+C | same | stderr `qwen4 PP plan: 2 bands`, `qwen4 native MTP speculator enabled`, `drafter=mtp`; loaded `pp=2, mtp_mode_effective=on` | greedy native PP MTP | 59.8 / 50.3 / 110.4 |
| `a3b-ar` | hiptrx, one card of B,C (same lane, after PP2) | Qwen3.6-35B-A3B **mq4r** (18,700,048,128 B) | stderr `drafter=off`; loaded `arch=qwen3_5_moe, pp=1, mtp_mode_effective=off` | arch-path smoke: it must load, run and commit the pinned ids | 11.5 / 26.7 / 38.4 |
| `xtx-ar` | hipx, xtx | Flash-Next mq4 split base (same file, hipx path) + `.ple-q4` | stderr `qwen4 D2 default (N host split layers): off (developer.qwen4_d2_default=false)`; loaded `arch=qwen4, mtp_mode_effective=off` | D2 forced OFF; shipped KV/state defaults | 116.9 / 34.9 / 152.5 (explicit-q8 variant, see XTX D2 policy) |
| `xtx-d2` | hipx, xtx | same | stderr `qwen4 D2 default (N host split layers): FORCED ON`, `CPU split (D2) armed on N layers` (N >= 1) | none (no equality invariant) | not measured |

PP2 AR and PP2 MTP each run in their own load because there is no request-level control that turns an
MTP-loaded daemon into the AR baseline; the same holds for fn1 and EP2.

Route proofs are positive checks (`stderr_require`, `loaded_expect`) rather than a global rejection of
`no closed route plan`: PP2, EP2 and A3B legitimately print that line before running their intended
routes. `stderr_fail_patterns` is empty; kernel faults are the hold watcher's job.

## Invariants

Equality pairs checked by `verdict`, each over the committed token ids (a failing pair is a FAIL naming
both cells and the first differing position):

| Pair | Why it is contractual |
|---|---|
| `fn1-ar == fn1-mtp` | Greedy native MTP equals AR under the default fp8 QSA state (CHANGELOG, 0.4.3 entry "Flash-Next (qwen4) on gfx1201: the default fp8 QSA K/V state now has a KLD gate", line 170 at `3866d41b1`: "greedy native MTP equals AR under fp8"). |
| `fn1-ar == ep2-ar` | EP output on exact gfx1201 is bit-identical to the same build on one card (`docs/multi-gpu.md` "Exactness", lines 356-365 at `3866d41b1`; the evidence is fixture-bound to exact gfx1201 at EP2/EP4; CHANGELOG line 524). |
| `fn1-mtp == ep2-mtp` | Same section: on EP2 the native-MTP greedy tokens, window results and trunk/head/drafter state match one card. |
| `ep2-ar == ep2-mtp` | Same section: greedy MTP equals AR on the mesh. Implied by the two pairs above; kept so a break is attributed to the EP pair directly. |
| `pp2-ar == pp2-mtp` | `docs/multi-gpu.md` lines 849-854 give the PP native-MTP exactness contract as equality with single-card native MTP on the same artifact, and the CHANGELOG records mq2-xts PP2 byte-identical to one card for AR and greedy MTP (lines 148-152 at `3866d41b1`) while single-card greedy MTP equals AR (line 170). So PP2 AR equals PP2 MTP transitively. The PP section does not word PP AR == PP MTP as a separate clause. |

Never added: any pair across architectures or quantizations. `pp2-* == fn1-*` is not an invariant: PP2
runs mq2-xts and fn1 runs mq4, different weights, different ids after the shared prefix. A3B and XTX
cells are in no pair (XTX D2 versus PCIe arithmetic is not contractual either).

Equality across builds (fixed build versus broken build, today versus last week) is not claimed by an
invariant; the pins carry that per cell, and a changed id list is CHANGED, not a silent update.

## Artifact exceptions and PLE q4

Three deliberate departures from "what the registry ships", all approved, all recorded in the plan's
`_comment` and each load's `note`:

- **PP2 uses the mq2-xts export.** The registry-default mq4 base alone (70.9 GB) is larger than the two
  cards' aggregate VRAM (68.4 GB), and PP keeps every band's experts resident (no host-mapped experts).
  The mq2-xts file is the prior-art export, not necessarily the newer registry split mq2 pin.
- **A3B uses the canonical mq4r.** The registry default `.mq4p` is not present on the lane host. The cell
  covers the `qwen3_5_moe` architecture path, not the missing default artifact.
- **PLE is pinned to q4 on every Flash-Next load** (`HIPFIRE_PLE_FORMAT=q4`; the gfx1201 default is q8).
  These are smoke pins, not claims about the full shipped defaults. KV mode, recurrent-state quantization
  and MTP depth are omitted from every load so the shipped defaults apply (QSA fp8 / GDN q8 and native
  MTP K=7 on R9700; A3B resolves auto KV to Q8).

`HIPFIRE_SESSION_CACHE_BYTES=0` is set on every load so a session snapshot can never carry state between
cells. `HIPFIRE_EMIT_TOKEN_IDS=1`, `HIPFIRE_NO_REGISTRY_FETCH=1` and a fresh `HOME` are installed by
`hold` for every daemon.

Model paths are `${SMOKE_FN_MQ4_SPLIT}`, `${SMOKE_FN_MQ2}` and `${SMOKE_A3B}` in the plan, resolved from
the per-host model-env file; `model_bytes` fails a load before spawn if the file size differs.

## XTX D2 policy

Main ruling (2026-10-18): the XTX bus has dropped under D2-armed loads for reasons not yet explained
(eight known drops that day, causation unproven). Therefore:

- `xtx-ar` runs on every land with D2 **forced off** (`HIPFIRE_QWEN4_D2_DEFAULT=0`).
- D2 is the **shipped mq4 XTX default** (a measured per-load choice). `xtx-ar` therefore does **not**
  cover the shipped D2 execution route; it covers the PCIe route on the same artifact.
- `xtx-d2` is disabled (`enabled: false`) and runs only when the land touches the D2 / decode-split
  paths: `run_if_paths` on `crates/hipfire-dispatch/src/cpu_split*`, `crates/hipfire-cpu/`,
  `crates/hipfire-arch-qwen4/src/d2_default.rs` and `.../gpu_forward/cpu_splice.rs`, and `run_if_diff`
  hunks in `gpu_forward.rs` (D2, cpu_split, routed_view), `carriers.rs` (D2, CPU split) and
  `crates/hip-bridge/src/` (mapped / host-register / cp_wait). Its route proofs are source-derived and have
  never been run on a GPU; the first run of `xtx-d2` is an unvalidated first run.
- `kv_mode` and `state_quant` are not set on the XTX loads (shipped gfx1100 defaults). The recorded
  validation (hold j1812, 153 s, 32 ids) used explicit q8 for both; the first land smoked with the final
  plan pins the shipped-default ids.
- A watcher hit or lost device on hipx makes the lane INCOMPLETE: stop, do not retry inside the campaign,
  report.

## Plan schema

`plan.json` is runner plan version 1; the runner's module docstring (`scripts/post_land_smoke.py`) is the
schema reference. Beyond the per-load fields (`model`, `model_bytes`, `load_params`, `env`, `deadline_s`,
`stderr_require`, `loaded_expect`, `enabled`, `run_if_paths`, `run_if_diff`, `cells`) and `invariants`:

- `model` may use `$VAR`, `${VAR}` and a leading `~`; an unresolved `$` fails the load before spawn
  (`unresolved model path: ...`, shown as UNRESOLVED by `plan` and `lanes`). `model_bytes` is checked
  against the expanded file's size (`model size mismatch: want N got M`).
- `kernel_fault_patterns` (optional): case-insensitive regexes the hold watcher matches on the kernel log;
  the default is the amdgpu fault, reset, ring timeout, bus-loss, PCIe AER and kernel-oops list.
- `lanes`: `{name, host, cards, mem_gb, hostmem_gb, hostmem_lock, max_secs, loads}`; every load belongs to
  exactly one lane. `host` is a free label matched against the orchestrator env file.

Runner subcommands: `plan` (what would run and why), `lanes` (JSON of lanes with at least one selected
load, plus their `fabric_args`), `run` (loads, no watcher), `hold` (what runs inside a fabric hold: kernel
watcher plus the lane's loads), `verdict`, `pin`.

## Running it

Env file (bash `KEY=VALUE`, sourced). Example for hiptrx (local) plus hipx (remote):

```bash
# /home/kaden/qcal/post-land-smoke.env
REPO=/home/kaden/ClaudeCode/warpfront/hipfire-beta   # the smoke only runs git rev-parse/show/diff/fetch here
BINS_ROOT=/home/kaden/qcal/release-0.4.3/beta-bins    # <full-sha>/{daemon,hipfire,SHA256SUMS[,LANDED.json]}
PINS=/home/kaden/qcal/post-land-smoke-state/pins.json
OUT_ROOT=/home/kaden/qcal/post-land-smoke-runs
FABRIC_RUN=/home/kaden/qcal/locks/fabric/fabric-run.sh
OWNER=PostLandSmoke
PRIORITY=P1
WAIT_SECS=3600
LOCAL_HOST=hiptrx
LOCAL_MODEL_ENV=/home/kaden/qcal/post-land-smoke-state/hiptrx-models.env
LOCAL_HOME_MODELS=/home/kaden/.hipfire/models
REMOTE_HIPX_SSH=hipx
REMOTE_HIPX_ROOT=/home/kaden/qcal/post-land-smoke      # absolute, must exist; one directory per run is created under it
REMOTE_HIPX_FABRIC_RUN=/home/kaden/qcal/locks/fabric/fabric-run.sh
REMOTE_HIPX_MODEL_ENV=/home/kaden/qcal/post-land-smoke/hipx-models.env
REMOTE_HIPX_HOME_MODELS=/home/kaden/.hipfire/models
```

Model-env files are `KEY=VALUE` lines (blank lines and `#` comments allowed, no quoting or expansion).
`hiptrx-models.env`:

```
SMOKE_FN_MQ4_SPLIT=/home/kaden/qcal/release-0.4.3/mq4split-stage/split/qwen3.8-flash-next-base.mq4-xts
SMOKE_FN_MQ2=/home/kaden/qcal/release-0.4.3/consumer24/artifacts/export/c24h6306-d2/qwen3.8-flash-next.mq2-xts
SMOKE_A3B=/home/kaden/.hipfire/models/qwen3.6-35b-a3b.mq4r
```

`hipx-models.env` (on hipx) has only
`SMOKE_FN_MQ4_SPLIT=/home/kaden/qcal/release-0.4.3/mq4split-pull/hipx-q4/home/models/qwen3.8-flash-next-base.mq4-xts`.
The `.ple-q4` sidecar must sit next to each Flash-Next file. Do not copy models and do not pull; see
`MODEL-STORE.md`.

```bash
scripts/post-land-smoke.sh --env /home/kaden/qcal/post-land-smoke.env --land <sha> --dry-run   # lanes + exact fabric commands, submits nothing
scripts/post-land-smoke.sh --env /home/kaden/qcal/post-land-smoke.env --land <sha> [--prev <sha>] [--out DIR] [--skip-host H]
scripts/post-land-smoke.sh --env /home/kaden/qcal/post-land-smoke.env --watch 120               # smoke every new origin/beta that has LANDED.json
```

`--land` steps: resolve the full sha; `PREV` is the last pinned sha in the pins history, else `LAND^`;
make `OUT_ROOT/<sha>-<utc>`; extract that land's own `scripts/post_land_smoke.py` and
`scripts/post-land-smoke/plan.json` with `git show` (each land is smoked with its own runner and plan);
verify `BINS_ROOT/<sha>/SHA256SUMS` with `sha256sum -c`, hardlink `daemon` and `hipfire` into `RUN/bin`
(never copy; a cross-device link fails) and verify again; write `changed.txt` and `land.diff`
(`git diff --name-only` / `-U0`); compute `lanes.json` and `plan.txt`; submit every lane; wait; merge with
`verdict`; apply the pin policy; write `SUMMARY.txt`. The last stdout line is
`POST-LAND-SMOKE <sha> <OVERALL> <RUN>/VERDICT.json`.

A lane whose host has no configuration is not submitted and the script exits 2, unless `--skip-host H`
is given; then that host's cells are FAIL "not run". `--watch` is single-instance (`flock` on
`OUT_ROOT/watch.lock`), fetches `origin beta`, and runs `--land` for a head that differs from
`OUT_ROOT/last-smoked` and has `BINS_ROOT/<sha>/LANDED.json`, then records it in `last-smoked`.

Remote lanes: the script builds `RUN/remote-H/` (runner, plan, daemon hardlink, `changed.txt`, `land.diff`,
generated `launch-<lane>.sh`, `verify.sh`, `manifest.sh`, `MANIFEST.sha256`), rsyncs it to
`REMOTE_H_ROOT/<run-id>/`, runs `ssh HOST <dir>/verify.sh` (a mismatch aborts before any submission), then
runs `ssh HOST <dir>/launch-<lane>.sh` in the background. Every remote action is a generated script
invoked by absolute path, never a free-text command. After the lane ends the results are rsynced back to
`RUN/lanes/<lane>/` (`rsync -a`, so each daemon HOME's `.hipfire/models` stays a symlink, with
`--exclude='.hipfire/models' --exclude='*.hfq' --exclude='*.mq4*'` so a model store or model file is never
pulled; never `scp -r`, which follows symlinks) and checked against the remote's `RESULTS-<lane>.sha256`;
results that do not verify are not judged. `SSH_BIN` and `RSYNC_BIN` override `ssh` / `rsync` (the tests
use fakes).

`HOLD_EXTRA_ARGS` in the env file appends words to every `hold` command (for example
`--watch-cmd sleep 3600` in CPU tests); leave it empty on real hosts.

## Fabric hold recipe

Each lane is one fabric hold (`class correct`, shared card lock); `plan.json` carries the admission numbers
and the runner prints them in `lanes` as `fabric_args`:

| Lane | Host | Fabric arguments |
|---|---|---|
| `hiptrx-fn1` | hiptrx | `--class correct --card A --mem-gb 30.3 --hostmem-gb 54 --hostmem-lock --max-secs 600` (fn1-ar, fn1-mtp) |
| `hiptrx-ep2` | hiptrx | `--class correct --card A,E --mem-gb 30.3 --hostmem-gb 54 --hostmem-lock --max-secs 600` (ep2-ar, ep2-mtp) |
| `hiptrx-pp-a3b` | hiptrx | `--class correct --card B,C --mem-gb 28 --hostmem-gb 4 --hostmem-lock --max-secs 600` (pp2-ar, pp2-mtp, a3b-ar) |
| `hipx-xtx` | hipx | `--class correct --card xtx --mem-gb 22.4 --hostmem-gb 64 --hostmem-lock --max-secs 600` (xtx-ar) |
| `hipx-xtx-d2` | hipx | same arguments (xtx-d2; submitted only when selected) |

Within a lane the loads run one after another. Every hold is capped at 600 s, and the runner rejects a plan
whose lane could outlast its cap: the sum over the lane's loads of `deadline_s` plus the 5 s kill grace,
plus 15 s of slack, must not exceed `max_secs`. A slow load therefore ends on its own deadline (a FAIL
naming it) and never on the fabric cap, which would make the lane INCOMPLETE. That rule is why fn1 and EP2
are two lanes and `xtx-d2` has its own: `hiptrx-fn1` and `hiptrx-ep2` both need card A and the Flash-Next
host budget (never two host-mapped Flash-Next loads at once), so fabric runs them one after the other.

The orchestrator adds `--priority P1 --owner OWNER --wait-secs WAIT_SECS`. P1 because a post-land check
that must finish in about ten minutes cannot wait behind hours of P2 queue (observed P2 waits on the
shared fabric were 38 to 81 minutes); P0 stays reserved for release gates. `hostmem` for the Flash-Next
lanes comes from measured pinned peaks (fn1 about 42.7 GiB, EP2 about 50.1 GiB) plus headroom; the PP2 lane
is resident-only (GTT about 0.4 GiB, RssAnon about 1.2 GiB). The PP2 lane and one Flash-Next lane together
declare 58 GiB host, below the roughly 61.4 GiB TTM cap; fabric still admits from live availability.

The command inside the hold is
`env <model vars> python3 post_land_smoke.py hold --plan ... --lane L --daemon RUN/bin/daemon --out RUN/lanes/L --changed-paths ... --diff ... --home-models DIR`.
`hold` refuses to run outside a fabric hold (`FABRIC_JOB` unset), starts the kernel watcher, runs the
lane's selected loads exactly as `run --only` would, and writes `hold.json` (status, watcher pid/pgid and
whether its group is gone, hits, `FABRIC_CARDS`, `ROCR_VISIBLE_DEVICES`, `HIPFIRE_DEVICES`). Each daemon
gets a fresh `HOME` whose `.hipfire/models` is a symlink to the canonical store.

## Budget

Execution time excluding queue. Warm-cache times are the recipe measurements (single runs); the cold
times are from the first end-to-end run (see Known state):

| Lane | Sequence | Warm | Deadlines (worst case) |
|---|---|---|---|
| `hiptrx-fn1` | fn1-ar, fn1-mtp | 191.3 s | 280 + 280 s (585 s) |
| `hiptrx-ep2` | ep2-ar, ep2-mtp (fixed build) | 337.8 s | 270 + 300 s (595 s) |
| `hiptrx-pp-a3b` | PP2 pair, then A3B | 234 s (cold run: 245 s) | 180 + 180 + 90 s (480 s) |
| `hipx-xtx` | one load | 153 s (cold run: 261 s) | 540 s (560 s) |

The deadlines are sized for a cold page cache, not the warm recipe runs. A post-land smoke usually finds the
70.9 GB mq4 split cold, and its host-mapped expert load then reads through page faults: on hipx the cold
`xtx-ar` load took 222.9 s (`weights` 216.7 s, storage read 69.8 GB at 0.32 GB/s, 564,872 major faults),
and on hiptrx, under host memory pressure, fn1 and EP2 loads did not reach `loaded` inside the earlier
180 / 210 s deadlines. Lanes on disjoint cards run concurrently; the two Flash-Next lanes share card A, so
hiptrx needs about 9 minutes of hold time warm and up to about 20 minutes when every load runs to its
deadline.

## Verdicts and exit codes

`verdict` merges all lane directories (each needs `run.json`) against the pins:

| Overall | Exit | Meaning |
|---|---|---|
| PASS | 0 | every selected cell passed its proofs, ids are NEW or MATCH the pin, every invariant holds |
| FAIL | 1 | a cell failed (load error, missing route proof, wrong token count, deadline, stderr/loaded mismatch), an invariant broke, or a lane did not run ("not run") |
| CHANGED | 3 | all cells ran clean but at least one id list differs from its pin |
| INCOMPLETE | 4 | a lane's `hold.json` is `incomplete` (kernel fault or watcher died) or still `running`; every cell of its loads is FAIL `INCOMPLETE: <reason>`; wins over FAIL |
| usage / infrastructure | 2 | bad arguments or config, binary checksum mismatch, no lane produced results, or the run was interrupted |

SKIPPED cells (a path-filtered load whose filter did not match) are neutral and never pinned. Results
fetched from a remote host that do not match its manifest are discarded and listed under "infrastructure
notes" in `SUMMARY.txt`; the lane's cells are then "not run".

## Pins

`PINS` is one JSON file: `cells.<cell>.ids` (the 32 ids), `ids_md5`, the sha and time of the pin and the
reason, plus `history` (one entry per pin run). Pin policy applied by the orchestrator:

- PASS: `pin` writes NEW and MATCH cells.
- FAIL (not INCOMPLETE): `pin --allow-failed-cells "post-land FAIL at <sha9>: <failed cells>"` pins the
  passing NEW/MATCH cells and records the reason and failures in the history. A FAIL never pins a failed
  cell and never accepts a CHANGED cell.
- CHANGED: nothing is pinned. After reviewing the id diff the owner accepts it:
  `python3 RUN/post_land_smoke.py pin --verdict RUN/VERDICT.json --pins PINS --accept-changed "<reason>"`.
- INCOMPLETE: never pinned.

The pins file is written atomically and `verdict` plus `pin` run under `flock` on `PINS.lock`.

## Safety

- The watcher is `journalctl -k -f` in its own session and process group, with stdin `/dev/null` and no
  inherited fd above 2. It is killed as a group on every exit path (normal end, exception, SIGTERM, SIGINT,
  SIGHUP); afterwards the hold checks the group is gone and records it in `hold.json`. A fault-pattern hit
  or an unexpectedly exiting watcher writes `STOP` in the lane directory, kills the running daemon's
  process group and runs no further loads: the lane is INCOMPLETE.
- **INCOMPLETE means stop and report.** Do not rerun inside a campaign; a kernel fault or lost device on
  these cards needs the owner of the host.
- Never kill by pattern (no `pkill -f`, `killall`, `kill $(pgrep ...)`): other holds on the same host run
  their own `journalctl -k -f` watchers and daemons. The orchestrator records the PID and PGID it launched
  (`RUN/lanes/<lane>.pid`, `PID PGID` with PID equal to PGID, verified at launch) and, on INT/TERM/HUP,
  forwards SIGTERM only to those groups and to its recorded ssh PIDs; fabric then tears down its own child.
  A lane that finishes starting after the signal is signalled as soon as it is recorded, and no further
  lane is submitted.
- Signal forwarding after start is LOCAL only: recorded local process groups and recorded local `ssh` client
  PIDs. Nothing is signalled on a remote host, and queued or remote fabric jobs are cancelled by the owner by
  exact fabric job id: `fabric-run.sh --cancel <ID>`. Job ids appear in `SUMMARY.txt` and in
  `RUN/lanes/<lane>.fabric.log` (`queued as jNNNN`).
- Only owned, recorded groups are signalled, and never after their leader is reaped: a daemon or watcher
  leader stays an unreaped zombie (waited with `waitid(WEXITED|WNOWAIT)`) until the last `killpg`, because a
  reaped PID/PGID can be reused by an unrelated group. A `killpg(pgid, 0)` probe is not an ownership proof;
  it is used only after the reap, to report `watcher.gone`.
- Ownership is checked before every group signal, the first SIGTERM included. If the leader is no longer ours
  (ECHILD, or reaped elsewhere) the group is NOT signalled, probed or retried by PID: the load records
  `exit_code` null (or the real rc, if one was known) and the error `process group ownership lost; not
  signalled` (FAIL), the watcher records `ownership_lost: true` and `gone: false`, and the hold is INCOMPLETE.
- An interrupted run is reported as INTERRUPTED (exit 2) and pins nothing.
- No step downloads or copies models, or writes outside `RUN`, the remote run directory and `PINS`.

## Known state (captures at 3866d41b1, end-to-end runs at a47d07326)

- **EP2 fails both cells** (`ep2-ar`, `ep2-mtp`): the sealed-MoE identity error above, no token committed.
  Fix `5dfe1517b` (a separately built binary, validated on both EP2 cells, not on `beta`) is pending. Until
  it lands the smoke of `beta` is FAIL. With the fixed build, `fn1-ar`, `fn1-mtp`, `ep2-ar` and `ep2-mtp`
  all commit the same 32 ids (md5 `73888ee0996542d34299857245d53bd7`); `pp2-*` commit md5
  `c6f894f3d2c2ebb38326335d00418ff1`; `a3b-ar` commits md5 `9391f77ee9b35147e6766bde99b5c540` (md5 over the
  compact JSON array of ids, recipe format; `ids_md5` in the verdict and pins is a different encoding, the
  md5 of the comma-joined ids).
- **`a3b-ar` re-pinned at the 0.4.3 RC (2026-10-23).** The A3B FP8 packet `c4c058a20f` changed the
  `.mq4r` prefill numerics deterministically (`HIPFIRE_GDN_MOE_PRE_FUSED` flips a near-tie at token 13;
  `HIPFIRE_FA_PREFILL_PM` with Q prep adds WT2 prefill KLD +0.0034, CI [+0.0009, +0.0058], within the A3B
  gate; decode unchanged). By Main's decision the cell is pinned to the RC ids, `ids_md5`
  `f4a33e160a48fb46728e640b6b488a15` (recipe-format md5 `7129643965bf3a7399ef50e6357851be`), from the RC run
  at `8949a6bea3`; the old `9391f77e…` above is the pre-packet capture. Evidence:
  `release-0.4.3/a3b-mq4r-divergence/`.
- **The PP2-MTP prompt-dependent window bug is not covered.** A known `beta` error
  (`forced-token advance: ... row capture cannot be resized while a window is live`) depends on the
  prompt and was not triggered by this prompt in any observed PP2 sample, so `pp2-mtp` passes on a build
  that still has it.
- `xtx-ar` was first GPU-validated in hold j1812 (explicit-q8 variant).
- `xtx-d2` has never run.
- **First end-to-end runs of the orchestrator** (beta `a47d07326` product binaries, 2026-10-18/19).
  - In the first run, fn1 and EP2 shared one 600 s lane. The cold fn1 loads hit their old 180 s deadlines,
    and the fabric cap cut the lane (INCOMPLETE). That run led to the lane split, the cold-cache deadlines and
    the plan check above.
  - With the current plan the hiptrx lanes ran clean, with no kernel-watcher line. `fn1-ar`, `fn1-mtp`,
    `pp2-*` and `a3b-ar` commit exactly the ids listed above.
  - Both EP2 cells FAIL with the sealed-MoE error, as expected until `5dfe1517b` lands.
  - `xtx-ar` (D2 off, shipped KV/state defaults, from the first run's hipx lane) commits the same 32 ids as
    fn1.
