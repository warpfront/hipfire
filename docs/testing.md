# Testing playbook

**Lifecycle:** shipped / reference. Moved out of `AGENTS.md` on 2026-11-15
(section numbers kept; § 0 hard rules and the § 4 DDTree removal, now under
"Deprecations and removals", stay in [`AGENTS.md`](../AGENTS.md)).

**Audience:** agents (or humans) running smoke / perf / correctness
tests on hipfire — particularly the MQ3 sub-4-bit Magnum Quant family,
the DFlash MQ3 cross-quant matrix, and the DFlash draft pull /
prompt-shape adaptation paths.

**Companion docs:** [`AGENTS.md`](../AGENTS.md) holds the project-wide hard
rules (§ 0) and deprecation status. [`docs/VALIDATION.md`](VALIDATION.md) is
the authority on which validation route a given change owes.
[`docs/methodology/lessons-0.4.5.md`](methodology/lessons-0.4.5.md) is the
authority on measurement hygiene and gate discipline (it supersedes older
rules here). Fixture identities are data in
[`fixtures/pins.toml`](../fixtures/pins.toml), explained in
[`docs/fixtures.md`](fixtures.md). This file holds the *testing playbook*: how
to verify the engine works, what to measure, what counts as pass/fail.

---

## Default behavior to be aware of

- **MQ3 is production on gfx11** (`gfx1100/1101/1102/1150/1151`) and
  gfx12 (`gfx1200/1201`). On gfx10 (RDNA1/2, wave32), MQ3 weights still
  load and run via the wave32 GEMV fallback — correct, just slower
  prefill. Wave64 GCN/CDNA (gfx906, gfx94x) has no MQ3 GEMV: a
  Qwen3.5/3.6 load whose weights use MQ3 (MQ3G256, MQ3-Lloyd, MQ3G256V2)
  or any other format with a wave32-only GEMV (e.g. MQ4G256V2, the
  `.mq4-xts` tier) is refused at load there (#1136); other model families
  are not gated at load for this. MoE/A3B + MQ3 is refused at load time
  (no MoE-branched WMMA path).
- **`dflash_mode=off` is the schema default.** Any test exercising DFlash
  on a model without a registry DFlash policy still needs
  `hipfire config set dflash_mode auto` or
  `HIPFIRE_DFLASH_DRAFT=<path>` first. The one exception is the
  Qwen3.8-27B tiers that declare a paired draft (`qwen3.8:27b`, `-mq3*`,
  `-mq4*`, including the canonical `qwen3.8:27b-mq4-xts` fixture): the
  registry layer sets `speculation.dflash=auto`, so a present paired file
  activates DFlash with no config, an absent file keeps MTP `auto` (which is
  AR here: Qwen35 declares no measured AR floor; `--spec mtp` opts in), and an
  explicit `dflash_mode off` (canonical `speculation.dflash=off`) is the
  kill switch. `dflash_block` stays `auto` (L16 singleton, L8 on a
  `serve.vmm_batch` serve with ≥ 4 lanes, IU4 verify gates unchanged).
  DFlash greedy may differ from AR near argmax ties; judge it against the
  MTP arm and the C1 == C8 state oracle, not byte-equality with AR.

---

## 1 · Setup (one-time)

### Pull the model + draft you want to test

`hipfire pull <tag>` fetches the target plus its registry-declared
DFlash draft sidecar (same mechanism as the MTP/DSpark sidecars):

```bash
# Canonical acceptance / dense validation fixture (Qwen3.8-27B MQ4XTS):
hipfire pull qwen3.8:27b-mq4-xts   # ~15 GB target + MQ4 DFlash draft sidecar
# lands at ~/.hipfire/models/qwen3.8-27b.mq4-xts
# measured draft identity (acceptance/perf pin):
#   a local copy of the registry draft (same bytes as the
#   qwen38-27b-dflash-mq4.hfq sidecar)
#   (pin:draft/qwen38-27b-dflash-mq4.hfq; see docs/fixtures.md)

# Smaller / faster smoke only (not the acceptance fixture):
hipfire pull qwen3.5:9b            # 5.3 GB target + 0.55 GB DFlash draft sidecar
hipfire pull qwen3.5:4b            # even smaller bring-up smoke
```

Standalone `*-draft` tags (`hipfire pull qwen3.8:27b-draft`) still work —
they address the registry draft file for anyone who wants the draft alone.
Legacy `qwen3.{5,6}:{9b,27b}-draft` tags remain pullable.

Files land at `~/.hipfire/models/<canonical-name>`.
**Do not rename.** Load resolves the draft by its registry-declared
filename; renaming breaks the pairing — an explicit `dflash_mode auto` then
runs AR (one warning line) and `on` fails the load; a Qwen3.8-27B tier
(registry `auto`) keeps MTP `auto` when the draft file is absent, and Qwen35's
`auto` is AR (no measured AR floor; `--spec mtp` opts in).

### Verify hashes after pull (paranoid mode)

For the **canonical dense fixture** (`qwen3.8-27b.mq4-xts`), verify
SHA-256 against `pin:model/qwen3.8-27b.mq4-xts` with
`scripts/check_fixture.sh [PATH]` — do not trust filename alone.

Registry-present smoke / historical artifacts (still in
`registry/models.json`; md5 pins `pin:draft/qwen35-9b-dflash-mq4.hfq` and
`pin:draft/qwen35-27b-dflash-mq4.hfq`, decimal sizes in their `size_mb`):

```
scripts/check-pins.py verify draft/qwen35-9b-dflash-mq4.hfq  PATH
scripts/check-pins.py verify draft/qwen35-27b-dflash-mq4.hfq PATH
```

Any mismatch = re-pull or report. The dense `qwen3.6:27b` (`qwen3.6-27b.mq4`)
and its draft `qwen3.6:27b-draft` (`qwen36-27b-dflash-mq4.hfq`) are retired in
0.5.0: `hipfire pull` refuses them (replacement `qwen3.8:27b`), and a local copy
verified against the compiled `RETIRED_MODELS` SHA-256 pin still loads.

> **Sizes here are decimal (MB = 10⁶ bytes, GB = 10⁹ bytes), matching
> Hugging Face's reported sizes and the `hipfire pull` progress bar.**
> `ls -lh` / `du -h` report **binary** units (MiB = 2²⁰, GiB = 2³⁰) but
> *label them* `M`/`G`, so a 919 MB file shows as `877M` in `ls`
> (919 × 10⁶ ÷ 2²⁰ ≈ 877 MiB) and a 15 GB file shows as `14G`. This is
> not a size mismatch or a truncated download — it's the same byte
> count in two unit systems. When a download looks "smaller than the
> manifest," divide by 1.048576 (MB→MiB) or 1.073742 (GB→GiB) before
> assuming corruption; confirm with the md5, not the human-readable
> size.

### Build from source (if you're on a dev branch)

```bash
cargo build --release
```

That is the whole command. The daemon is a crate (`hipfire-daemon`, binary
`target/release/daemon`), not a `--example`, and the product no longer needs a
feature incantation. If you find yourself typing `--example daemon`, you are
following a pre-saddle instruction.

---

## 2 · Test surface (quant formats, DFlash, prompt-shape)

### A. MQ3 (sub-4-bit Magnum Quant)

MQ3 = FWHT-rotated 3-bit weight format, 104 B/group (3.25 bpw vs MQ4's
4 bpw at 136 B/group). What's wired:

- **K4-unrolled GEMV decode + fused residual** on gfx1100 — decode matches
  MQ4 within ~2% (9B ~141 vs MQ4 ~129 tok/s).
- **WMMA prefill family** (`gemm_qkvza/qkv/gate_up/residual hfq3`), arch-gated
  to gfx11 wave32 WMMA; gfx12 ships a K4 variant.
- **DFlash cross-quant matrix** — MQ3↔MQ3, MQ3↔MQ4, MQ4↔MQ3 all valid for
  dense models. MoE/A3B + MQ3 is refused at daemon load.

Sweep harness (quality + perf, md5-stamped): `./scripts/mq3-mq2-sweep.sh`.

### B. Cache invalidation on model swap

`Gpu::unload_model` drains `mmq_screen_cache` + `fp16_shadow_cache` and tears
down captured hipGraphs (verify, replay, AR forward) — a pointer-keyed
silent-corruption class. **Smoke test:** a rapid `hipfire serve` model-swap
loop must NOT emit garbage on the new model's first decode.

### C. Prompt-shape adaptation (default ON)

Engine-side `\n{3,}` → `\n\n` collapse before tokenize (drops the rare BPE
token 1358 `\n\n\n` for the HOT token 271 `\n\n` on Qwen3.5/3.6 vocab).
**Default ON** — worth +14–27% tok/s on PEP-8-style code prompts containing
`\n{3,}`, zero effect otherwise, zero correctness cost. Opt out only when raw
`\n{3,}` whitespace is semantically load-bearing:

- Env: `HIPFIRE_NORMALIZE_PROMPT=0`
- TUI: `hipfire config set prompt_normalize false`
- Per-model: `hipfire config qwen3.8:27b-mq4-xts set prompt_normalize false`


**Verify:** see §3 prompt-shape A/B test.

### D. Token-heat diagnostic

`HIPFIRE_PROMPT_TOKEN_HEAT=1` triggers `Tokenizer::dump_prompt_heat()` at every
encode site (stderr pretty, or stdout JSON with `HIPFIRE_PROMPT_HEAT_JSON=1`).
Standalone: `cargo run --release -p hipfire-runtime --example encode_prompt -- MODEL.hfq PROMPT.txt --heat`. (An example, so it is not in `target/release/` after a plain `cargo build --release`.)

### E. DFlash draft endpoints (HuggingFace)

**Current acceptance fixture** (dense Qwen3.8-27B MQ4XTS):

- Target: `hipfire-models/qwen3.8-27b` / `qwen3.8-27b.mq4-xts`
  (registry tag `qwen3.8:27b-mq4-xts` → `~/.hipfire/models/qwen3.8-27b.mq4-xts`)
- Registry draft sidecar: `qwen38-27b-dflash-mq4.hfq`
  (`hipfire pull qwen3.8:27b-mq4-xts` or `hipfire pull qwen3.8:27b-draft`)
- Measured draft identity (acceptance/perf pin):
  a local copy of the registry draft (same bytes as the
  `qwen38-27b-dflash-mq4.hfq` sidecar; `pin:draft/qwen38-27b-dflash-mq4.hfq`,
  see [`docs/fixtures.md`](fixtures.md))

Still pullable (smaller smoke / historical):

- `hipfire-models/qwen3.5-9b/qwen35-9b-dflash-mq4.hfq`
- `hipfire-models/qwen3.5-27b/qwen35-27b-dflash-mq4.hfq`
(The 3.6 27B draft and target, `qwen36-27b-dflash-mq4.hfq` /
`qwen3.6-27b.mq4`, are retired: no longer pullable, see § "Verify hashes".)

`hipfire pull <target>` fetches the target plus its draft sidecar;
standalone drafts stay pullable via `hipfire pull qwen3.8:27b-draft`
(and legacy `qwen3.5:{9b,27b}-draft`).

---

## 3 · Smoke tests (run these to validate)

### 3.1 — `hipfire bench` is the benchmark tool

**Use `hipfire bench`. Do not bench through `examples/dflash_spec_demo`.**

It drives the model through the native daemon protocol — the same path a user's
request takes — and owns warmup, repetition and reporting:

```bash
hipfire bench <model> --runs 5 --warmups 3 --max-tokens 128 --json
```

| flag | why it matters |
|---|---|
| `--runs` | measurement runs (default 5) |
| `--warmups` | discarded warmup runs (default 10) — DPM/thermal state settles here |
| `--json` | machine-readable; keep the whole object, the `samples` array is the evidence |
| `--spec` | `off`/`dflash`/`mtp`/`ngram`/`dspark`/`auto` |
| `--backend` | `noslots` (sequential daemon) / `slots` / `batch` / `both` |
| `--workload` | `stateless` / `multiturn` / `both` |
| `--prompt-file PATH` | verbatim prompt bytes for the run; JSON records `prompt_tokens`/`prompt_md5`/`prompt_chars` plus a `warnings` caveat below 256 tokens |
| `--kv-mode`, `--kv-backend` | KV format and allocator |
| `--reasoning-on` | off by default: a reasoning model cannot close `<think>` inside the token budget, and the daemon fails that turn closed |

Pin `--backend` and `--workload` explicitly for any A/B. The default is `both`,
which measures two things at once and is not a comparison.

**Do not use `--concurrency` for a single-stream comparison.** It is the
closed-loop serving sweep (K lanes, unique request tags, an overlap check per
K>1 cell) and answers the concurrent-serving question, not the single-stream
one. A C>1 claim is measured on the served path under the rules of
[`docs/methodology/lessons-0.4.5.md`](methodology/lessons-0.4.5.md) § 4
(distinct prompts per stream, EOS-on and fixed-length cells reported
separately, at least 4 interleaved fresh loads per arm, bootstrap CIs).

Still true regardless of tool: run in a fresh process, and record the model md5
and both binary md5s with any number you report. There is no universal noise
band: characterize it on the fixture with an A/A pair before calling a delta
real. The older figures (gfx1100 within-session ±10–15 % from DPM/thermal
state; `HIPFIRE_VERIFY_GRAPH=0` ~0.1 % spread versus 1.5–3 % with graph capture
on) are historical observations on that card, not bands to reuse.

### 3.2 — Prompt-shape A/B

Prompt structure moves tau by up to 17 %, so an A/B must differ in exactly one
thing. Use a committed prompt file and record its md5:

```bash
HIPFIRE_NORMALIZE_PROMPT=0 hipfire bench <model> --runs 5 --json   # A
HIPFIRE_NORMALIZE_PROMPT=1 hipfire bench <model> --runs 5 --json   # B
```

Canonical bench config (post-2026-04-26) for 27B-3.5 LRU code DFlash
(from `CLAUDE.md`; the prompt-structure measurements behind it are in
[`docs/perf-checkpoints/2026-11-15-agents-md-measurements.md`](perf-checkpoints/2026-11-15-agents-md-measurements.md)):

```
max=256 --no-chatml --kv-mode q8
PEP-8 strict prompt (\n\n\n between top-level defs)
prompt_normalize=true (default)
```

### 3.3 — Speculation comparison

```bash
hipfire bench <model> --spec off    --runs 5 --backend noslots --json
hipfire bench <model> --spec dflash --runs 5 --backend noslots --json
```

Report the median with the run count and a bootstrap CI of the difference, plus
GPU model, ROCm version, model md5 and binary md5.
Older pre-q8 DFlash numbers are not authoritative for current perf triage.

### 3.4 — DFlash-by-genre matrix (full sweep)

```bash
./scripts/sweep_dflash_full.sh   # 3 model × 2 mode × 3 genre × 3 runs
```

Reference numbers in `README.md` "DFlash speculative decode" section.
Code prompts: 4× win on 27B / 2.6-3× on 9B. Prose prompts: tie or
small loss on 9B (-20%, draft-target alignment issue, NOT a bug).

### 3.5 — Correctness route before any DFlash claim

There is no single script to run here; `./scripts/coherence-gate-dflash.sh`
is **retired and absent** (see [`AGENTS.md`](../AGENTS.md) § 0 rule 1). Select the route from
[`docs/VALIDATION.md`](VALIDATION.md). For a DFlash/spec-decode claim
that means the user-facing serve path:

```bash
python3 scripts/serve_harness.py battery --model <model> ...   # varied prompts
python3 scripts/serve_harness.py chain   --model <model> ...   # related turns
```

and, when the change touches kernels, dispatch, graph capture, or Redline
replay, additionally:

```bash
python3 scripts/redline_daemon_harness.py ...                  # capture + PM4 parity
```

**Always eyeball the decoded output.** Token attractors and single-token
degeneracy pass every statistical threshold as fake wins, and a
suspiciously tight stddev or an unusually high τ is a warning sign, not
reassurance.

### 3.6 — Pull flow end-to-end

If you're testing an actual user UX flow:

```bash
hipfire pull qwen3.8:27b-mq4-xts   # target + registry draft sidecar
# Qwen3.8-27B paired tiers need no config: the registry resolves dflash=auto.
# Other models: hipfire config set dflash_mode auto    # opt in (schema default: off)
# Acceptance draft pin when measuring (optional override of registry sidecar):
# export HIPFIRE_DFLASH_DRAFT=/path/to/local-copy-of-registry-draft.hfq   # same bytes as the qwen38-27b-dflash-mq4.hfq sidecar
hipfire run qwen3.8:27b-mq4-xts "Write a Python function to find the longest substring without repeating characters"
# expected: loader logs 'DFlash draft loaded: ...'
# on-disk target: ~/.hipfire/models/qwen3.8-27b.mq4-xts
```


Without the `dflash_mode auto` config, `hipfire run` runs pure AR
even when a paired draft is on disk. `dflash_mode on` instead requires
the sidecar and fails the load when it is missing; `developer.dflash_draft`
or `run --model-draft` overrides the sidecar. This is the "I pulled
the draft but DFlash isn't firing" pitfall.

---

## 5 · Reporting findings

### Where to put bench results

- **Numerical perf-checkpoints:** a dated record in
  [`docs/perf-checkpoints/`](perf-checkpoints/), plus the numbers in the
  commit message body of the commit that produced them (or the PR
  description). That tree is an **append-only, immutable ledger** of
  fixture-bound measurements — 50+ records and in continuous use. Read
  [`docs/perf-checkpoints/README.md`](perf-checkpoints/README.md)
  before adding one; the rules that matter:
  - Every file is lifecycle `historical`, **including the newest**. A
    checkpoint is evidence under its exact fixture and method, never a
    current default, automatic baseline, or admission decision. Newest
    file != current baseline.
  - **Never modify or delete an existing checkpoint.** Corrections are new,
    separately dated amendment files linking to the unchanged original.
  - Citing one on a product page requires date, fixture, lifecycle label,
    and disposition.
  - Current product claims live in [`docs/BENCHMARKS.md`](BENCHMARKS.md)
    (admission-gated), routes in [`docs/VALIDATION.md`](VALIDATION.md),
    protocol in
    [`docs/methodology/perf-benchmarking.md`](methodology/perf-benchmarking.md).
- **Forensic discoveries (e.g. "I found X regresses Y"):** in the
  commit message of the fix (or the bisect commit). For longer
  writeups, the PR description. Local-only scratch goes to
  `.codeinsight+research/` (gitignored).
- **Regression vs. last-shipped baseline:** include the model md5, the
  `hipfire` binary md5, and the daemon binary md5 (`target/release/daemon`).
  Without these the result is unreproducible. `hipfire bench --json` output
  should be pasted whole rather than summarised — its `samples` array is what
  lets a reader judge whether a delta cleared the noise.

### Don't claim a perf win without

- At least 4 fresh-process loads per arm, interleaved (ABBA or Latin square),
  with a throwaway arm first; 6–12 per arm when the effect is under about
  1 ms/step
- A bootstrap 95% CI of the difference; a bar claim quotes the pooled median,
  the ratio's CI and how many loads cleared the bar
- Prompt md5 recorded; distinct prompts or nonces in any C>1 greedy arm
- Binary md5 recorded; the number measured on the build that lands
- A log line proving the lever was active in the measured run
- Claim-scoped `serve_harness` / VALIDATION route pass (retired coherence-gate scripts are **not** acceptance)
- For a kernel change, a served A/B, not only an isolated bench (unless it is
  bitwise and below served resolution; see
  [`lessons-0.4.5.md`](methodology/lessons-0.4.5.md) § 3)

- Eyeball check on decoded output (especially when τ is unusually high)

### Don't claim a perf regression without

- The same load count, interleaving and CI as a win (same prompt, same env),
  and a host-load check of the runs that show it
- Bisect to a specific commit (use `scripts/probe_commits.sh COMMIT_BEFORE COMMIT_AFTER`)
- Confirmation that the regression appears across genres (not just one
  prompt that happens to hit a different distribution)
- A reducer check: lead, warm-up and fixed-length bursts are not pooled into
  the cell (a τ "regression" was exactly that, lessons § 4.2)

---

## 6 · Common pitfalls (history of what bit us)

| Symptom | Real cause | Fix |
|---|---|---|
| "DFlash got slower overnight" | Prompt structure changed (one newline added/removed) | Use byte-identical prompts via `benchmarks/prompts/*.txt` |
| `τ=9.42` on first run, `τ=8.07` on next | Different prompt — see above | Same fix |
| "DFlash 102 tok/s on prose vs 124 AR" | Draft-target argmax disagreement on prose tokens, τ collapses to ~1.2 | Expected with z-lab drafts; no retraining fix is planned — Path C (custom draft training, `feat/mtp-dflash-training`) was a failed month-1 experiment and is dead. Use AR or accept the genre-conditional behaviour. |
| 3.6-A3B DFlash 68.6 tok/s vs AR 135 tok/s (50% loss) | 3.6 draft trained on 3.5 traces; target distribution mismatch on code. τ=1.22 on hard code. | Use AR mode for 3.6-A3B. Draft mismatch is expected and no 3.6 retrain is planned — Path C (`feat/mtp-dflash-training`) is dead/out-of-scope, not a forthcoming fix. 3.5-A3B DFlash works (τ=4.91). |
| `hipMalloc out of memory` at hidden_rb | Contiguous DFlash (the draft declares no window) sizes the draft context structures to the context, capped at 8192 rows; long ctx + 27B can exhaust 24 GB | Use a draft that declares a sliding window (the default windowed draft pins draft VRAM at W); lower ctx. KV headroom: `fwht3` is the optional compact mode on every arch |
| `tok/s` below expected on long-ctx | KV cache growth — prefill is fine but decode slows past ~2K | Test at small ctx first, then scale |
| daemon doesn't pair a pulled draft | Renamed draft file, or pulled before the sidecar existed | Don't rename files after pull; re-run `hipfire pull <tag>` to fetch the registry-declared sidecar |
| `[hipfire-daemon] dflash_mode=off — skipping draft load` | Default flipped to `off` in 35265c6 (post-2026-04-26). Pulling a draft does NOT auto-enable DFlash anymore. | `hipfire config set dflash_mode auto` (or `on`); or per-model `hipfire config qwen3.8:27b-mq4-xts set dflash_mode on` |
| "Numbers don't match the README" | Forgot `HIPFIRE_NORMALIZE_PROMPT=1` (pre-2026-04-26) | Now default ON. Pull latest. If you opted out via `prompt_normalize=false`, that overrides the default — flip back. |
| "27B DFlash regressed 30-40% suddenly" | PR #32 (cleanup-dead-wmma-kernels) on master removed `gemm_hfq4g256_residual_wmma{,2,_k4}.hip` thinking dead. Dispatch fell back to slower variants. | Verify against canonical 199 tok/s @ max=120 with default flags. If kernel files missing in `kernels/src/`, `git checkout` from a known-good commit (see commit 9a2c667 for the full recovery context). |
| `HIPFIRE_GRAPH=1` reports plausible tok/s but output is garbage | Dangling stack-pointer kernargs from raw `self.hip.launch_kernel(...)` calls in `forward_scratch_layers` (kv_cache_write_*, attention_flash_*, fused_qkv_hfq4g256, rmsnorm_batched, rope_partial_interleaved_f32, gated_delta_net_q8, etc.) — captured pointers dangle past `end_graph_capture` | Bench tok/s alone never proves graph correctness. Always eyeball under `HIPFIRE_GRAPH=1` and run the claim-scoped VALIDATION serve route — never retired coherence-gate scripts as acceptance. Fix: migrate every raw-launch helper used in forward_scratch_layers to `launch_maybe_blob` (model after `conv1d_silu_split_f32_n`). |
| `memory.offload_exec` modes disagree on greedy output, or `passback` differs between two identical runs | Expected: the modes run different GEMV engines, and `passback`'s row split is scheduled per process (probe + online arm timings). Measured (2B): `cpu`×2 and `pcie`×2 byte-identical; `cpu` vs `pcie` differ; `passback`×2 differ; `HIPFIRE_OFFLOAD_PASSBACK_SHARE=0` byte-identical to `cpu` | NEVER assert a greedy/golden reference across a mode pair (`cpu` vs `pcie` is not bit-identical either), and NEVER assert `passback` output with the share on `auto` — pin `memory.offload_passback_share`, or use `0` to reach the exact `cpu` twin. See [`docs/perf-checkpoints/2026-10-02-offload-passback-model-spread-gfx1201-amendment-1.md`](perf-checkpoints/2026-10-02-offload-passback-model-spread-gfx1201-amendment-1.md) § 5. |
| Qwen3.5-class (arch 5/6) model with `speculation.mtp=auto` runs AR, resident or offloaded (bench `spec_route` null, serve `timings.tau`/`mtp` null, no `MTP head loaded` line) | Native MTP `auto` arms only a producer that declares a measured AR floor, and Qwen35 declares none, so the load logs `MTP: off by default (no measured AR floor); opt in with --spec mtp` and no head is loaded. Where a floor-declaring producer (Qwen4) does attach under `auto`, a host-mapped routed-expert spill still declines it on any arch but its admitted ones (gfx1100, gfx1201) (the `pcie`/`cpu` decline line logs the count, seam and `opt in with --spec mtp`). Unset/null layer/expert budgets keep all weights resident; only explicit `auto`/`-1` or manual `N` requests placement changes | Use `--spec mtp` / `speculation.mtp=on`, subject to metadata-based refusal of uniform MQ6G256 V1 (qt15) Qwen35 MoE on gfx1200/gfx1201 with FP16 WMMA enabled whenever a head would attach. `--spec off` / `speculation.mtp=off`, `auto` and an auto-declined spill remain AR; `HIPFIRE_FP16=0` permits explicit MTP. Mixed/graded MQ4+MQ6, MQ6V2, gfx1151 and dense are outside the refusal. `speculation.dflash=on` still refuses without a DFlash draft and never aliases to MTP. No root WMMA fix is claimed |

---

## 7 · Quick-reference flag table

| Env var | Purpose | Default |
|---|---|---|
| `HIPFIRE_NORMALIZE_PROMPT` | Phase 1 `\n{3,}` collapse | **ON (since 2026-04-26)** — set `0` to opt out |
| `HIPFIRE_PROMPT_TOKEN_HEAT` | Per-position BPE merge-rank dump | OFF |
| `HIPFIRE_PROMPT_HEAT_JSON` | JSON output for heat dump | OFF |
| `HIPFIRE_PROMPT_HEAT_LIMIT` | Max rows in heat dump | 64 |
| `HIPFIRE_KV_MODE` | Override `kv_cache` config. `fwht3` is the optional compact mode; `asymN`, `turboN` and `legacy-asymN` were removed in 0.5.0 and are rejected | `auto`: fp8 on gfx1201 (Qwen H24/Hkv4/D256, single GPU, no adaptive), q8 otherwise incl. gfx1100/gfx1151 (`kv_mode.rs:28-46,194-215`) |
| `HIPFIRE_ATTN_FLASH` | Override flash_mode config | (config) |
| `HIPFIRE_OOM_GUARD` | Memory preflight OOM guard (`kv_slots::preflight_alloc`, SlotPool arena, bench-sweep headroom check). `auto`: on for unified-memory APUs (Strix Halo — overshoot is a global OOM), off for discrete GPUs, swap-decided for GPU-less processes | `auto` (`memory.oom_guard`) |
|`HIPFIRE_DFLASH_DRAFT`|Force a specific draft path, overriding the registry sidecar. Empty string = explicit opt-out|(unset: registry sidecar when `dflash_mode` is `auto`/`on`)|
|`HIPFIRE_DFLASH_WINDOW`|Windowed draft context: sliding attention over the last W rows (DFlash2: every layer; otherwise layers 0..n-2 plus a full-attention last layer). Draft VRAM pins at W regardless of `max_seq`; past-W requests degrade τ instead of falling back to AR. Unset = the draft's declared `sliding_window` (`dflash_spec.rs:170-200`); `<rows>` = override (warns on mismatch); `0` is refused (legacy contiguous DFlash removed in 0.5.0).|draft-declared window (fixed 8192-row contiguous cap if none)|
| `HIPFIRE_LM_HEAD_F16` | `auto`/`native` keeps qt=1 lm_head as F16; `f32`/`legacy` expands to F32 | auto/native |
| `HIPFIRE_LOCAL` | Force local-spawn (skip serve HTTP) | OFF |
| `HIPFIRE_HOST_TIMING` | Per-cycle host timing probe | OFF |
| `HIPFIRE_VERIFY_GRAPH` | Verify-forward graph capture (0 = off) | ON |
| `HIPFIRE_GPU_LAYER_BUDGET` | Resident-layer budget for partial GPU offload: `N` keeps the last `N` layers on the GPU and spills the prefix to host RAM. Counts layers **on** the GPU — `3` on a 64-layer model spills 61. Unset/null/empty/unparseable leaves weights device-resident on every model. Explicit `auto`/`-1` opts into fit-capacity placement, routed experts before whole layers. Manual `N` remains a hard pin. Sibling of `HIPFIRE_OFFLOAD_EXEC` (unchanged default `pcie`). See [docs/CONFIG.md](CONFIG.md#memory). | unset (fully resident) |
| `HIPFIRE_OFFLOAD_EXEC` | Which engine multiplies a spilled layer's host-mapped weights: `pcie` (default), `cpu`, or `passback` (scheduled co-inference — both engines run the same spilled step concurrently, divided by output rows). Sibling of `memory.gpu_layer_budget`; `cpu` executes the weight-reading GEMVs on the CPU instead of reading them over PCIe, and `passback` runs the `pcie` path on the step's first output rows and the `cpu` path on the rest, concurrently. See [docs/plans/partial-gpu-offload-design.md](plans/partial-gpu-offload-design.md) § 6.2.1 and § 6.2.2. | `pcie` |
| `HIPFIRE_OFFLOAD_PASSBACK_SHARE` | Schedule of a spilled step's output rows between the two engines under co-inference (`memory.offload_exec=passback` only): the GPU takes the first rows (the `pcie` arm), the CPU the rest (the `cpu` arm). `auto` (default) schedules it from this host's measured engine rates and refines it online; a number in `(0, 0.5]` pins it; `0` collapses it onto the CPU engine alone, so the mode is byte-identical to `cpu`. Inert (one notice at load) in any other mode. Measure or override with `hipfire offload-bench`. See [docs/plans/partial-gpu-offload-design.md](plans/partial-gpu-offload-design.md) § 6.2.2. | `auto` |
| `HIPFIRE_SERVE_UI` | Serve the embedded chat UI at `/ui` on the serve listener (same as `hipfire serve --ui`; `--open` also launches the browser). Unauthenticated — warn-level on non-loopback binds. Chat history lives in the browser's IndexedDB; nothing is stored server-side. | OFF |

| `hipfire bench` flag | Purpose |
|---|---|
| `--ar-baseline` | Skip DFlash, greedy-decode via target only |
| `--no-chatml` | Bare prompts (raw-text drafts) |
| `--no-adaptive-b` | Fix B at the draft's trained block size |

---

## 8 · Open questions agents can investigate

Open performance questions are tracked against the 0.4.5 Flash-Next EP2 record
([`lessons-0.4.5.md`](methodology/lessons-0.4.5.md) and the cp20
checkpoint); read both before proposing a lever. Still open there:

1. **EP2 TTFT** — the dominant gap at every cell (0.065-0.263 s median stream
   TTFT on 2x R9700 vs about half that for another open engine on the same host).
2. **Story / low-τ decode** — story cells run at τ about 2.2; per-stream decode
   is below that engine's at C1-C8.

Closed, do not reopen: Path C (target-aligned custom DFlash draft training) is
dead, and DDTree tree verify was removed in 0.5.0.
