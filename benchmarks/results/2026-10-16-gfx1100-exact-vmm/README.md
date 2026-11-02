# gfx1100 Target Exact VMM AR validation

Date: 2026-10-16. Baseline: upstream beta
`272c4dcddd50a03d5876371f463a279c6aa5fabe`.

## Claim and scope

This is an opt-in functional port, not a throughput claim. On exact gfx1100,
eligible dense uniform MQ4G256V2 targets execute a complete singleton lowered
layer program per request, preserving binding-local fused prep/prerotation
state. Weights are shared; KV, recurrent, conv and sampling state are private.
Projection launches are not merged. The existing gfx1201 route is unchanged.
`serve.vmm_batch` stays default-off; the serve loader does not install
cross-request DFlash/MTP verification on the new gfx1100 Exact route.

Q8 recurrent state must use the default-on error feedback for serve admission.
EF-off remains on the existing singleton route with an explicit diagnostic.
There is no new `Architecture` trait surface or kernel arithmetic.

## Fixture and identities

- GPU1: Radeon Pro W7900 Dual Slot, gfx1100, PCI `0000:9e:00.0` (state oracle).
- GPU0: Radeon Pro W7900, gfx1100, PCI `0000:53:00.0` (HTTP service).
- HIP 7.15. Per-card locks held; other processes were not stopped.
- Target: `hipfire-models/qwen3.8-27b/qwen3.8-27b.mq4-xts`.
  Size 14,987,185,152 bytes; SHA-256
  `3e38ccbae3776470eb5a89344d300e9279d6b9ab6c31fd40ca1758c4f7c6f8ae`.
  Verified with `scripts/check_fixture.sh`.
- Q8 K/V, automatic VMM backend, Q8 recurrent state with default error feedback,
  speculation off, greedy sampling, thinking off. No backend/max-seq override;
  the oracle loaded ACK reports `max_seq_bound=262144`, `kv_backend_legacy=false`.
- Core oracle MD5: `06c0a7ed4a1a0d909c8fb2506abf0498`.
- Core daemon MD5: `e6fe5d5a9032804e357e1af3b9a0b7a6`.
- Core CLI MD5: `28d07f6eae2e54e97eb3cfdc072cb780`.
- Final source: `209568655` (EF-off guard uses explicit `StateQuant::Q8`,
  with a CPU truth table ensuring FP32/Q4 and empty state are not refused).
  Final daemon MD5 `53d358e71699b80946f716db35237ccf`, CLI
  `605d7fed0793fa9eea32cff05a75787e`, oracle
  `df9c1d384f53eb13d20f1f95280ab257`.

Core runs used the semantic source subsequently committed as `18f23074b`,
before the mechanical format commit. Formatting all three tested source files
produces byte-identical source to that commit's files; no semantic changes
were introduced by format/conflict resolution. The later `3ea16143a` and
`209568655` add the loaded-state EF-off serve refusal, precision-specific CPU
test and documentation; final-build admission receipts are recorded separately,
not attributed to core bins. The final oracle binary also passed the short
C1/C2 state matrix; the full 256-step matrix remains attributed to core bins.

## Results

| Check | Coverage | Result |
|---|---|---|
| State oracle | C1–C8, 256 greedy steps, mixed 512/8192/32768/256-token prefixes | PASS: independent singleton parity and request isolation, including logits, hidden, KV, recurrent/conv state, IDs and positions; 14m32s wall |
| Lifecycle controls | Stop, stop under memory pressure, cancel, stale epoch/commit, poisoned tail/plan, slot reuse | PASS |
| HTTP battery | C1–C8, 180 requests, max_tokens=512 | 180/180 stop; zero bad-output/stream flags; exactly-once terminals; zero isolation differences |
| HTTP chain | C1–C8, 180 multi-turn requests, max_tokens=512 | 180/180 stop; zero bad-output/stream flags; exactly-once terminals; zero isolation differences |
| Flag-off battery | Five singleton prompts | PASS; texts equal batching-on C1 |
| Flag-off Redline | Three captures; 32 decode / 8 shadow iterations | PASS: stable 659 dispatches / 16 kernels / `70ae0f331f24e1d7`, valid AQL contracts, HIP/PM4 shadow and GDN frame exact |
| EF-off, default graph | Short C1/C2, 8 steps | FAIL: singleton reference itself is non-repeatable; isolation still passes |
| EF-off, graph off | Same short matrix, eager HIP | PASS: both isolation and singleton parity |
| Final default-on admission | Final daemon, C1/C2 battery, max_tokens=512 | 15/15 stop, zero bad flags/terminal errors; outputs equal C1 |
| Final EF-off admission refusal | Final daemon, five singleton prompts, max_tokens=512 | 5/5 stop, zero bad flags/terminal errors; explicit refusal receipt, no VMM lane store staged (KV itself stays VMM) |
| Loaded DFlash2 sidecar | Final daemon, C2 battery, max_tokens=512 | 10/10 stop, zero bad flags/terminal errors; texts equal final AR C1, shared verify not installed |

The state oracle cycles eight request-unique synthetic prefixes through the
four lengths: this is **not** every lane simultaneously at 32K, nor a quality
benchmark. The separate service outputs were inspected: sorted-list merge code,
210-mile arithmetic, axial-tilt explanation, a coherent four-sentence story,
and five code-maintenance tips. Full decoded service text is in the raw JSON.

The EF-off result is not evidence of a newly introduced frame-counting bug:
the singleton reference already differs from itself. Its cached hipGraph
replays captured stochastic requant frames, while eager execution reserves new
ones. Disabling graph removes the difference. This PR does not repair that
legacy singleton graph behavior, and does not certify injected GPU-error
recovery with EF-off; the new serve admission conservatively refuses it.

The loaded-sidecar check uses `qwen38-27b-dflash-mq4.hfq`, SHA-256
`d0a74a232a0e2166d889f823e91e0fbf778d21dd9668d7de055cdecb065401bc`.
Its startup log proves the DFlash2 draft is loaded alongside a staged VMM AR
store and the AR-only receipt; all ten requests have no DFlash/MTP execution
identity. This checks staging/ownership, not batched DFlash correctness.

Initial 128-token admission probes truncated the arithmetic answer and set
the harness's `runaway` flag (one EF-off row, two loaded-sidecar rows). They
are not quality acceptance evidence; the final 512-token reruns above all
terminate normally. Original short-probe logs remain local and unchanged.

The service harness reports its resolved `kv_mode` as `UNOBSERVED`; this is
not presented as a measured Q8 field. Q8 K/V are explicit in the request and
in the original serve startup allocation logs; backend `vmm` is observed in
every service row, and the state oracle's loaded ACK independently reports Q8.

Timings include cold JIT and concurrent CPU checks. They are diagnostic only,
not a speedup, slowdown or peak-VRAM measurement.

## CPU and build checks

- Full workspace lib tests on the rebased core: 6,353 passed, 0 failed,
  79 ignored across 57 library suites. Final precision-specific loader guard:
  111 passed, including the new guard test.
- `scripts/pre-commit-ci.sh origin/beta`: PASS for the formatting, functional
  and EF-off guard commits. Only the three touched Rust files were formatted;
  this is isolated in `3d21a676b`, not mixed into the functional commit.
- Full release workspace/all-target build: PASS at the final code revision
  (`cargo build --release --workspace --all-targets --locked`), followed by
  a product rebuild; product binary MD5s still match the tested binaries.
- `scripts/no-gpu-ci.sh`: serial Rust steps pass; Python has 563 passed / 22
  failed. All 22 failures reproduce on the clean current beta: 21 hw-gate
  review cases where `scripts/hw-gate/select.py` shadows stdlib `select`, plus
  the registry reproducibility test's missing Flash-Next MQ3-XTS PLE inventory
  sidecar. With diagnostic `PYTHONSAFEPATH=1`, all 42 review tests pass.
- An initial parallel no-GPU invocation hit process-spawn `EFAULT` in the
  unchanged compiler batch test; the serial rerun passes. Its cause is not
  established. No original failed log was overwritten.
- Remaining CPU steps after the Python failure were invoked separately:
  Redline unittest 179, jev_eval 45, install 17, uninstall 6, GPU lock 4 pass.

No CI/Python/registry policy was changed to hide the baseline failures.

## Reproduce

Build the oracle and product separately:

```sh
cargo build --release --locked -p hipfire-arch-qwen35 --features deltanet --example cb_vmm_state_oracle
cargo build --release --locked -p hipfire-daemon -p hipfire-cli
```

After selecting and locking a gfx1100 GPU, use a private `HIPFIRE_HOME` and
fresh output paths (the oracle refuses to overwrite reference artifacts):

```sh
HIPFIRE_KV_MODE=q8 target/release/examples/cb_vmm_state_oracle \
  ~/.hipfire/models/qwen3.8-27b.mq4-xts \
  --ks 1,2,3,4,5,6,7,8 --contexts 512,8192,32768 --short 256 \
  --steps 256 --spec off --out /absolute/fresh-results/state.json

HIPFIRE_SERVE_VMM_BATCH=1 HIPFIRE_CONTINUOUS_BATCH_SIZE=8 \
HIPFIRE_DAEMON_BIN="$PWD/target/release/daemon" HIPFIRE_CLI_BIN="$PWD/target/release/hipfire" \
python3 scripts/serve_harness.py \
  --model ~/.hipfire/models/qwen3.8-27b.mq4-xts --tag qwen3.8:27b-mq4-xts \
  --mode battery --concurrent-clients 1,2,3,4,5,6,7,8 \
  --speculation off --sampling greedy --thinking off --kv q8 --max-tokens 512 \
  --home /absolute/fresh-results/home --port 11730 \
  --serve-log /absolute/fresh-results/serve.log --out /absolute/fresh-results/battery.json
```

Repeat with `--mode chain` and a different home/result path. Require the VMM
store staged and gfx1100 AR-only receipts, not a successful singleton fallback.

Raw reports are losslessly compressed as `*.json.gz`; decompress with
`gzip -dc FILE.json.gz`. Original state arrays/logs remain local (about 19 GiB
for this full oracle) and are not added to Git. Reports from the earlier
`5880f3eff` validation remain separate and are not attributed to these binaries.
