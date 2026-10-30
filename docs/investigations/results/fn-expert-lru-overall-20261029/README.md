# Frozen Flash-Next exact-expert LRU validation

Tested source: `b0fda0da1ee62fdbf59d2448866330c53893bded`, based on beta `6c99bef7f8fcb966c4104aab2583e894f20c9d10`. Product implementation is `2e1dc0542`; the final validation commit adds this evidence only after the binary was frozen. No product or validation-source change followed the measurements.

## Results and limits

`overall-receipts.json`: `mandatory_checks_pass=true`, `matrix_evidence_complete=true`, **`all_pass=false`**. The optional baseline 8-turn coding session fails the unchanged expected-substring gate: turn 8 reaches 512 tokens before saying `chunk`, ending at Error Handling. Its JSON, log, driver failure and `session-cap-limitation.json` remain here. No larger-cap retry or session-quality pass is claimed. That failure shut down A1; A1extra reloads for staged/long decode only, so baseline full-sequence continuity is not claimed.

| Workload | Cache off | Cache on | Candidate change |
|---|---:|---:|---:|
| Battery weighted decode, median of 3 fresh processes/arm | 33.62 tok/s | 35.05 tok/s | +4.25% |
| Battery total prefill, median | 1.680 s | 3.540 s | +110.7% time |
| Battery total request wall time, median | 22.849 s | 23.849 s | +4.38% time |
| 5103-token staged prefill, one observation/arm | 1563.5 tok/s | 1507.5 tok/s | -3.58% |
| 1024-token decode, one observation/arm | 30.6 tok/s | 38.2 tok/s | +24.84% |
| Same 1024-token request wall time | 33.948 s | 27.687 s | -18.44% time |

The five battery prompts produce 711 generated tokens per process; every answer terminates naturally. Decode is reconstructed as `sum(gen)/sum(gen/decode_tok_s)` from server rates rounded to 0.1 tok/s. Baseline range: 33.14–33.62; candidate: 34.68–35.12. Each fresh server first runs the same five 16-token warmup requests. Timing order is A1, B1, B2, A2, A3, B3; an extra baseline process between A1/B1 supplies only long-workload evidence after the optional session stopped A1. It is excluded from the battery statistics.

All six battery cells have byte-identical prompts, request hashes and decoded answers. Chain, staged-prefill and long-decode transcripts also match across arms. Request IDs join to successful PM4 proof markers; responses have one terminal, DONE, no late bytes or stream error, and no empty/attractor output. A 128-token dirtying request precedes the 5103-token prefill; candidate restore counters increment before that long request. The 1024-token Rust response reaches the cap: this proves exact bounded decode, not a completed/compiled parser implementation.

These are measured tradeoffs on one R9700 gfx1201: short-request latency regresses because of donor restoration, while the single long-decode workload improves. The path remains experimental and default-off. No general-model, fresh-pull, or default-on speed claim is made.

The actual tier has 4015 fixed core experts plus 4016 evictable slots. Its additional immutable host/GTT mirror is 10,486,579,200 bytes (about 10000 MiB). Candidate whole-process device-global sampled peaks range 31.14–32.87 GB VRAM and 55.33–55.38 GB GTT. Baseline ranges are 31.12–31.17 GB VRAM and 44.85–55.28 GB GTT. These 100 ms sysfs samples include startup and delayed release from the preceding server; they are observed global peaks, not per-process allocator peaks or a clean memory-delta measurement. Use the placement receipt for the explicit mirror cost. B1's last periodic cache sample at token 2560 reports 536631 cache hits, 202316 host misses, 78160 published fills, and peak inflight 64; the full counters/logs are retained. Timing counters are nested wall intervals and must not be summed.

Both arms use one identical frozen binary; only the cache flag changes. They share the GTT-provider correction, so this is cache-on versus cache-off in this branch, not a separate untouched upstream binary. The fixture is a grandfathered 125 GB monolith with explicit Q4LM row-order PLE, a bound ranked hotset, AR/speculation off, decode-share off, PCIe execution, 20000 MiB resident-expert budget, tail 50%, inflight 64, chunk 4096 and max_seq 16384. These are explicit overrides, not the fresh-pull/default PLE route.

Service harness KV mode/backend/reason fields remain unobserved because its generic parser does not recognize the Qwen4-specific startup marker. Each retained serve log explicitly records `qwen4 state: QSA fp8 K/V (vmm context storage), GDN q8 recurrent`; Redline's loaded ACK independently reports FP8/VMM and `kv_backend_legacy=false`. No legacy override was used. `HIPFIRE_DN_STATE_EF=1` was present but is not this family's state selector; the actual Qwen4 `auto` format is plain Q8. Live Q8Ef GPU qualification is not claimed.

## Correctness/build evidence

- 182 unique focused Rust tests, 184 successful executions; eight Python harness tests pass. No full local workspace unit-test or workspace release build is claimed.
- Two pre-commit runs pass, including workspace/all-target Clippy and static gates. Scoped release CLI/daemon and lab oracle builds pass with existing warnings.
- `publication-gpu-execute.json`: 15 residency/output checks and 3 ID-copy checks pass, including partial-fill, stale-generation and partial-publication controls. This mechanism oracle alone does not exercise the full production manager or model.
- `redline-candidate-trial01/report.json.gz`: 3 stable captures, 1149 typed dispatches/31 AQL contracts; 16 distinct reported positions 513–528 compare ordinary HIP, recorded HIP and PM4 exactly for tokens, logits and captured state. Actual state is native FP8/VMM plus plain GDN Q8. EF-only divergence has CPU coverage, not live Q8Ef GPU coverage.
- The induced retained-replay error is a pre-submit prepared-position boundary refusal followed by exact HIP recovery. No injected GPU kernel fault or unknown-quiescence GPU run is claimed.

## Artifact identities and reproduction

`runtime/MANIFEST.json` binds the exact CLI, renamed daemon and lab executable hashes, clean tested source, model stat and hotset identity. Model SHA is the existing verified bound-hotset identity, not a new 125 GB rehash. `ARTIFACTS.json` records every retained input's original hash and stored hash; logs and memory traces are gzip-only, with no text sanitization. Model bytes, executables and HOME configs are excluded.

Developer runners are retained exactly as run, with X570 paths. Adjust their ROOT/OUT/MODEL constants and binary locations for another checkout. `run-overall-initial.py` is the initial battery/chain/session attempt. `run-overall-frozen.py` is the fixed-scope continuation after the preserved optional failure, including the separate A1extra long-workload process. Both use the canonical `scripts/serve_harness.py` and only defer successful warmup cleanup to reuse the same Rust server. Startup retry cleanup and canonical verdicts remain active. The GPU calls were run under `scripts/gpu-lock.sh`; acquire the same shared lock before replaying them.

`verify-overall-receipts.py` performs posthoc checks without loading a model. After decompressing retained `serve.log.gz` files in a working copy, run it with this result directory as its `--root` argument (see `--help`); the original optional-session failure intentionally leaves its exit status nonzero. Inspect `mandatory_checks_pass` separately rather than erasing that failure. Native Redline commands and exact environment are in `redline-candidate-trial01/run.json` and `run-redline-frozen.py`.

Historical S0/offline evidence in the sibling `fn-expert-cache-s0-20261025` directory keeps its earlier source/binary labels. It is not relabeled as this current full-model qualification.
