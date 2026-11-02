# PR #742 final-head G4 evidence

This directory records the final-head G4 acceptance run for the Qwen 3.5 route and its cancellation, commit, continuous-batch, DFlash, and fault-recovery seams. It is evidence, not a performance report.

## Run identity and scope

- UTC campaign date: `2026-09-10`
- Worktree: `/home/bjoern/hipfire/.claude/worktrees/g4-next-integration`
- Branch: `replan/g4-next-integration`
- HEAD: `e78694c85c09e3c0db747e2abb89aa24eff6c586`
- Feature parent: `17f84e57ecf2f4381aabc414670eefe7657718ce`
- Beta parent: `5773f62497d603ef8678ac09beb72869517a0ee2`
- No source files were edited for this campaign. The durable changes are this evidence directory only; ignored `target/` helpers and receipts are not acceptance artifacts.
- No aggregate G4 promotion is claimed. No throughput, latency, or quality comparison is claimed.

The campaign deliberately separates a first continuous-batch failure caused by an incomplete local JIT-cache mirror from the corrected run. The first result is an environmental packaging failure, not a route verdict; the corrected result is the route evidence.

## Host, binary, and model identity

- GPU: AMD Radeon 8060S, `gfx1151`
- VRAM: 131.1 GB reported by the test harness; rocminfo GPU pool `128000000 KB`
- HIP: 7.2 (`hipconfig`: `7.2.53211-9999`)
- HSA runtime: 1.18
- ROCm root: `/nix/store/lqklrnx2bc9k765jyxc0d8q6h15wlybb-clr-7.2.3`
- Build command:
  `CARGO_TARGET_DIR=/tmp/hipfire-g4-e786-target cargo build --release -p hipfire-cli -p hipfire-daemon --features hipfire-daemon/serve-fault-inject,hipfire-runtime/serve-fault-inject,hipfire-generate/serve-fault-inject`
- `hipfire --version`: `hipfire 0.3.1 (e78694c85c09; replan/g4-next-integration)`
- CLI SHA-256: `97cc75e69c053dd4d9cd529025e7bb61fcdd6eefe8c172655b4b2c04e198eea6`
- daemon SHA-256: `82c45928b85318d1760855d4c0f40f20508618c586c9cf910edbee9f03f1b9b7`

Canonical artifacts:

| artifact | bytes | SHA-256 |
| --- | ---: | --- |
| `/home/bjoern/.hipfire/models/qwen3.5-27b.mq4` | 14,984,158,208 | `ea615949ddf6a180eee03ff6fde39f7e51148f153b1b05f82258b9953088576e` |
| `/home/bjoern/.hipfire/models/qwen35-27b-dflash-mq4.hfq` | 919,401,472 | `3d428b97c1911a9ad815cc52fbee080306852c1dafad6b1b17bb70bd68010301` |

Exploratory artifacts (not acceptance artifacts):

| artifact | bytes | SHA-256 |
| --- | ---: | --- |
| `/home/bjoern/.hipfire/models/qwen3.6-27b-vl.mq4` | 15,908,670,469 | `6017171d6441d9dd6083a3732c7a0e1679ea936198772746e0b5411f2185f993` |
| `/home/bjoern/.hipfire/models/dots-ocr.q8.hfq` | 4,420,477,952 | `eec256b12ec11b118422cb49fbfd49b14c653374af8bc31b08a2a3b1b6f5b268` |
| `benchmarks/vision/images/scene_1.jpg` | 168,376 | `549784c349af8a87d53e2fec897d03aa800ed73a64bb9d8de60485cdd71d1e98` |
| `benchmarks/images/dots_ocr_smoke_001.jpg` | 772,990 | `90345584ccc2c4a883779e5d47693276e8cf3fe752700af4f03b3142ab46cfa2` |

The canonical prompt hashes are preserved in `raw/dflash-prompt-bundle.sha256` and the command/console receipts. The five direct prompt hashes are, respectively, `3837e57d...`, `d0a1db1...`, `d671894...`, `107b33a...`, and `056c3f2...` (full values are in the raw receipts).

## Protected daemon

The only durably committed listener capture is the post-run socket capture `raw/protection-after.txt`, which records PID `1278900` on `127.0.0.1:11524`. No independent before capture was found in the campaign artifacts, so “before/unchanged/no-signal” is an operator observation, not independently established by committed evidence. The campaign launched and cleaned only isolated daemon processes it owned; this evidence does not claim it touched the protected listener. The durable fault-seam receipt is `raw/component-seams-manifest.txt`.

## Acceptance results

### Qwen 3.5 lifecycle and G4.5 semantics — pass

The direct stdio run loaded the canonical MQ4 target, exercised fresh snapshot, generate/commit/done, reset, same-session reuse, unload/reload, and post-reload generation. It also covered ordinary stop, max-token length, explicit open-think validation, next-turn reuse, and controlled prefill abort. The lifecycle snapshots show `replay_clean: true` before the run, after reset, after fault recovery, and after the controlled cancellation. Raw evidence: `raw/direct-stdio-summary.json`, `raw/direct-stdio-trace.jsonl`, `raw/g45-and-prefill-summary.json`, and `raw/g45-supplement-console.txt`.

### G4.6 singleton commit gate and key reuse — pass

The exact commit lane, early commit, wrong attempt, wrong id, late commit, duplicate generate, same-key reuse, timeout, decode abort, and post-prefill fault cases were exercised. A valid `commit_ready` followed by the matching `{id, attempt_id}` commit produced exactly one `done`; stale controls did not produce a second terminal. Raw evidence: `raw/direct-stdio-summary.json` and `raw/direct-stdio-trace.jsonl`.

### Continuous-batch first run — environmental failure, retained separately

The first two-lane run loaded with `continuous_batch_capable: true` and both lanes started, but both GPU drives failed before `commit_ready` because the cached generated source included `"kv_slot_desc.h"` without that header present beside the source:

```text
batch GPU error: forward_decode_batch: HipError(0): hipcc compilation failed for kv_cache_write_q8_0_independent
.../kv_cache_write_q8_0_independent.ae4df8f416708260.hip:6:10: fatal error: 'kv_slot_desc.h' file not found
```

The failures were marked retryable and rolled back. The resulting snapshot reported `replay_clean: false`; explicit reset, unload, reload, and a post-reload singleton generation returned `ready` with one matching `done`. This is an environmental cache-packaging observation, not a continuous-batch route rejection. Exact lanes/attempts were `lane-abort/301` and `lane-commit/302`. Raw evidence: `raw/direct-stdio-summary.json` and `raw/direct-stdio-stderr.jsonl`.

### Continuous-batch corrected run — pass

The failed mirror was copied to `/tmp/hipfire-g4-home-g46-batch-corrected`, `kernels/src/kv_slot_desc.h` was staged exactly beside the cached generated source, and the HSACO was compiled with the exact failed-run device flags. The corrected header SHA-256 is `f862ec11dfc3ba051f17430a25895b7b73830fc9a32ba18731d82766c0c62976`; the staged HSACO SHA-256 is `99070ab255c8b8645619c4906fda01331e6aedfd5ea787956b00e18545ce7b56`; the cache hash file SHA-256 is `37a72f29dd8abd603853f9e3ffaecbea2e0d693ec069bfbc5ec102a9a3ffad9c`.

The corrected two-lane matrix used:

- `lane-commit-a`, attempt `601`: `commit_ready` then matching `done`, finish `stop`, `continuous_batch_independent`, lane `0`.
- `lane-commit-b`, attempt `602`: `commit_ready` then matching `done`, finish `stop`, `continuous_batch_independent`, lane `1`.
- Both lanes started; both reached `commit_ready`; both matching commits were sent; exactly one `done` was observed per lane; attempt correlation held.
- A stale abort was sent before re-admitting the same key `lane-commit-a/601`; the same-key reuse returned `REUSED` with one matching `done`, finish not `aborted`, and correct attempt correlation.

Raw evidence: `raw/batch-corrected-command.txt`, `raw/batch-corrected-summary.json`, `raw/batch-corrected-trace.jsonl`, `raw/batch-corrected-stderr.jsonl`, and `raw/batch-corrected-console.txt`.

### Cancellation cardinality audit — pass after contract classification

The raw summary helper labels every `aborted` JSON object as a generic terminal, so its `terminal_count: 2` for cancellation is not the lifecycle cardinality. The wire contract and consumer classify the pair as one logical terminal:

- `aborted` is a correlated mid-stream control acknowledgement: `{type:"aborted", id, reason:"client_cancelled", attempt_id}`.
- `done` is the sole terminal envelope: `{type:"done", id, finish_reason:"aborted", attempt_id, ...}`.
- Both events carry the same id and attempt. The `done` envelope carries generated completion-token count and zeroed prompt/decode timings.
- `done` is the only event latched by `SemanticEventFold`; unknown/control events, including `aborted`, are forwarded without latching the terminal. The consumer therefore sees exactly one terminal lifecycle outcome.

Exact observed cancellation sequences:

```json
[
  {"type":"gen_start","id":"g46-abort-prefill-controlled","attempt_id":501},
  {"type":"aborted","id":"g46-abort-prefill-controlled","reason":"client_cancelled","attempt_id":501},
  {"type":"done","id":"g46-abort-prefill-controlled","finish_reason":"aborted","prompt_tokens":0,"completion_tokens":0,"prefill_ms":0,"decode_ms":0,"attempt_id":501}
]
```

```json
[
  {"type":"gen_start","id":"g46-abort-decode","attempt_id":212},
  {"type":"token","id":"g46-abort-decode","text":"#","attempt_id":212},
  {"type":"aborted","id":"g46-abort-decode","reason":"client_cancelled","attempt_id":212},
  {"type":"done","id":"g46-abort-decode","finish_reason":"aborted","prompt_tokens":0,"completion_tokens":1,"prefill_ms":0,"decode_ms":0,"attempt_id":212}
]
```

```json
[
  {"type":"gen_start","id":"g46-timeout","attempt_id":208},
  {"type":"commit_ready","id":"g46-timeout","attempt_id":208,"finish_reason":"stop"},
  {"type":"aborted","id":"g46-timeout","reason":"client_cancelled","attempt_id":208},
  {"type":"done","id":"g46-timeout","finish_reason":"aborted","prompt_tokens":0,"completion_tokens":18,"prefill_ms":0,"decode_ms":0,"attempt_id":208}
]
```

The controlled prefill sequence proves that no token preceded the abort (`g46-abort-prefill-controlled/501`). The implementation evidence is `crates/hipfire-engine/src/emit.rs:344-364`, `crates/hipfire-runtime/src/semantic.rs:408-455`, and `crates/hipfire-cli/src/serve/complete.rs:956-1015`. The targeted contract tests passed:

- `hipfire-runtime` `semantic::tests::wire_gen_start_and_aborted_helpers_are_correlated`
- `hipfire-generate` `wire_helpers_used_by_gen_start_and_cancel_writers`
- `hipfire-cli` `serve::complete::tests::semantic_fold_error_and_abort_terminals_expose_no_calls`
- `hipfire-cli` `tests::nonstream_client_disconnect_aborts_and_releases_admission`

The final test receipts are `raw/postcheck-runtime-abort-wire.txt`, `raw/postcheck-qwen-abort-wire.txt`, `raw/postcheck-cli-abort-fold.txt`, and `raw/postcheck-cli-abort-drain.txt`. The last test also asserts that admission does not reach zero before the daemon done/aborted marker has drained, then admits a follow-up request.

### Fault seams — pass

Ten module-qualified ignored GPU seam tests passed (`1 passed; 0 failed` each): four Qwen DFlash construction owner-reuse cases, four DeepSeek4 DSpark rollback/retry cases, and two Qwen35 dense/MoE owner-reclaim cases. The complete command and result receipt is the durable `raw/component-seams-manifest.txt`; the initial unqualified invocation that matched zero tests is explicitly excluded by that receipt.

### DFlash battery and chain — semantic pass, no performance claim

The canonical Qwen target/draft loaded and both the battery and two-prompt chain reported `dflash: true`, non-empty semantic answer text, and no ATEM leak. They hit configured length caps (`finish: length`, `runaway: true`), so the evidence proves route activation and semantic output only; it is not a quality or speed promotion. Raw evidence: `raw/dflash-battery.json`, `raw/dflash-chain.json`, their command/console/server receipts, and `raw/dflash-prompt-bundle.sha256`.

### Vision explorations — failed/noncanonical, excluded

- Qwen3.6-VL loaded its F16 vision tower and completed the gfx1151 vision forward, but returned empty content with `tokens: 32` and `finish_reason: null`. It is failed exploratory evidence, not acceptance.
- Dots OCR loaded its vision tower and completed the vision forward, but returned a truncated 128-token table fragment with `finish_reason: null`. It is failed/noncanonical exploratory evidence, not acceptance.

Raw evidence: `raw/qwen36-vl-command.txt`, `raw/qwen36-vl-console.txt`, `raw/dots-ocr-command.txt`, and `raw/dots-ocr-console.txt`. VL empty/null and truncated output remain failed/noncanonical; they are not converted into a pass. No Qwen3.8 canonical artifact was available, and malformed-tool cases were not run as deterministic acceptance routes.

## Postchecks

The four cancellation contract tests above all passed. `git diff --check` produced no output. The isolated final build and strict clippy receipts are preserved even when their result is environmental/project-wide:

- `raw/postcheck-build.txt`: feature build in `/tmp/hipfire-g4-final-check-target` (the earlier isolated campaign build succeeded and produced the hashed binaries above).
- `raw/postcheck-clippy.txt`: strict affected-package clippy; it is blocked by existing `hipfire-config` warnings (`doc_overindented_list_items`, `redundant_guards`, and `obfuscated_if_else`), outside this evidence-only change.
- `raw/postcheck-fmt.txt`: the repository's `cargo fmt` alias is configured to reject direct invocation (`use-scripts-fmt-changed-sh-instead-of-cargo-fmt`); this is preserved as an environment/tooling observation. A standalone cargo-fmt check is recorded separately if available.
- `raw/postcheck-diff-check.txt`: no whitespace errors.

No source workaround or warning suppression was added for these postcheck observations.

## Raw evidence inventory

All campaign stdout/stderr, commands, summaries, traces, cache provenance, postchecks, and the protected socket capture are under `raw/`. `ledger.jsonl` is the machine-readable one-row-per-verdict index. `MANIFEST.sha256` covers the durable README, ledger, and every raw receipt; regenerate it only after all evidence files are final.

The key raw files are:

- direct lifecycle/G4.5/G4.6: `direct-stdio-trace.jsonl`, `direct-stdio-stderr.jsonl`, `direct-stdio-summary.json`, `direct-driver-command.txt`, `direct-driver-console.txt`
- supplemental G4.5/prefill abort: `g45-supplement-command.txt`, `g45-supplement-console.txt`, `g45-and-prefill-summary.json`
- corrected continuous batch: `batch-corrected-command.txt`, `batch-corrected-console.txt`, `batch-corrected-summary.json`, `batch-corrected-trace.jsonl`, `batch-corrected-stderr.jsonl`
- DFlash: `dflash-battery-*`, `dflash-chain-*`, `dflash-prompt-bundle.sha256`
- exploratory vision: `qwen36-vl-*`, `dots-ocr-*`
- contract/postchecks: `postcheck-*`
- protection: `protection-after.txt`

## Limitations and closeout

The following blockers remain open:

- Only one physical GPU was available; no RCCL multi-GPU proof was captured.
- Local DeepSeek4 target/draft digest(s) do not match the canonical full-82GB fixture, and no canonical full route ran.
- The canonical Qwen3.5 A3B artifact was absent.
- Production direct-Qwen fault injection does not cover every required load boundary (embedding/completed-layer/final-norm/output publication).
- No deterministic canonical malformed-tool fixture exists.
- No canonical accepted VL fixture exists; the available VL explorations failed.
- No numerical or mutable-state oracle was captured; aggregate G4 promotion is not established.

This evidence does not establish performance, aggregate G4 promotion, vision acceptance, malformed-tool acceptance, Qwen3.8 acceptance, or a clean strict-clippy baseline. The initial continuous-batch failure is retained as an environment/cache packaging event and is distinct from the corrected route pass. The logical cancellation lifecycle is exactly one terminal (`done`); the preceding `aborted` object is an expected correlated nonterminal control acknowledgement. No running daemon or unrelated worktree was modified.
