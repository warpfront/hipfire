# Exact expert LRU: mapped short-prefill qualification

This follows the frozen Draft PR #1022 implementation, keeping beta base `6c99bef7f8fcb966c4104aab2583e894f20c9d10`. Tested candidate is `db5abf9b5` plus the exact patch in `candidate-source.patch`, hash bound by FREEZE/runtime manifests. The final product commit folds in that patch without changing runtime source; only changelog wording and evidence are added afterward. Old source remains available under checkpoint/flash-next-lru-before-prefill-20261029.

## Behavior

Old LRU restored the entire roughly 10000 MiB donor tail whenever a dirty decode cache entered prefill. The candidate reuses existing immutable BorrowPrefill(0) tables for dirty short tiles: core experts remain on device, donors and cold experts read immutable mapped host bytes. Neither mutable decode tables nor donor slots are read, so pending fills and LRU metadata remain valid. G2 staging and retained decode replay are excluded from that short view. Pristine and staging-width tiles keep the original Resident path; long prefill drains fills and restores the donor tail. A clean short remainder of a long prefill stays Resident.

No expert payload or GPU table allocation is added. The third runtime adds CPU metadata; admission's conservative host metadata reserve rises from 64 to 128 MiB. This is a reserve, not a measured host peak. Existing owner tests prove the immutable read spans are disjoint from every donor slot, with Resident as the intersecting negative control. Generation/guard/mutable admission and the routed top-10 selections are unchanged.

## Results

`prefill-comparison-receipts.json`: PASS, all_pass=true, no missing evidence or forced longer-cap retry. Seven small CPU positive/negative controls check that verifier's failure behavior.

| Metric | Old LRU | Mapped short-prefill LRU | Change |
|---|---:|---:|---:|
| Short battery P, median of 3 fresh processes/arm | 53.09 tok/s | 81.83 tok/s | +54.14% |
| Battery total prefill | 3.5414 s | 2.2975 s | -35.12% time |
| Battery weighted D | 34.92 tok/s | 34.70 tok/s | -0.63% |
| Battery total request wall time | 23.915 s | 22.803 s | -4.65% time |
| 5103-token prefill, single observation/arm | 1488.3 tok/s | 1503.7 tok/s | +1.03% |
| 1024-token decode, single observation/arm | 38.1 tok/s | 38.5 tok/s | +1.05% |

The primary battery contains five byte-identical prompts (188 input tokens, 711 generated tokens per cell), with the same five 16-token warmups in each fresh server. Process order: old1, new1, new2, old2, old3, new3. Rates are reconstructed from harness values rounded to 0.1 tok/s; request wall times are rounded to 0.001 s. New D range 34.60–34.81, old 34.71–35.06. No statistically significant decode change is claimed. The single-observation long metrics are scope checks, not three-process speed claims.

Both arms enable exact cache with tail 50%, inflight 64, resident budget 20000 MiB, chunk 4096, max_seq 16384, explicit Q4LM row-order PLE, greedy seed123, thinking/speculation/decode-share off, PCIe execution and a shared content-keyed kernel cache. The actual stage threshold logged here is 2048 rows; there is no new threshold knob or sweep. Each first server continues from battery through chain, dirty128 → long-prefill5103, and longdecode1024 in the same process. No optional session was retried in this comparison.

The old/new five-prompt battery, chain, staged-prefill and long-decode assistant output, prompt hashes and request hashes match exactly. Request IDs join to PM4 markers; terminal/DONE/late-byte/empty/attractor checks pass. Before the long request, the candidate logs 15 admitted mapped-prefill calls/568 rows and zero bulk restorations. The long request increments restoration exactly once by 10,486,579,200 bytes. These are logical restore-byte counters, not total PCIe traffic: mapped prefill still reads some expert weights over PCIe. Stage execution itself has no explicit per-request counter in the logs and remains marked UNOBSERVED by the verifier.

The remaining short-prefill cost is visible: the earlier cache-off reference measured about 111.9 tok/s. That historical reference uses a different binary/source receipt and is not a contemporaneous three-arm comparison. The new path materially reduces the earlier regression, but does not eliminate the extra host reads or establish a default-on win for all prompts/models.

## Correctness and scope

- 15 focused cache/config/placement/short-prefill policy tests pass. 79 decode-share regression tests pass (two repeat owner tests from the focused filter). Relevant rustfmt/map/env/static checks pass. Full local workspace lib tests and a full workspace release build are not claimed.
- Default-feature CLI/daemon release build passes with warnings. Runtime manifests identify the copied old and new executables by SHA256/MD5, model stat and the existing verified bound-hotset identity; the 125 GB model is not rehashed.
- New Redline report passes: native FP8/VMM, plain Q8 GDN, 16 distinct reported positions 513–528 with ordinary HIP/recorded HIP/PM4 byte parity. Three captures are stable with 1149 typed launches/31 contracts. The pre-submit boundary refusal recovers to clean HIP exactly. This is not a kernel-fault injection or live Q8Ef GPU qualification.
- The older optional baseline session failure is preserved in the sibling fn-expert-lru-overall-20261029 directory. This new matrix deliberately does not retry that session or relabel its failed quality gate.
- 100 ms memory traces are device-global samples covering startup and delayed prior-server release, not per-process allocator peaks. No clean peak-memory delta is inferred from them. The host mirror remains 10,486,579,200 bytes; no new model-weight storage is claimed.

## Reproduction and identity

Runners are retained as run with X570 paths: adjust ROOT/OUT/binary constants for another checkout and acquire scripts/gpu-lock.sh. run-prefill-comparison.py runs canonical serve_harness with same-process phases, preserves startup cleanup and all canonical gates, and stops further model loads if the first pair fails the preset gain check. run-redline-new.py records its exact environment in redline-new-trial01/run.json. No parameter search occurred.

Decompress serve.log.gz files in a working copy before running verify-prefill-comparison.py --root <result-dir>; it only reads retained inputs and writes a report, without loading a model. All retained artifacts' original and stored hashes are in ARTIFACTS.json. Compression is gzip-only; HOME configs, executable/model bytes and credentials are excluded. The older source/binary receipts remain historical evidence, separately labelled.

The standalone metadata sizing probe is archived byte-for-byte as `metadata/metadata-layout.rs.gz`; it is evidence source rather than a crate target.
