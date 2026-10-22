# Historical TP2 quality outputs for PR 940

These are historical raw QC receipts from the original beta `66db8cb528444b28751bee98fba2e24cfc024acc` experiment. They are not measurements of the later merge onto beta `88e9f0d6f`. The repository owner explicitly authorized publication of the complete generated answers, reasoning text, and original output JSON on 2026-10-22. Model weights, executable binaries, raw F32 tensors, and benchmark input dataset contexts are excluded.

## Results and exact arms

| Suite | A and B | Result |
|---|---|---|
| LongBench-v2 hard30 | Both head and scratch selectors off versus explicitly on | 30/30 stop and 17/30 correct on both; all answers and reasoning equal |
| Five long-output tasks, 65536-token budget | Head off versus narrow default, scratch off | 5/5 stop on both; all answers and reasoning equal |
| Earlier five-task run, 32768-token budget | Both selectors off versus on | Snake, Tetris and radix reach length on both; these are incomplete finals |
| Greedy serve battery | Head off versus narrow default, scratch off | 5/5 stop on both; all assistant text equal |

Long-task sampling is temperature 1, top-p .95, top-k 20, seed 7, thinking enabled with xhigh effort and no separate thinking cap. LongBench uses max_seq 65536; the completed five-task run uses max_seq 131072. JSONL must be split on physical LF, not Unicode line separators: one LongBench row contains U+2028. Byte equality is a change-regression check, not proof that every task answer is correct. Generated games were not certified by browser execution.

## Files and provenance

The flat `longbench-combined-*-results.jsonl` and `longdecode-head-*-results.jsonl` files are convenient exact copies of the full results. The dated subdirectories additionally preserve the original result, load, identity and summary JSON files, including the earlier length-limited outputs. Neither copy includes the LongBench input request bodies. `battery-*.json` contains the complete native serve battery report. Decode/prefill JSON and memory summaries retain the original timing receipts; no old timings are relabelled as current-head evidence.

`head-default-binary-source.json` identifies the head-only source (`4cc190ea8` plus diff MD5 `c11370bebb69603dfd2607c2126f6a36`, daemon MD5 `ea98ede36ab9c987a1e46cc886d666b6`). `combined-qc-binary-source.json` identifies the earlier combined source (daemon MD5 `6b45820bfe90db9ea265aa607b2a36a8`). The pinned target SHA256 is `3e38ccbae3776470eb5a89344d300e9279d6b9ab6c31fd40ca1758c4f7c6f8ae`.

`archived-evidence-summary.json` is preserved verbatim, including its then-local publication status and original source-match/CI fields. Those fields describe its creation time, not publication status or current-head validation. `numeric-diagnosis.json` and `tp2-default-logits-status.json` preserve the independent TP1/TP2 failure and the separate diagnostic scope. No numerical gate is waived by the equal task outputs.

`manifest.json` indexes each published payload by byte count and SHA256; it excludes itself to avoid a circular digest. The separate [current-head receipts](../current-head/README.md) carry refreshed source/build identities and pending maintainer disposition. This raw-evidence-only commit can be omitted when selecting implementation commits for mainline.
