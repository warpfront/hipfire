# Long-context and sustained-decode A/B correctness

Compared old exact LRU with the mapped-short-prefill candidate, **both cache=1**. This tests the incremental prefill change; it is not a cache-off versus cache-on experiment. Exact frozen executables are identified in the run receipts and in the previous prefill result manifests. No product source changed during this campaign.

## Result and scope

- **35/35 paired outputs match exactly**, including request/prompt hashes, visible/reasoning/tool text, generation count, context count and finish. The per-request PM4 and transport/DONE/terminal/late-byte checks pass. The verifier's relative regression gate passes.
- Fixed LongBench-v2 hard30: **11/30 versus 11/30** using supplied official gold letters and conservative formatted-final extraction. This is a selected subset under a 512-token, no-thinking budget, not an upstream official full-benchmark score.
- Both arms have the same 13 truncated/unparsed questions and 3 raw attractor flags on numeric utility-array enumerations. Canonical LB harness exitsFAIL on those flags. Every response was already saved by the per-row checkpoint; neither flags nor failures were erased. The flagged texts are structured calculations, but their arithmetic was not thereby certified correct.
- Longdecode: **5/5 versus 5/5 naturally terminate**, with generated counts 12,888, 13,438, 2,585, 3,603, 3,438 (35,952 per arm; 71,904 total). The 16,384 cap is an upper bound, not a claimed actual output length. The unrelated27B/TP2 reference responses were not used as gold.
- Absolute quality is **not all-pass**. The same Snake source passes 15 offline Chromium checks. Tetris has a shared boot error (`clearRect` called on a canvas element); two observations reflect one root cause and nine later checks are blocked. The 1,903-word story satisfies the reviewed length/scenes/dialogue/ending requirements. The KV/radix tutorials contain shared arithmetic/ownership/pseudocode errors, detailed in their review. No generated answer was fixed or retried to obtain a pass.

## Protocol

Both arms use the same Flash-Next monolith/hotset, greedy temperature 0/seed 123, thinking/speculation/decode-share off, normalization off, native FP8/VMM and plain GDN Q8, explicit Q4LM row-order PLE, chunk 4096, **context 65,536 and resident budget 18,000 MiB**. The larger context requires the common lower resident budget; these settings differ from the earlier 16K/20,000 MiB performance campaign. This campaign makes no performance claim. Actual question input lengths are 20,405–30,250 tokens; fixture raw-token counts are not substituted for actual runtime usage.

JSONL was parsed on physical LF, preserving the embedded U+2028 and original prompt bytes/MD5s. Raw request bodies were checked against their canonical request MD5s, fixture bytes, temperature, seed and caps. Responses all have complete per-request receipts. The old/new suite histories match: fiction 16-token warmup, then the same fixed question order. The LB and LD suites use separate resident processes. Four loads were needed after the original baseline LB process closed at its late canonical attractor gate; the 30 baseline questions were not rerun. No allow-incoherent environment override or cap escalation was used.

## Evidence and reproduction limits

`SUMMARY.json` is the compact result. `MANIFEST.json` hashes every member of `evidence.tar.gz`. The archive retains canonical rows/partial rows, source labels, run/command/settings receipts, logs, the relative verifier, all generated outputs, manual requirement reviews, and original HTML/browser checks/screenshots. Partial LB files contain all 30 actual responses and are clearly labelled with the failed canonical verdict.

Input LongBench contexts, prompt lists, actual HTTP request bodies, HOME configs and model/executable bytes are excluded from the public bundle. They remain in `/home/husrcf/Code/hipfireQC` for full wire re-audit. To rerun the complete verifier, use those original fixtures/private request bodies with its CLI; source/binary/fixture identities and wire hashes are retained publicly. The generated HTML and offline browser checks are independently runnable after extracting the archive (adjust the recorded X570/tmp paths and install the recorded Node/Chromium versions). Browser scenarios use explicit state fixtures and the unchanged original algorithms; this is bounded functional evidence, not exhaustive gameplay certification or an objective 5-task gold score.

The verifier’s no-execution field describes that data-only script; the separate browser checks executed reviewed, unchanged generated HTML.
