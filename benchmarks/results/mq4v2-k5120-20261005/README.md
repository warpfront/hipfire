# gfx1100 MQ4V2 K5120 gate/up AR decode

Scope: opt-in specialization of the existing gate/up GEMV for exact gfx1100,
M=17408 per output and K=5120. No weight reformat, requantization or additional
resident weight allocation. Independent of split-KV attention/verifier work.

## GPU1 long decode

W7900 Dual Slot (gfx1100), physical GPU1, HIP 7.15. Qwen3.8-27B MQ4-XT;
Q8 KV, VMM, speculation off, temperature 0, repeat penalty 1.0,
`max_think_tokens=1`, max context 8192, output cap 4096. Every arm starts a
fresh server, waits through a 10-second DPM warmup, generates 128 excluded
warmup tokens from a separate prompt, then measures the same long request.
There is a 15-second idle interval between arms.

| Arm | K5120 flag | Generated tokens | TTFT s | Decode tok/s |
|---|---:|---:|---:|---:|
| A1 | 0 | 4096 | 0.166 | 38.9 |
| B1 | 1 | 4096 | 0.167 | 39.6 |
| B2 | 1 | 4096 | 0.167 | 39.6 |
| A2 | 0 | 4096 | 0.167 | 38.8 |

Off median **38.85**, on median **39.60 tok/s**, **+1.93% (1.0193x)**.
This is a small, fixture-specific observation from two samples per arm,
not a statistically established gain across workloads. The metric is the
daemon-reported decode rate, not request wall time or prefill throughput.

All arms have zero cached prompt tokens, zero reasoning output, nonempty
visible output, clean stream termination and identical decoded text. The
requested story exceeds the output budget, so `finish=length` is expected.
Text parity does not establish token-ID/logit equality or broad answer quality.
Separate prior serving battery and HIP/blob/PM4 parity are independent existing
validation, not conclusions from these four requests. Their original JSON
(`serve-0.json`, `serve-1.json`, `replay-candidate.json`) will accompany the
evidence archive. No per-arm profiler route trace was collected during
the performance test.

## Identity

- Tested production commit: `cb2f84ae3`, plus tests/docs commit `671e88e02`.
- Base beta: `45f1abd1d`.
- Model SHA256: `80e7c624424fd1d363ba86681d3dc1e5ac5534e0e064306a32be204c4843d0f3`.
- CLI MD5: `894739f626fe3bf5cd4e077bf7743a6d`.
- Daemon MD5: `529a278af991815057690ff384cd5846`.
- Prompt MD5: `9f64b7bc79c38aaa65cccc511b167390`.
- Request MD5: `5fcd75a0625638d146903088b2247f4e`.
- SHA256 of JSON-encoded `[reasoning_content, content]` (UTF-8, ensure_ascii=False):
  `a0d1f4d7b9878111ad48620cf49a944da7686ef73c8557d0439190ba9b0696cc`.

## Reproduce

Build the product CLI and daemon with the repository's supported toolchain.
Use an idle gfx1100 device; the engine GPU lease is authoritative. K5120 is
default-off and a compiler-less install needs a pack containing its new symbol.

```bash
GPU_ID=1 MODEL=/path/to/qwen3.8-27b.mq4-xt OUT=/tmp/k5120-abba-new \
  bash benchmarks/scripts/mq4v2_k5120_abba.sh
```

The output directory must not already exist. The scripts reuse the official
`serve_harness.py`, explicitly disable reasoning and speculation, save per-arm
responses/configuration/warmup and binary/model/prompt/script hashes. Compare
`decode_tok_s` in A1/A2 against B1/B2 only after checking 4096 generated tokens,
zero cached prompt tokens, matching request/output text, and clean termination.
Do not mix results from different physical GPUs, model hashes or reasoning modes.

The recorded run used the local precursor of these portable scripts with the
same prompt/request/warmup schedule. Portable scripts add stronger warmup
assertions, signal cleanup and script hashing; they are not claimed to have
produced the historical run. The attachment retains precursor snapshots.

## Evidence and CI

Raw transcripts and logs will be delivered separately as a PR attachment, following
`benchmarks/results/RETENTION.md`; they are not committed here. The attachment
excludes weights, caches, daemon homes, full environments and unrelated research.
Local CI results:

| Check | Result |
|---|---|
| Official `scripts/no-gpu-ci.sh` | Passed with `PYTHONSAFEPATH=1 PYTHONPATH="$PWD"`; Python suite 562 passed |
| `cargo test --locked --lib --workspace` (GPUs hidden) | Passed |
| Crate maps | Passed after refreshing the two modified crates' generated maps |
| Scratch-growth check | Passed |
| Ratchet-diff against beta | Passed; no thresholds raised |
| Leanup ratchets | Fails identically on clean beta: daemon_lines=5392 exceeds 5377 |
| Released changelog check | Fails identically on clean beta: historical v0.1.6/v0.1.7-alpha.2 sections differ |

The initial default-Conda Python run had 21 failures because hw-gate's
`select.py` shadowed Python's standard-library `select`; the complete safe-path
rerun passed without source changes. Both attempts are retained in the raw
evidence. The two structural baseline failures were reproduced in a clean
`45f1abd1d` worktree; they are not claimed green and no policy was weakened.
This local evidence does not replace the required remote CI jobs or maintainer
review. Attachment publication is pending PR creation.
