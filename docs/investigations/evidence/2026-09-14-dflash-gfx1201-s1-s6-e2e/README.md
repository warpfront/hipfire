# gfx1201 DFlash S1-S6 E2E evidence

## Scope

- Tested code commit: `2814cceb9` (`perf(dflash): port fa batch fusions to gfx1201`)
- Host: `X570`
- GPU: AMD Radeon AI PRO R9700 (`gfx1201`, 34.2 GB VRAM)
- HIP runtime: `7.14`
- Target KV: Q8; initial serving checks use contiguous KV, PR LongBench/long-decode matrices use VMM
- Candidate: S1-S6 defaults enabled on the validated gfx1201 path
- Control: all six fusions disabled with their existing `*_OFF=1` environment controls

The control disables:

```text
HIPFIRE_DN_SNAPSHOT_BULK_OFF=1
HIPFIRE_HIDDEN_SCATTER_FUSE_OFF=1
HIPFIRE_MQ_F16_PROJECTION_OFF=1
HIPFIRE_MQ_F16_RESIDUAL_OFF=1
HIPFIRE_GDN_PRE_FUSE_OFF=1
HIPFIRE_FA_BATCH_FUSE_OFF=1
```

## Production E2E results

### Qwen3.5-9B MQ4 + matched DFlash draft

The five-genre battery completed under candidate, control, and ordinary AR. Candidate and control DFlash transcripts were byte-identical. A multi-turn chain also completed with real prefix-cache hits (`456` and `601` cached tokens), and its candidate/control transcript was byte-identical.

An 8K NIAH fixture produced a 5,489-token request and recalled `mauve-velociraptor-7741` under both arms (`recall=1/1`, byte-identical output).

For the fixed 3,798-token prompt and 512-token response, after excluding each arm's first JIT-contaminated run:

| Metric | S1-S6 off | S1-S6 on | Delta |
|---|---:|---:|---:|
| Prefill median | 2,034.7 tok/s | 2,094.5 tok/s | +2.94% |
| Decode median | 65.5 tok/s | 67.9 tok/s | +3.66% |

Warm samples:

```text
off prefill: 2040.7, 2034.7, 2031.7 tok/s
on  prefill: 2107.7, 2089.4, 2094.5 tok/s
off decode : 65.6, 65.5, 65.4 tok/s
on  decode : 67.9, 67.9, 67.8 tok/s
```

### Qwen3.8-27B MQ4V2 + matched DFlash draft

Qwen3.8 uses an effort-native prompt contract. The valid no-think request therefore sets both `reasoning_effort=none` and `max_think_tokens=1`; using only the named `thinking=off` budget is rejected because it may leave an open think span at the generation cap.

The fixed 3,798-token prompt generated 512 tokens under both arms with byte-identical output and DFlash `tau=2.15`. A five-request same-daemon run then exercised prefix-cache restore: requests 2-5 each reused 2,304 prompt tokens and produced candidate/control transcripts that were byte-identical turn-for-turn.

Steady cached requests 2-5:

| Metric | S1-S6 off | S1-S6 on | Delta |
|---|---:|---:|---:|
| Prefill median | 646.3 tok/s | 656.5 tok/s | +1.59% |
| Decode median | 32.05 tok/s | 33.5 tok/s | +4.52% |

Samples:

```text
off cached prefill: 643.5, 646.1, 646.4, 647.2 tok/s
on  cached prefill: 661.4, 656.9, 656.1, 655.6 tok/s
off cached decode : 30.7, 32.1, 32.0, 32.1 tok/s
on  cached decode : 33.6, 33.5, 33.5, 33.5 tok/s
```

Repeated identical requests change the DFlash operating point after prefix-cache restore (`tau=2.15` on the cold-prefix request and `tau=0.91` on cached requests). These cached-request numbers must not be mixed with cold-prefix decode measurements.

### Qwen3.8-27B LongBench hard30 A/B

The committed 30-question, approximately 32K-context fixture was run once per arm with no thinking, greedy sampling, a 768-token cap, Q8 VMM KV, and the matched DFlash draft. The only difference between the arms was the six `*_OFF=1` controls listed above.

| Metric | S1-S6 off | S1-S6 on | Delta |
|---|---:|---:|---:|
| Completed | 30/30 | 30/30 | equal |
| Multiple-choice accuracy | 14/30 | 14/30 | equal |
| Exact output parity | - | 30/30 | exact |
| Prefill median | 470.05 tok/s | 470.20 tok/s | +0.03% |
| Decode median | 75.10 tok/s | 76.75 tok/s | +2.20% |
| Wall median | 59.5466 s | 59.4835 s | +0.11% |

Most answers terminate after roughly eight generated tokens, so this matrix is primarily correctness and long-context coverage, not the main decode-throughput claim.

### Qwen3.8-27B long-decode A/B

Five heterogeneous long-output tasks were run with the same target, draft, KV mode, and S1-S6 control boundary. All five candidate outputs were byte-identical to control.

| Case | Tokens | Off decode | On decode | Delta | Off wall | On wall |
|---|---:|---:|---:|---:|---:|---:|
| `kv-cache-guide` | 1,536 | 74.7 tok/s | 78.2 tok/s | +4.69% | 20.762 s | 19.831 s |
| `sglang-guide` | 1,536 | 58.3 tok/s | 61.1 tok/s | +4.80% | 26.540 s | 25.301 s |
| `open-story` | 1,214 | 32.4 tok/s | 34.1 tok/s | +5.25% | 37.699 s | 35.758 s |
| `snake-python` | 2,212 | 128.0 tok/s | 134.5 tok/s | +5.08% | 17.479 s | 16.638 s |
| `tetris-python` | 4,846 | 134.7 tok/s | 141.2 tok/s | +4.83% | 36.246 s | 34.585 s |
| **Median** | - | **74.7 tok/s** | **78.2 tok/s** | **+4.69%** | **26.540 s** | **25.301 s** |

Short-prompt prefill was neutral within noise (`646.7` versus `643.7 tok/s`, `-0.46%`). This matches the implementation boundary: S1-S6 are DFlash ChainVerify/commit launch fusions and do not replace ordinary AR execution.

## PR-template validation

- `BASE_REF=upstream/beta ./scripts/fmt-changed.sh`: pass, no resulting diff.
- `cargo build --release`: pass.
- `HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 cargo test --lib --workspace`: pass.
- `scripts/serve_harness.py --mode battery`, Qwen3.8-27B target plus matched DFlash draft: 5/5 turns completed, no empty output, attractor, or retrieval miss. One reasoning fixture reached the explicit 256-token generation cap.
- `scripts/speed-gate.sh --verbose` on gfx1201: 10 passed, 5 regressed, 0 skipped. All three DFlash rows exceeded their locked floors: 27B LRU `+49.5%`, 27B merge-sort `+49.2%`, and 9B merge-sort `+90.7%`. The five below-floor rows are ordinary MQ4 prefill rows outside this DFlash-only route (`0.8B pp32/pp128`, `4B pp32/pp128`, and `9B pp32`), so the global speed-gate checkbox remains unchecked rather than being represented as a branch pass.
- `./scripts/no-gpu-ci.sh`: Rust/no-GPU checks pass. Its Python phase reports five pre-existing `tests/test_mq4c_repack.py` failures because the current beta test expects `HfqmError`, `main`, and `parse_hfqm_index` exports absent from the current beta tool. Both the test and tool are byte-unchanged by this branch.

Artifacts:

```text
target sha256: 9f91556f7e0431a077d03756a7102d0154108757289e6e5fe9a2d204c0c9eeb7
draft  sha256: d0a74a232a0e2166d889f823e91e0fbf778d21dd9668d7de055cdecb065401bc
daemon md5:    b0572cc8d3f6c163aa86e2f664cc9a7c
dataset sha256: 839a19be0b3b1c801a0ca58d388996faff34704e34dd196d54346adf17a1dce9
```

## State-oracle boundary

`scripts/redline_daemon_harness.py --dflash-verify-shadow --pm4` currently fails on gfx1201 for both candidate and the S1-S6-disabled control. Both recorded-HIP and PM4 states diverge from ordinary HIP after the initial position. The control reported 347 failed checks versus 333 for the candidate. This therefore does not isolate a regression in the S1-S6 port; it is an upstream gfx1201 shadow-harness/capture baseline issue that needs separate investigation.

The production serving checks above remain positive: no crash, no empty response under the valid Qwen3.8 reasoning contract, correct NIAH recall, real cache reuse, and byte-identical candidate/control transcripts.

## Artifacts

`raw/` contains the harness JSON, A/B summaries, generated outputs, and daemon logs. HIPRTC code objects and per-home caches are intentionally excluded.
