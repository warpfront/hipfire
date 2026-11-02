# Asym3-Givens D256 current-master validation

Hardware and software:

- AMD Radeon Pro W7900, exact `gfx1100`
- ROCm 7.14
- Qwen3.6-27B MQ4
- Asym3-Givens K and Q8 V cache
- PP8192, batch 1, no speculative decoding
- branch base: upstream master `aaf5e3211`

Command:

```bash
BUILD=0 GPU_ID=0 KV_MODE=asym3 PREFILL=8192 RUNS=5 SLEEP_SECS=10 \
  ./scripts/bench_ck_q8_prefill_ab.sh
```

Results after two warmups per arm:

| Arm | Samples (tok/s) | Median |
| --- | --- | ---: |
| Native | `271.1, 582.2, 576.7, 572.5, 571.7` | `572.5` |
| CK | `806.4, 799.5, 797.4, 795.0, 794.3` | `797.4` |

The first native sample contains one-time JIT contamination; retaining it does
not change the five-run median. CK improves the median by `39.28%`. Both arms
produce next token `248046`.

The rocprof run records both an untimed JIT warmup pass and one profiled pass.
Dividing CK-specific aggregate dispatch times by two gives:

| Component | Time per PP8192 pass |
| --- | ---: |
| CK D256 FMHA | `283.6 ms` |
| Asym3 K decode | `28.5 ms` |
| Q8 V decode | `38.2 ms` |
| F16 output to F32 | `5.2 ms` |
| F32 Q Givens transform to F16 | `2.6 ms` |
| Total CK chain | `358.1 ms` |

This is about `3.3%` of the `10.8 s` profiled application wall after CK is
enabled. It bounds the benefit available from additional staging or CK-tile
work and keeps packed-MQ4 performance claims outside this PR.

## Repository checks

- `cargo build --release --workspace --features deltanet`: passed.
- `cargo check -p rdna-compute --features flash-attn-ck`: passed.
- `cargo test -p rdna-compute --features flash-attn-ck flash_attn_ck --lib`:
  passed.
- `cargo check -p hipfire-dispatch --features flash-attn-ck`: passed.
- `cargo check -p hipfire-dispatch`: passed without the optional feature.
- A clean sidecar rebuild and the gfx1100 GPU smoke passed. The Asym3-Givens
  D256 cell had `max_abs=6.110966e-05` and `mean_abs=1.009769e-05`; unsupported
  Givens D512 and FWHT cells remained recognized but fail-closed.

`cargo test --lib --workspace --features deltanet` and `./scripts/no-gpu-ci.sh`
both reached one existing failure:

```text
sampling::slot_sample_tests::a_zero_seed_never_reaches_the_xorshift_dead_state
assertion `left != right` failed: left=0, right=0
```

The exact test fails identically in a clean worktree at
`upstream/master@aaf5e3211`, so this is not introduced by the CK change.

`tools.change_gate` selected eight routes against the exact upstream base. Five
passed. Its three failures were: the same upstream sampling test; a Qwen3.5 4B
serve-battery configuration rejected before model startup because its thinking
budget exceeded `max_tokens`; and the locked gfx1100 decode speed floor. The
last item was also reproduced on clean upstream master on this W7900:

| Tree | PP32 prefill | Decode |
| --- | ---: | ---: |
| PR branch | `1204.8 tok/s` | `140.2 tok/s` |
| Clean `upstream/master@aaf5e3211` | `1169.2 tok/s` | `140.8 tok/s` |

Both decode results are below the shared gfx1100 floor of `169.1 tok/s`, while
the PR and clean-master measurements agree within `0.5%`. The locked floor does
not distinguish Radeon Pro W7900 from RX 7900 XTX captures, so this result is
reported as a machine-baseline mismatch rather than a CK regression.

## LongBench boundary diagnostic

The 30-case hard set with a 768-token output cap completed without runtime
errors. Case `longbench-16` was the only parsed-answer asymmetry: native emitted
`C`, while CK reached the 768-token cap without a final choice. A CK-only rerun
isolated the cause:

| Output cap | Generated | Finish | Parsed choice |
| ---: | ---: | --- | --- |
| `768` | `768` | length | none |
| `2048` | `900` | stop | `C` |

The longer CK run therefore reaches the same `C` choice as native. The gold
answer is `A`, so this is a truncation diagnosis rather than a correctness gain.
