# gfx1201 MQ4V2 K5120 gate/up: opt-in evidence

This is a small exact-shape specialization, not a new weight format or a general gfx12 performance claim. R9700 / gfx1201, ROCm 7.14.60850, Qwen3.8-27B MQ4-XT (qt44), artifact SHA256 `80e7c624424fd1d363ba86681d3dc1e5ac5534e0e064306a32be204c4843d0f3` (14,987,185,152 bytes). No extra resident weight buffers. The existing DEV/RT load policy, wave32, block32 and arithmetic are unchanged.

## Configuration

```bash
hipfire config set kernel.gfx12_mq4v2_gateup_k5120 true
# Or one process:
HIPFIRE_GFX12_MQ4V2_GATEUP_K5120=1 hipfire serve ...
```

Default off, exact gfx1201 and gate_m=up_m=17408, K=5120 only. Restart the daemon after changing the setting. The upstream gfx1100 default-on `kernel.mq4v2_gateup_k5120` is untouched. Compiler-less installations need the corresponding gfx1201 kernel-pack symbol.

## Initial serving ABBA

Measured on beta `5d172b6639b24a67505bd4b0453ad97278f66799` plus the initial patch. This pre-rebase experiment used the original shared opt-in flag; the final PR has an independent gfx12 flag to avoid inheriting the subsequent upstream gfx1100 default-on change. See final validation below rather than assuming old routing evidence covers the new setting.

AR, no speculation, Q8 requested, VMM observed, temperature 0, thinking off, 126 prompt tokens, 4096 output tokens; fresh server per arm, excluded 128-token warmup, 10s DPM warmup, 15s between arms. Prompt: `benchmarks/prompts/mq4v2_k5120_long_ar.txt`. Raw official serving-harness rows are in `initial-abba/`; `summary.json` contains hashes and aggregation. `kv_mode` was unobserved in harness fields, not independently certified; no legacy KV backend was reported.

| Arm | Specialization | Decode tok/s | TTFT s | Wall s |
| --- | ---: | ---: | ---: | ---: |
| A1 | off | 40.0 | 0.103 | 102.587 |
| B1 | on | 40.2 | 0.102 | 101.981 |
| B2 | on | 40.2 | 0.101 | 101.904 |
| A2 | off | 40.1 | 0.102 | 102.298 |

Median 40.05 -> 40.20 tok/s, **+0.3745%**. Only two samples per arm and daemon rates rounded to 0.1 tok/s: this does not establish statistical significance. All four outputs have identical SHA256 `078e1ea2e2d9f2a58e9c34fa8e44ec7705a71a1205f707567f25fecea2229131`, with no stream errors or attractors. Intentional `finish=length` is not a completed-story quality claim. TTFT is not a prefill speedup claim.

## Local gate/up diagnostic

The probe loads the actual archived generic/specialized HSACOs, not recompiled kernels. Synthetic MQ4V2 weights, three input seeds and sixteen slots: 1,671,168 finite output values were byte-identical. Ring 1 repeats 94,699,520 bytes; ring 16 rotates 1,515,192,320 bytes. Neither is asserted to fit cache. Each mode excludes 5s warmup then runs ten alternating ABBA/BAAB blocks, 128 graph calls per event sample, 20 samples per arm.

| Ring | Generic median us | K5120 median us | Local speedup | Positive paired blocks |
| --- | ---: | ---: | ---: | ---: |
| 1 | 157.484 | 156.398 | +0.694% | 9/10 |
| 16 | 157.314 | 156.439 | +0.560% | 10/10 |

Raw data: `local-probe.log`; analyzer: `benchmarks/scripts/mq4v2_k5120_local_report.py`. Event times include graph gaps. Nominal weight bytes/time is not measured DRAM bandwidth. Ring-16 projects only 0.056 ms saved over 64 calls, not a measured token-level saving. This supports a small local effect, not a hidden large kernel win.

Actual object audit: VGPR 76 -> 76, SGPR 22 -> 19, scratch 0 -> 0. Generic object SHA256 `8c18c9dd3785fe24ea9d591a0a88a0d2de84c46ae9c2318e7e3b19ed14e6537e`; specialized `39a7c19b52c246c6fbda25983c476f381b143c245de1ac8c337bb7f404775b63`.

## Correctness and limitations

Initial PP64 plus fixed teacher-forced tokens 42,43,44,45: 993,280 FP32 logits byte-identical, hash in `logits.sha256`. Both official Redline replay runs passed four-position HIP/PM4 parity, including GDN state. Every capture had 64 expected gate/up symbols and none from the other arm.

The initial full GPU rdna-compute serial suite had 438 passed / 2 failed / 13 ignored. The two GDN Q8 failures (`gdn_q8_state_tracks_f32_on_every_route`, `gdn_q8_verify_matches_single_row`) also reproduced on unmodified beta 5d172b663; see `base-gdn-q8.log`. No GDN implementation changes are included. This is not a globally green GPU-suite claim.

## Reproduce

Build `hipfire-cli`, `hipfire-daemon`, and the `saddle-lab` example `probe_k5120_logits` in release mode. On an idle gfx1201 with the pinned artifact:

```bash
MODEL=/path/to/qwen3.8-27b.mq4-xt \
OUT=tmp/gfx1201-k5120-new GPU_ID=0 \
bash benchmarks/scripts/mq4v2_k5120_gfx1201_validate.sh
```

This takes the cooperative GPU lock, checks logits and capture routing, then invokes the existing AR ABBA script with the independent gfx12 flag. Keep all raw outputs. ISA capture helper and standalone local probe are under `benchmarks/scripts/mq4v2_k5120_*`.

## Final rebased validation

Completed on beta `7878272585a53d5bd46178b2828f902ce9483331` plus commit `f3039d135`, using the independent gfx12 flag. The remote experiment checkout is a source overlay: its recorded git HEAD still points at the earlier base; `rebased/working.diff.gz` records that overlay. The source commit named here is the tested source, not that stale checkout HEAD. Raw successful validation is in `rebased/` (gzip-compressed capture JSON and CI log; use `gzip -dc` to inspect).

| Arm | Specialization | Decode tok/s | TTFT s | Wall s |
| --- | ---: | ---: | ---: | ---: |
| A1 | off | 40.1 | 0.108 | 102.209 |
| B1 | on | 40.2 | 0.103 | 101.921 |
| B2 | on | 40.2 | 0.103 | 102.080 |
| A2 | off | 40.1 | 0.102 | 102.296 |

Median **40.10 -> 40.20 tok/s (+0.2494%)**. Do not pool this with the initial run across different source baselines. All four output hashes and all logits match the initial run. Both replay reports pass with the expected 64 routed symbols. A no-GPU CPU test run overlapped the early serving test; no other GPU workload was launched by this experiment. This is another reason not to overinterpret a sub-percent E2E difference.

The final serialized `scripts/no-gpu-ci.sh` passed after serving stopped, including Rust checks/tests, Python tests, config/env lifecycle and no-process-leak checks. The first attempt lacked pytest/numpy; the second passed tests but detected the intentionally concurrent benchmark daemon at its final leak check. Dependencies were installed in an isolated experiment venv; the final rerun passed without bypassing checks. Targeted K5120 tests: 6 passed; config: 107 passed. Crate maps and `git diff --check` also passed. Global speed-gate and full-workspace GPU tests are not claimed.
