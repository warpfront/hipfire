# gfx1201 A4C2 landing gate against official `0f6cea0dd`

**PASS; request landing.** Candidate `g12-a4c2` @ `b33819157` (source `44ca584ad`) makes the gfx1201 one-pass two-candidate A4 producer default-on with `HIPFIRE_G12_A4C2=0` restoring RTN. This intentionally changes numerical output and improves WT2 KLD, rather than preserving the exact RTN digest. Model `/home/kaden/qcal/qat/v25/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq`. The isolated gate worktree was built with `cargo build --release -p hipfire-cli -p hipfire-daemon` and `cargo build --release -p hipfire-runtime --example eval_hipfire --features deltanet`. Baseline binaries came from the previously certified `0f6cea0dd` worktree and their digests were pinned by the runner.

## Card-E fresh-process ABBA × 2

`python3 scratch-g12a4gate/run_abba.py`: one discarded fresh-process warmup per arm, then **A B B A A B B A**, every entry a fresh CLI and daemon process. A is official `0f6cea0dd`, B is candidate `b33819157`. Matrix: `--pp 512,8192 --ctx 128 --tg 128 --spec off --runs 3 --warmups 1 --kv-mode fp8 --json`; private arm homes/caches; graph on; card-E `GPU-05f92432f2312a0e`. Every daemon log confirms gfx1201 and `KV cache: Fp8 vmm`.

| Process | Arm | pp512 tok/s | pp8192 tok/s | tg128 tok/s |
|---|---|---:|---:|---:|
| p1 | A | 3716.3 | 4010.9 | 36.47663 |
| p2 | B | 3706.9 | 4003.4 | 36.51076 |
| p3 | B | 3707.7 | 4001.8 | 36.49496 |
| p4 | A | 3685.4 | 3998.1 | 36.50334 |
| p5 | A | 3692.1 | 3995.5 | 36.50205 |
| p6 | B | 3708.6 | 4002.3 | 36.48395 |
| p7 | B | 3706.1 | 4003.0 | 36.48562 |
| p8 | A | 3692.4 | 3995.0 | 36.48541 |
| **A median** | | **3692.25** | **3996.80** | **36.49373** |
| **B median** | | **3707.30** | **4002.65** | **36.49029** |
| **B vs A** | | **+0.408%** | **+0.146%** | **−0.009%** |

**Pass:** pp8192 is not >0.5% slower; all four B rows exceed 99% of the A process-median baseline (lowest B 4001.8 vs 99% ×3996.8 =3956.8); fast decode well within 1%. `abba/summary.json` and per-process raw JSON/logs include all 3 samples and both binary digests. The absolute ~4000 tok/s level varies from the earlier gfx1201 gate's ~3993 result because the comparison was freshly paired; only the contemporaneous A/B delta is a claim.

## Card-D c24 quality

`python3 scratch-g12a4gate/run_wt2.py` on shared card D (`GPU-6109a4cb5f833235`) used `eval_hipfire --kv-mode fp8 --kv-v q8 --scoring-mode prefill --max-chunks 24`, legacy KV, graph off, and `/home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin`. **Candidate KLD 0.076071, `.kldseq` MD5 `5ac9c345e4483b7c317f48e265c351f5`**, identical to the candidate A4C2 reference. This improves official RTN KLD **0.083278 → 0.076071**, intentionally changing the old MD5 `9d0e860f41db992820ebdc9483c0a041`. The evaluator log asserts gfx1201 and legacy fp8 KV; `wt2/summary.json`, `wt2/candidate.log` record the result (generated binary `.kldseq` not checked in).

## Card-E manual serve and card-B guard

The candidate's manually launched `hipfire serve` on E confirmed `KV cache: Fp8 vmm`. `python3 scripts/serve_harness.py --no-spawn --kv fp8 --max-think-tokens 1 --max-tokens 512` returned **5/5 coherent, nonempty, stop-terminated answers**: correct Python merge, correct 210-mile arithmetic, seasons due to axial tilt, a complete short story, and five coding guidelines; no runaway/attractor/retrieval miss. `battery.json` records the answers. On the same server the greedy `benchmarks/prompts/ttft_511.txt` fixture yielded a coherent 135-token clarification of its truncated 511-token merge prompt and stopped normally; `ttft-511.json` records the full response. The manual server was stopped afterwards.

`python3 scripts/guard_gfx1201_baseline.py --output-dir scratch-g12a4gate/guard --home /home/kaden/.hipfire-homes/g12a4gate-b` on guard-only card B (`GPU-e475645fe0200397`) passed the actual fp8 VMM check and **3620 tok/s floor**: three fresh-process pp8192 medians **4007.3 / 3991.5 / 3985.8**, minimum **3985.8**; decode medians 36.506 / 36.463 / 36.492, all within 1% of 36.5. See `guard/summary.json` and per-process samples/logs. Post-run `rocm-smi --showpids` showed no KFD processes.

## gfx11 isolation and deployment caveat

Only `crates/hipfire-config/src/lib.rs`, `crates/rdna-compute/src/feature_flags.rs`, `docs/env-vars.md`, and shared `kernels/src/block_i4_128_quant.hip` change in the candidate code. `git diff --exit-code 0f6cea0dd HEAD -- 'kernels/src/*gfx11*'` is empty. The new A4C2 flag is appended only inside `if arch == "gfx1201"`; the shared kernel's optimized path requires gfx1201 plus two candidates. The candidate's `scratch-a4c2/isa_identity.txt` reports **ISA_IDENTITY_PASS** for all 34 affected JIT sources: gfx1100/gfx1151 instruction-identical with both default two-candidate and eight-candidate builds, and gfx1201 kill-switch arm identical; only the expected gfx1201 two-candidate producer sources differ. Its bitwise oracles and same-binary rocprof are in `scratch-a4c2/REPORT.md` (+1.807 ms for four producers at pp8192).

Deployment caveat from the candidate report: adding the default gfx1201 JIT flag changes cache keys. A packaged gfx1201 blob compiled without it is stale and recompiles when hipcc is available; without hipcc the existing runtime fallback can load the old RTN blob. This is the same existing behavior as gfx11's default A4 flag, but installs expecting guaranteed A4C2 need runtime compilation available.
